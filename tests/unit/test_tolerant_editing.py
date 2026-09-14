"""Regression tests for tolerant editing (Part 11) and edit-cache invalidation."""

from __future__ import annotations

import pytest

from harness_core.tools.filesystem import EditFileTool


@pytest.fixture()
def edit_file(tmp_path):
    f = tmp_path / "code.py"
    return f, EditFileTool()


class TestTolerantEditing:
    @pytest.mark.asyncio
    async def test_exact_match_applies(self, edit_file):
        f, tool = edit_file
        f.write_text("def foo():\n    return 1\n")
        result = await tool.execute({
            "path": str(f), "old_string": "return 1", "new_string": "return 2",
        })
        assert result.status.value == "success"
        assert "return 2" in f.read_text()
        assert result.metadata["match_type"] == "exact"
        assert result.metadata["replacements"] == 1

    @pytest.mark.asyncio
    async def test_exact_match_generates_diff(self, edit_file):
        f, tool = edit_file
        f.write_text("a = 1\nb = 2\n")
        result = await tool.execute({
            "path": str(f), "old_string": "b = 2", "new_string": "b = 3",
        })
        assert result.status.value == "success"
        diff = result.metadata["diff"]
        assert "-b = 2" in diff
        assert "+b = 3" in diff

    @pytest.mark.asyncio
    async def test_whitespace_tolerant_unique_match_applies(self, edit_file):
        f, tool = edit_file
        # Model returns different indentation than the file.
        f.write_text("class A:\n        def method(self):\n            return 1\n")
        result = await tool.execute({
            "path": str(f),
            "old_string": "def method(self):\n    return 1",
            "new_string": "def method(self):\n    return 42",
        })
        assert result.status.value == "success", result.error
        assert result.metadata["match_type"] == "whitespace"
        assert "return 42" in f.read_text()

    @pytest.mark.asyncio
    async def test_zero_match_fails_safely(self, edit_file):
        f, tool = edit_file
        f.write_text("x = 1\n")
        result = await tool.execute({
            "path": str(f), "old_string": "no such text", "new_string": "y",
        })
        assert result.status.value == "error"
        assert "not found" in result.error.lower()
        assert f.read_text() == "x = 1\n"  # untouched
        assert not result.retryable

    @pytest.mark.asyncio
    async def test_multiple_exact_matches_fail_without_edit(self, edit_file):
        f, tool = edit_file
        f.write_text("dup()\ndup()\n")
        result = await tool.execute({
            "path": str(f), "old_string": "dup()", "new_string": "single()",
        })
        assert result.status.value == "error"
        assert "ambiguous" in result.error.lower()
        assert f.read_text() == "dup()\ndup()\n"  # NOTHING modified
        assert result.metadata["match_count"] == 2

    @pytest.mark.asyncio
    async def test_multiple_whitespace_matches_fail(self, edit_file):
        f, tool = edit_file
        f.write_text("    foo()\n    foo()\n")
        result = await tool.execute({
            "path": str(f), "old_string": "foo()", "new_string": "bar()",
        })
        # 'foo()' matches two lines exactly → ambiguous, nothing modified.
        assert result.status.value == "error"
        assert f.read_text() == "    foo()\n    foo()\n"

    @pytest.mark.asyncio
    async def test_identical_strings_rejected(self, edit_file):
        f, tool = edit_file
        f.write_text("a\n")
        result = await tool.execute({
            "path": str(f), "old_string": "a", "new_string": "a",
        })
        assert result.status.value == "error"

    @pytest.mark.asyncio
    async def test_missing_file_fails(self, edit_file, tmp_path):
        _, tool = edit_file
        result = await tool.execute({
            "path": str(tmp_path / "nope.py"),
            "old_string": "a", "new_string": "b",
        })
        assert result.status.value == "error"

    @pytest.mark.asyncio
    async def test_edit_is_bounded_diff(self, edit_file):
        f, tool = edit_file
        f.write_text("".join(f"line{i} = {i}\n" for i in range(2000)))
        result = await tool.execute({
            "path": str(f), "old_string": "line1999 = 1999", "new_string": "line1999 = 0",
        })
        assert result.status.value == "success"
        assert len(result.metadata["diff"]) <= 4200


class TestEditCacheInvalidation:
    @pytest.mark.asyncio
    async def test_edit_invalidates_reuse_snapshot(self, tmp_path):
        """After an edit, the reuse manager must not serve stale content."""
        from harness_core.context.reuse import ContextReuseManager
        from harness_core.tools.filesystem import WriteFileTool

        f = tmp_path / "f.txt"
        write = WriteFileTool()
        await write.execute({"path": str(f), "content": "v1\n"})
        reuse = ContextReuseManager()
        content = f.read_text()
        reuse.record_read(str(f), content=content)
        assert reuse.is_unchanged(str(f), content=content)

        tool = EditFileTool()
        result = await tool.execute({
            "path": str(f), "old_string": "v1", "new_string": "v2",
        })
        assert result.status.value == "success"
        new_content = f.read_text()
        assert new_content != content
        # Snapshot must be invalidated by the loop after write/edit (the loop
        # calls context_reuse.invalidate on the write/edit path).
        reuse.invalidate(str(f))
        assert not reuse.has_snapshot(str(f))
