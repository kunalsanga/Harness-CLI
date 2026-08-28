"""Unit tests for the tool system."""

import pytest
import tempfile
from pathlib import Path

from harness_core.agent.types import ToolResultStatus
from harness_core.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListFilesTool
from harness_core.tools.search import GlobTool, GrepTool


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestReadFileTool:
    """Tests for ReadFileTool."""

    @pytest.mark.asyncio
    async def test_read_existing_file(self, tmp_dir: Path):
        test_file = tmp_dir / "test.txt"
        test_file.write_text("hello world\nline 2\nline 3")

        tool = ReadFileTool()
        result = await tool.execute({"path": str(test_file)})

        assert result.status == ToolResultStatus.SUCCESS
        assert "hello world" in result.output
        assert "line 2" in result.output

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_dir: Path):
        tool = ReadFileTool()
        result = await tool.execute({"path": str(tmp_dir / "missing.txt")})

        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_read_with_offset(self, tmp_dir: Path):
        test_file = tmp_dir / "lines.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5")

        tool = ReadFileTool()
        result = await tool.execute({"path": str(test_file), "offset": 3})

        assert result.status == ToolResultStatus.SUCCESS
        assert "line 3" in result.output
        assert "line 1" not in result.output

    @pytest.mark.asyncio
    async def test_read_with_limit(self, tmp_dir: Path):
        test_file = tmp_dir / "lines.txt"
        test_file.write_text("line 1\nline 2\nline 3\nline 4\nline 5")

        tool = ReadFileTool()
        result = await tool.execute({"path": str(test_file), "limit": 2})

        assert result.status == ToolResultStatus.SUCCESS
        assert "line 1" in result.output
        assert "line 5" not in result.output


class TestWriteFileTool:
    """Tests for WriteFileTool."""

    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_dir: Path):
        tool = WriteFileTool()
        result = await tool.execute(
            {"path": str(tmp_dir / "new.txt"), "content": "hello"}
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert (tmp_dir / "new.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_overwrite_file(self, tmp_dir: Path):
        test_file = tmp_dir / "existing.txt"
        test_file.write_text("old content")

        tool = WriteFileTool()
        result = await tool.execute(
            {"path": str(test_file), "content": "new content"}
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert test_file.read_text() == "new content"

    @pytest.mark.asyncio
    async def test_write_creates_directories(self, tmp_dir: Path):
        tool = WriteFileTool()
        result = await tool.execute(
            {"path": str(tmp_dir / "a" / "b" / "c.txt"), "content": "nested"}
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert (tmp_dir / "a" / "b" / "c.txt").read_text() == "nested"


class TestEditFileTool:
    """Tests for EditFileTool."""

    @pytest.mark.asyncio
    async def test_edit_file(self, tmp_dir: Path):
        test_file = tmp_dir / "edit.txt"
        test_file.write_text("hello world")

        tool = EditFileTool()
        result = await tool.execute(
            {
                "path": str(test_file),
                "old_string": "world",
                "new_string": "python",
            }
        )

        assert result.status == ToolResultStatus.SUCCESS
        assert test_file.read_text() == "hello python"

    @pytest.mark.asyncio
    async def test_edit_nonexistent_string(self, tmp_dir: Path):
        test_file = tmp_dir / "edit.txt"
        test_file.write_text("hello world")

        tool = EditFileTool()
        result = await tool.execute(
            {
                "path": str(test_file),
                "old_string": "missing",
                "new_string": "replaced",
            }
        )

        assert result.status == ToolResultStatus.ERROR


class TestListFilesTool:
    """Tests for ListFilesTool."""

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_dir: Path):
        (tmp_dir / "a.txt").write_text("a")
        (tmp_dir / "b.txt").write_text("b")

        tool = ListFilesTool()
        result = await tool.execute({"path": str(tmp_dir)})

        assert result.status == ToolResultStatus.SUCCESS
        assert "a.txt" in result.output
        assert "b.txt" in result.output

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, tmp_dir: Path):
        tool = ListFilesTool()
        result = await tool.execute({"path": str(tmp_dir)})

        assert result.status == ToolResultStatus.SUCCESS


class TestGlobTool:
    """Tests for GlobTool."""

    @pytest.mark.asyncio
    async def test_glob_matches(self, tmp_dir: Path):
        (tmp_dir / "test.py").write_text("x")
        (tmp_dir / "test.js").write_text("y")
        (tmp_dir / "other.txt").write_text("z")

        tool = GlobTool()
        result = await tool.execute({"pattern": "*.py", "path": str(tmp_dir)})

        assert result.status == ToolResultStatus.SUCCESS
        assert "test.py" in result.output
        assert "test.js" not in result.output


class TestGrepTool:
    """Tests for GrepTool."""

    @pytest.mark.asyncio
    async def test_grep_finds_pattern(self, tmp_dir: Path):
        (tmp_dir / "code.py").write_text("def hello():\n    pass\n")

        tool = GrepTool()
        result = await tool.execute({"pattern": "hello", "path": str(tmp_dir)})

        assert result.status == ToolResultStatus.SUCCESS
        assert "hello" in result.output
