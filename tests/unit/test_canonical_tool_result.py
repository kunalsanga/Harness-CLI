"""Tests for the canonical ToolResult contract.

These tests pin the contract that every tool implementation must follow.
The contract is the `ToolResults` factory namespace plus the underlying
`ToolResult` dataclass: all tools MUST go through these factories, and
the runtime (loop.py) MAY construct `ToolResult(...)` directly for
synthetic cases (unknown tool, missing args, etc.) — that path is
already covered by the existing test_tool_failure_propagation.py.

Coverage:
  - Each factory produces a well-formed ToolResult.
  - retryable defaults are correct (transient vs. permanent).
  - The factory fields are populated as documented.
  - Migrated tools (`shell`, `filesystem`, `search`, `git`, `parallel`)
    return well-formed, consistent results.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from harness_core.agent.types import (
    ToolResultStatus,
    ToolResults,
)
from harness_core.tools.base import Tool, ToolSchema
from harness_core.tools.filesystem import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from harness_core.tools.git import (
    GitCommitTool,
    GitDiffTool,
    GitIdentityTool,
    GitLogTool,
    GitPushTool,
    GitRemoteTool,
    GitStatusTool,
)
from harness_core.tools.parallel import ParallelToolExecutor
from harness_core.tools.search import GlobTool, GrepTool
from harness_core.tools.shell import RunCommandTool


# ─── Factory: success ─────────────────────────────────────────────────────


class TestSuccessFactory:
    def test_returns_success(self):
        r = ToolResults.success("ok")
        assert r.status == ToolResultStatus.SUCCESS
        assert r.output == "ok"
        assert r.error is None
        assert r.retryable is True  # default
        assert r.exit_code is None
        assert r.stderr is None
        assert r.metadata == {}

    def test_metadata_default_empty(self):
        r = ToolResults.success("ok", metadata=None)
        assert r.metadata == {}

    def test_with_metadata(self):
        r = ToolResults.success("ok", metadata={"k": 1})
        assert r.metadata == {"k": 1}

    def test_with_exit_code_and_stderr(self):
        r = ToolResults.success("ok", exit_code=0, stderr="noise")
        assert r.exit_code == 0
        assert r.stderr == "noise"


# ─── Factory: error ───────────────────────────────────────────────────────


class TestErrorFactory:
    def test_returns_error(self):
        r = ToolResults.error("bad")
        assert r.status == ToolResultStatus.ERROR
        assert r.output == ""
        assert r.error == "bad"
        assert r.retryable is True  # default: transient

    def test_retryable_false(self):
        r = ToolResults.error("bad", retryable=False)
        assert r.retryable is False

    def test_exit_code_preserved(self):
        r = ToolResults.error("bad", exit_code=2, stderr="x")
        assert r.exit_code == 2
        assert r.stderr == "x"

    def test_execution_failed_property(self):
        r = ToolResults.error("bad", exit_code=1)
        assert r.execution_failed
        assert r.failure_category == "execution_error"

    def test_retryable_false_means_not_transient(self):
        r = ToolResults.error("bad", retryable=False)
        assert r.is_final
        assert not r.is_transient

    def test_retryable_true_means_transient(self):
        r = ToolResults.error("bad", retryable=True)
        assert r.is_transient
        assert not r.is_final


# ─── Factory: permission_denied ──────────────────────────────────────────


class TestPermissionDeniedFactory:
    def test_default(self):
        r = ToolResults.permission_denied()
        assert r.status == ToolResultStatus.PERMISSION_DENIED
        assert r.retryable is False
        assert r.error == "Permission denied"
        assert not r.execution_failed
        assert r.failure_category == "permission_denied"

    def test_custom_error(self):
        r = ToolResults.permission_denied("Out of scope: /etc/passwd")
        assert r.error == "Out of scope: /etc/passwd"
        assert r.retryable is False


# ─── Factory: timeout ─────────────────────────────────────────────────────


class TestTimeoutFactory:
    def test_returns_timeout(self):
        r = ToolResults.timeout("timed out")
        assert r.status == ToolResultStatus.TIMEOUT
        assert r.retryable is True
        assert r.execution_failed
        assert r.failure_category == "timeout"

    def test_timeout_seconds_in_metadata(self):
        r = ToolResults.timeout("timed out", timeout_seconds=42.0)
        assert r.metadata == {"timeout_seconds": 42.0}

    def test_no_timeout_seconds(self):
        r = ToolResults.timeout("timed out")
        assert r.metadata == {}


# ─── Factory: network_failure ─────────────────────────────────────────────


class TestNetworkFailureFactory:
    def test_retryable(self):
        r = ToolResults.network_failure("connection refused")
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is True
        assert r.error == "connection refused"


# ─── Factory: git_failure ────────────────────────────────────────────────


class TestGitFailureFactory:
    def test_retryable_default_false(self):
        r = ToolResults.git_failure("status", "git error")
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False
        assert r.metadata["operation"] == "status"

    def test_exit_code_and_stderr(self):
        r = ToolResults.git_failure("status", "x", exit_code=128, stderr="y")
        assert r.exit_code == 128
        assert r.stderr == "y"

    def test_metadata_merged(self):
        r = ToolResults.git_failure("push", "x", metadata={"branch": "main"})
        assert r.metadata == {"operation": "push", "branch": "main"}


# ─── Factory: from_exception ─────────────────────────────────────────────


class TestFromExceptionFactory:
    def test_captures_message(self):
        r = ToolResults.from_exception(ValueError("bad input"))
        assert r.status == ToolResultStatus.ERROR
        assert r.error == "bad input"
        assert r.retryable is True

    def test_retryable_false(self):
        r = ToolResults.from_exception(KeyError("missing"), retryable=False)
        assert r.retryable is False


# ─── Factory: unknown_tool / missing_argument ────────────────────────────


class TestSpecialFactories:
    def test_unknown_tool(self):
        r = ToolResults.unknown_tool("foo")
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False
        assert "Unknown tool: foo" in r.error

    def test_missing_argument(self):
        r = ToolResults.missing_argument("write_file", ["path", "content"])
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False
        assert "write_file" in r.error
        assert "path" in r.error
        assert "content" in r.error


# ─── Cross-tool consistency: every tool uses factories ────────────────────


class TestMigratedToolContract:
    """Every migrated tool must return results with consistent semantics."""

    @pytest.fixture
    def tmp_dir(self, tmp_path: Path) -> Path:
        return tmp_path

    # ── shell ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_shell_success_has_exit_code_zero(self, tmp_dir: Path):
        tool = RunCommandTool(working_directory=str(tmp_dir))
        r = await tool.execute({"command": "echo hi"})
        assert r.status == ToolResultStatus.SUCCESS
        assert r.exit_code == 0
        assert r.error is None
        assert not r.execution_failed

    @pytest.mark.asyncio
    async def test_shell_failure_has_exit_code_and_retryable(self, tmp_dir: Path):
        tool = RunCommandTool(working_directory=str(tmp_dir))
        r = await tool.execute({"command": "this_command_does_not_exist_xyz"})
        assert r.status == ToolResultStatus.ERROR
        assert r.exit_code is not None
        assert r.exit_code != 0
        assert r.retryable is True
        assert r.execution_failed
        assert r.failure_category == "execution_error"

    @pytest.mark.asyncio
    async def test_shell_dangerous_command_is_not_retryable(self, tmp_dir: Path):
        tool = RunCommandTool(working_directory=str(tmp_dir))
        r = await tool.execute({"command": "rm -rf / something"})
        assert r.status == ToolResultStatus.ERROR
        # Bug fix: dangerous commands must not be marked retryable.
        assert r.retryable is False
        assert r.execution_failed
        assert r.failure_category == "tool_error"

    @pytest.mark.asyncio
    async def test_shell_timeout(self, tmp_dir: Path):
        tool = RunCommandTool(working_directory=str(tmp_dir))
        # tiny timeout; command that would block
        import sys
        script = tmp_dir / "sleep.py"
        script.write_text("import time\ntime.sleep(5)")
        r = await tool.execute({"command": "python sleep.py", "timeout": 0.1})
        assert r.status == ToolResultStatus.TIMEOUT
        assert r.retryable is True
        assert r.execution_failed
        assert r.failure_category == "timeout"

    # ── filesystem ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_read_file_missing_is_not_retryable(self, tmp_dir: Path):
        r = await ReadFileTool().execute({"path": str(tmp_dir / "missing")})
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False
        assert r.execution_failed

    @pytest.mark.asyncio
    async def test_write_file_exception_is_not_retryable(self, tmp_dir: Path):
        # Try to write to an invalid path (parent doesn't exist and is a file)
        blocker = tmp_dir / "blocker"
        blocker.write_text("x")
        invalid = blocker / "inside"  # cannot be created (parent is a file)
        r = await WriteFileTool().execute({"path": str(invalid), "content": "x"})
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False

    @pytest.mark.asyncio
    async def test_edit_file_missing_string_is_not_retryable(self, tmp_dir: Path):
        f = tmp_dir / "x.txt"
        f.write_text("hello")
        r = await EditFileTool().execute(
            {"path": str(f), "old_string": "missing", "new_string": "x"}
        )
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False

    @pytest.mark.asyncio
    async def test_list_files_missing_path_is_not_retryable(self, tmp_dir: Path):
        r = await ListFilesTool().execute({"path": str(tmp_dir / "nope")})
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False

    # ── search ─────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_glob_missing_path_is_not_retryable(self, tmp_dir: Path):
        r = await GlobTool().execute({"pattern": "*.py", "path": str(tmp_dir / "nope")})
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False

    @pytest.mark.asyncio
    async def test_grep_missing_path_is_not_retryable(self, tmp_dir: Path):
        r = await GrepTool().execute({"pattern": "x", "path": str(tmp_dir / "nope")})
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False

    # ── git ────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_git_identity_returns_deterministic_result(self, tmp_dir: Path):
        """GitIdentityTool returns SUCCESS when global identity is configured,
        ERROR when it is not. The test adapts to whatever the environment has."""
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=str(tmp_dir), check=True)

        r = await GitIdentityTool(working_directory=str(tmp_dir)).execute({})

        # Verify the result is well-formed regardless of environment.
        assert r.status in (ToolResultStatus.SUCCESS, ToolResultStatus.ERROR)
        assert isinstance(r.metadata, dict)
        # ERROR results are not retryable; SUCCESS results use the default (True).
        if r.status == ToolResultStatus.ERROR:
            assert r.retryable is False
        else:
            assert r.retryable is True

    @pytest.mark.asyncio
    async def test_git_status_in_clean_repo_succeeds(self, tmp_dir: Path):
        import subprocess
        subprocess.run(["git", "init", "-q"], cwd=str(tmp_dir), check=True)
        r = await GitStatusTool(working_directory=str(tmp_dir)).execute({})
        assert r.status == ToolResultStatus.SUCCESS
        assert r.metadata.get("clean") is True

    @pytest.mark.asyncio
    async def test_git_commit_forbidden_pattern_is_not_retryable(self, tmp_dir: Path):
        import subprocess
        subprocess.run(["git", "init", "-q"], cwd=str(tmp_dir), check=True)
        r = await GitCommitTool(working_directory=str(tmp_dir)).execute(
            {"message": "git config user.name \"bot\""}
        )
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False
        assert "identity" in r.error.lower()

    @pytest.mark.asyncio
    async def test_git_push_no_remote_is_not_retryable(self, tmp_dir: Path):
        import subprocess
        subprocess.run(["git", "init", "-q"], cwd=str(tmp_dir), check=True)
        r = await GitPushTool(working_directory=str(tmp_dir)).execute({})
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False
        assert "remote" in r.error.lower()

    # ── parallel executor ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_parallel_unknown_tool(self):
        executor = ParallelToolExecutor(tools={})
        r = await executor.execute_single("does_not_exist", {})
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is False
        assert "Unknown tool" in r.error

    @pytest.mark.asyncio
    async def test_parallel_permission_denied(self):
        tool = ReadFileTool()
        executor = ParallelToolExecutor(tools={"read_file": tool})

        def deny_all(name, args):
            return False

        r = await executor.execute_single(
            "read_file", {"path": "x"}, permission_check=deny_all
        )
        assert r.status == ToolResultStatus.PERMISSION_DENIED
        assert r.retryable is False
        assert not r.execution_failed

    @pytest.mark.asyncio
    async def test_parallel_catches_exception(self):
        class Boom(Tool):
            @property
            def schema(self) -> ToolSchema:
                return ToolSchema(name="boom", description="x", parameters={})

            async def execute(self, args):
                raise RuntimeError("explode")

        executor = ParallelToolExecutor(tools={"boom": Boom()})
        r = await executor.execute_single("boom", {})
        assert r.status == ToolResultStatus.ERROR
        assert "explode" in r.error
        # The exception is unexpected/transient from the runtime's POV.
        assert r.retryable is True


# ─── Cross-tool: retryable semantics are consistent ──────────────────────


class TestRetryableContract:
    """retryable is the most important semantic: tools must not lie about it."""

    @pytest.mark.asyncio
    async def test_filesystem_errors_are_permanent(self, tmp_path: Path):
        """File-not-found, string-not-found, etc. are NEVER transient."""
        for tool_call in [
            (ReadFileTool(), {"path": str(tmp_path / "missing")}),
            (EditFileTool(), {
                "path": str(tmp_path / "x"),
                "old_string": "a", "new_string": "b",
            }),
            (ListFilesTool(), {"path": str(tmp_path / "missing")}),
            (GlobTool(), {"pattern": "*.py", "path": str(tmp_path / "missing")}),
        ]:
            tool, args = tool_call
            r = await tool.execute(args)
            assert r.status == ToolResultStatus.ERROR
            assert r.retryable is False, (
                f"{tool.__class__.__name__} should not be retryable on this error"
            )

    @pytest.mark.asyncio
    async def test_shell_exit_code_and_exception_are_retryable(self, tmp_path: Path):
        """Shell failures (non-zero exit, exceptions) are transient."""
        tool = RunCommandTool(working_directory=str(tmp_path))
        r = await tool.execute({"command": "false"})
        assert r.status == ToolResultStatus.ERROR
        assert r.retryable is True
        assert r.execution_failed


# ─── Cross-tool: existing properties still work ──────────────────────────


class TestExistingPropertiesPreserved:
    """The migration must not break the existing ToolResult properties."""

    def test_execution_failed_still_works(self):
        r = ToolResults.error("x", exit_code=1)
        assert r.execution_failed

    def test_is_perm_denied_still_works(self):
        r = ToolResults.permission_denied()
        assert r.is_perm_denied
        assert not r.execution_failed

    def test_is_transient_still_works(self):
        # retryable=True + ERROR -> transient
        r = ToolResults.error("x", retryable=True)
        assert r.is_transient
        # retryable=False + ERROR -> not transient
        r = ToolResults.error("x", retryable=False)
        assert not r.is_transient
        # timeout is transient
        r = ToolResults.timeout("x")
        assert r.is_transient

    def test_is_final_still_works(self):
        r = ToolResults.permission_denied()
        assert r.is_final
        r = ToolResults.error("x", retryable=False)
        assert r.is_final
        r = ToolResults.error("x", retryable=True)
        assert not r.is_final

    def test_failure_category_still_works(self):
        assert ToolResults.success("x").failure_category == "success"
        assert ToolResults.permission_denied().failure_category == "permission_denied"
        assert ToolResults.timeout("x").failure_category == "timeout"
        assert ToolResults.error("x", exit_code=1).failure_category == "execution_error"
        assert ToolResults.error("x").failure_category == "tool_error"


# ─── The contract is canonical across the whole runtime ──────────────────


class TestContractIsCanonical:
    """types.py is the only module allowed to construct ToolResult directly.

    Everything else must go through a ToolResults factory, so result
    semantics (status, retryability, metadata) are defined in exactly one
    place. This is the guard that keeps the contract canonical as new tools
    and call sites are added.
    """

    def test_no_direct_toolresult_construction_outside_types(self):
        import harness_core

        root = Path(harness_core.__file__).parent
        allowed = {root / "agent" / "types.py"}

        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            if path in allowed:
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if line.strip().startswith("#"):
                    continue
                # ToolResults.<factory>( is the sanctioned form and does not match.
                if re.search(r"\bToolResult\(", line):
                    offenders.append(
                        f"{path.relative_to(root)}:{lineno}: {line.strip()}"
                    )

        assert not offenders, (
            "direct ToolResult(...) construction must be replaced with a "
            "ToolResults factory:\n" + "\n".join(offenders)
        )


# ─── Git failures carry their diagnosis ──────────────────────────────────


class TestGitFailureDiagnosis:
    @pytest.mark.asyncio
    async def test_git_failure_carries_operation_and_category(self, tmp_path: Path):
        """git status outside a repository fails with a diagnosed result."""
        tool = GitStatusTool(working_directory=str(tmp_path))
        r = await tool.execute({"cwd": str(tmp_path)})

        if r.status == ToolResultStatus.SUCCESS:
            pytest.skip("tmp_path unexpectedly inside a git repository")

        assert r.status == ToolResultStatus.ERROR
        assert r.metadata.get("operation") == "status"
        assert r.metadata.get("category") == "git_failure"
        assert r.retryable is False
        assert r.exit_code not in (0, None)
