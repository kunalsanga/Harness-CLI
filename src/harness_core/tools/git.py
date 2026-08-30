"""Structured Git tools for the agent.

Each tool returns structured information (success, operation, stdout,
stderr, exit_code, and operation-specific fields such as commit_hash,
branch, remote) instead of a bare exit code. Failures are diagnosed into
human-readable reasons (identity missing, authentication failed, no remote,
working tree clean) rather than "Command failed with exit code 1."

All tools execute inside a confined working directory (workspace_root).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from harness_core.agent.types import ToolResult, ToolResultStatus
from harness_core.tools.base import Tool, ToolSchema
from harness_core.tools.diagnosis import classify_git_failure


async def _run_git(args: list[str], cwd: str | None, timeout: float = 15.0) -> tuple[int, str, str]:
    """Run a git command in the given directory. Returns (rc, stdout, stderr)."""
    process = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout_b, stderr_b = await asyncio.wait_for(process.communicate(), timeout=timeout)
    return (
        process.returncode or 0,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


def _cwd(arguments: dict[str, Any], working_directory: str | None) -> str:
    return arguments.get("cwd") or working_directory or os.getcwd()


def _git_error(operation: str, stdout: str, stderr: str, exit_code: int) -> ToolResult:
    """Build a diagnosed ERROR ToolResult for a failed git operation."""
    diagnosis = classify_git_failure(operation, stdout, stderr, exit_code)
    return ToolResult(
        status=ToolResultStatus.ERROR,
        output=stdout,
        error=diagnosis.reason,
        exit_code=exit_code,
        stderr=stderr if stderr else None,
        metadata={"operation": operation},
    )


class GitStatusTool(Tool):
    """Get git status (structured)."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_status",
            description="Get the current git status",
            parameters={"type": "object", "properties": {
                "cwd": {"type": "string", "description": "Working directory (optional)"},
            }},
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cwd = _cwd(arguments, self.working_directory)
        try:
            rc, stdout, stderr = await _run_git(["status", "--short"], cwd)
        except asyncio.TimeoutError:
            return ToolResult(status=ToolResultStatus.TIMEOUT, output="", error="git status timed out")
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
        if rc != 0:
            return _git_error("status", stdout, stderr, rc)
        files = [line for line in stdout.splitlines() if line.strip()]
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=stdout or "(working tree clean)",
            metadata={"operation": "status", "clean": not files, "files": files},
        )


class GitDiffTool(Tool):
    """Get git diff."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_diff",
            description="Show git diff",
            parameters={
                "type": "object",
                "properties": {
                    "args": {"type": "string", "description": "Additional diff arguments"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                },
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cwd = _cwd(arguments, self.working_directory)
        args = ["diff"]
        extra = arguments.get("args", "")
        if extra:
            args.extend(extra.split())
        try:
            rc, stdout, stderr = await _run_git(args, cwd)
        except asyncio.TimeoutError:
            return ToolResult(status=ToolResultStatus.TIMEOUT, output="", error="git diff timed out")
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
        if rc != 0:
            return _git_error("diff", stdout, stderr, rc)
        if len(stdout) > 5000:
            stdout = stdout[:5000] + "\n... (truncated)"
        return ToolResult(status=ToolResultStatus.SUCCESS, output=stdout, metadata={"operation": "diff"})


class GitLogTool(Tool):
    """Get git log."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_log",
            description="Show recent git log",
            parameters={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of commits"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                },
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cwd = _cwd(arguments, self.working_directory)
        count = arguments.get("count", 10)
        try:
            rc, stdout, stderr = await _run_git(["log", "--oneline", f"-{count}"], cwd)
        except asyncio.TimeoutError:
            return ToolResult(status=ToolResultStatus.TIMEOUT, output="", error="git log timed out")
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
        if rc != 0:
            return _git_error("log", stdout, stderr, rc)
        return ToolResult(status=ToolResultStatus.SUCCESS, output=stdout or "(no commits)", metadata={"operation": "log"})


class GitIdentityTool(Tool):
    """Check git identity configuration."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_identity",
            description="Check if git user.name and user.email are configured",
            parameters={"type": "object", "properties": {
                "cwd": {"type": "string", "description": "Working directory (optional)"},
            }},
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cwd = _cwd(arguments, self.working_directory)
        try:
            _, name_out, _ = await _run_git(["config", "user.name"], cwd)
            user_name = name_out.strip()
            _, email_out, _ = await _run_git(["config", "user.email"], cwd)
            user_email = email_out.strip()

            if user_name and user_email:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output=f"Git identity configured:\n  user.name: {user_name}\n  user.email: {user_email}",
                    metadata={"user_name": user_name, "user_email": user_email, "configured": True},
                )
            missing = []
            if not user_name:
                missing.append("user.name")
            if not user_email:
                missing.append("user.email")
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error=(
                    f"Git identity is not configured. Missing: {', '.join(missing)}.\n"
                    "Configure your OWN identity with:\n"
                    "  git config user.name \"Your Name\"\n"
                    "  git config user.email \"your.email@example.com\""
                ),
                metadata={"user_name": user_name or None, "user_email": user_email or None, "configured": False},
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class GitRemoteTool(Tool):
    """Get git remote information (structured)."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_remote",
            description="Show git remote configuration",
            parameters={"type": "object", "properties": {
                "cwd": {"type": "string", "description": "Working directory (optional)"},
            }},
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cwd = _cwd(arguments, self.working_directory)
        try:
            rc, stdout, stderr = await _run_git(["remote", "-v"], cwd)
        except asyncio.TimeoutError:
            return ToolResult(status=ToolResultStatus.TIMEOUT, output="", error="git remote timed out")
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
        if rc != 0:
            return _git_error("remote", stdout, stderr, rc)
        remotes: dict[str, str] = {}
        for line in stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                remotes.setdefault(parts[0], parts[1])
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=stdout or "(no remotes)",
            metadata={"operation": "remote", "remotes": remotes},
        )


class GitAddTool(Tool):
    """Stage files (git add)."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_add",
            description="Stage files for commit",
            parameters={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to stage (omit to stage all changes)",
                    },
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                },
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cwd = _cwd(arguments, self.working_directory)
        files = arguments.get("files") or []
        args = ["add", "-A"] if not files else ["add", *files]
        try:
            rc, stdout, stderr = await _run_git(args, cwd)
        except asyncio.TimeoutError:
            return ToolResult(status=ToolResultStatus.TIMEOUT, output="", error="git add timed out")
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
        if rc != 0:
            return _git_error("add", stdout, stderr, rc)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=stdout or "Staged changes.",
            metadata={"operation": "add", "files": files or ["-A"]},
        )


class GitCommitTool(Tool):
    """Create a git commit (structured result with diagnosis)."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_commit",
            description="Create a git commit",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"},
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to stage (omit to stage all changes)",
                    },
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["message"],
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cwd = _cwd(arguments, self.working_directory)
        message = arguments["message"]
        files = arguments.get("files", [])

        # BLOCK: Never allow git config user.name/email to be set by the agent
        forbidden_patterns = ["git config user.name", "git config user.email"]
        for pattern in forbidden_patterns:
            if pattern in message.lower():
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=(
                        "Harness must never invent Git identity.\n"
                        "Configure user.name and user.email manually with:\n"
                        "  git config user.name \"Your Name\"\n"
                        "  git config user.email \"your.email@example.com\""
                    ),
                )

        try:
            # Stage files (all changes when no explicit files given)
            add_args = ["add", "-A"] if not files else ["add", *files]
            await _run_git(add_args, cwd)

            # Commit
            rc, stdout, stderr = await _run_git(["commit", "-m", message], cwd)

            combined = f"{stdout}\n{stderr}".lower()
            if rc != 0 and ("nothing to commit" in combined or "no changes added" in combined):
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output="Working tree already clean. Nothing to commit.",
                    metadata={"operation": "commit", "nothing_to_commit": True},
                )
            if rc != 0 and ("please tell me who you are" in combined or "author identity unknown" in combined):
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=(
                        "Git identity is not configured. Configure your OWN identity:\n"
                        "  git config user.name \"Your Name\"\n"
                        "  git config user.email \"your.email@example.com\""
                    ),
                    metadata={"operation": "commit"},
                )
            if rc != 0:
                return _git_error("commit", stdout, stderr, rc)

            commit_hash = self._parse_hash(stdout)
            branch = self._parse_branch(stdout)
            short = commit_hash[:7] if commit_hash else ""
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output=f"Commit created {short} {message}".strip(),
                metadata={"operation": "commit", "commit_hash": commit_hash, "branch": branch},
            )
        except asyncio.TimeoutError:
            return ToolResult(status=ToolResultStatus.TIMEOUT, output="", error="git commit timed out")
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))

    @staticmethod
    def _bracket_content(output: str) -> str:
        import re
        m = re.search(r"\[([^\]]+)\]", output)
        return m.group(1) if m else ""

    @staticmethod
    def _parse_hash(output: str) -> str:
        import re
        content = GitCommitTool._bracket_content(output)
        m = re.search(r"\b([0-9a-f]{7,40})\b", content)
        return m.group(1) if m else ""

    @staticmethod
    def _parse_branch(output: str) -> str:
        content = GitCommitTool._bracket_content(output)
        parts = content.split()
        return parts[0] if parts else ""


class GitPushTool(Tool):
    """Push commits to a remote repository (first-class operation)."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_push",
            description="Push commits to a remote repository",
            parameters={
                "type": "object",
                "properties": {
                    "remote": {"type": "string", "description": "Remote name (default: auto-detect)"},
                    "branch": {"type": "string", "description": "Branch name (default: current branch)"},
                    "force": {"type": "boolean", "description": "Force push (use with caution)"},
                    "cwd": {"type": "string", "description": "Working directory (optional)"},
                },
            },
            permission_required="allow",
            timeout_seconds=30.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cwd = _cwd(arguments, self.working_directory)
        try:
            remote = arguments.get("remote", "")
            branch = arguments.get("branch", "")
            force = arguments.get("force", False)

            # Auto-detect remote
            if not remote:
                rc, stdout, _ = await _run_git(["remote"], cwd)
                remotes = [r.strip() for r in stdout.splitlines() if r.strip()]
                if rc != 0 or not remotes:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        output="",
                        error="No git remote configured.\nAdd a remote with: git remote add origin <url>",
                        metadata={"operation": "push"},
                    )
                remote = "origin" if "origin" in remotes else remotes[0]

            # Auto-detect branch
            if not branch:
                rc, stdout, _ = await _run_git(["branch", "--show-current"], cwd)
                branch = stdout.strip()
                if rc != 0 or not branch:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        output="",
                        error="Not on any branch (detached HEAD). Switch to a branch first.",
                        metadata={"operation": "push"},
                    )

            args = ["push", "-u", remote, branch]
            if force:
                args = ["push", "--force-with-lease", "-u", remote, branch]

            rc, stdout, stderr = await _run_git(args, cwd, timeout=30.0)

            if rc == 0:
                # Verify remote state: resolve the pushed commit hash
                _, rev_out, _ = await _run_git(["rev-parse", "HEAD"], cwd)
                commit_hash = rev_out.strip()
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output=f"Pushed to {remote}/{branch}" + (f" ({commit_hash[:7]})" if commit_hash else ""),
                    exit_code=0,
                    metadata={
                        "operation": "push",
                        "remote": remote,
                        "branch": branch,
                        "commit_hash": commit_hash,
                    },
                )

            error_msg = stderr or stdout
            combined = error_msg.lower()
            if any(k in combined for k in ("authentication", "401", "403", "could not read username", "permission")):
                error_msg += (
                    "\n\nGitHub authentication failed.\n"
                    "Git credentials are not configured on this machine.\n"
                    "Options:\n"
                    "  - Use GitHub CLI: gh auth login\n"
                    "  - Set up SSH keys\n"
                    "  - Configure Git credential manager"
                )
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=stdout,
                error=error_msg,
                exit_code=rc,
                stderr=stderr if stderr else None,
                metadata={"operation": "push", "remote": remote, "branch": branch},
            )
        except asyncio.TimeoutError:
            return ToolResult(status=ToolResultStatus.TIMEOUT, output="", error="Push timed out after 30s")
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
