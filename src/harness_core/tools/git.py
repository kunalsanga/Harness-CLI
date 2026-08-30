"""Git tools for the agent."""

from __future__ import annotations

import asyncio
from typing import Any

from harness_core.agent.types import ToolResult, ToolResultStatus
from harness_core.tools.base import Tool, ToolSchema


class GitStatusTool(Tool):
    """Get git status."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_status",
            description="Get the current git status",
            parameters={"type": "object", "properties": {}},
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return await self._run_git(["status", "--short"])

    async def _run_git(self, args: list[str]) -> ToolResult:
        try:
            process = await asyncio.create_subprocess_exec(
                "git", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            output = stdout.decode("utf-8", errors="replace")
            return ToolResult(
                status=ToolResultStatus.SUCCESS if process.returncode == 0 else ToolResultStatus.ERROR,
                output=output,
                error=stderr.decode("utf-8", errors="replace") if process.returncode != 0 else None,
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class GitDiffTool(Tool):
    """Get git diff."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_diff",
            description="Show git diff",
            parameters={
                "type": "object",
                "properties": {
                    "args": {
                        "type": "string",
                        "description": "Additional diff arguments",
                    },
                },
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        args = ["diff"]
        extra = arguments.get("args", "")
        if extra:
            args.extend(extra.split())
        return await self._run_git(args)

    async def _run_git(self, args: list[str]) -> ToolResult:
        try:
            process = await asyncio.create_subprocess_exec(
                "git", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            output = stdout.decode("utf-8", errors="replace")
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return ToolResult(
                status=ToolResultStatus.SUCCESS if process.returncode == 0 else ToolResultStatus.ERROR,
                output=output,
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class GitLogTool(Tool):
    """Get git log."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_log",
            description="Show recent git log",
            parameters={
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of commits"},
                },
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        count = arguments.get("count", 10)
        try:
            process = await asyncio.create_subprocess_exec(
                "git", "log", f"--oneline", f"-{count}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            output = stdout.decode("utf-8", errors="replace")
            return ToolResult(
                status=ToolResultStatus.SUCCESS if process.returncode == 0 else ToolResultStatus.ERROR,
                output=output or "(no commits)",
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class GitIdentityTool(Tool):
    """Check git identity configuration."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_identity",
            description="Check if git user.name and user.email are configured",
            parameters={"type": "object", "properties": {}},
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            # Check user.name
            name_proc = await asyncio.create_subprocess_exec(
                "git", "config", "user.name",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            name_out, _ = await name_proc.communicate()
            user_name = name_out.decode("utf-8", errors="replace").strip()

            # Check user.email
            email_proc = await asyncio.create_subprocess_exec(
                "git", "config", "user.email",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            email_out, _ = await email_proc.communicate()
            user_email = email_out.decode("utf-8", errors="replace").strip()

            if user_name and user_email:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output=f"Git identity configured:\n  user.name: {user_name}\n  user.email: {user_email}",
                    metadata={"user_name": user_name, "user_email": user_email, "configured": True},
                )
            else:
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
                        "Configure with:\n"
                        "  git config user.name \"Your Name\"\n"
                        "  git config user.email \"your.email@example.com\""
                    ),
                    metadata={"user_name": user_name or None, "user_email": user_email or None, "configured": False},
                )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class GitCommitTool(Tool):
    """Create a git commit."""

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
                        "description": "Files to stage",
                    },
                },
                "required": ["message"],
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
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
            # Stage files
            if files:
                for f in files:
                    process = await asyncio.create_subprocess_exec(
                        "git", "add", f,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await process.communicate()
            else:
                process = await asyncio.create_subprocess_exec(
                    "git", "add", "-A",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await process.communicate()

            # Commit
            process = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", message,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            output = stdout.decode("utf-8", errors="replace")
            return ToolResult(
                status=ToolResultStatus.SUCCESS if process.returncode == 0 else ToolResultStatus.ERROR,
                output=output,
                error=stderr.decode("utf-8", errors="replace") if process.returncode != 0 else None,
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class GitPushTool(Tool):
    """Push commits to a remote repository."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_push",
            description="Push commits to a remote repository",
            parameters={
                "type": "object",
                "properties": {
                    "remote": {
                        "type": "string",
                        "description": "Remote name (default: auto-detect)",
                    },
                    "branch": {
                        "type": "string",
                        "description": "Branch name (default: current branch)",
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Force push (use with caution)",
                    },
                },
            },
            permission_required="allow",
            timeout_seconds=30.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            # Auto-detect remote if not specified
            remote = arguments.get("remote", "")
            branch = arguments.get("branch", "")
            force = arguments.get("force", False)

            if not remote:
                # Find the origin remote
                proc = await asyncio.create_subprocess_exec(
                    "git", "remote",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                remotes = stdout.decode("utf-8", errors="replace").strip().split("\n")
                remotes = [r.strip() for r in remotes if r.strip()]
                if not remotes:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        output="",
                        error=(
                            "No git remote configured.\n"
                            "Add a remote with: git remote add origin <url>"
                        ),
                    )
                remote = "origin" if "origin" in remotes else remotes[0]

            if not branch:
                # Get current branch
                proc = await asyncio.create_subprocess_exec(
                    "git", "branch", "--show-current",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                branch = stdout.decode("utf-8", errors="replace").strip()
                if not branch:
                    return ToolResult(
                        status=ToolResultStatus.ERROR,
                        output="",
                        error="Not on any branch (detached HEAD). Switch to a branch first.",
                    )

            # Build push command
            args = ["push", "-u", remote, branch]
            if force:
                args = ["push", "--force-with-lease", "-u", remote, branch]

            process = await asyncio.create_subprocess_exec(
                "git", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=30
            )
            output = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            if process.returncode == 0:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output=output or f"Pushed to {remote}/{branch}",
                    exit_code=0,
                )
            else:
                error_msg = stderr_str or output
                # Provide helpful error messages
                if "Authentication" in error_msg or "401" in error_msg or "403" in error_msg:
                    error_msg += (
                        "\n\nGitHub authentication required.\n"
                        "Git credentials are not configured on this machine.\n"
                        "Options:\n"
                        "  - Use GitHub CLI: gh auth login\n"
                        "  - Set up SSH keys\n"
                        "  - Configure Git credential manager"
                    )
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output=output,
                    error=error_msg,
                    exit_code=process.returncode,
                    stderr=stderr_str if stderr_str else None,
                )
        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TIMEOUT,
                output="",
                error="Push timed out after 30s",
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class GitRemoteTool(Tool):
    """Get git remote information."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="git_remote",
            description="Show git remote configuration",
            parameters={"type": "object", "properties": {}},
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        return await self._run_git(["remote", "-v"])

    async def _run_git(self, args: list[str]) -> ToolResult:
        try:
            process = await asyncio.create_subprocess_exec(
                "git", *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10)
            output = stdout.decode("utf-8", errors="replace")
            return ToolResult(
                status=ToolResultStatus.SUCCESS if process.returncode == 0 else ToolResultStatus.ERROR,
                output=output or "(no remotes)",
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
