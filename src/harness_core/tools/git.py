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
            permission_required="ask",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        message = arguments["message"]
        files = arguments.get("files", [])

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
