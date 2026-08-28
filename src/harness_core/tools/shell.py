"""Shell tool for running commands."""

from __future__ import annotations

import asyncio
import os
import platform
import subprocess
from typing import Any

from harness_core.agent.types import ToolResult, ToolResultStatus
from harness_core.tools.base import Tool, ToolSchema


class RunCommandTool(Tool):
    """Run a shell command."""

    def __init__(self, working_directory: str | None = None) -> None:
        self.working_directory = working_directory or os.getcwd()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_command",
            description="Execute a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "cwd": {
                        "type": "string",
                        "description": "Working directory (optional)",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 30)",
                    },
                },
                "required": ["command"],
            },
            permission_required="ask",
            timeout_seconds=30.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments["command"]
        cwd = arguments.get("cwd", self.working_directory)
        timeout = arguments.get("timeout", 30.0)

        # Dangerous command detection
        dangerous_patterns = ["rm -rf /", "mkfs", "> /dev/sda", "dd if="]
        for pattern in dangerous_patterns:
            if pattern in command:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=f"Dangerous command detected: {pattern}",
                )

        try:
            # Use platform-appropriate shell
            if platform.system() == "Windows":
                shell_cmd = ["cmd", "/c", command]
            else:
                shell_cmd = ["bash", "-c", command]

            process = await asyncio.create_subprocess_exec(
                *shell_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.communicate()
                return ToolResult(
                    status=ToolResultStatus.TIMEOUT,
                    output="",
                    error=f"Command timed out after {timeout}s",
                )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Truncate large output
            max_output = 10000
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + "\n... (truncated)"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + "\n... (truncated)"

            output = stdout_str
            if stderr_str:
                output += f"\n[stderr]\n{stderr_str}"

            return ToolResult(
                status=ToolResultStatus.SUCCESS if process.returncode == 0 else ToolResultStatus.ERROR,
                output=output,
                metadata={"return_code": process.returncode or 0},
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
