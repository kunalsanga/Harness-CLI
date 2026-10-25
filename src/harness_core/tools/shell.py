"""Shell tool for running commands."""

from __future__ import annotations

import asyncio
import os
import platform
import re
import shutil
import subprocess
from typing import Any

from harness_core.agent.types import ToolResult, ToolResults
from harness_core.tools.base import Tool, ToolSchema


def normalize_windows_command(cmd: str, cwd: str) -> str:
    """Normalize Windows shell commands:
    1. Convert POSIX drive notation /c/Users/... -> C:/Users/...
    2. Unescape bash-escaped spaces in paths (e.g. New\\ folder -> New folder)
    3. Strip redundant leading cd to cwd, ., or ./
    4. Rewrite python3 -> python if python3 is unavailable on Windows
    """
    c = (cmd or "").strip()

    # Convert POSIX drive notation /c/... -> C:/...
    c = re.sub(r'(^|[\s"\'])/([a-zA-Z])/(?=[a-zA-Z0-9_\-\./\\])', r'\1\2:/', c)

    # Convert bash-escaped spaces
    c = c.replace(r"\ ", " ")

    # Strip redundant leading cd to cwd, ., or ./
    norm_cwd = os.path.normpath(cwd).lower().replace("\\", "/")
    cd_match = re.match(
        r'^\s*cd\s+(?:/d\s+)?(?:["\']([^"\']+)["\']|([^\s;&|]+(?:\s+[^\s;&|]+)*))\s*(&&|;|\|)\s*(.+)$',
        c,
        re.IGNORECASE,
    )
    if cd_match:
        target_dir = (cd_match.group(1) or cd_match.group(2) or "").strip()
        rest = cd_match.group(4).strip()
        target_dir_clean = re.sub(r'^/([a-zA-Z])/', r'\1:/', target_dir)
        target_norm = os.path.normpath(target_dir_clean).lower().replace("\\", "/")
        if target_norm in (".", "", norm_cwd) or norm_cwd.endswith(target_norm) or target_norm.endswith(norm_cwd):
            c = rest

    # Rewrite python3 -> python if python3 is not available
    if re.search(r'\bpython3\b', c):
        if not shutil.which("python3") and shutil.which("python"):
            c = re.sub(r'\bpython3\b', 'python', c)

    return c


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

        # Dangerous command detection — these are never retryable.
        dangerous_patterns = ["rm -rf /", "mkfs", "> /dev/sda", "dd if="]
        for pattern in dangerous_patterns:
            if pattern in command:
                return ToolResults.error(
                    f"Dangerous command detected: {pattern}",
                    retryable=False,
                )

        try:
            # Use platform-appropriate shell
            if platform.system() == "Windows":
                exec_cmd = normalize_windows_command(command, cwd)
                ps_exe = (
                    shutil.which("powershell.exe")
                    or shutil.which("powershell")
                    or shutil.which("pwsh.exe")
                    or shutil.which("pwsh")
                )
                if ps_exe:
                    pwsh_script = (
                        f"{exec_cmd}\n"
                        f"if ($LASTEXITCODE) {{ exit $LASTEXITCODE }}\n"
                        f"if (-not $?) {{ exit 1 }}"
                    )
                    shell_cmd = [
                        ps_exe,
                        "-NoProfile",
                        "-NonInteractive",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-Command",
                        pwsh_script,
                    ]
                else:
                    shell_cmd = ["cmd.exe", "/c", exec_cmd]
            else:
                exec_cmd = command
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
                return ToolResults.timeout(
                    f"Command timed out after {timeout}s",
                    timeout_seconds=timeout,
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

            return_code = process.returncode or 0

            if return_code == 0:
                return ToolResults.success(
                    output,
                    metadata={"return_code": return_code},
                    exit_code=return_code,
                    stderr=stderr_str if stderr_str else None,
                )

            # Non-zero exit — retryable: same command may succeed on a retry
            # (race condition, transient resource issue, etc.).
            is_file_inspection = bool(
                re.search(r"^\s*(cat|type|head|tail|ls|dir)\b", command.strip(), re.I)
            ) or " | tail" in command or " | head" in command
            inspection_tip = ""
            if is_file_inspection:
                inspection_tip = (
                    "\nTip: Use the dedicated 'read_file' or 'list_files' tool "
                    "instead of shell commands to inspect files."
                )

            err_msg = (
                f"Command failed with exit code {return_code}.\n"
                f"Command: {command}\n"
                f"Exit code: {return_code}"
                + (f"\nstderr: {stderr_str[:2000]}" if stderr_str else "")
                + inspection_tip
            )
            return ToolResults.error(
                err_msg,
                output=output,
                exit_code=return_code,
                stderr=stderr_str if stderr_str else None,
                retryable=True,
                metadata={"return_code": return_code},
            )
        except Exception as e:
            return ToolResults.from_exception(e, retryable=True)
