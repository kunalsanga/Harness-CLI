"""Filesystem tools for the agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness_core.agent.types import ToolResult, ToolResultStatus
from harness_core.tools.base import Tool, ToolSchema


class ReadFileTool(Tool):
    """Read a file from the filesystem."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="read_file",
            description="Read the contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                    },
                },
                "required": ["path"],
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = Path(arguments["path"])
            if not path.exists():
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=f"File not found: {path}",
                )
            if not path.is_file():
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=f"Not a file: {path}",
                )

            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            offset = arguments.get("offset", 1)
            limit = arguments.get("limit")

            start = max(0, offset - 1)
            end = start + limit if limit else len(lines)
            selected = lines[start:end]

            output = "\n".join(selected)
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output=output,
                metadata={"total_lines": len(lines), "showing_lines": f"{start+1}-{min(end, len(lines))}"},
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class WriteFileTool(Tool):
    """Write content to a file."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="write_file",
            description="Write content to a file (creates or overwrites)",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = Path(arguments["path"])
            content = arguments["content"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output=f"Written {len(content)} bytes to {path}",
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class EditFileTool(Tool):
    """Edit a file by replacing a string."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="edit_file",
            description="Edit a file by replacing an exact string match",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "old_string": {"type": "string", "description": "Exact string to replace"},
                    "new_string": {"type": "string", "description": "Replacement string"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = Path(arguments["path"])
            if not path.exists():
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=f"File not found: {path}",
                )

            content = path.read_text(encoding="utf-8")
            old_string = arguments["old_string"]
            new_string = arguments["new_string"]

            if old_string not in content:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=f"String not found in {path}",
                )

            new_content = content.replace(old_string, new_string, 1)
            path.write_text(new_content, encoding="utf-8")

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output=f"Edited {path}",
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class ListFilesTool(Tool):
    """List files in a directory."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_files",
            description="List files and directories in a path",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively",
                    },
                },
                "required": [],
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = Path(arguments.get("path", "."))
            recursive = arguments.get("recursive", False)

            if not path.exists():
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=f"Path not found: {path}",
                )

            if recursive:
                entries = sorted(
                    str(p.relative_to(path))
                    for p in path.rglob("*")
                    if not p.name.startswith(".")
                )
            else:
                entries = sorted(
                    str(p.relative_to(path))
                    for p in path.iterdir()
                    if not p.name.startswith(".")
                )

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output="\n".join(entries) if entries else "(empty)",
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
