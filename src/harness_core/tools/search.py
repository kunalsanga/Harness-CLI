"""Search tools for the agent."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

from harness_core.agent.types import ToolResult, ToolResultStatus
from harness_core.tools.base import Tool, ToolSchema


class GlobTool(Tool):
    """Find files matching a glob pattern."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="glob",
            description="Find files matching a glob pattern",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '**/*.py')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                },
                "required": ["pattern"],
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            pattern = arguments["pattern"]
            search_path = Path(arguments.get("path", "."))

            if not search_path.exists():
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=f"Path not found: {search_path}",
                )

            matches = sorted(
                str(p.relative_to(search_path))
                for p in search_path.rglob("*")
                if fnmatch.fnmatch(p.name, pattern.split("/")[-1])
                and not any(part.startswith(".") for part in p.parts)
            )

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output="\n".join(matches[:200]) if matches else "(no matches)",
                metadata={"count": len(matches)},
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))


class GrepTool(Tool):
    """Search for text patterns in files."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="grep",
            description="Search for a pattern in files",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                    },
                    "include": {
                        "type": "string",
                        "description": "File pattern to include (e.g., '*.py')",
                    },
                },
                "required": ["pattern"],
            },
            permission_required="allow",
            timeout_seconds=10.0,
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            pattern = arguments["pattern"]
            search_path = Path(arguments.get("path", "."))
            include = arguments.get("include", "*")

            if not search_path.exists():
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output="",
                    error=f"Path not found: {search_path}",
                )

            results = []
            files_searched = 0

            for file_path in search_path.rglob("*"):
                if not file_path.is_file():
                    continue
                if any(part.startswith(".") for part in file_path.parts):
                    continue
                if not fnmatch.fnmatch(file_path.name, include):
                    continue

                files_searched += 1
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern.lower() in line.lower():
                            rel = file_path.relative_to(search_path)
                            results.append(f"{rel}:{i}: {line.strip()}")
                            if len(results) >= 100:
                                break
                except Exception:
                    continue

                if len(results) >= 100:
                    break

            output = "\n".join(results) if results else "(no matches)"
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output=output,
                metadata={"files_searched": files_searched, "matches": len(results)},
            )
        except Exception as e:
            return ToolResult(status=ToolResultStatus.ERROR, output="", error=str(e))
