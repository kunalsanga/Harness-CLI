"""Filesystem tools for the agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from harness_core.agent.types import ToolResult, ToolResults
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
                return ToolResults.error(
                    f"File not found: {path}",
                    retryable=False,
                )
            if not path.is_file():
                return ToolResults.error(
                    f"Not a file: {path}",
                    retryable=False,
                )

            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()

            offset = arguments.get("offset", 1)
            limit = arguments.get("limit")

            start = max(0, offset - 1)
            end = start + limit if limit else len(lines)
            selected = lines[start:end]

            output = "\n".join(selected)
            return ToolResults.success(
                output,
                metadata={"total_lines": len(lines), "showing_lines": f"{start+1}-{min(end, len(lines))}"},
            )
        except Exception as e:
            return ToolResults.from_exception(e, retryable=False)


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
            return ToolResults.success(f"Written {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResults.from_exception(e, retryable=False)


class EditFileTool(Tool):
    """Edit a file by replacing a string — tolerant matching (Part 11).

    Matching ladder (never silently modifies multiple locations):

        1. exact match           → apply
        2. whitespace-tolerant   → apply to the unique line-block match
        3. zero matches          → fail safely with a clear error
        4. multiple matches      → fail safely, require disambiguation

    Results include a bounded unified diff and the number of replaced
    occurrences so the runtime/verification can audit the change.
    """

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="edit_file",
            description=(
                "Edit a file by replacing a string. Matching is whitespace-tolerant: "
                "if the exact string is not found, a unique match ignoring indentation "
                "differences is applied. Never edits when the old string matches "
                "multiple locations — narrow the match instead."
            ),
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

    # ── matching helpers (module-level logic, unit-testable) ──────────

    @staticmethod
    def find_match_count(content: str, old_string: str) -> int:
        """Count exact, non-overlapping occurrences."""
        if not old_string:
            return 0
        return content.count(old_string)

    @staticmethod
    def _normalized_lines(text: str) -> list[str]:
        """Lines with all whitespace runs collapsed to single spaces."""
        return [" ".join(line.split()) for line in text.splitlines()]

    @classmethod
    def find_whitespace_matches(
        cls, content: str, old_string: str
    ) -> list[int]:
        """Find unique line-block matches ignoring whitespace differences.

        Returns the 0-based starting line numbers of matches. Only whole
        line-blocks are considered — never partial lines — so a match is
        unambiguous about where it starts and ends.
        """
        if not old_string.strip():
            return []
        old_lines = [l for l in cls._normalized_lines(old_string) if l]
        if not old_lines:
            return []
        norm_content = cls._normalized_lines(content)
        n = len(old_lines)
        matches: list[int] = []
        for i in range(len(norm_content) - n + 1):
            window = norm_content[i : i + n]
            # Skip windows that are entirely empty (would match anywhere).
            if all(not l for l in window):
                continue
            if window == old_lines:
                matches.append(i)
        return matches

    @staticmethod
    def build_diff(old_content: str, new_content: str, path: str) -> str:
        """Bounded unified diff for events/verification."""
        import difflib

        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=2,
        )
        text = "".join(diff)
        if len(text) > 4000:
            text = text[:4000] + "\n... [diff truncated]"
        return text

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            path = Path(arguments["path"])
            if not path.exists():
                return ToolResults.error(
                    f"File not found: {path}",
                    retryable=False,
                )

            content = path.read_text(encoding="utf-8")
            old_string = arguments["old_string"]
            new_string = arguments["new_string"]

            if old_string == new_string:
                return ToolResults.error(
                    "old_string and new_string are identical; nothing to edit",
                    retryable=False,
                )

            # ── Ladder 1: exact match ─────────────────────────────────
            exact_count = self.find_match_count(content, old_string)
            if exact_count == 1:
                new_content = content.replace(old_string, new_string, 1)
                path.write_text(new_content, encoding="utf-8")
                return ToolResults.success(
                    f"Edited {path} (exact match)",
                    metadata={
                        "match_type": "exact",
                        "replacements": 1,
                        "diff": self.build_diff(content, new_content, path.name),
                    },
                )
            if exact_count > 1:
                return ToolResults.error(
                    f"Ambiguous edit: old_string matches {exact_count} locations in "
                    f"{path}. Include more surrounding lines to make the match unique. "
                    "Nothing was modified.",
                    metadata={"match_type": "ambiguous", "match_count": exact_count},
                    retryable=False,
                )

            # ── Ladder 2: whitespace-tolerant unique line-block match ─
            ws_matches = self.find_whitespace_matches(content, old_string)
            if len(ws_matches) == 1:
                start = ws_matches[0]
                n = len([l for l in self._normalized_lines(old_string) if l])
                end = start + n
                new_content_lines = (
                    content.splitlines(keepends=True)[:start]
                    + self._reindented_replacement(
                        content.splitlines(keepends=True)[start:end],
                        old_string,
                        new_string,
                    )
                    + content.splitlines(keepends=True)[end:]
                )
                new_content = "".join(new_content_lines)
                path.write_text(new_content, encoding="utf-8")
                return ToolResults.success(
                    f"Edited {path} (whitespace-tolerant match at line {start + 1})",
                    metadata={
                        "match_type": "whitespace",
                        "replacements": 1,
                        "line_start": start + 1,
                        "diff": self.build_diff(content, new_content, path.name),
                    },
                )
            if len(ws_matches) > 1:
                return ToolResults.error(
                    f"Ambiguous edit: old_string matches {len(ws_matches)} locations "
                    f"(ignoring whitespace) in {path}. Nothing was modified.",
                    metadata={"match_type": "ambiguous", "match_count": len(ws_matches)},
                    retryable=False,
                )

            # ── Ladder 3: zero matches → fail safely ──────────────────
            return ToolResults.error(
                f"String not found in {path} (exact and whitespace-tolerant search). "
                "Re-read the file and retry with the exact current content. "
                "Nothing was modified.",
                metadata={"match_type": "none"},
                retryable=False,
            )
        except Exception as e:
            return ToolResults.from_exception(e, retryable=False)

    @staticmethod
    def _reindented_replacement(
        original_block: list[str], old_string: str, new_string: str
    ) -> list[str]:
        """Build the replacement block preserving the file's indentation.

        The whitespace-tolerant match ignored indentation; the replacement
        is aligned back onto the original block's leading whitespace so the
        edit lands cleanly.
        """
        # Detect the original block's base indentation from its first line.
        base_indent = ""
        for line in original_block:
            stripped = line.lstrip()
            if stripped:
                base_indent = line[: len(line) - len(stripped)]
                break
        # If new_string is just a whitespace-variant of old content, keep the
        # original block's lines but with normalized whitespace only when the
        # content is semantically identical; otherwise use new_string lines
        # aligned to base_indent.
        new_lines = new_string.splitlines(keepends=True)
        out: list[str] = []
        for line in new_lines:
            body = line.strip("\n")
            eol = "\n" if line.endswith("\n") else ""
            if body and not body.startswith((" ", "\t")):
                out.append(base_indent + body + eol)
            else:
                out.append(line if eol else line + eol)
        return out


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
                return ToolResults.error(
                    f"Path not found: {path}",
                    retryable=False,
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

            return ToolResults.success(
                "\n".join(entries) if entries else "(empty)",
            )
        except Exception as e:
            return ToolResults.from_exception(e, retryable=False)
