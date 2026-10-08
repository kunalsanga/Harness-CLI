"""Search tools for the agent."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, TYPE_CHECKING

from harness_core.agent.types import ToolResult, ToolResults
from harness_core.tools.base import Tool, ToolSchema

if TYPE_CHECKING:
    from harness_core.intelligence import ProjectIntelligence


def _rebase_or_none(root: Path, display: str, base: Path) -> Path | None:
    """Convert an index display path (root-relative) to *base*-relative."""
    p = root / display
    try:
        rel = p.relative_to(base)
    except ValueError:
        return None
    if any(part.startswith(".") for part in rel.parts):
        return None
    return rel


class GlobTool(Tool):
    """Find files matching a glob pattern.

    When a :class:`ProjectIntelligence` instance is provided and the requested
    directory lies inside the indexed workspace, matches are answered from the
    index (no directory walk). Falls back to ``rglob`` otherwise; the output
    contract (paths relative to the search path, filename-glob semantics) is
    identical in both modes.
    """

    def __init__(self, intelligence: "ProjectIntelligence | None" = None) -> None:
        self.intelligence = intelligence

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
                return ToolResults.error(
                    f"Path not found: {search_path}",
                    retryable=False,
                )

            matches = self._glob_via_walk(pattern, search_path)
            if self.intelligence is not None:
                fast = self._glob_via_intelligence(pattern, search_path)
                if fast is not None:
                    matches = fast

            return ToolResults.success(
                "\n".join(matches[:200]) if matches else "(no matches)",
                metadata={"count": len(matches)},
            )
        except Exception as e:
            return ToolResults.from_exception(e, retryable=False)

    def _glob_via_intelligence(
        self, pattern: str, search_path: Path
    ) -> list[str] | None:
        """Indexed filename matching (fast path).

        Returns ``None`` when the fast path cannot serve the request (cold
        index or a directory outside the indexed workspace) so the caller
        falls back to the walk.
        """
        intel = self.intelligence
        assert intel is not None
        root = Path(intel.root)
        try:
            base = search_path.resolve()
            base.relative_to(root)
        except (OSError, ValueError):
            return None
        if not intel.file_list(limit=1):
            return None  # cold index → walk

        # Historical semantics: match the file NAME against the last
        # component of the pattern (so "**/*.py" matches any .py file).
        name_pattern = pattern.split("/")[-1]
        matches: list[str] = []
        for display in intel.file_list(limit=0):
            rel = _rebase_or_none(root, display, base)
            if rel is None:
                continue
            if fnmatch.fnmatch(Path(display).name, name_pattern):
                matches.append(str(rel))
        return sorted(matches)

    def _glob_via_walk(self, pattern: str, search_path: Path) -> list[str]:
        return sorted(
            str(p.relative_to(search_path))
            for p in search_path.rglob("*")
            if fnmatch.fnmatch(p.name, pattern.split("/")[-1])
            and not any(part.startswith(".") for part in p.parts)
        )


class GrepTool(Tool):
    """Search for text patterns in files.

    With a :class:`ProjectIntelligence` instance, uses the native search
    engine with cache-backed memoization. Falls back to a pure-Python line
    scan otherwise; the output contract (``rel:line: content`` rows,
    case-insensitive literal matching, 100-result cap) is identical.
    """

    def __init__(self, intelligence: "ProjectIntelligence | None" = None) -> None:
        self.intelligence = intelligence

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
                return ToolResults.error(
                    f"Path not found: {search_path}",
                    retryable=False,
                )

            results, files_searched = self._grep_via_walk(pattern, search_path, include)
            if self.intelligence is not None:
                fast = self._grep_via_intelligence(pattern, search_path, include)
                if fast is not None:
                    results, files_searched = fast

            output = "\n".join(results) if results else "(no matches)"
            return ToolResults.success(
                output,
                metadata={"files_searched": files_searched, "matches": len(results)},
            )
        except Exception as e:
            return ToolResults.from_exception(e, retryable=False)

    def _grep_via_intelligence(
        self, pattern: str, search_path: Path, include: str
    ) -> tuple[list[str], int] | None:
        """Native/cached search (fast path).

        Returns ``None`` when the fast path cannot serve the request (cold
        index or a directory outside the indexed workspace) so the caller
        falls back to the walk.
        """
        intel = self.intelligence
        assert intel is not None
        root = Path(intel.root)
        try:
            base = search_path.resolve()
            base.relative_to(root)
        except (OSError, ValueError):
            return None
        if not intel.file_list(limit=1):
            return None  # cold index → walk

        raw = intel.search(
            pattern,
            path=str(base),
            max_results=300,
            case_insensitive=True,
            regex=False,
        )
        results: list[str] = []
        files_seen: set[str] = set()
        for m in raw:
            rel = _rebase_or_none(root, m.get("file", ""), base)
            if rel is None:
                continue
            if not fnmatch.fnmatch(Path(str(rel)).name, include):
                continue
            files_seen.add(str(rel))
            results.append(f"{rel}:{m.get('line', '')}: {m.get('content', '').strip()}")
            if len(results) >= 100:
                break
        return results, len(files_seen)

    def _grep_via_walk(
        self, pattern: str, search_path: Path, include: str
    ) -> tuple[list[str], int]:
        results: list[str] = []
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
        return results, files_searched
