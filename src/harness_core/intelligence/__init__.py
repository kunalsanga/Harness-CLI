"""Project intelligence — the canonical project brain.

One facade over the existing building blocks (no duplicate systems):

    ProjectIntelligence
        ├── scan()              fast repository scan (native when available)
        ├── refresh()           incremental update (mtime/hash diffing)
        ├── search()            layered text/regex search (cache-aware)
        ├── glob()              indexed filename matching (cache-aware)
        ├── symbols()           symbol lookup (SymbolIndex)
        ├── dependencies()      import/dependency graph queries
        ├── related()           dependents+dependencies of a file
        ├── context_candidates() ranked relevant files for a task
        └── invalidate()        file-scoped or full invalidation

Wires together EXISTING components:
    - indexing.symbols.SymbolIndex      regex symbol/import extraction
    - indexing.dependency_graph.DependencyGraph  import edges
    - analysis.relevance.RelevanceRanker         deterministic ranking
    - cache.search_cache.SearchCache    search result memoization
    - cache.file_cache.FileContentCache content memoization
    - harness_core.native fast_*        Rust engine (Python fallback)

The facade is synchronous and thread-safe; tools call it directly, the
context pipeline uses ``context_candidates()`` for candidate discovery.
"""

from __future__ import annotations

import fnmatch
import os
import re
import time
from pathlib import Path
from typing import Any

from harness_core.analysis.relevance import RelevanceRanker, RelevanceScore
from harness_core.cache.search_cache import SearchCache
from harness_core.indexing.dependency_graph import DependencyGraph
from harness_core.indexing.symbols import ImportInfo, Symbol, SymbolIndex
from harness_core.native import (
    fast_batch_hash,
    fast_file_index,
    fast_glob,
    fast_grep,
    is_native_available,
)

# Source extensions that participate in symbol/dependency indexing.
_SOURCE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
})

# Skip-list for traversal and indexing (mirrors RepositoryAnalyzer.SKIP_DIRS
# plus caches; kept local to avoid coupling the hot path to analysis).
_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".tox",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", ".harness",
    "dist", "build", "target", ".next", ".nuxt", ".eggs",
    ".uv-cache", ".tmp-pytest", ".freebuff", ".kilo",
})

_MAX_INDEX_BYTES = 1_000_000  # skip files > 1MB for symbol extraction


class _FileInfo:
    """Incremental-update record for one tracked file."""

    __slots__ = ("path", "mtime", "size", "hash")

    def __init__(self, path: str, mtime: float, size: int, content_hash: str) -> None:
        self.path = path
        self.mtime = mtime
        self.size = size
        self.hash = content_hash


class ProjectIntelligence:
    """Canonical project intelligence facade for one workspace root."""

    def __init__(self, root: str | Path, max_cache_entries: int = 500) -> None:
        self.root = str(Path(root).resolve())
        self.symbols = SymbolIndex()
        self.dependencies = DependencyGraph()
        self.ranker = RelevanceRanker()
        self.search_cache = SearchCache(max_entries=max_cache_entries, ttl_seconds=300.0)
        self._files: dict[str, _FileInfo] = {}
        self._last_scan: float = 0.0
        # Performance observability (durations in ms).
        self.last_scan_ms: float = 0.0
        self.last_refresh_ms: float = 0.0
        self.last_search_ms: float = 0.0
        self.last_index_update_ms: float = 0.0

    # ── paths ─────────────────────────────────────────────────────────

    def _abs(self, path: str | Path) -> str:
        p = Path(path)
        if not p.is_absolute():
            p = Path(self.root) / p
        return str(p.resolve())

    def _is_source(self, path: str) -> bool:
        return Path(path).suffix.lower() in _SOURCE_EXTENSIONS

    # ── scan / incremental refresh ────────────────────────────────────

    def scan(self, force: bool = False) -> dict[str, Any]:
        """Scan the workspace and index metadata + symbols + dependencies.

        Full scan only when the index is empty or ``force=True``; otherwise
        behaves as an incremental refresh. Returns index stats including
        durations for performance observability.
        """
        if force or not self._files:
            return self._full_scan()
        return self.refresh()

    def _full_scan(self) -> dict[str, Any]:
        started = time.monotonic()
        entries = fast_file_index(self.root, max_files=0, respect_gitignore=True)
        new_files: dict[str, _FileInfo] = {}

        # Metadata pass (batch hash only for source files worth indexing).
        to_hash: list[str] = []
        for entry in entries:
            path = str(entry.get("path", ""))
            if not path or not self._should_track(path):
                continue
            to_hash.append(path)
            new_files[path] = _FileInfo(path, float(entry.get("mtime", 0.0)), int(entry.get("size", 0)), "")

        hashes = fast_batch_hash(to_hash) if to_hash else {}
        for path, info in new_files.items():
            info.hash = hashes.get(path, "")

        # Track the file set BEFORE symbol/dependency extraction so that
        # _index_file can resolve imports against the full module candidate map.
        self._files = new_files

        # Symbol + dependency extraction per source file.
        index_started = time.monotonic()
        for path, info in new_files.items():
            if self._is_source(path) and info.size <= _MAX_INDEX_BYTES:
                self._index_file(path, info)
        self.last_index_update_ms = round((time.monotonic() - index_started) * 1000, 1)

        self._last_scan = time.monotonic()
        self.last_scan_ms = round((time.monotonic() - started) * 1000, 1)
        return self.stats

    def _should_track(self, path: str) -> bool:
        """Whether a file participates in the index at all."""
        try:
            rel = Path(path).relative_to(self.root)
        except ValueError:
            return False
        if any(part in _SKIP_DIRS for part in rel.parts):
            return False
        name = Path(path).name.lower()
        if name in {".env", ".env.local"} or name.endswith((".pem", ".key")):
            return False  # secrets are never indexed
        return True

    def _index_file(self, path: str, info: _FileInfo) -> None:
        """(Re)index one source file's symbols and dependency edges."""
        self.symbols.remove_file(path)
        # Re-index rebuilds only this file's own imports; incoming edges are
        # owned by the importing files and must survive the rebuild.
        self.dependencies.remove_outgoing_edges(path)
        self.symbols.index_file(path)
        # Build dependency edges from the file's imports.
        imports = self.symbols.get_imports_in_file(path)
        modules = self.dependencies.registered_modules()
        module_candidates: dict[str, str] = {}
        for tracked in self._files:
            if self._is_source(tracked):
                rel = Path(tracked).relative_to(self.root)
                module_candidates[_module_name(rel)] = tracked
        # Merge pre-registered modules.
        module_candidates.update(modules)
        for imp in imports:
            target = _resolve_import(imp, self.root, module_candidates)
            if target and target != path:
                self.dependencies.add_edge(path, target, kind="import", line_number=imp.line_number)
            elif imp.module:
                self.dependencies.register_module(imp.module, path)

    def refresh(self) -> dict[str, Any]:
        """Incremental update: only changed/new/deleted files are reprocessed."""
        started = time.monotonic()
        entries = fast_file_index(self.root, max_files=0, respect_gitignore=True)
        current: dict[str, tuple[float, int]] = {}
        to_hash: list[str] = []
        candidate_new: list[tuple[str, float, int]] = []

        for entry in entries:
            path = str(entry.get("path", ""))
            if not path or not self._should_track(path):
                continue
            mtime = float(entry.get("mtime", 0.0))
            size = int(entry.get("size", 0))
            current[path] = (mtime, size)
            known = self._files.get(path)
            if known is None or known.mtime != mtime or known.size != size:
                candidate_new.append((path, mtime, size))
                to_hash.append(path)

        hashes = fast_batch_hash(to_hash) if to_hash else {}

        # Deletions.
        deleted = [p for p in self._files if p not in current]
        index_started = time.monotonic()
        for path in deleted:
            self._drop_file(path)

        # New/changed files: hash-guarded (same hash → metadata-only update).
        changed = 0
        for path, mtime, size in candidate_new:
            content_hash = hashes.get(path, "")
            known = self._files.get(path)
            if known and known.hash and known.hash == content_hash:
                known.mtime, known.size = mtime, size
                continue
            info = _FileInfo(path, mtime, size, content_hash)
            self._files[path] = info
            if self._is_source(path) and size <= _MAX_INDEX_BYTES:
                self._index_file(path, info)
                self.search_cache.invalidate_for_file(path)
            changed += 1
        self.last_index_update_ms = round((time.monotonic() - index_started) * 1000, 1)

        self._last_scan = time.monotonic()
        self.last_refresh_ms = round((time.monotonic() - started) * 1000, 1)
        return {
            **self.stats,
            "changed": changed,
            "deleted": len(deleted),
            "incremental": True,
        }

    def _drop_file(self, path: str) -> None:
        """Remove a deleted file from every index structure."""
        self._files.pop(path, None)
        self.symbols.remove_file(path)
        self.dependencies.remove_file(path)
        self.search_cache.invalidate_for_file(path)

    def invalidate(self, path: str | Path | None = None) -> None:
        """Invalidate one file's index entries, or everything when None."""
        if path is None:
            self._files.clear()
            self.symbols.clear()
            self.dependencies.clear()
            self.search_cache.invalidate_all()
        else:
            self._drop_file(self._abs(path))

    def on_file_changed(self, path: str | Path) -> None:
        """Event hook for tools/watchers after a write: re-index just that file."""
        abs_path = self._abs(path)
        try:
            stat = Path(abs_path).stat()
        except OSError:
            self._drop_file(abs_path)
            return
        if not self._should_track(abs_path):
            return
        info = _FileInfo(abs_path, stat.st_mtime, stat.st_size, "")
        self._files[abs_path] = info
        if self._is_source(abs_path) and stat.st_size <= _MAX_INDEX_BYTES:
            self._index_file(abs_path, info)
        self.search_cache.invalidate_for_file(abs_path)

    # ── search layers ─────────────────────────────────────────────────

    def glob(self, pattern: str, path: str | None = None, limit: int = 200) -> list[str]:
        """Layer 1/2: indexed filename matching with cache."""
        base = self._abs(path) if path else self.root
        key = SearchCache.make_key("glob", root=base, pattern=pattern, limit=limit)
        cached = self.search_cache.get(key)
        if cached is not None:
            return list(cached)
        rel_base = str(Path(base).relative_to(self.root)) if base != self.root else ""
        results: list[str] = []
        # Pure filename glob ("*.py") answered from the index; path-aware
        # patterns fall back to the native walk (gitignore-aware).
        if "/" in pattern or "**" in pattern or path:
            matches = fast_glob(base, pattern, max_files=limit, respect_gitignore=True)
            results = [self._display(m) for m in matches]
        else:
            for tracked in self._files:
                if rel_base and not tracked.startswith(base):
                    continue
                if fnmatch.fnmatch(Path(tracked).name, pattern):
                    results.append(self._display(tracked))
                    if len(results) >= limit:
                        break
            results.sort()
        self.search_cache.put(key, results, related_files=list(results)[:50])
        return results

    def search(
        self,
        pattern: str,
        path: str | None = None,
        include: str | None = None,
        regex: bool = False,
        case_insensitive: bool = True,
        max_results: int = 100,
    ) -> list[dict[str, str]]:
        """Layer 3/4: fast text/regex search (native engine + cache).

        Results are relative display paths with 1-based line numbers,
        matching the historical GrepTool contract.
        """
        base = self._abs(path) if path else self.root
        key = SearchCache.make_key(
            "grep", root=base, pattern=pattern, include=include or "",
            regex=regex, ci=case_insensitive, limit=max_results,
        )
        cached = self.search_cache.get(key)
        if cached is not None:
            return [dict(m) for m in cached]

        started = time.monotonic()
        effective = pattern if regex else _literal_to_regex(pattern)
        raw = fast_grep(
            base,
            effective,
            path_filter=include,
            max_results=max_results,
            case_insensitive=case_insensitive,
            respect_gitignore=True,
        )
        results = [
            {
                "file": self._display(m.get("file", "")),
                "line": str(m.get("line", "")),
                "content": str(m.get("content", ""))[:200],
            }
            for m in raw
        ]
        self.last_search_ms = round((time.monotonic() - started) * 1000, 1)
        if results:
            # Positive results only: a cached empty result cannot be linked
            # to any file (no related_files), so it would survive file-change
            # invalidation and poison later searches. Negative caching is
            # deliberately skipped for correctness.
            self.search_cache.put(
                key, results,
                related_files=[m["file"] for m in results[:50]],
            )
        return results

    def find_symbols(self, query: str, kind: str | None = None, limit: int = 25) -> list[Symbol]:
        """Layer 5: symbol search across the indexed source files."""
        hits = self.symbols.search(query)
        if kind:
            hits = [s for s in hits if s.kind == kind]
        return hits[:limit]

    def find_definitions(self, name: str, limit: int = 10) -> list[Symbol]:
        """Definition lookup for a symbol name."""
        return self.symbols.find_definition(name)[:limit]

    def dependencies_of(self, path: str | Path, transitive: bool = False) -> list[str]:
        """Files that *path* imports (optionally transitively)."""
        p = self._abs(path)
        if transitive:
            return self.dependencies.get_transitive_dependencies(p)
        return self.dependencies.get_dependencies(p)

    def dependents_of(self, path: str | Path, transitive: bool = False) -> list[str]:
        """Files that import *path* (optionally transitively)."""
        p = self._abs(path)
        if transitive:
            return self.dependencies.get_transitive_dependents(p)
        return self.dependencies.get_dependents(p)

    def related(self, path: str | Path, max_depth: int = 2) -> list[str]:
        """Dependency-linked files in both directions (tests included)."""
        return self.dependencies.get_related_files(self._abs(path), max_depth)

    def find_related_tests(self, path: str | Path) -> list[str]:
        """Test files related to *path* by naming convention or imports."""
        p = Path(self._abs(path))
        stem = p.stem
        if stem.startswith("test_"):
            stem = stem[5:]
        candidates: list[str] = []
        for pattern in (f"test_{stem}.py", f"{stem}_test.py", f"test_{stem}.ts", f"{stem}.test.ts", f"{stem}.test.js"):
            for tracked in self._files:
                if Path(tracked).name == pattern:
                    candidates.append(tracked)
        # Imported-by test files via the dependency graph.
        for dep in self.dependencies.get_dependents(str(p)):
            if "test" in Path(dep).name.lower() and dep not in candidates:
                candidates.append(dep)
        return [self._display(c) for c in candidates]

    def context_candidates(
        self,
        task: str,
        search_matches: dict[str, list[str]] | None = None,
        seed_files: list[str] | None = None,
        max_results: int = 20,
    ) -> list[RelevanceScore]:
        """Layer 6/7: ranked relevant files for a task.

        Seeds = direct matches + dependency neighbors of matches; ranking is
        the existing deterministic RelevanceRanker over the full index.
        """
        tracked = list(self._files.keys())
        if not tracked:
            return []
        seed_set: set[str] = set(seed_files or [])
        for match_path in list((search_matches or {}).keys()):
            abs_match = self._abs(match_path)
            seed_set.add(abs_match)
            for rel in self.dependencies.get_related_files(abs_match, max_depth=1):
                seed_set.add(rel)
        display_seeds = {self._display(p) for p in seed_set}
        ranked = self.ranker.rank_files(
            [self._display(p) for p in tracked],
            task,
            search_matches=search_matches,
            max_results=max_results * 3,
        )
        # Boost dependency-linked seeds; they carry structural evidence.
        boosted: list[RelevanceScore] = []
        for score in ranked:
            if score.path in display_seeds:
                score.total_score += 0.15
                score.signals["dependency_seed"] = 1.0
            boosted.append(score)
        boosted.sort(key=lambda s: s.total_score, reverse=True)
        return boosted[:max_results]

    def file_list(self, limit: int = 500) -> list[str]:
        """Indexed files as workspace-relative display paths.

        ``limit <= 0`` returns every indexed file (used by tools that need
        the full set); a positive limit truncates.
        """
        items = [self._display(p) for p in sorted(self._files)]
        return items[:limit] if limit > 0 else items

    # ── helpers ───────────────────────────────────────────────────────

    def _display(self, abs_path: str) -> str:
        try:
            return str(Path(abs_path).relative_to(self.root)).replace("\\\\", "/")
        except ValueError:
            return abs_path.replace("\\\\", "/")

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files": len(self._files),
            "source_files": sum(1 for p in self._files if self._is_source(p)),
            "symbols": self.symbols.stats,
            "dependencies": self.dependencies.stats,
            "native": is_native_available(),
            "last_scan_ms": self.last_scan_ms,
            "last_refresh_ms": self.last_refresh_ms,
            "last_search_ms": self.last_search_ms,
            "last_index_update_ms": self.last_index_update_ms,
            "search_cache": self.search_cache.to_dict(),
        }


def _literal_to_regex(pattern: str) -> str:
    """Escape a literal search string into an equivalent regex."""
    return re.escape(pattern)


def _module_name(rel_path: Path) -> str:
    """File path → dotted module name (py) or path-like module (js/ts)."""
    parts = list(rel_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_import(
    imp: ImportInfo,
    root: str,
    module_candidates: dict[str, str],
) -> str | None:
    """Best-effort resolution of one import to a tracked file."""
    module = imp.module
    if not module:
        return None
    # from package import name → the specific submodule (package.name)
    # is a better target than the package __init__, so try it first.
    if imp.is_from:
        for name in imp.names:
            if name == "*":
                continue
            candidate = f"{module}.{name}"
            if candidate in module_candidates:
                return module_candidates[candidate]
            tail = candidate.split(".")[-1]
            for mod_key, file_path in module_candidates.items():
                if mod_key.endswith(f".{tail}") or mod_key == tail:
                    return file_path
    # Direct hit.
    if module in module_candidates:
        return module_candidates[module]
    for name in imp.names:
        candidate = f"{module}.{name}" if not imp.is_from else f"{module}.{name}"
        if candidate in module_candidates:
            return module_candidates[candidate]
        tail = candidate.split(".")[-1]
        for mod_key, file_path in module_candidates.items():
            if mod_key.endswith(f".{tail}") or mod_key == tail:
                return file_path
    # Relative suffix match (namespace packages, src layouts).
    tail = module.split(".")[-1]
    for mod_key, file_path in module_candidates.items():
        if mod_key.endswith(f".{tail}") or mod_key == tail:
            return file_path
    return None
