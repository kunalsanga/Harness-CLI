"""
Lightweight repository dependency graph.

Tracks import/dependency relationships between files to help the context
engine identify related files. Supports Python, JS/TS, Rust, and Go.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Dependency:
    """A directed dependency edge."""
    source: str  # file that imports
    target: str  # file/module imported
    kind: str = 'import'  # 'import', 'include', 'use', 'require'
    line_number: int = 0


class DependencyGraph:
    """Lightweight dependency graph for repository files.

    Maintains forward (imports) and reverse (imported-by) maps to answer:
    - What does file X depend on?
    - What files depend on file X?
    - What is the transitive closure of file X's dependencies?
    """

    def __init__(self) -> None:
        self._forward: dict[str, set[str]] = defaultdict(set)  # file -> {imports}
        self._reverse: dict[str, set[str]] = defaultdict(set)  # file -> {imported by}
        self._edges: list[Dependency] = []
        self._module_to_file: dict[str, str] = {}  # module path -> file path
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_edge(
        self,
        source: str,
        target: str,
        kind: str = 'import',
        line_number: int = 0,
    ) -> None:
        """Add a dependency edge from source to target."""
        source = str(Path(source).resolve())
        target = str(Path(target).resolve())
        with self._lock:
            self._forward[source].add(target)
            self._reverse[target].add(source)
            self._edges.append(Dependency(
                source=source, target=target,
                kind=kind, line_number=line_number,
            ))

    def register_module(self, module_path: str, file_path: str) -> None:
        """Register a mapping from module path to file path."""
        with self._lock:
            self._module_to_file[module_path] = str(Path(file_path).resolve())

    def get_dependencies(self, path: str) -> list[str]:
        """Get files that *path* directly depends on."""
        path = str(Path(path).resolve())
        with self._lock:
            return sorted(self._forward.get(path, set()))

    def get_dependents(self, path: str) -> list[str]:
        """Get files that directly depend on *path*."""
        path = str(Path(path).resolve())
        with self._lock:
            return sorted(self._reverse.get(path, set()))

    def get_transitive_dependencies(
        self, path: str, max_depth: int = 5
    ) -> list[str]:
        """Get all transitive dependencies up to max_depth."""
        path = str(Path(path).resolve())
        visited: set[str] = set()
        result: list[str] = []

        def _dfs(current: str, depth: int) -> None:
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            if depth > 0:
                result.append(current)
            with self._lock:
                deps = self._forward.get(current, set())
            for dep in deps:
                _dfs(dep, depth + 1)

        _dfs(path, 0)
        return sorted(set(result))

    def get_transitive_dependents(
        self, path: str, max_depth: int = 3
    ) -> list[str]:
        """Get all files that transitively depend on *path*."""
        path = str(Path(path).resolve())
        visited: set[str] = set()
        result: list[str] = []

        def _dfs(current: str, depth: int) -> None:
            if depth > max_depth or current in visited:
                return
            visited.add(current)
            if depth > 0:
                result.append(current)
            with self._lock:
                deps = self._reverse.get(current, set())
            for dep in deps:
                _dfs(dep, depth + 1)

        _dfs(path, 0)
        return sorted(set(result))

    def resolve_module(self, module_path: str) -> Optional[str]:
        """Resolve a module path to a file path."""
        with self._lock:
            return self._module_to_file.get(module_path)

    def get_related_files(
        self, path: str, max_depth: int = 2
    ) -> list[str]:
        """Get files related to *path* through dependencies in both directions."""
        deps = self.get_transitive_dependencies(path, max_depth)
        dependents = self.get_transitive_dependents(path, max_depth)
        all_related = set(deps) | set(dependents)
        all_related.discard(str(Path(path).resolve()))
        return sorted(all_related)

    def has_cycle(self) -> bool:
        """Check if the graph has cycles (DFS-based)."""
        with self._lock:
            return self._has_cycle_unlocked()

    def _has_cycle_unlocked(self) -> bool:
        """Cycle detection without lock. Caller must hold lock."""
        white = set(self._forward.keys()) | set(self._reverse.keys())
        gray: set[str] = set()
        black: set[str] = set()

        def _visit(node: str) -> bool:
            white.discard(node)
            gray.add(node)
            for neighbor in self._forward.get(node, set()):
                if neighbor in gray:
                    return True
                if neighbor in white and _visit(neighbor):
                    return True
            gray.discard(node)
            black.add(node)
            return False

        while white:
            node = next(iter(white))
            if _visit(node):
                return True
        return False

    def clear(self) -> None:
        """Clear the entire graph."""
        with self._lock:
            self._forward.clear()
            self._reverse.clear()
            self._edges.clear()
            self._module_to_file.clear()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                'files': len(set(self._forward.keys()) | set(self._reverse.keys())),
                'edges': len(self._edges),
                'modules': len(self._module_to_file),
                'has_cycle': self._has_cycle_unlocked(),
            }
