"""Native performance extensions for Harness Engineering CLI.

Provides high-performance filesystem operations:
  - fast_glob: .gitignore-aware glob with parallel traversal
  - fast_grep: Regex text search with binary file avoidance
  - fast_file_index: File metadata collection
  - fast_hash / fast_batch_hash: Content hashing for deduplication
  - fast_count_files: Quick file counting

When the Rust native extension is unavailable, falls back to pure Python.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

# Try to load native extension
_NATIVE_AVAILABLE = False
try:
    from harness_fs import (
        fast_glob,
        fast_file_index,
        fast_grep,
        fast_hash,
        fast_batch_hash,
        fast_count_files,
    )
    _NATIVE_AVAILABLE = True
except ImportError:
    pass


def is_native_available() -> bool:
    """Check if native Rust extension is loaded."""
    return _NATIVE_AVAILABLE


# ── Python Fallback Implementations ───────────────────────────────────────


def _py_fast_glob(
    root: str,
    pattern: str,
    max_files: int = 0,
    respect_gitignore: bool = True,
    include_hidden: bool = False,
) -> list[str]:
    """Python fallback for fast_glob."""
    import fnmatch

    root_path = Path(root)
    if not root_path.exists():
        return []

    # Load gitignore patterns
    gitignore_patterns = set()
    if respect_gitignore:
        gitignore_file = root_path / ".gitignore"
        if gitignore_file.exists():
            try:
                for line in gitignore_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        gitignore_patterns.add(line)
            except Exception:
                pass

    results = []
    for entry in root_path.rglob("*"):
        if not entry.is_file():
            continue

        # Hidden files
        if not include_hidden and any(part.startswith(".") for part in entry.parts):
            continue

        # Gitignore check
        rel = entry.relative_to(root_path)
        rel_str = str(rel).replace("\\", "/")
        if any(fnmatch.fnmatch(rel_str, pat) for pat in gitignore_patterns):
            continue

        # Pattern match
        if fnmatch.fnmatch(entry.name, pattern) or fnmatch.fnmatch(rel_str, pattern):
            results.append(str(entry))
            if max_files > 0 and len(results) >= max_files:
                break

    return results


def _py_fast_grep(
    root: str,
    pattern: str,
    path_filter: str | None = None,
    max_results: int = 1000,
    case_insensitive: bool = False,
    respect_gitignore: bool = True,
) -> list[dict[str, str]]:
    """Python fallback for fast_grep."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    regex = re.compile(pattern, re.IGNORECASE if case_insensitive else 0)
    path_regex = re.compile(path_filter) if path_filter else None
    results = []

    for entry in root_path.rglob("*"):
        if not entry.is_file():
            continue
        if len(results) >= max_results:
            break

        # Path filter
        if path_regex and not path_regex.search(str(entry)):
            continue

        try:
            content = entry.read_text(encoding="utf-8", errors="ignore")
            if "\x00" in content[:1000]:
                continue  # Binary file

            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    results.append({
                        "file": str(entry),
                        "line": str(line_num),
                        "content": line.strip()[:200],
                        "match": regex.search(line).group() if regex.search(line) else "",
                    })
                    if len(results) >= max_results:
                        break
        except Exception:
            continue

    return results


def _py_fast_file_index(
    root: str,
    max_files: int = 0,
    respect_gitignore: bool = True,
) -> list[dict[str, Any]]:
    """Python fallback for fast_file_index."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    results = []
    for entry in root_path.rglob("*"):
        if not entry.is_file():
            continue
        try:
            stat = entry.stat()
            results.append({
                "path": str(entry),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "is_dir": False,
            })
            if max_files > 0 and len(results) >= max_files:
                break
        except Exception:
            continue

    return results


def _py_fast_hash(path: str) -> str:
    """Python fallback for fast_hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _py_fast_batch_hash(paths: list[str]) -> dict[str, str]:
    """Python fallback for fast_batch_hash."""
    return {p: _py_fast_hash(p) for p in paths if os.path.exists(p)}


def _py_fast_count_files(
    root: str,
    respect_gitignore: bool = True,
    extensions: list[str] | None = None,
) -> int:
    """Python fallback for fast_count_files."""
    root_path = Path(root)
    if not root_path.exists():
        return 0

    count = 0
    for entry in root_path.rglob("*"):
        if not entry.is_file():
            continue
        if extensions:
            if entry.suffix not in extensions:
                continue
        count += 1

    return count


# ── Exports — use native if available, else Python ────────────────────────

if _NATIVE_AVAILABLE:
    # Native Rust implementations are already imported
    pass
else:
    # Use Python fallbacks
    fast_glob = _py_fast_glob  # type: ignore
    fast_grep = _py_fast_grep  # type: ignore
    fast_file_index = _py_fast_file_index  # type: ignore
    fast_hash = _py_fast_hash  # type: ignore
    fast_batch_hash = _py_fast_batch_hash  # type: ignore
    fast_count_files = _py_fast_count_files  # type: ignore


__all__ = [
    "is_native_available",
    "fast_glob",
    "fast_grep",
    "fast_file_index",
    "fast_hash",
    "fast_batch_hash",
    "fast_count_files",
]
