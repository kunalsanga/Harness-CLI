"""
File content cache with LRU eviction and staleness detection.

Caches file contents indexed by (path, mtime). Automatically invalidates
stale entries when a file's modification time changes. Uses LRU eviction
to bound memory usage.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Directories to always skip when caching
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.tox', '.mypy_cache'}

# Patterns for sensitive files that should never be cached
SENSITIVE_PATTERNS = {'.env', '.env.*', '*.pem', '*.key', '*.secret', '*credentials*'}


@dataclass
class CacheStats:
    """Statistics for cache performance tracking."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    secret_skips: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'invalidations': self.invalidations,
            'secret_skips': self.secret_skips,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0,
        }


@dataclass
class FileMetadata:
    """Metadata about a cached file."""
    path: str
    size: int
    mtime: float
    content_hash: str
    line_count: int = 0
    is_binary: bool = False
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class _CacheEntry:
    """Internal cache entry."""
    content: str
    metadata: FileMetadata
    raw_bytes: bytes


class FileContentCache:
    """LRU file content cache with staleness detection.

    Args:
        max_entries: Maximum number of files to cache. 0 = unlimited.
        max_bytes: Maximum total bytes to cache. 0 = unlimited.
        workspace_root: Root directory for the workspace. Used to resolve
            relative paths and skip sensitive directories.
        ignore_patterns: Additional glob patterns for files that should
            never be cached.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        workspace_root: str | Path | None = None,
        ignore_patterns: Optional[set[str]] = None,
    ):
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self._stats = CacheStats()

        self._ignore_patterns = set(SENSITIVE_PATTERNS)
        if ignore_patterns:
            self._ignore_patterns |= ignore_patterns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, path: str | Path) -> Optional[str]:
        """Return cached text for *path*, or None on miss/stale/binary/missing.

        If the file has changed on disk since it was cached, the entry is
        invalidated and None is returned (treated as a cache miss).
        """
        path = self._resolve(path)

        with self._lock:
            entry = self._cache.get(path)
            if entry is None:
                self._stats.misses += 1
                return None

            # Validate staleness
            try:
                stat = os.stat(path)
            except OSError:
                # File deleted — evict
                self._evict_entry(path)
                self._stats.misses += 1
                return None

            if stat.st_mtime != entry.metadata.mtime or stat.st_size != entry.metadata.size:
                # Stale
                self._evict_entry(path)
                self._stats.invalidations += 1
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(path)
            self._stats.hits += 1
            return entry.content

    def put(self, path: str | Path, content: str | None = None) -> Optional[str]:
        """Read and cache the file at *path*, or store *content* directly.

        If *content* is provided, it is stored directly. Otherwise the file
        is read from disk.

        Returns the cached content, or None if the file should be skipped
        (binary, sensitive, missing, etc.).
        """
        path = self._resolve(path)

        # Check skip conditions
        if self._should_skip(path):
            return None

        if content is not None:
            # Store provided content directly
            raw_bytes = content.encode('utf-8')
            return self._store(path, raw_bytes)

        # Read from disk
        try:
            stat = os.stat(path)
            raw_bytes = Path(path).read_bytes()
        except (OSError, PermissionError):
            return None

        return self._store(path, raw_bytes)

    def get_metadata(self, path: str | Path) -> Optional[FileMetadata]:
        """Return metadata for a cached file, or None if not cached."""
        path = self._resolve(path)
        with self._lock:
            entry = self._cache.get(path)
            return entry.metadata if entry else None

    def contains(self, path: str | Path) -> bool:
        """Check if *path* is in the cache."""
        path = self._resolve(path)
        with self._lock:
            return path in self._cache

    def invalidate(self, path: str | Path) -> bool:
        """Remove *path* from the cache. Returns True if it was present."""
        path = self._resolve(path)
        with self._lock:
            return self._evict_entry(path)

    def invalidate_all(self) -> int:
        """Drop all cached entries. Returns count of entries removed."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._total_bytes = 0
            return count

    def clear(self) -> None:
        """Drop all cached entries."""
        self.invalidate_all()

    @property
    def size(self) -> int:
        """Number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                'entries': len(self._cache),
                'total_bytes': self._total_bytes,
                'stats': self._stats.to_dict(),
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, path: str | Path) -> str:
        """Resolve to absolute path string."""
        p = Path(path)
        if not p.is_absolute():
            p = self._workspace_root / p
        return str(p.resolve())

    def _store(self, path: str, raw_bytes: bytes) -> Optional[str]:
        """Store raw bytes in the cache. Returns decoded text or None."""
        is_binary = b'\x00' in raw_bytes[:8192]
        if is_binary:
            return None

        try:
            text = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            return None

        content_hash = hashlib.md5(raw_bytes).hexdigest()
        line_count = text.count('\n') + (1 if text and not text.endswith('\n') else 0)

        try:
            stat = os.stat(path)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            return None

        metadata = FileMetadata(
            path=path,
            size=size,
            mtime=mtime,
            content_hash=content_hash,
            line_count=line_count,
            is_binary=is_binary,
        )

        entry = _CacheEntry(content=text, metadata=metadata, raw_bytes=raw_bytes)

        with self._lock:
            old = self._cache.get(path)
            if old:
                self._total_bytes -= len(old.raw_bytes)
                del self._cache[path]

            self._cache[path] = entry
            self._total_bytes += len(raw_bytes)
            self._evict_lru()

        return text

    def _evict_entry(self, path: str) -> bool:
        """Evict a single entry. Must be called with lock held."""
        entry = self._cache.pop(path, None)
        if entry:
            self._total_bytes -= len(entry.raw_bytes)
            return True
        return False

    def _evict_lru(self) -> None:
        """Evict least-recently-used entries until within limits. Lock held."""
        while self._max_entries and len(self._cache) > self._max_entries:
            _path, entry = self._cache.popitem(last=False)
            self._total_bytes -= len(entry.raw_bytes)
            self._stats.evictions += 1

        while self._max_bytes and self._total_bytes > self._max_bytes:
            _path, entry = self._cache.popitem(last=False)
            self._total_bytes -= len(entry.raw_bytes)
            self._stats.evictions += 1

    def _should_skip(self, path: str) -> bool:
        """Check if path should be skipped (sensitive, binary dirs, etc.)."""
        from fnmatch import fnmatch

        # Check skip directories
        parts = Path(path).parts
        for part in parts:
            if part in SKIP_DIRS:
                return True

        # Check sensitive patterns
        basename = os.path.basename(path)
        if any(fnmatch(basename, pat) for pat in self._ignore_patterns):
            self._stats.secret_skips += 1
            return True

        return False
