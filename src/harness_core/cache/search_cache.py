"""
Search result cache for glob and grep results.

Caches search operations keyed by deterministic hash of parameters.
Supports TTL-based expiry, LRU eviction, and bulk invalidation.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _CacheEntry:
    """A single cached search result."""
    result: Any
    timestamp: float = field(default_factory=time.monotonic)
    file_mtimes: dict[str, float] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.timestamp


class SearchCache:
    """LRU cache for search results (glob, grep, repository discovery).

    Args:
        max_entries: Maximum cached search results.
        ttl_seconds: Time-to-live for cache entries. 0 = no expiry.
    """

    def __init__(
        self,
        max_entries: int = 500,
        ttl_seconds: float = 60.0,
    ):
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        """Return cached result for *key*, or None on miss/expiry."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            if self._ttl and entry.age_seconds > self._ttl:
                self._cache.pop(key, None)
                self._misses += 1
                return None

            self._cache.move_to_end(key)
            self._hits += 1
            return entry.result

    def put(
        self,
        key: str,
        result: Any,
        related_files: Optional[list[str]] = None,
    ) -> None:
        """Store *result* under *key*."""
        file_mtimes: dict[str, float] = {}
        if related_files:
            import os
            for fp in related_files:
                try:
                    file_mtimes[fp] = os.stat(fp).st_mtime
                except OSError:
                    pass

        entry = _CacheEntry(result=result, file_mtimes=file_mtimes)

        with self._lock:
            old = self._cache.get(key)
            if old:
                del self._cache[key]

            self._cache[key] = entry
            self._evict_lru()

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if present."""
        with self._lock:
            entry = self._cache.pop(key, None)
            return entry is not None

    def invalidate_stale(self) -> int:
        """Remove all expired entries. Returns count removed."""
        count = 0
        with self._lock:
            to_remove = []
            for key, entry in self._cache.items():
                if self._ttl and entry.age_seconds > self._ttl:
                    to_remove.append(key)
            for key in to_remove:
                del self._cache[key]
                count += 1
        return count

    def invalidate_for_file(self, filepath: str) -> int:
        """Invalidate all cache entries that reference *filepath*."""
        count = 0
        with self._lock:
            to_remove = [
                key for key, entry in self._cache.items()
                if filepath in entry.file_mtimes
            ]
            for key in to_remove:
                del self._cache[key]
                count += 1
        return count

    def invalidate_all(self) -> int:
        """Drop all cached entries. Returns count removed."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
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
    def hit_rate(self) -> float:
        """Hit rate as a float between 0 and 1."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize cache state for inspection."""
        with self._lock:
            return {
                'entries': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': self.hit_rate,
            }

    @staticmethod
    def make_key(prefix: str, **kwargs: Any) -> str:
        """Build a deterministic cache key from a prefix and parameters."""
        parts = [prefix]
        for k in sorted(kwargs.keys()):
            v = kwargs[k]
            if isinstance(v, (list, tuple)):
                v = ','.join(str(x) for x in v)
            parts.append(f'{k}={v}')
        raw = '|'.join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        while self._max_entries and len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)
