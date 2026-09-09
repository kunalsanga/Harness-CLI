"""Memory store — pluggable persistence backends.

The :class:`MemoryStore` ABC defines the interface; the only shipped
implementation today is :class:`LocalMemoryStore` (JSON-on-disk).  Future
implementations (Chroma, Qdrant, Weaviate, PGVector) plug in by
implementing the same interface — no other module needs to change.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from harness_core.memory.domain import MemoryEntry, MemoryType


# ── Abstract interface ─────────────────────────────────────────────────────


class MemoryStore(ABC):
    """Pluggable persistence interface for memory entries."""

    @abstractmethod
    async def save(self, entry: MemoryEntry) -> str:
        """Persist an entry.  Returns the entry ID."""

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None:
        """Fetch an entry by ID, or None if not found."""

    @abstractmethod
    async def update(self, entry: MemoryEntry) -> None:
        """Replace an existing entry (raises KeyError if missing)."""

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Hard-delete an entry.  Returns True if it existed."""

    @abstractmethod
    async def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count entries matching the given filters."""

    @abstractmethod
    async def all_entries(self) -> list[MemoryEntry]:
        """Return every entry (used during index rebuild)."""

    async def search(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        """Cosine-similarity search.  Subclasses may override for efficiency.

        The default implementation iterates all entries and ranks them in
        pure Python.  Vector-database backends (Chroma/Qdrant/PGVector)
        should override this to push the search to the database.
        """
        scored: list[tuple[MemoryEntry, float]] = []
        for entry in await self.all_entries():
            if not self._matches_filters(entry, filters):
                continue
            if not entry.embedding:
                continue
            score = _cosine(query_embedding, entry.embedding)
            if score >= min_score:
                scored.append((entry, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _matches_filters(entry: MemoryEntry, filters: dict[str, Any] | None) -> bool:
        if not filters:
            return True
        # Type filter
        types = filters.get("memory_types")
        if types and entry.type not in types:
            return False
        # Project filter
        pid = filters.get("project_id")
        if pid is not None and entry.project_id != pid:
            return False
        # Agent role filter
        role = filters.get("agent_role")
        if role is not None and entry.agent_role != role:
            return False
        # Tags filter — entry must share at least one tag
        tags = filters.get("tags")
        if tags and not any(t in entry.tags for t in tags):
            return False
        # Archive filter
        archived = filters.get("archived", False)
        if entry.archived != archived:
            return False
        return True


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; both vectors should be unit-length."""
    if len(a) != len(b) or not a:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


# ── Local JSON implementation ──────────────────────────────────────────────


class LocalMemoryStore(MemoryStore):
    """JSON-file backed store.

    All entries live in a single JSON file (``.harness/memory/entries.json``).
    Writes are atomic (write to temp, rename).  Loads the entire index in
    memory on construction — fine for the expected scale of tens of
    thousands of entries.  Use Chroma/Qdrant for larger corpora.
    """

    def __init__(self, storage_path: Path | str) -> None:
        self.storage_path = Path(storage_path)
        self._lock = asyncio.Lock()
        # id -> MemoryEntry
        self._cache: dict[str, MemoryEntry] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load all entries from disk; tolerant of missing or corrupt files."""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            return
        try:
            raw = self.storage_path.read_text(encoding="utf-8")
            data = json.loads(raw) if raw.strip() else {"entries": []}
        except (json.JSONDecodeError, OSError):
            # Corrupt file: log and start fresh (don't crash the harness)
            return
        for entry_data in data.get("entries", []):
            try:
                entry = MemoryEntry.from_dict(entry_data)
                self._cache[entry.id] = entry
            except (KeyError, ValueError):
                # Skip malformed entries rather than failing the whole load
                continue

    def _atomic_write(self) -> None:
        """Write the current cache to disk atomically."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"entries": [e.to_dict() for e in self._cache.values()]}
        tmp = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.storage_path)

    # ── Public API ────────────────────────────────────────────────────────

    async def save(self, entry: MemoryEntry) -> str:
        async with self._lock:
            self._cache[entry.id] = entry
            self._atomic_write()
        return entry.id

    async def get(self, entry_id: str) -> MemoryEntry | None:
        return self._cache.get(entry_id)

    async def update(self, entry: MemoryEntry) -> None:
        async with self._lock:
            if entry.id not in self._cache:
                raise KeyError(f"Memory entry {entry.id!r} not found")
            self._cache[entry.id] = entry
            self._atomic_write()

    async def delete(self, entry_id: str) -> bool:
        async with self._lock:
            if entry_id not in self._cache:
                return False
            del self._cache[entry_id]
            self._atomic_write()
            return True

    async def count(self, filters: dict[str, Any] | None = None) -> int:
        if not filters:
            return len(self._cache)
        return sum(
            1 for e in self._cache.values() if self._matches_filters(e, filters)
        )

    async def all_entries(self) -> list[MemoryEntry]:
        return list(self._cache.values())

    # ── Inspection / test helpers ──────────────────────────────────────────

    def file_size_bytes(self) -> int:
        try:
            return self.storage_path.stat().st_size
        except OSError:
            return 0


# ── Pluggability hook ──────────────────────────────────────────────────────


def memory_store_from_url(url: str, *, fallback_path: Path | None = None) -> MemoryStore:
    """Factory that picks a backend based on a URL scheme.

    Schemes:
      - ``file://...`` / ``json://...`` / no scheme → :class:`LocalMemoryStore`
      - ``chroma://...`` / ``qdrant://...`` / ``weaviate://...`` /
        ``pgvector://...`` → not yet implemented; falls back to local.
    """
    if url.startswith(("chroma://", "qdrant://", "weaviate://", "pgvector://")):
        # Future: dispatch to the appropriate backend. For now, fall back.
        if fallback_path is None:
            fallback_path = Path.cwd() / ".harness" / "memory" / "entries.json"
        return LocalMemoryStore(fallback_path)
    # Strip optional scheme
    path = url.split("://", 1)[-1] if "://" in url else url
    return LocalMemoryStore(path)
