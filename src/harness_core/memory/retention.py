"""Memory retention policy: archive, summarise, and delete stale entries.

The policy prevents unbounded memory growth by enforcing per-entry rules:
  - Age: entries older than ``max_age_days`` are candidates for deletion.
  - Access frequency: entries with ``access_count == 0`` and high age are deleted.
  - Importance: entries below ``min_importance_to_keep`` are deleted.
  - Summary threshold: entries older than ``summarize_threshold_age_days`` are
    replaced with a summarised version (using the model provider if available).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from harness_core.memory.domain import MemoryEntry


@dataclass
class RetentionConfig:
    """Configuration for the retention policy."""

    max_entries: int = 10_000
    max_age_days: float = 90.0
    min_access_count_to_keep: int = 1
    min_importance_to_keep: float = 0.1
    summarize_threshold_age_days: float = 30.0
    archive_path: Path | None = None


@dataclass
class PruneReport:
    """Result of a retention policy application."""

    scanned: int = 0
    archived: int = 0
    deleted: int = 0
    summarized: int = 0
    freed_bytes: int = 0


class MemoryRetentionPolicy:
    """Determines which entries to archive, delete, or summarise."""

    def __init__(self, config: RetentionConfig | None = None) -> None:
        self.config = config or RetentionConfig()

    def should_archive(self, entry: MemoryEntry, now: float) -> bool:
        """Archive entries that are old enough but still worth keeping."""
        age_days = self._age_days(entry, now)
        if age_days < self.config.summarize_threshold_age_days:
            return False
        # Keep but archive (not delete): moderate importance or accessed once
        if entry.importance_score >= 0.3 or entry.access_count >= 1:
            return True
        return False

    def should_delete(self, entry: MemoryEntry, now: float) -> bool:
        """Hard-delete entries that are not worth keeping."""
        age_days = self._age_days(entry, now)
        # Over max age → delete
        if age_days > self.config.max_age_days:
            return True
        # Never accessed AND low importance → delete
        if (
            entry.access_count < self.config.min_access_count_to_keep
            and entry.importance_score < self.config.min_importance_to_keep
            and age_days > 7.0
        ):
            return True
        return False

    def should_summarise(self, entry: MemoryEntry, now: float) -> bool:
        """Replace old entries with summarised versions to save space."""
        age_days = self._age_days(entry, now)
        if age_days < self.config.summarize_threshold_age_days:
            return False
        # Only summarise if it's large and moderately important
        if len(entry.content) > 500 and entry.importance_score >= 0.3:
            return True
        return False

    def should_summarize(self, entry: MemoryEntry, now: float) -> bool:
        """Alias of :meth:`should_summarise` (American spelling)."""
        return self.should_summarise(entry, now)

    @staticmethod
    def _age_days(entry: MemoryEntry, now: float) -> float:
        return max(0.0, (now - entry.created_at) / 86400.0)

    async def summarize(
        self, entry: MemoryEntry, model: Any = None
    ) -> str:
        """Generate a summary of an entry's content.

        If `model` is a :class:`ModelProvider`, uses it to summarise.
        Otherwise returns the first 200 characters (fallback).
        """
        if model is None:
            return entry.content[:200] + "..." if len(entry.content) > 200 else entry.content
        try:
            prompt = (
                f"Summarise the following memory entry in ≤200 characters, "
                f"preserving all key facts:\n\n{entry.content}"
            )
            from harness_core.providers.base import CompletionRequest
            req = CompletionRequest(
                model="",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            resp = await model.generate(req)
            return resp.content[:500]
        except Exception:
            # Fallback: truncate
            return entry.content[:200] + "..." if len(entry.content) > 200 else entry.content

    async def apply(self, manager: Any) -> PruneReport:
        """Apply the retention policy to a MemoryManager.

        Args:
            manager: Must expose ``store`` with ``all_entries()``/``get()``/
                ``update()``/``delete()``.

        Order of operations per entry:
          1. delete  — if the entry is past its useful life;
          2. summarise — otherwise, if old and verbose but still valuable;
          3. archive  — otherwise, if old but worth keeping.
        """
        report = PruneReport()
        now = time.time()

        entries = await manager.store.all_entries()
        report.scanned = len(entries)

        to_delete: list[str] = []
        to_archive: list[str] = []
        to_summarise: list[str] = []  # entry ids

        for entry in entries:
            if entry.archived:
                continue  # already archived; never touch again
            if self.should_delete(entry, now):
                to_delete.append(entry.id)
            elif self.should_summarize(entry, now):
                to_summarise.append(entry.id)
            elif self.should_archive(entry, now):
                to_archive.append(entry.id)

        report.deleted = len(to_delete)
        report.archived = len(to_archive)
        report.summarized = len(to_summarise)

        for eid in to_delete:
            await manager.store.delete(eid)
        for eid in to_archive:
            existing = await manager.store.get(eid)
            if existing:
                await manager.store.update(replace(existing, archived=True))
        for eid in to_summarise:
            existing = await manager.store.get(eid)
            if existing:
                summary = await self.summarize(existing)
                await manager.store.update(
                    replace(
                        existing,
                        content=summary,
                        tags=list(dict.fromkeys([*(existing.tags or []), "summarised"])),
                        importance_score=round(existing.importance_score * 0.8, 4),
                        content_hash=MemoryEntry.compute_hash(summary),
                    )
                )

        return report
