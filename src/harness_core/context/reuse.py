"""Context reuse manager (Phase 10.5, Part 17).

Tracks file content snapshots so the runtime can avoid re-reading files that
have not changed since they were last read.  This is a *runtime* concern —
agents should not blindly re-inject unchanged file contents into every model
request.

Pure and deterministic: operates on paths + content bytes/hashes, no I/O.
The engine performs the actual filesystem reads; this manager decides whether
a read is worth repeating based on the snapshot it last observed.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileSnapshot:
    """Last observed state of one file."""

    path: str
    content_hash: str
    size: int
    mtime_ns: int
    at: float = field(default_factory=time.time)


class ContextReuseManager:
    """Snapshot-aware context reuse for file reads.

    ``is_unchanged(path, size, mtime_ns, content)`` is the single decision
    point: it compares against the snapshot recorded by ``record_read`` and
    returns True only when nothing changed.  ``invalidate`` drops a snapshot
    (e.g. after a write/edit), and ``invalidate_all`` resets everything.
    """

    def __init__(self, *, max_snapshots: int = 512) -> None:
        self._snapshots: dict[str, FileSnapshot] = {}
        self.max_snapshots = max_snapshots

    @staticmethod
    def content_hash(content: bytes | str) -> str:
        data = content.encode("utf-8") if isinstance(content, str) else content
        return hashlib.sha256(data).hexdigest()[:24]

    def record_read(
        self,
        path: str,
        *,
        content: bytes | str,
        size: int | None = None,
        mtime_ns: int | None = None,
    ) -> FileSnapshot:
        """Record that `path` was read with the given state. Returns the snapshot."""
        if size is None:
            size = len(content) if isinstance(content, str) else len(content)
        snap = FileSnapshot(
            path=path,
            content_hash=self.content_hash(content),
            size=size,
            mtime_ns=mtime_ns or 0,
        )
        self._snapshots[path] = snap
        if len(self._snapshots) > self.max_snapshots:
            # Drop oldest by insertion time (dicts preserve insertion order).
            for oldest in list(self._snapshots)[: len(self._snapshots) - self.max_snapshots]:
                self._snapshots.pop(oldest, None)
        return snap

    def is_unchanged(
        self,
        path: str,
        *,
        content: bytes | str | None = None,
        size: int | None = None,
        mtime_ns: int | None = None,
    ) -> bool:
        """True when the file's current state matches the last recorded read.

        Prefer passing `content` (hash is authoritative); size/mtime are used
        as a fast-path pre-filter when content is unavailable.
        """
        snap = self._snapshots.get(path)
        if snap is None:
            return False
        if mtime_ns and snap.mtime_ns and mtime_ns != snap.mtime_ns:
            return False
        if size is not None and size != snap.size:
            return False
        if content is not None:
            if self.content_hash(content) != snap.content_hash:
                return False
        return True

    def has_snapshot(self, path: str) -> bool:
        return path in self._snapshots

    def snapshot(self, path: str) -> FileSnapshot | None:
        return self._snapshots.get(path)

    def invalidate(self, path: str) -> None:
        """Drop the snapshot for `path` (file changed or was written)."""
        self._snapshots.pop(path, None)

    def invalidate_all(self) -> None:
        self._snapshots = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self._snapshots),
            "paths": sorted(self._snapshots),
        }
