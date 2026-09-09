"""Memory retrieval with similarity search and role-based access control.

The retriever is the *read* surface of the memory subsystem.  Models and
agents can read memories via high-level helpers (``retrieve_failures``,
``retrieve_successes``, etc.) but cannot write directly.  All retrievals
go through sanitisation before being returned to a prompt.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from harness_core.memory.domain import (
    MemoryEntry,
    MemoryType,
    RetrievalQuery,
    RetrievalResult,
)

if TYPE_CHECKING:
    from harness_core.memory.indexer import MemoryIndexer
    from harness_core.memory.sanitizer import MemorySanitizer
    from harness_core.memory.store import MemoryStore


# Role → set of memory types that role is permitted to read.
# Spec 8I: BACKEND/FRONTEND/DATABASE → SUCCESS/FAILURE/PROJECT/EPISODIC.
# ARCHITECT, PLANNER, REVIEWER, SECURITY_REVIEWER, ANALYZER, DEBUGGER
# can read broader types.  All agents can read SEMANTIC (learned facts).
_READABLE_TYPES_BY_ROLE: dict[str, set[MemoryType]] = {
    "backend": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    },
    "frontend": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    },
    "database": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    },
    "coder": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    },
    "architect": set(MemoryType),  # all
    "planner": set(MemoryType),  # all
    "reviewer": set(MemoryType),  # all, with redaction
    "security_reviewer": set(MemoryType),  # all, with extra redaction
    "analyzer": set(MemoryType),
    "debugger": set(MemoryType),
    "researcher": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
        MemoryType.ARCHITECTURE,
    },
    "ui_designer": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
    },
    "tester": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    },
    "integration": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
        MemoryType.SEMANTIC,
    },
    "verifier": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.SEMANTIC,
    },
    "git_release": {
        MemoryType.SUCCESS,
        MemoryType.FAILURE,
        MemoryType.PROJECT,
        MemoryType.EPISODIC,
    },
    "orchestrator": set(MemoryType),  # full visibility
}


def _types_for_role(role: str) -> set[MemoryType]:
    """Return the set of types a role can read (defaults to all for unknown)."""
    return _READABLE_TYPES_BY_ROLE.get(role, set(MemoryType))


class MemoryRetriever:
    """Read-only access to the memory store with similarity search."""

    def __init__(
        self,
        store: "MemoryStore",
        indexer: "MemoryIndexer",
        sanitizer: "MemorySanitizer",
    ) -> None:
        self.store = store
        self.indexer = indexer
        self.sanitizer = sanitizer

    # ── Core search ──────────────────────────────────────────────────────

    async def _search(
        self,
        query_text: str,
        memory_types: list[MemoryType] | None = None,
        agent_role: str | None = None,
        project_id: str | None = None,
        tags: list[str] | None = None,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> list[RetrievalResult]:
        """Run a similarity search and return ranked, sanitised results."""
        query_emb = self.indexer.embed(query_text)
        filters: dict = {"archived": False}
        if memory_types:
            filters["memory_types"] = memory_types
        if project_id is not None:
            filters["project_id"] = project_id
        if agent_role is not None:
            filters["agent_role"] = agent_role
        if tags:
            filters["tags"] = tags

        scored = await self.store.search(
            query_embedding=query_emb,
            filters=filters,
            top_k=top_k,
            min_score=min_score,
        )
        # Compute matched terms for the result wrapper
        query_tokens = set(self.indexer._tokenise(query_text))  # noqa: SLF001
        results: list[RetrievalResult] = []
        for entry, score in scored:
            entry_tokens = set(self.indexer._tokenise(entry.content))  # noqa: SLF001
            matched = sorted(query_tokens & entry_tokens)
            results.append(
                RetrievalResult(entry=entry, score=score, matched_terms=matched)
            )
        # Update access tracking (fire-and-forget; ignore errors)
        try:
            await self._record_access([r.entry for r in results])
        except Exception:
            pass
        return results

    async def _record_access(self, entries: list[MemoryEntry]) -> None:
        now = time.time()
        for entry in entries:
            entry.access_count += 1
            entry.last_accessed_at = now
            try:
                await self.store.update(entry)
            except KeyError:
                # Entry may have been deleted concurrently; ignore
                pass

    # ── Spec-required query helpers ──────────────────────────────────────

    async def retrieve_similar(self, query: RetrievalQuery) -> list[RetrievalResult]:
        return await self._search(
            query_text=query.query,
            memory_types=query.memory_types,
            agent_role=query.agent_role,
            project_id=query.project_id,
            tags=query.tags,
            top_k=query.top_k,
            min_score=query.min_score,
        )

    async def retrieve_failures(
        self, project_id: str = "", top_k: int = 5
    ) -> list[RetrievalResult]:
        pid = project_id if project_id else None
        # Empty query → broad failure memory dump
        scored = await self._list_by_type(
            memory_type=MemoryType.FAILURE, project_id=pid, top_k=top_k
        )
        return scored

    async def retrieve_successes(
        self, project_id: str = "", top_k: int = 5
    ) -> list[RetrievalResult]:
        pid = project_id if project_id else None
        return await self._list_by_type(
            memory_type=MemoryType.SUCCESS, project_id=pid, top_k=top_k
        )

    async def retrieve_architecture_decisions(
        self, project_id: str = "", top_k: int = 5
    ) -> list[RetrievalResult]:
        pid = project_id if project_id else None
        return await self._list_by_type(
            memory_type=MemoryType.ARCHITECTURE, project_id=pid, top_k=top_k
        )

    async def retrieve_project_context(
        self, project_id: str = "", top_k: int = 10
    ) -> list[RetrievalResult]:
        """Mixed-type overview of everything known about a project."""
        if not project_id:
            return []
        all_entries = await self.store.all_entries()
        candidates = [
            e for e in all_entries if e.project_id == project_id and not e.archived
        ]
        # Score by recency + importance
        now = time.time()
        scored: list[RetrievalResult] = []
        for e in candidates:
            age_days = max(0.0, (now - e.created_at) / 86400.0)
            recency = 1.0 / (1.0 + age_days / 7.0)  # weekly decay
            score = 0.6 * e.importance_score + 0.4 * recency
            scored.append(RetrievalResult(entry=e, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    async def retrieve_for_role(
        self, query: str, role: str, project_id: str = "", top_k: int = 5
    ) -> list[RetrievalResult]:
        """Role-aware retrieval: filters to types the role can read.

        Ranking is by similarity to ``query`` first; if nothing clears the
        similarity threshold, the top readable entries for the project are
        returned instead (ranked by importance + recency) so an agent still
        receives useful historical context rather than an empty block.
        """
        allowed = _types_for_role(role)
        # Empty allowed set → no access at all
        if not allowed:
            return []
        results = await self._search(
            query_text=query,
            memory_types=list(allowed),
            project_id=project_id if project_id else None,
            top_k=top_k,
        )
        if not results:
            results = await self._list_by_types(
                memory_types=list(allowed),
                project_id=project_id,
                top_k=top_k,
            )
        # Sanitise every entry before returning
        return [
            RetrievalResult(
                entry=self.sanitizer.sanitize_entry(r.entry),
                score=r.score,
                matched_terms=r.matched_terms,
            )
            for r in results
        ]

    # ── Internal ─────────────────────────────────────────────────────────

    async def _list_by_type(
        self,
        memory_type: MemoryType,
        project_id: str | None,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return top-k entries of a single type, ranked by importance + recency."""
        return await self._list_by_types(
            memory_types=[memory_type], project_id=project_id or "", top_k=top_k
        )

    async def _list_by_types(
        self,
        memory_types: list[MemoryType],
        project_id: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Return top-k entries of any of the given types, ranked by
        importance + recency.  Used as the fallback for role-scoped retrieval
        and by the single-type query helpers."""
        all_entries = await self.store.all_entries()
        candidates = [
            e for e in all_entries if e.type in memory_types and not e.archived
        ]
        if project_id:
            candidates = [e for e in candidates if e.project_id == project_id]
        now = time.time()
        scored: list[RetrievalResult] = []
        for e in candidates:
            age_days = max(0.0, (now - e.created_at) / 86400.0)
            recency = 1.0 / (1.0 + age_days / 14.0)
            score = 0.7 * e.importance_score + 0.3 * recency
            scored.append(RetrievalResult(entry=e, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        # Try to record access (best effort)
        try:
            await self._record_access([r.entry for r in scored[:top_k]])
        except Exception:
            pass
        return scored[:top_k]

    # ── Formatted output for prompt injection ────────────────────────────

    @staticmethod
    def format_for_prompt(results: list[RetrievalResult]) -> str:
        """Render a list of results as a markdown block for prompt injection."""
        if not results:
            return ""
        lines: list[str] = []
        for r in results:
            tag = r.entry.type.value
            score = f"{r.score:.2f}"
            lines.append(f"- [{tag} · score={score}] {r.entry.content}")
        return "\n".join(lines)
