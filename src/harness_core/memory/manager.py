"""MemoryManager — the central, runtime-controlled write surface for memory.

This is the *only* module that writes to the memory store.  Models and
agents can never write memories directly; they can only request reads via
the retriever.  This enforces the security boundary defined in Phase 8I.

Usage::

    mm = MemoryManager(store, graph, retriever, indexer, sanitizer, policy)
    id = await mm.record_success(task, result, project_id="my-project")
    ctx = await mm.get_context_for_planner("build auth system")
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from harness_core.memory.domain import (
    EdgeType,
    GraphEdge,
    GraphNode,
    MemoryEntry,
    MemoryType,
    NodeType,
    RetrievalQuery,
)
from harness_core.memory.graph import KnowledgeGraph
from harness_core.memory.indexer import MemoryIndexer
from harness_core.memory.retention import MemoryRetentionPolicy, PruneReport, RetentionConfig
from harness_core.memory.retriever import MemoryRetriever
from harness_core.memory.sanitizer import MemorySanitizer
from harness_core.memory.store import LocalMemoryStore, MemoryStore

# ── Singleton ──────────────────────────────────────────────────────────────

_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager | None:
    """Return the current singleton, or None if not initialised."""
    return _memory_manager


def init_memory_manager(
    workspace_path: Path,
    *,
    enabled: bool = False,
    config: dict[str, Any] | None = None,
) -> MemoryManager:
    """Initialise (or re-initialise) the global MemoryManager singleton.

    Args:
        workspace_path: Root of the workspace (used to resolve .harness/memory/).
        enabled: If False, the manager is a no-op stub (writes are silently dropped).
        config: Optional overrides for RetentionConfig keys.
    """
    global _memory_manager
    _memory_manager = MemoryManager.from_workspace(
        workspace_path, enabled=enabled, config=config
    )
    return _memory_manager


def init_memory_manager_from_project(
    workspace_path: Path,
    config: dict[str, Any] | None = None,
    *,
    default_enabled: bool = True,
) -> MemoryManager | None:
    """Best-effort bootstrap hook used by application/CLI startup paths.

    Resolves the enable flag from project config (``memory.enabled`` under the
    top-level ``config`` dict, defaulting to ``default_enabled``) and calls
    :func:`init_memory_manager`.  It **never raises**: if the store cannot be
    initialised the manager degrades to the disabled no-op singleton, or
    ``None`` if even that fails — memory is an enhancement to execution and
    must never be able to crash normal agent startup.

    Returns:
        An enabled ``MemoryManager``, a disabled no-op manager, or ``None``
        when memory is opted out / unavailable.  All runtime consumers treat
        ``None`` and a disabled manager identically.
    """
    cfg = config or {}
    mem_cfg = cfg.get("memory") or {}
    if not isinstance(mem_cfg, dict):
        mem_cfg = {}
    enabled = bool(mem_cfg.get("enabled", default_enabled))

    if not enabled:
        # Opted out: keep the singleton unset/disabled without side effects.
        return None

    try:
        return init_memory_manager(workspace_path, enabled=True)
    except Exception:
        # Fall back to the inert disabled manager; if even that fails (e.g.
        # the workspace is not writable) return None rather than crash.
        try:
            return init_memory_manager(workspace_path, enabled=False)
        except Exception:
            return None


# ── Memory Manager ────────────────────────────────────────────────────────


class MemoryManager:
    """Central facade for the memory subsystem.

    All write operations go through this class so that:
      1. Every entry is sanitised before persistence.
      2. Deduplication is enforced (same content_hash + project → single entry).
      3. The knowledge graph is kept in sync.
      4. Retention policy is applied on demand.
    """

    def __init__(
        self,
        store: MemoryStore,
        graph: KnowledgeGraph,
        retriever: MemoryRetriever,
        indexer: MemoryIndexer,
        sanitizer: MemorySanitizer,
        retention_policy: MemoryRetentionPolicy,
        *,
        enabled: bool = True,
    ) -> None:
        self.store = store
        self.graph = graph
        self.retriever = retriever
        self.indexer = indexer
        self.sanitizer = sanitizer
        self.retention_policy = retention_policy
        self.enabled = enabled

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        *,
        enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> MemoryManager:
        base = workspace_path / ".harness" / "memory"
        cfg = config or {}
        ret_cfg = RetentionConfig(
            max_entries=cfg.get("max_entries", 10_000),
            max_age_days=cfg.get("max_age_days", 90.0),
            min_importance_to_keep=cfg.get("min_importance_to_keep", 0.1),
            summarize_threshold_age_days=cfg.get("summarize_threshold_age_days", 30.0),
        )
        store = LocalMemoryStore(base / "entries.json")
        graph = KnowledgeGraph(storage_path=base / "graph.json")
        indexer = MemoryIndexer()
        sanitizer = MemorySanitizer()
        retriever = MemoryRetriever(store=store, indexer=indexer, sanitizer=sanitizer)
        policy = MemoryRetentionPolicy(config=ret_cfg)
        return cls(
            store=store,
            graph=graph,
            retriever=retriever,
            indexer=indexer,
            sanitizer=sanitizer,
            retention_policy=policy,
            enabled=enabled,
        )

    # ── Write operations (runtime-controlled only) ─────────────────────────

    async def record_success(
        self,
        task_id: str,
        description: str,
        outcome: str,
        agent_role: str,
        project_id: str = "",
        tags: list[str] | None = None,
        importance: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a successful task outcome as a SUCCESS memory entry."""
        if not self.enabled:
            return ""
        content = f"Task: {description}\nOutcome: {outcome}"
        return await self._record(
            content=content,
            mtype=MemoryType.SUCCESS,
            agent_role=agent_role,
            project_id=project_id,
            task_id=task_id,
            tags=tags,
            importance=importance,
            metadata=metadata,
        )

    async def record_failure(
        self,
        task_id: str,
        description: str,
        error: str,
        agent_role: str,
        project_id: str = "",
        tags: list[str] | None = None,
        importance: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a task failure as a FAILURE memory entry."""
        if not self.enabled:
            return ""
        content = f"Task: {description}\nError: {error}"
        return await self._record(
            content=content,
            mtype=MemoryType.FAILURE,
            agent_role=agent_role,
            project_id=project_id,
            task_id=task_id,
            tags=tags,
            importance=importance,
            metadata=metadata,
        )

    async def record_architecture(
        self,
        decision: str,
        rationale: str,
        agent_role: str,
        project_id: str = "",
        task_id: str = "",
        tags: list[str] | None = None,
        importance: float = 0.9,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record an architecture decision record (ADR)."""
        if not self.enabled:
            return ""
        content = f"Decision: {decision}\nRationale: {rationale}"
        return await self._record(
            content=content,
            mtype=MemoryType.ARCHITECTURE,
            agent_role=agent_role,
            project_id=project_id,
            task_id=task_id,
            tags=tags or ["architecture", "adr"],
            importance=importance,
            metadata=metadata,
        )

    async def record_decision(
        self,
        decision: str,
        evidence: str,
        agent_role: str,
        project_id: str = "",
        tags: list[str] | None = None,
        importance: float = 0.7,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a reviewer-validated decision."""
        if not self.enabled:
            return ""
        content = f"Decision: {decision}\nEvidence: {evidence}"
        return await self._record(
            content=content,
            mtype=MemoryType.DECISION,
            agent_role=agent_role,
            project_id=project_id,
            task_id="",
            tags=tags or ["decision"],
            importance=importance,
            metadata=metadata,
        )

    async def record_episode(
        self,
        description: str,
        outcome: str,
        agent_role: str,
        project_id: str = "",
        tags: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Record a general episodic event."""
        if not self.enabled:
            return ""
        content = f"Event: {description}\nOutcome: {outcome}"
        return await self._record(
            content=content,
            mtype=MemoryType.EPISODIC,
            agent_role=agent_role,
            project_id=project_id,
            task_id="",
            tags=tags,
            importance=importance,
            metadata=metadata,
        )

    # ── Internal write helper ─────────────────────────────────────────────

    async def _record(
        self,
        content: str,
        mtype: MemoryType,
        agent_role: str,
        project_id: str,
        task_id: str,
        tags: list[str] | None,
        importance: float,
        metadata: dict[str, Any] | None,
    ) -> str:
        # 1. Sanitise
        safe_content = self.sanitizer.sanitize(content, role=agent_role)
        safe_hash = MemoryEntry.compute_hash(content)  # hash on original

        # 2. Check for dedup
        if project_id:
            candidates = await self.store.all_entries()
            for candidate in candidates:
                if (
                    candidate.content_hash == safe_hash
                    and candidate.project_id == project_id
                    and not candidate.archived
                ):
                    # Boost importance of existing entry
                    candidate.importance_score = min(
                        1.0, candidate.importance_score + 0.1
                    )
                    await self.store.update(candidate)
                    return candidate.id

        # 3. Build entry
        entry = MemoryEntry(
            id=uuid.uuid4().hex[:16],
            type=mtype,
            content=safe_content,
            content_hash=safe_hash,
            tags=tags or [],
            project_id=project_id,
            task_id=task_id,
            agent_role=agent_role,
            importance_score=importance,
            metadata=metadata or {},
        )

        # 4. Index and save
        entry.embedding = self.indexer.index(entry)
        await self.store.save(entry)

        # 5. Sync graph
        if task_id:
            task_node = GraphNode(
                id=task_id,
                type=NodeType.TASK,
                label=description_from_content(content),
                properties={"outcome": mtype.value},
            )
            self.graph.add_node(task_node)
            if mtype == MemoryType.FAILURE:
                self.graph.add_edge(
                    GraphEdge(
                        src=task_id,
                        dst=task_id,  # self-loop for failure
                        type=EdgeType.CAUSED_BY,
                        properties={"error": error_from_content(content)},
                    )
                )
            elif mtype == MemoryType.SUCCESS:
                self.graph.add_edge(
                    GraphEdge(
                        src=task_id,
                        dst=task_id,
                        type=EdgeType.CREATED,
                    )
                )
        return entry.id

    # ── High-level retrieval (for RAG injection) ──────────────────────────

    async def get_context_for_planner(
        self, user_request: str, project_id: str = ""
    ) -> str:
        """Return formatted context for the Planner (Phase 8F)."""
        if not self.enabled:
            return ""
        pid = project_id if project_id else None
        # Gather failures, successes, and architecture decisions
        failures = await self.retriever.retrieve_failures(project_id=project_id or "", top_k=3)
        successes = await self.retriever.retrieve_successes(project_id=project_id or "", top_k=3)
        arch = await self.retriever.retrieve_architecture_decisions(
            project_id=project_id or "", top_k=2
        )
        # Also do a semantic similarity search on the user's request
        similar = await self.retriever.retrieve_similar(
            RetrievalQuery(query=user_request, project_id=pid, top_k=3, min_score=0.1)
        )

        parts: list[str] = []
        if similar:
            parts.append("## Prior Similar Projects\n" + MemoryRetriever.format_for_prompt(similar))
        if failures:
            parts.append("## Known Past Failures (avoid these)\n" + MemoryRetriever.format_for_prompt(failures))
        if successes:
            parts.append("## Prior Successful Approaches\n" + MemoryRetriever.format_for_prompt(successes))
        if arch:
            parts.append("## Architecture Decisions\n" + MemoryRetriever.format_for_prompt(arch))
        return "\n\n".join(parts)

    async def get_context_for_worker(
        self, objective: str, role: str, project_id: str = ""
    ) -> str:
        """Return formatted context for a WorkerAgent (Phase 8G)."""
        if not self.enabled:
            return ""
        pid = project_id if project_id else None
        results = await self.retriever.retrieve_for_role(
            query=objective,
            role=role,
            project_id=project_id,
            top_k=5,
        )
        if not results:
            return ""
        # Separate by type for readability
        failures = [r for r in results if r.entry.type == MemoryType.FAILURE]
        successes = [r for r in results if r.entry.type == MemoryType.SUCCESS]
        other = [r for r in results if r.entry.type not in (MemoryType.FAILURE, MemoryType.SUCCESS)]
        parts: list[str] = []
        if failures:
            parts.append("## Past Failures to Avoid\n" + MemoryRetriever.format_for_prompt(failures))
        if successes:
            parts.append("## Prior Successes\n" + MemoryRetriever.format_for_prompt(successes))
        if other:
            parts.append("## Relevant Context\n" + MemoryRetriever.format_for_prompt(other))
        return "\n\n".join(parts)

    async def get_failures_for_recovery(
        self, task_description: str, project_id: str = ""
    ) -> str:
        """Return failure context for RecoveryOrchestrator."""
        if not self.enabled:
            return ""
        results = await self.retriever.retrieve_failures(project_id=project_id or "", top_k=3)
        if not results:
            return ""
        return "## Prior Failures for This Task\n" + MemoryRetriever.format_for_prompt(results)

    # ── Admin ────────────────────────────────────────────────────────────

    async def prune(self) -> PruneReport:
        """Run the retention policy and return a report."""
        return await self.retention_policy.apply(self)

    async def stats(self) -> dict[str, Any]:
        """Return memory subsystem statistics."""
        all_entries = await self.store.all_entries()
        by_type: dict[str, int] = {}
        for e in all_entries:
            by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
        return {
            "total_entries": len(all_entries),
            "by_type": by_type,
            "graph_stats": self.graph.stats(),
            "file_size_bytes": (
                self.store.file_size_bytes() if hasattr(self.store, "file_size_bytes") else 0
            ),
        }


# ── Helpers ────────────────────────────────────────────────────────────────


def description_from_content(content: str) -> str:
    try:
        return content.split("\n", 1)[0].replace("Task: ", "").replace("Event: ", "")
    except Exception:
        return content[:80]


def error_from_content(content: str) -> str:
    try:
        for line in content.split("\n"):
            if line.startswith("Error: "):
                return line[7:]
    except Exception:
        pass
    return ""
