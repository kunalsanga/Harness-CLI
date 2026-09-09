"""
Persistent memory subsystem for Phase 8.

Provides long-term, permission-aware memory across independent executions.

Key types:
    MemoryEntry      — a single memory record
    MemoryType       — EPISODIC / SEMANTIC / PROJECT / FAILURE / SUCCESS / ARCHITECTURE / DECISION
    MemoryManager    — the only write surface (runtime-controlled); also provides RAG helpers
    MemoryStore      — pluggable persistence (LocalMemoryStore ships; others planned)
    KnowledgeGraph    — relationship graph between tasks, files, decisions, failures
    MemoryRetriever  — similarity search + role-based retrieval
    MemorySanitizer  — strips secrets/credentials before persistence
    MemoryRetentionPolicy — prevents unbounded growth

Usage::

    from harness_core.memory import init_memory_manager, get_memory_manager

    # Initialise once at startup
    mm = init_memory_manager(Path("/path/to/workspace"), enabled=True)

    # Use anywhere in the codebase
    mm = get_memory_manager()
    if mm:
        ctx = await mm.get_context_for_planner("build auth system")
        id = await mm.record_success(task_id="...", description="...", outcome="...")
"""

from harness_core.memory.domain import (
    EdgeType,
    GraphEdge,
    GraphNode,
    MemoryEntry,
    MemoryType,
    NodeType,
    RetrievalQuery,
    RetrievalResult,
)
from harness_core.memory.graph import KnowledgeGraph
from harness_core.memory.indexer import MemoryIndexer
from harness_core.memory.manager import (
    MemoryManager,
    get_memory_manager,
    init_memory_manager,
    init_memory_manager_from_project,
)
from harness_core.memory.retention import (
    MemoryRetentionPolicy,
    PruneReport,
    RetentionConfig,
)
from harness_core.memory.retriever import MemoryRetriever
from harness_core.memory.sanitizer import MemorySanitizer
from harness_core.memory.store import (
    LocalMemoryStore,
    MemoryStore,
    memory_store_from_url,
)

__all__ = [
    # Domain
    "MemoryEntry",
    "MemoryType",
    "RetrievalQuery",
    "RetrievalResult",
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    # Store
    "MemoryStore",
    "LocalMemoryStore",
    "memory_store_from_url",
    # Retrieval
    "MemoryRetriever",
    # Indexer
    "MemoryIndexer",
    # Sanitizer
    "MemorySanitizer",
    # Graph
    "KnowledgeGraph",
    # Retention
    "MemoryRetentionPolicy",
    "RetentionConfig",
    "PruneReport",
    # Manager
    "MemoryManager",
    "init_memory_manager",
    "init_memory_manager_from_project",
    "get_memory_manager",
]
