"""Domain models for the persistent memory subsystem."""

from __future__ import annotations

import enum
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


# ── Memory Types ───────────────────────────────────────────────────────────


class MemoryType(str, enum.Enum):
    """Kinds of memory stored in the system."""

    #: "We did X" — event records from task execution
    EPISODIC = "episodic"
    #: "X works because Y" — learned facts
    SEMANTIC = "semantic"
    #: Project-level facts and metadata
    PROJECT = "project"
    #: "This approach failed because Z"
    FAILURE = "failure"
    #: "This approach worked"
    SUCCESS = "success"
    #: Architecture decision records (ADRs)
    ARCHITECTURE = "architecture"
    #: Reviewer-validated decisions
    DECISION = "decision"


# ── Core Memory Entry ──────────────────────────────────────────────────────


@dataclass
class MemoryEntry:
    """A single memory record stored in the system.

    Entries are persisted via MemoryStore. All writes go through
    MemoryManager, never directly to the store, so sanitisation
    and deduplication are always applied.
    """

    #: Unique identifier (uuid hex)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    #: Kind of memory
    type: MemoryType = MemoryType.EPISODIC
    #: Human-readable content
    content: str = ""
    #: TF-IDF / token-hash embedding vector.  Used for similarity search.
    embedding: list[float] = field(default_factory=list)
    #: Free-form topic tags
    tags: list[str] = field(default_factory=list)
    #: Unix timestamp when the entry was created
    created_at: float = field(default_factory=time.time)
    #: "" = global; non-empty = project-scoped
    project_id: str = ""
    #: SubTask.task_id that produced this entry ("" if manual)
    task_id: str = ""
    #: AgentRole.value that produced this entry
    agent_role: str = ""
    #: Arbitrary structured metadata (e.g. error category, file paths)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: 0.0–1.0 importance; used for retention scoring
    importance_score: float = 0.5
    #: How many times this entry was retrieved
    access_count: int = 0
    #: Last retrieval timestamp
    last_accessed_at: float | None = None
    #: Soft-deleted entries are excluded from normal retrieval
    archived: bool = False
    #: SHA-256 of the raw (pre-sanitisation) content for dedup
    content_hash: str = ""

    # ── Convenience ──────────────────────────────────────────────────────

    @staticmethod
    def compute_hash(content: str) -> str:
        """Return SHA-256 hex of content for deduplication."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "embedding": self.embedding,
            "tags": self.tags,
            "created_at": self.created_at,
            "project_id": self.project_id,
            "task_id": self.task_id,
            "agent_role": self.agent_role,
            "metadata": self.metadata,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "archived": self.archived,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            embedding=data["embedding"],
            tags=data["tags"],
            created_at=data["created_at"],
            project_id=data.get("project_id", ""),
            task_id=data.get("task_id", ""),
            agent_role=data.get("agent_role", ""),
            metadata=data.get("metadata", {}),
            importance_score=data.get("importance_score", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed_at=data.get("last_accessed_at"),
            archived=data.get("archived", False),
            content_hash=data.get("content_hash", ""),
        )


# ── Retrieval ─────────────────────────────────────────────────────────────


@dataclass
class RetrievalQuery:
    """A query submitted to the retriever."""

    query: str = ""
    #: Restrict to these types; None = all types
    memory_types: list[MemoryType] | None = None
    #: Restrict to entries created by this role; None = any
    agent_role: str | None = None
    #: Restrict to this project; None = any
    project_id: str | None = None
    #: Restrict to entries sharing at least one tag; None = any
    tags: list[str] | None = None
    #: Maximum number of results
    top_k: int = 5
    #: Minimum cosine similarity score to return
    min_score: float = 0.1


@dataclass
class RetrievalResult:
    """A single result from the retriever."""

    entry: MemoryEntry
    #: Cosine similarity score (0.0–1.0)
    score: float
    #: Tokens from the query that matched the entry
    matched_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "score": round(self.score, 4),
            "matched_terms": self.matched_terms,
        }


# ── Knowledge Graph ───────────────────────────────────────────────────────


class NodeType(str, enum.Enum):
    """Types of nodes in the knowledge graph."""

    TASK = "task"
    AGENT = "agent"
    DECISION = "decision"
    FILE = "file"
    ARCHITECTURE = "architecture"
    FAILURE = "failure"
    SOLUTION = "solution"


class EdgeType(str, enum.Enum):
    """Types of edges in the knowledge graph."""

    CREATED = "created"
    FIXED = "fixed"
    DEPENDS_ON = "depends_on"
    RELATED_TO = "related_to"
    IMPLEMENTED_BY = "implemented_by"
    CAUSED_BY = "caused_by"


@dataclass
class GraphNode:
    """A node in the knowledge graph."""

    id: str
    type: NodeType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "label": self.label,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphNode:
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            label=data["label"],
            properties=data.get("properties", {}),
        )


@dataclass
class GraphEdge:
    """A directed edge in the knowledge graph."""

    src: str
    dst: str
    type: EdgeType
    properties: dict[str, Any] = field(default_factory=dict)
    #: Higher weight → stronger relationship; used in ranking
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "type": self.type.value,
            "properties": self.properties,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphEdge:
        return cls(
            src=data["src"],
            dst=data["dst"],
            type=EdgeType(data["type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
        )
