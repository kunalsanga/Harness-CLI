"""Canonical models for the context intelligence pipeline.

These types are the stable contract between context discovery (filesystem,
index, memory, reuse cache, AI relevance) and the agent loop. They are pure
data — no I/O happens in this module — so they are safe to import from
anywhere without cycles.

Architecture (Phase 2.0 "context intelligence"):

    USER TASK
       ↓
    ContextPipeline.discover(request: ContextRequest)
       ├── candidate sources (deterministic)
       │     • existing context reuse (ContextReuseManager snapshots)
       │     • repository index (ContextEngine.discover_project)
       │     • path/name matching (RelevanceRanker)
       │     • memory (optional injected provider)
       │     • AI relevance (optional model-assisted re-ranking)
       ├── candidate ranking
       ├── dependency/relationship expansion where available
       └── budget fit (ContextBudgetManager)

The agent loop consumes a ContextSelection; it never talks to candidate
sources directly. Future Engineering Brain code can reuse these models
without touching the loop.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any


class CandidateSource(str, Enum):
    """Where a context candidate came from.

    Deterministic sources are always available; the AI source is only
    present when model-assisted ranking succeeded.
    """

    REUSE_CACHE = "reuse_cache"        # served from ContextReuseManager snapshot
    REPOSITORY_INDEX = "repo_index"    # ContextEngine project discovery
    PATH_MATCH = "path_match"          # deterministic name/path scoring
    GREP = "grep"                      # content search hits
    SYMBOL = "symbol"                  # symbol index hits
    DEPENDENCY = "dependency"          # import/dependency expansion
    RECENT = "recent"                  # recently read/touched files
    MEMORY = "memory"                  # persistent memory store
    AI_RANKING = "ai_ranking"          # model-assisted relevance stage


class Freshness(str, Enum):
    """Freshness state of a piece of context, per the reuse contract."""

    FRESH = "fresh"          # read from disk this cycle
    CACHED = "cached"        # unchanged since last read (reuse snapshot valid)
    INVALIDATED = "invalidated"  # snapshot dropped (file changed on disk)
    SUMMARIZED = "summarized"    # content replaced by a compaction summary
    STALE = "stale"          # snapshot predates a known mutation


@dataclass
class ContextEvidence:
    """Why a candidate is believed relevant.

    Evidence is deliberately structured (source + score + optional detail)
    so the UI and future Engineering Brain can audit *why* a file was
    selected instead of trusting an opaque score.
    """

    source: CandidateSource
    score: float = 0.0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "score": round(self.score, 4),
            "detail": self.detail,
        }


@dataclass
class ContextCandidate:
    """One file (or piece) proposed as context for a task."""

    path: str                      # workspace-relative path, forward slashes
    score: float = 0.0             # aggregate relevance score
    evidence: list[ContextEvidence] = field(default_factory=list)
    freshness: Freshness = Freshness.FRESH
    tokens_estimate: int = 0
    exists: bool = True            # validated against the filesystem
    snippet: str = ""              # optional short excerpt (bounded)

    def add_evidence(self, ev: ContextEvidence) -> None:
        """Attach evidence and fold its score into the aggregate.

        Aggregate uses a max-plus-bonus fold: the strongest evidence
        dominates and each additional distinct source adds a small
        corroborating bonus, so one strong signal beats many weak ones
        but multi-source agreement still wins.
        """
        self.evidence.append(ev)
        if ev.score >= self.score:
            self.score = ev.score + 0.05 * len(
                {e.source for e in self.evidence}
            ) - 0.05
        self.score = round(min(self.score, 1.5), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "score": round(self.score, 4),
            "evidence": [e.to_dict() for e in self.evidence],
            "freshness": self.freshness.value,
            "tokens_estimate": self.tokens_estimate,
        }


@dataclass
class ContextRequest:
    """A request for context assembled around a user task."""

    task: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    agent_id: str = ""
    max_candidates: int = 24          # bounded candidate discovery
    max_files: int = 8                # bounded selected files
    max_file_tokens: int = 2000       # per-file content cap
    include_ai_ranking: bool = True   # model-assisted stage allowed
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextSnapshot:
    """Immutable record of what context was given to the model.

    Emitted as ``context.snapshot`` metadata and persisted with the run so
    post-hoc analysis can answer "which files were selected and why".
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task: str = ""
    candidates_considered: int = 0
    selected: list[ContextCandidate] = field(default_factory=list)
    tokens_used: int = 0
    tokens_budget: int = 0
    ai_ranking_used: bool = False
    ai_ranking_failed: bool = False
    duration_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    def file_paths(self) -> list[str]:
        return [c.path for c in self.selected]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task[:200],
            "candidates_considered": self.candidates_considered,
            "selected": [c.to_dict() for c in self.selected],
            "tokens_used": self.tokens_used,
            "tokens_budget": self.tokens_budget,
            "ai_ranking_used": self.ai_ranking_used,
            "ai_ranking_failed": self.ai_ranking_failed,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ContextSelection:
    """The result of context discovery: selected files + full snapshot."""

    snapshot: ContextSnapshot
    files: list[str] = field(default_factory=list)      # workspace-relative
    absolute_paths: list[str] = field(default_factory=list)
    fresh_contents: dict[str, str] = field(default_factory=dict)  # path → content read this cycle

    def to_context_pieces(self) -> list[SimpleNamespace]:
        """Convert selected files into engine-style context pieces.

        Returns lightweight objects with the same shape as
        ``ContextEngine.ContextPiece`` (source/content/priority/
        tokens_estimate/metadata) so the agent loop can treat them
        uniformly without an import cycle.
        """
        pieces: list[SimpleNamespace] = []
        for cand in self.snapshot.selected:
            content = self.fresh_contents.get(cand.path)
            if content is None:
                continue
            pieces.append(
                SimpleNamespace(
                    source=f"file:{cand.path}",
                    content=content,
                    priority=50.0 + cand.score,
                    tokens_estimate=cand.tokens_estimate,
                    metadata={"path": cand.path},
                )
            )
        return pieces
