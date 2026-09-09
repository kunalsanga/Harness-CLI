"""Authoritative runtime state for the unified engineering runtime (Phase 9).

There is exactly ONE ProjectState per execution and it is owned by
``EngineeringRuntime``.  It deliberately does **not** duplicate TaskGraph
state: task progress, readiness and per-task results live in the TaskGraph
(the authority for task state).  ProjectState references the graph and
derives its summary numbers from it.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness_core.agents.domain import TaskGraph
    from harness_core.runtime.requirements import Requirements, TraceabilityIndex


class RuntimeStage(enum.Enum):
    """The explicit project execution lifecycle.

    Not every stage must run for every project: the runtime advances through
    the stages its TaskGraph actually exercises.
    """

    UNKNOWN = "unknown"
    DISCOVER = "discover"
    UNDERSTAND = "understand"
    PLAN = "plan"
    DESIGN = "design"
    DECOMPOSE = "decompose"
    EXECUTE = "execute"
    INTEGRATE = "integrate"
    TEST = "test"
    DEBUG = "debug"
    REVIEW = "review"
    VERIFY = "verify"
    DELIVER = "deliver"
    # Terminal outcomes (not traversable forward stages):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


# Ordered stages used to derive stage transitions for state snapshots.
_STAGE_ORDER: list[RuntimeStage] = [
    RuntimeStage.UNKNOWN,
    RuntimeStage.DISCOVER,
    RuntimeStage.UNDERSTAND,
    RuntimeStage.PLAN,
    RuntimeStage.DESIGN,
    RuntimeStage.DECOMPOSE,
    RuntimeStage.EXECUTE,
    RuntimeStage.INTEGRATE,
    RuntimeStage.TEST,
    RuntimeStage.DEBUG,
    RuntimeStage.REVIEW,
    RuntimeStage.VERIFY,
    RuntimeStage.DELIVER,
]


class RuntimeStatus(enum.Enum):
    """Terminal status of a project execution.

    Failure semantics (Phase 9 §18): a task is never promoted from FAILED to
    COMPLETED without a *new* execution (the scheduler's retest tasks), and a
    model's claim of success never yields SUCCESS without going through the
    verify stage.
    """

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class VerificationStatus(enum.Enum):
    """Project-level verification posture."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class StageRecord:
    """One lifecycle-stage entry."""

    stage: RuntimeStage
    started_at: float = field(default_factory=time.time)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage.value, "started_at": self.started_at, "detail": self.detail}


@dataclass
class ArtifactRef:
    """A structured reference to a produced artifact.

    Agents exchange *references* to artifacts (paths + kind + summary)
    instead of copying large content around.  Memory may persist these.
    """

    # api_contract | database_schema | architecture_decision | ui_spec |
    # test_report | failure_report | review_report | source
    kind: str
    path: str
    produced_by_task: str = ""
    summary: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "produced_by_task": self.produced_by_task,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        return cls(
            kind=data.get("kind", "source"),
            path=data.get("path", ""),
            produced_by_task=data.get("produced_by_task", ""),
            summary=data.get("summary", ""),
        )


@dataclass
class ActiveAgent:
    """A running agent as seen by the runtime (name + role + status)."""

    name: str
    role: str
    status: str = "running"
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "task_id": self.task_id,
        }


@dataclass
class ProjectState:
    """Runtime-owned view of one project execution.

    Task-level truth lives in ``task_graph``; this object holds project-level
    truth (lifecycle stage, requirements, traces, artifacts, warnings) plus
    derived convenience counters computed from the graph on demand.
    """

    project_id: str = ""
    workspace: str = ""
    original_request: str = ""
    status: RuntimeStatus = RuntimeStatus.RUNNING
    stage: RuntimeStage = RuntimeStage.UNKNOWN
    stage_history: list[StageRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    requirements: Requirements | None = None
    traceability: TraceabilityIndex | None = None

    # References to authoritative components (not copies).
    task_graph: TaskGraph | None = None

    artifacts: list[ArtifactRef] = field(default_factory=list)
    # task_id -> ActiveAgent, maintained from scheduler task.* events.
    active_agents: dict[str, ActiveAgent] = field(default_factory=dict)

    recovery_attempts: int = 0
    recovery_exhausted: bool = False
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    verification_status: VerificationStatus = VerificationStatus.NOT_STARTED
    verification_summary: str = ""
    verification_results: list[dict[str, Any]] = field(default_factory=list)
    failed_task_ids: list[str] = field(default_factory=list)

    # ── stage transitions ────────────────────────────────────────────────

    def set_stage(self, stage: RuntimeStage, detail: str = "") -> RuntimeStage:
        """Record a lifecycle stage transition. Returns the new stage."""
        previous = self.stage
        if previous != stage:
            self.stage_history.append(StageRecord(stage=stage, detail=detail))
        self.stage = stage
        return stage

    # ── derived state (TaskGraph stays authoritative) ────────────────────

    @property
    def task_progress(self) -> tuple[int, int]:
        """(completed, total) from the authoritative TaskGraph."""
        if self.task_graph is None:
            return 0, 0
        return self.task_graph.get_completed_count(), self.task_graph.get_total_count()

    @property
    def completed_tasks(self) -> int:
        return self.task_progress[0]

    @property
    def total_tasks(self) -> int:
        return self.task_progress[1]

    @property
    def failed_count(self) -> int:
        if self.task_graph is None:
            return 0
        return self.task_graph.get_failed_count()

    @property
    def active_task_ids(self) -> list[str]:
        """Tasks that are RUNNING or QUEUED right now (from the graph)."""
        if self.task_graph is None:
            return []
        from harness_core.agents.domain import TaskStatus

        return [
            t.task_id
            for t in self.task_graph.tasks.values()
            if t.status in (TaskStatus.RUNNING, TaskStatus.QUEUED)
        ]

    def register_artifact(self, artifact: ArtifactRef) -> None:
        existing = [
            a for a in self.artifacts if a.path == artifact.path and a.kind == artifact.kind
        ]
        if not existing:
            self.artifacts.append(artifact)

    def register_warning(self, warning: str) -> None:
        if warning and warning not in self.warnings:
            self.warnings.append(warning)

    def register_blocker(self, blocker: str) -> None:
        if blocker and blocker not in self.blockers:
            self.blockers.append(blocker)

    # ── snapshot for the CLI ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Structured state a terminal/CLI can render without a UI redesign."""
        completed, total = self.task_progress
        terminal = self.stage in (
            RuntimeStage.SUCCEEDED,
            RuntimeStage.FAILED,
            RuntimeStage.BLOCKED,
            RuntimeStage.CANCELLED,
        )
        return {
            "project_id": self.project_id,
            "workspace": self.workspace,
            "status": self.status.value,
            "terminal": terminal,
            "stage": self.stage.value,
            "stage_history": [r.to_dict() for r in self.stage_history],
            "request": self.original_request,
            "progress": {"completed": completed, "total": total},
            "active": [a.to_dict() for a in self.active_agents.values()],
            "active_task_ids": self.active_task_ids,
            "failed_tasks": self.failed_task_ids,
            "recovery": {"attempts": self.recovery_attempts, "exhausted": self.recovery_exhausted},
            "verification": {
                "status": self.verification_status.value,
                "summary": self.verification_summary,
                "results": self.verification_results,
            },
            "artifacts": [a.to_dict() for a in self.artifacts],
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "duration_ms": round((self.completed_at - self.started_at) * 1000, 1)
            if self.completed_at
            else None,
        }
