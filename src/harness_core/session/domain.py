"""
Session domain model — the durable unit of engineering work in M4.

Entities:
  Session — top-level container for engineering work
  Run — a single agent execution within a session
  Checkpoint — snapshot of state at a safe boundary
  ContextSnapshot — cached repository context references
  MemoryItem — structured memory (decisions, discoveries, constraints, etc.)
  SessionEvent — persisted event log entry
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Enums ────────────────────────────────────────────────────────────────

class SessionStatus(enum.Enum):
    """Explicit session lifecycle states."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    ARCHIVED = "archived"


class RunStatus(enum.Enum):
    """Run lifecycle states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class MemoryType(enum.Enum):
    """Categories of structured session memory."""
    DECISION = "decision"
    DISCOVERY = "discovery"
    CONSTRAINT = "constraint"
    TODO = "todo"
    WARNING = "warning"
    ERROR = "error"
    SOLUTION = "solution"
    NOTE = "note"


# ── Valid state transitions ──────────────────────────────────────────────

# Session: which transitions are legal
SESSION_TRANSITIONS: dict[SessionStatus, set[SessionStatus]] = {
    SessionStatus.ACTIVE: {SessionStatus.PAUSED, SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.ABORTED},
    SessionStatus.PAUSED: {SessionStatus.ACTIVE, SessionStatus.ABORTED, SessionStatus.ARCHIVED},
    SessionStatus.COMPLETED: {SessionStatus.ARCHIVED},  # can archive, but not restart
    SessionStatus.FAILED: {SessionStatus.ACTIVE, SessionStatus.ARCHIVED},  # allow retry
    SessionStatus.ABORTED: {SessionStatus.ARCHIVED},
    SessionStatus.ARCHIVED: set(),  # terminal
}

RUN_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.INTERRUPTED},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
    RunStatus.INTERRUPTED: set(),
}


def can_transition(current: enum.Enum, target: enum.Enum, transitions: dict) -> bool:
    """Check if a state transition is valid."""
    allowed = transitions.get(current, set())
    return target in allowed


# ── Domain objects ───────────────────────────────────────────────────────

@dataclass
class Session:
    """Top-level container for engineering work."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    workspace_path: str = ""
    title: str = ""
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def transition_to(self, new_status: SessionStatus) -> None:
        """Transition to a new status with validation."""
        if not can_transition(self.status, new_status, SESSION_TRANSITIONS):
            raise ValueError(
                f"Invalid session transition: {self.status.value} → {new_status.value}"
            )
        self.status = new_status
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace_path": self.workspace_path,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class Run:
    """A single agent execution within a session."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    task: str = ""
    status: RunStatus = RunStatus.PENDING
    model_id: str = ""
    provider: str = ""
    task_type: str = ""
    outcome: str = ""
    verification_passed: bool = False

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    duration_ms: float = 0.0

    # Execution details
    iterations: int = 0
    tool_calls: int = 0
    failed_tool_calls: int = 0
    recovery_attempts: int = 0

    # Tokens
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    # Results
    result_summary: str = ""
    error_message: str = ""

    # Fallback
    fallback_used: bool = False
    fallback_model: str = ""

    def transition_to(self, new_status: RunStatus) -> None:
        """Transition to a new status with validation."""
        if not can_transition(self.status, new_status, RUN_TRANSITIONS):
            raise ValueError(
                f"Invalid run transition: {self.status.value} → {new_status.value}"
            )
        self.status = new_status
        if new_status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED):
            self.completed_at = time.time()
            self.duration_ms = (self.completed_at - self.started_at) * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "task": self.task,
            "status": self.status.value,
            "model_id": self.model_id,
            "provider": self.provider,
            "task_type": self.task_type,
            "outcome": self.outcome,
            "verification_passed": self.verification_passed,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": round(self.duration_ms, 1),
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "recovery_attempts": self.recovery_attempts,
            "total_tokens": self.total_tokens,
            "estimated_cost": round(self.estimated_cost, 6),
            "result_summary": self.result_summary,
            "error_message": self.error_message,
            "fallback_used": self.fallback_used,
            "fallback_model": self.fallback_model,
        }


@dataclass
class Checkpoint:
    """Snapshot of state at a safe boundary."""

    checkpoint_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    run_id: str = ""
    created_at: float = field(default_factory=time.time)

    # Repository state
    git_head: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    changed_files: list[str] = field(default_factory=list)

    # Agent state
    current_iteration: int = 0
    current_task: str = ""
    model_id: str = ""
    tool_history_summary: str = ""

    # Verification
    verification_status: str = ""  # pass/fail/pending
    tests_passing: int = 0
    tests_total: int = 0

    # Context references (NOT full file contents)
    context_files: list[str] = field(default_factory=list)
    context_file_hashes: dict[str, str] = field(default_factory=dict)
    repository_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "git_head": self.git_head,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "changed_files": self.changed_files,
            "current_iteration": self.current_iteration,
            "current_task": self.current_task,
            "model_id": self.model_id,
            "verification_status": self.verification_status,
            "tests_passing": self.tests_passing,
            "tests_total": self.tests_total,
            "context_files": self.context_files,
            "repository_summary": self.repository_summary,
        }


@dataclass
class MemoryItem:
    """A single memory entry — structured, deterministic, searchable."""

    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    memory_type: MemoryType = MemoryType.NOTE
    content: str = ""
    importance: float = 0.5  # 0.0–1.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source_run_id: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "session_id": self.session_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "importance": self.importance,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_run_id": self.source_run_id,
            "tags": self.tags,
        }


@dataclass
class SessionEvent:
    """A persisted event log entry."""

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    event_type: str = ""
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
        }
