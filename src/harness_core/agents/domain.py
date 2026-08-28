"""
Agent domain model for M5 — multi-agent software engineering system.

Core entities:
  AgentRole — specialized agent type
  AgentStatus — lifecycle state
  SubTask — decomposed work unit
  TaskGraph — dependency graph of subtasks
  AgentResult — structured output from an agent
  AgentMessage — inter-agent communication
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Enums ────────────────────────────────────────────────────────────────

class AgentRole(enum.Enum):
    """Specialized agent roles."""
    ORCHESTRATOR = "orchestrator"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYZER = "analyzer"
    CODER = "coder"
    TESTER = "tester"
    REVIEWER = "reviewer"
    DEBUGGER = "debugger"


class AgentStatus(enum.Enum):
    """Agent lifecycle states."""
    IDLE = "idle"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskStatus(enum.Enum):
    """Subtask lifecycle states."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MessageType(enum.Enum):
    """Inter-agent message types."""
    REQUEST = "request"
    RESULT = "result"
    QUESTION = "question"
    WARNING = "warning"
    ERROR = "error"
    HANDOFF = "handoff"
    REVIEW = "review"
    APPROVAL = "approval"


class ReviewVerdict(enum.Enum):
    """Reviewer agent verdict."""
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"


# ── Domain objects ───────────────────────────────────────────────────────

@dataclass
class SubTask:
    """A decomposed work unit within a task graph."""

    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_task_id: str = ""
    description: str = ""
    role: AgentRole = AgentRole.CODER
    dependencies: list[str] = field(default_factory=list)
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str = ""
    result: str = ""
    error: str = ""
    files_changed: list[str] = field(default_factory=list)
    model_id: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    duration_ms: float = 0.0
    tool_calls: int = 0
    iterations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "role": self.role.value,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "priority": self.priority,
            "assigned_agent": self.assigned_agent,
            "files_changed": self.files_changed,
            "model_id": self.model_id,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
        }


@dataclass
class TaskGraph:
    """Dependency graph of subtasks.

    Supports:
    - Topological ordering
    - Dependency validation
    - Ready-task detection
    - Parallel-safe task identification
    """

    tasks: dict[str, SubTask] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_task(self, task: SubTask) -> None:
        """Add a task to the graph."""
        self.tasks[task.task_id] = task

    def get_task(self, task_id: str) -> SubTask | None:
        return self.tasks.get(task_id)

    def get_ready_tasks(self) -> list[SubTask]:
        """Get tasks whose dependencies are all completed."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self.tasks.get(dep) is not None
                and self.tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.dependencies
            )
            if deps_met:
                task.status = TaskStatus.READY
                ready.append(task)
        return ready

    def get_completed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)

    def get_failed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

    def get_total_count(self) -> int:
        return len(self.tasks)

    def is_complete(self) -> bool:
        """Check if all tasks are completed, failed, or skipped."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for t in self.tasks.values()
        )

    def has_failures(self) -> bool:
        return self.get_failed_count() > 0

    def get_dependents(self, task_id: str) -> list[SubTask]:
        """Get tasks that depend on the given task."""
        return [
            t for t in self.tasks.values()
            if task_id in t.dependencies
        ]

    def validate(self) -> list[str]:
        """Validate the graph for issues. Returns list of error messages."""
        errors = []

        # Check all dependencies exist
        for task in self.tasks.values():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task {task.task_id} depends on non-existent task {dep}")

        # Check for cycles
        if self._has_cycle():
            errors.append("Circular dependency detected in task graph")

        return errors

    def _has_cycle(self) -> bool:
        """DFS-based cycle detection."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {tid: WHITE for tid in self.tasks}

        def visit(tid: str) -> bool:
            color[tid] = GRAY
            task = self.tasks[tid]
            for dep in task.dependencies:
                if dep in self.tasks:
                    if color[dep] == GRAY:
                        return True
                    if color[dep] == WHITE and visit(dep):
                        return True
            color[tid] = BLACK
            return False

        for tid in self.tasks:
            if color[tid] == WHITE:
                if visit(tid):
                    return True
        return False

    def topological_sort(self) -> list[SubTask]:
        """Return tasks in topological order."""
        WHITE, BLACK = 0, 2
        color: dict[str, int] = {tid: WHITE for tid in self.tasks}
        result: list[SubTask] = []

        def visit(tid: str) -> None:
            color[tid] = 1  # GRAY
            for dep in self.tasks[tid].dependencies:
                if dep in self.tasks and color[dep] == WHITE:
                    visit(dep)
            color[tid] = BLACK
            result.append(self.tasks[tid])

        for tid in self.tasks:
            if color[tid] == WHITE:
                visit(tid)

        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.get_total_count(),
            "completed": self.get_completed_count(),
            "failed": self.get_failed_count(),
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
        }


@dataclass
class AgentResult:
    """Structured output from an agent execution."""

    agent_id: str = ""
    role: AgentRole = AgentRole.CODER
    status: AgentStatus = AgentStatus.COMPLETED
    summary: str = ""
    artifacts: list[str] = field(default_factory=list)
    files_changed: list[str] = field(default_factory=list)
    tests_passed: int = 0
    tests_total: int = 0
    findings: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    review_verdict: ReviewVerdict | None = None
    duration_ms: float = 0.0
    tool_calls: int = 0
    iterations: int = 0
    model_id: str = ""
    tokens_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "status": self.status.value,
            "summary": self.summary,
            "files_changed": self.files_changed,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "findings": self.findings,
            "errors": self.errors,
            "recommendations": self.recommendations,
            "review_verdict": self.review_verdict.value if self.review_verdict else None,
            "duration_ms": round(self.duration_ms, 1),
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "model_id": self.model_id,
        }


@dataclass
class AgentMessage:
    """Structured inter-agent communication."""

    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    sender: str = ""
    receiver: str = ""
    task_id: str = ""
    message_type: MessageType = MessageType.RESULT
    content: str = ""
    artifacts: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "task_id": self.task_id,
            "message_type": self.message_type.value,
            "content": self.content[:200],
            "timestamp": self.timestamp,
        }
