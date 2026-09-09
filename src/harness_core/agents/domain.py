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
from typing import Any

from harness_core.agents.locks import WorkspaceResource

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
    ARCHITECT = "architect"
    UI_DESIGNER = "ui_designer"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    INTEGRATION = "integration"
    SECURITY_REVIEWER = "security_reviewer"
    VERIFIER = "verifier"
    GIT_RELEASE = "git_release"


class WorkspaceScope(enum.Enum):
    """Logical workspace scopes."""
    PROJECT = "project"
    FRONTEND = "frontend"
    BACKEND = "backend"
    DATABASE = "database"
    TESTS = "tests"
    DOCS = "docs"
    READ_ONLY = "read_only"


class AgentStatus(enum.Enum):
    """Agent lifecycle states."""
    CREATED = "created"
    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING = "waiting"
    BLOCKED = "blocked"
    REVIEWING = "reviewing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(enum.Enum):
    """Subtask lifecycle states."""
    CREATED = "created"
    BLOCKED = "blocked"
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    REVIEWING = "reviewing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(enum.Enum):
    """Inter-agent message types."""
    HANDOFF = "handoff"
    RESULT = "result"
    REQUEST = "request"
    INFORMATION = "information"
    BLOCKED = "blocked"
    REVIEW_REQUEST = "review_request"
    VERIFICATION_REQUEST = "verification_request"


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
    status: TaskStatus = TaskStatus.CREATED
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
    resources: list[WorkspaceResource] = field(default_factory=list)
    # Phase 9: runtime-assembled context for this task (requirements, traces,
    # artifact refs, handoffs). Provided by EngineeringRuntime; the task graph
    # remains the single source of task state — this is auxiliary prompt data.
    runtime_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "role": self.role.value,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "result": self.result,
            "error": self.error,
            "files_changed": self.files_changed,
            "model_id": self.model_id,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "resources": [r.to_dict() for r in self.resources],
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
            if task.status != TaskStatus.CREATED:
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

    def update_task_status(self, task_id: str, status: TaskStatus, error: str = "") -> None:
        """Update a task's status and handle cascading effects."""
        task = self.tasks.get(task_id)
        if not task:
            return

        task.status = status
        if error:
            task.error = error

        if status == TaskStatus.FAILED:
            self._propagate_status(task_id, TaskStatus.BLOCKED, f"Dependency {task_id} failed.")
        elif status == TaskStatus.CANCELLED:
            self._propagate_status(task_id, TaskStatus.CANCELLED, f"Dependency {task_id} cancelled.")

    def _propagate_status(self, task_id: str, status: TaskStatus, reason: str) -> None:
        """Recursively apply a status to all dependents."""
        dependents = self.get_dependents(task_id)
        for dep in dependents:
            if dep.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED) and dep.status != status:
                dep.status = status
                dep.error = reason
                self._propagate_status(dep.task_id, status, reason)

    def get_completed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)

    def get_failed_count(self) -> int:
        return sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)

    def get_total_count(self) -> int:
        return len(self.tasks)

    def is_complete(self) -> bool:
        """Check if all tasks are completed, failed, or cancelled."""
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
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

    def insert_recovery_sequence(self, failed_task_id: str, recovery_tasks: list[SubTask]) -> None:
        """
        Mutate graph to insert recovery tasks.
        The recovery tasks will execute, and the last recovery task will re-trigger
        dependents of the failed task.
        """
        if failed_task_id not in self.tasks:
            raise ValueError(f"Failed task {failed_task_id} not found in graph.")

        failed_task = self.tasks[failed_task_id]

        # We don't overwrite the original task status if it's FAILED (immutable history).
        # We just insert new tasks.

        # Let's make the first recovery task have NO dependencies on the FAILED task,
        # but depend on whatever the failed task depended on, so it can start immediately.
        if recovery_tasks:
            recovery_tasks[0].dependencies = list(failed_task.dependencies)

            # Chain them together
            for i in range(1, len(recovery_tasks)):
                recovery_tasks[i].dependencies.append(recovery_tasks[i-1].task_id)

            for t in recovery_tasks:
                self.tasks[t.task_id] = t

            # Update dependents to point to the last recovery task
            dependents = [t for t in self.tasks.values() if failed_task_id in t.dependencies and t.task_id not in [rt.task_id for rt in recovery_tasks]]
            last_recovery_task = recovery_tasks[-1]

            for dep in dependents:
                dep.dependencies.remove(failed_task_id)
                dep.dependencies.append(last_recovery_task.task_id)

                # We also need to recursively unblock them!
                self._unblock_dependents(dep.task_id)

    def _unblock_dependents(self, task_id: str) -> None:
        """Recursively reset BLOCKED tasks back to CREATED."""
        task = self.tasks.get(task_id)
        if not task:
            return
        if task.status in (TaskStatus.BLOCKED, TaskStatus.CANCELLED, TaskStatus.FAILED):
            task.status = TaskStatus.CREATED
            task.error = ""
            for dep in self.get_dependents(task_id):
                self._unblock_dependents(dep.task_id)

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

    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    sender_agent_id: str = ""
    sender_task_id: str = ""
    recipient_agent_id: str = ""
    recipient_task_id: str = ""
    message_type: MessageType = MessageType.INFORMATION
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    correlation_id: str = ""
    parent_message_id: str = ""
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_agent_id": self.sender_agent_id,
            "sender_task_id": self.sender_task_id,
            "recipient_agent_id": self.recipient_agent_id,
            "recipient_task_id": self.recipient_task_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "created_at": self.created_at,
            "correlation_id": self.correlation_id,
            "parent_message_id": self.parent_message_id,
            "schema_version": self.schema_version,
        }


@dataclass
class AgentContract:
    """Explicit contract defining an agent's execution parameters and constraints."""

    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    role: AgentRole = AgentRole.CODER
    task_id: str = ""
    objective: str = ""
    inputs: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    denied_tools: list[str] = field(default_factory=list)
    workspace_scope: WorkspaceScope = WorkspaceScope.PROJECT
    dependencies: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    model_policy: str = "auto"
    timeout_seconds: float = 300.0
    budget_cost: float = 1.0
    system_instructions: str = ""
    output_requirements: list[str] = field(default_factory=list)
    resources: list[WorkspaceResource] = field(default_factory=list)
    # Phase 9: runtime-assembled project context injected into the prompt.
    runtime_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "task_id": self.task_id,
            "objective": self.objective,
            "inputs": self.inputs,
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "workspace_scope": self.workspace_scope.value,
            "dependencies": self.dependencies,
            "expected_outputs": self.expected_outputs,
            "success_criteria": self.success_criteria,
            "model_policy": self.model_policy,
            "timeout_seconds": self.timeout_seconds,
            "budget_cost": self.budget_cost,
            "system_instructions": self.system_instructions,
            "output_requirements": self.output_requirements,
            "resources": [r.to_dict() for r in self.resources],
        }
