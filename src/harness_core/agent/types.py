"""Core types for the agent system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """Status of an engineering task."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    EVALUATING = "evaluating"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(Enum):
    """Roles for specialized agents."""

    BUILD = "build"
    PLAN = "plan"
    RESEARCH = "research"
    DEBUG = "debug"
    TEST = "test"
    REVIEW = "review"
    ARCHITECT = "architect"
    EXPERIMENT = "experiment"


class ToolResultStatus(Enum):
    """Status of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """Result of a tool execution.

    The runtime is the source of truth for execution results. Never override
    these fields based on model text responses.
    """

    status: ToolResultStatus
    output: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retryable: bool = True
    exit_code: int | None = None
    stderr: str | None = None

    @property
    def is_perm_denied(self) -> bool:
        return self.status == ToolResultStatus.PERMISSION_DENIED

    @property
    def is_transient(self) -> bool:
        return self.status in (ToolResultStatus.TIMEOUT, ToolResultStatus.ERROR) and self.retryable

    @property
    def is_final(self) -> bool:
        return self.status == ToolResultStatus.PERMISSION_DENIED or (
            self.status == ToolResultStatus.ERROR and not self.retryable
        )

    @property
    def execution_failed(self) -> bool:
        """True when a shell command exited non-zero or the tool errored.

        This is the definitive signal that the operation did not succeed.
        The agent loop must treat this as a failure regardless of what the
        model's text response says.
        """
        if self.status == ToolResultStatus.SUCCESS:
            return False
        if self.status == ToolResultStatus.PERMISSION_DENIED:
            return False  # blocked, not a failed execution
        if self.status == ToolResultStatus.TIMEOUT:
            return True
        # ERROR: check exit_code for shell commands
        if self.exit_code is not None and self.exit_code != 0:
            return True
        # ERROR without exit_code is a tool-level failure
        return True

    @property
    def failure_category(self) -> str:
        """Classify the failure for retry/recovery decisions."""
        if self.status == ToolResultStatus.SUCCESS:
            return "success"
        if self.status == ToolResultStatus.PERMISSION_DENIED:
            return "permission_denied"
        if self.status == ToolResultStatus.TIMEOUT:
            return "timeout"
        if self.exit_code is not None and self.exit_code != 0:
            return "execution_error"
        return "tool_error"


@dataclass
class ToolCall:
    """A tool call made by an agent."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: ToolResult | None = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


@dataclass
class TaskExecutionStats:
    """Tracks execution accounting for truthful task completion."""

    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    recovered: int = 0  # failures that were later succeeded
    unresolved: int = 0  # failures with no subsequent success
    _failed_tools: dict[str, int] = field(default_factory=dict)  # tool_name -> count of consecutive failures

    def record_attempt(self) -> None:
        self.attempted += 1

    def record_success(self, tool_name: str) -> None:
        self.succeeded += 1
        # Check if this tool previously failed -> it's a recovery
        if self._failed_tools.get(tool_name, 0) > 0:
            self.recovered += self._failed_tools[tool_name]
            self.unresolved = max(0, self.unresolved - self._failed_tools[tool_name])
            del self._failed_tools[tool_name]

    def record_failure(self, tool_name: str) -> None:
        self.failed += 1
        self.unresolved += 1
        self._failed_tools[tool_name] = self._failed_tools.get(tool_name, 0) + 1

    def record_permission_denied(self, tool_name: str) -> None:
        """Permission denied is not a failure per se, but blocks progress."""
        self.attempted += 1  # already counted in record_attempt
        # Don't count as failed or unresolved — it's a constraint

    @property
    def has_unresolved_failures(self) -> bool:
        return self.unresolved > 0

    @property
    def success_rate(self) -> float:
        if self.attempted == 0:
            return 0.0
        return self.succeeded / self.attempted

    def summary(self) -> str:
        parts = [f"Attempted: {self.attempted}", f"Succeeded: {self.succeeded}"]
        if self.failed > 0:
            parts.append(f"Failed: {self.failed}")
        if self.recovered > 0:
            parts.append(f"Recovered: {self.recovered}")
        if self.unresolved > 0:
            parts.append(f"Unresolved: {self.unresolved}")
        return ", ".join(parts)


class TodoStatus(Enum):
    """Status of a TODO item."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TodoItem:
    """A single TODO item in the task plan."""

    description: str = ""
    status: TodoStatus = TodoStatus.PENDING

    @property
    def symbol(self) -> str:
        return {
            TodoStatus.PENDING: "☐",
            TodoStatus.ACTIVE: "◐",
            TodoStatus.COMPLETED: "☑",
            TodoStatus.FAILED: "✗",
            TodoStatus.SKIPPED: "—",
        }[self.status]

    def display(self) -> str:
        return f"{self.symbol} {self.description}"


@dataclass
class TaskPlan:
    """Dynamic task plan with live status tracking."""

    items: list[TodoItem] = field(default_factory=list)

    def add(self, description: str) -> TodoItem:
        item = TodoItem(description=description)
        self.items.append(item)
        return item

    def complete(self, description: str) -> None:
        for item in self.items:
            if item.description == description and item.status != TodoStatus.COMPLETED:
                item.status = TodoStatus.COMPLETED
                return

    def activate(self, description: str) -> None:
        for item in self.items:
            if item.description == description and item.status == TodoStatus.PENDING:
                item.status = TodoStatus.ACTIVE
                return

    def fail(self, description: str) -> None:
        for item in self.items:
            if item.description == description and item.status != TodoStatus.COMPLETED:
                item.status = TodoStatus.FAILED
                return

    def display(self) -> list[str]:
        return [item.display() for item in self.items]

    @property
    def completed_count(self) -> int:
        return sum(1 for i in self.items if i.status == TodoStatus.COMPLETED)

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def is_complete(self) -> bool:
        return all(i.status in (TodoStatus.COMPLETED, TodoStatus.SKIPPED) for i in self.items)


@dataclass
class Task:
    """An engineering task."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    status: TaskStatus = TaskStatus.PENDING
    plan: list[str] = field(default_factory=list)
    task_plan: TaskPlan = field(default_factory=TaskPlan)
    thinking: str = ""  # High-level execution status message
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 30
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: str | None = None
    error: str | None = None
    execution_stats: TaskExecutionStats = field(default_factory=TaskExecutionStats)
    verification_passed: bool | None = None  # None = not verified, True/False = outcome
    verification_summary: str = ""


class TaskPhase(Enum):
    """Phases of a task lifecycle for progress tracking."""

    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    DIAGNOSING = "diagnosing"
    FIXING = "fixing"
    RECOVERING = "recovering"
    VERIFYING = "verifying"
    COMPLETE = "complete"


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    role: AgentRole = AgentRole.BUILD
    max_iterations: int = 30
    max_tool_calls: int = 100
    timeout_seconds: float = 300.0
    model_preference: str | None = None
    routing_mode: str = "auto"
    permissions: dict[str, str] = field(default_factory=dict)
    autonomous_mode: bool = True
    verbose: bool = False
    verify_on_complete: bool = True
