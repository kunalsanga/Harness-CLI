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
class Task:
    """An engineering task."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    status: TaskStatus = TaskStatus.PENDING
    plan: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 30
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: str | None = None
    error: str | None = None


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
