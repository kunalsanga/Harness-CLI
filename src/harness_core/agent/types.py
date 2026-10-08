"""Core types for the agent system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    """Status of an engineering task."""

    IDLE = "idle"
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    EVALUATING = "evaluating"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    # Phase 10.6: work ended with required tasks still unresolved and
    # nothing *failed*. NEVER rendered as success — only PARTIAL is honest.
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionState(Enum):
    """Canonical user-facing lifecycle state for one task."""

    IDLE = "idle"
    UNDERSTANDING = "understanding"
    PLANNING = "planning"
    THINKING = "thinking"
    WORKING = "working"
    REPAIRING = "repairing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class TaskExecutionTimeline:
    """Monotonic wall-clock accounting for a single task.

    Phase durations are exclusive wall-clock intervals. Model and tool time
    are nested diagnostics and are never added to phase or task elapsed time.
    """

    state: ExecutionState = ExecutionState.IDLE
    started_at: float | None = None
    finished_at: float | None = None
    transitioned_at: float | None = None
    phase_durations: dict[str, float] = field(default_factory=dict)
    model_attempt_duration: float = 0.0
    tool_duration: float = 0.0
    _phase_started_at: float | None = field(default=None, repr=False)

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def transition(self, state: ExecutionState, at: float | None = None) -> bool:
        now = time.monotonic() if at is None else at
        if self.state == state:
            return False
        if self.state in {
            ExecutionState.COMPLETED, ExecutionState.FAILED,
            ExecutionState.CANCELLED, ExecutionState.PAUSED,
        }:
            return False
        if self.started_at is None:
            self.started_at = now
        if self._phase_started_at is not None:
            key = self.state.value
            self.phase_durations[key] = self.phase_durations.get(key, 0.0) + max(
                0.0, now - self._phase_started_at
            )
        self.state = state
        self.transitioned_at = now
        self._phase_started_at = now if state not in {
            ExecutionState.COMPLETED, ExecutionState.FAILED,
            ExecutionState.CANCELLED, ExecutionState.PAUSED,
            ExecutionState.IDLE,
        } else None
        if state in {
            ExecutionState.COMPLETED, ExecutionState.FAILED,
            ExecutionState.CANCELLED, ExecutionState.PAUSED,
        }:
            self.finished_at = now
        return True

    def record_model_attempt(self, duration: float) -> None:
        self.model_attempt_duration += max(0.0, duration)

    def record_tool(self, duration: float) -> None:
        self.tool_duration += max(0.0, duration)

    def snapshot(self) -> dict[str, Any]:
        durations = dict(self.phase_durations)
        if self._phase_started_at is not None:
            durations[self.state.value] = durations.get(self.state.value, 0.0) + max(
                0.0, (self.finished_at or time.monotonic()) - self._phase_started_at
            )
        return {
            "state": self.state.value,
            "started_at": self.started_at,
            "transitioned_at": self.transitioned_at,
            "finished_at": self.finished_at,
            "elapsed": self.elapsed,
            "phase_durations": durations,
            "model_attempt_duration": self.model_attempt_duration,
            "tool_duration": self.tool_duration,
        }


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


class FailureReason(Enum):
    """Typed failure reasons for clean terminal UX."""
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_RATE_LIMITED = "model_rate_limited"
    PROVIDER_AUTH_FAILURE = "provider_auth_failure"
    PAYMENT_REQUIRED = "payment_required"
    TOOL_FAILURE = "tool_failure"
    TEST_FAILURE = "test_failure"
    VERIFICATION_FAILURE = "verification_failure"
    USER_CANCELLED = "user_cancelled"
    TASK_STAGNATION = "task_stagnation"
    COMPLETION_INVARIANT = "completion_invariant"
    PERMISSION_DENIED = "permission_denied"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNKNOWN = "unknown"


class ToolResultStatus(Enum):
    """Status of a tool execution.

    Extended classification for structured retry/recovery decisions.
    """

    SUCCESS = "success"
    ERROR = "error"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    RETRYABLE_FAILURE = "retryable_failure"    # transient, safe to retry
    NOT_FOUND = "not_found"                    # resource/path does not exist
    INVALID_REQUEST = "invalid_request"        # malformed arguments
    RATE_LIMITED = "rate_limited"              # 429 from tool/API
    AUTH_FAILURE = "auth_failure"              # 401/403 from tool/API


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
        """Classify the failure for retry/recovery decisions.

        Returns one of: success, permission_denied, timeout,
        retryable_failure, not_found, invalid_request, rate_limited,
        auth_failure, execution_error, tool_error.
        """
        if self.status == ToolResultStatus.SUCCESS:
            return "success"
        if self.status == ToolResultStatus.PERMISSION_DENIED:
            return "permission_denied"
        if self.status == ToolResultStatus.TIMEOUT:
            return "timeout"
        if self.status == ToolResultStatus.RETRYABLE_FAILURE:
            return "retryable_failure"
        if self.status == ToolResultStatus.NOT_FOUND:
            return "not_found"
        if self.status == ToolResultStatus.INVALID_REQUEST:
            return "invalid_request"
        if self.status == ToolResultStatus.RATE_LIMITED:
            return "rate_limited"
        if self.status == ToolResultStatus.AUTH_FAILURE:
            return "auth_failure"
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

    def record_permission_denied(self, tool_name: str = "") -> None:
        """Permission denied is not a failure per se, but blocks progress."""
        # tool_name is accepted for interface symmetry with record_success
        # / record_failure so callers can swap recorders in one place.
        del tool_name
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
    """Status of a TODO item. Runtime-owned — never set from model prose."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ACTIVE = IN_PROGRESS  # backward-compatible alias


@dataclass
class TodoItem:
    """A single TODO item in the task plan.

    `required=True` means the completion invariant refuses to mark the
    task COMPLETED until this TODO is resolved. The default is True so
    every TODO in the runtime plan is a hard requirement unless the
    planner explicitly opts it out.
    """

    description: str = ""
    status: TodoStatus = TodoStatus.PENDING
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    evidence: dict[str, Any] | None = None
    error: str | None = None
    # Runtime-owned fields
    required: bool = True
    category: str = "other"
    expected_operations: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[str, ...] = field(default_factory=tuple)

    @property
    def title(self) -> str:
        return self.description

    @property
    def symbol(self) -> str:
        return {
            TodoStatus.PENDING: "☐",
            TodoStatus.IN_PROGRESS: "◐",
            TodoStatus.COMPLETED: "✓",
            TodoStatus.FAILED: "✗",
            TodoStatus.SKIPPED: "—",
        }.get(self.status, "☐")

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

    def add_spec(
        self,
        description: str,
        *,
        category: str = "other",
        expected_operations: tuple[str, ...] = (),
        dependencies: tuple[str, ...] = (),
        required: bool = True,
    ) -> TodoItem:
        """Add a TODO with structured metadata (preferred over raw .add)."""
        item = TodoItem(
            description=description,
            category=category,
            expected_operations=expected_operations,
            dependencies=dependencies,
            required=required,
        )
        self.items.append(item)
        return item

    def get(self, todo_id: str) -> TodoItem | None:
        for item in self.items:
            if item.id == todo_id:
                return item
        return None

    def by_title(self, title: str) -> TodoItem | None:
        for item in self.items:
            if item.description == title:
                return item
        return None

    def complete(self, description: str, evidence: dict[str, Any] | None = None) -> None:
        for item in self.items:
            if item.description == description and item.status != TodoStatus.COMPLETED:
                self._mark_completed(item, evidence)
                return

    def complete_id(self, todo_id: str, evidence: dict[str, Any] | None = None) -> TodoItem | None:
        item = self.get(todo_id)
        if item and item.status != TodoStatus.COMPLETED:
            self._mark_completed(item, evidence)
            return item
        return None

    def activate(self, description: str) -> None:
        for item in self.items:
            if item.description == description and item.status == TodoStatus.PENDING:
                self._mark_started(item)
                return

    def activate_id(self, todo_id: str) -> TodoItem | None:
        item = self.get(todo_id)
        if item and item.status == TodoStatus.PENDING:
            self._mark_started(item)
            return item
        return None

    def fail(self, description: str, error: str | None = None) -> None:
        for item in self.items:
            if item.description == description and item.status != TodoStatus.COMPLETED:
                self._mark_failed(item, error)
                return

    def fail_id(self, todo_id: str, error: str | None = None) -> TodoItem | None:
        item = self.get(todo_id)
        if item and item.status not in (TodoStatus.COMPLETED, TodoStatus.SKIPPED):
            self._mark_failed(item, error)
            return item
        return None

    def skip_id(
        self,
        todo_id: str,
        reason: str = "",
        *,
        authorized: bool = False,
    ) -> TodoItem | None:
        """Mark a TODO as SKIPPED.

        Phase 10.6 trust rule: a REQUIRED TODO can only be skipped when the
        caller is runtime code with evidence (``authorized=True``). A model
        text response can never waive required work — it would let 3/5 render
        as COMPLETE.
        """
        item = self.get(todo_id)
        if item is None or item.status not in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS):
            return None
        if item.required and not authorized:
            return None  # required work cannot be waived without authorization
        item.status = TodoStatus.SKIPPED
        item.completed_at = time.time()
        if reason:
            item.error = reason
        return item

    def skip_dependents(self, todo_id: str, reason: str = "dependency failed") -> list[TodoItem]:
        """Mark every TODO whose dependency chain includes `todo_id` as SKIPPED.

        Used when a required step fails: blind execution of later steps
        is forbidden, so dependents are explicitly skipped with a clear
        reason rather than left dangling.
        """
        skipped: list[TodoItem] = []
        # BFS through the dependency graph
        affected = {todo_id}
        frontier = [todo_id]
        while frontier:
            current = frontier.pop()
            for item in self.items:
                if (
                    item.status in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS)
                    and current in item.dependencies
                    and item.id not in affected
                ):
                    affected.add(item.id)
                    frontier.append(item.id)
        for item_id in affected:
            if item_id == todo_id:
                continue
            # Dependency failure is runtime evidence: dependents are
            # legitimately skipped rather than blindly executed.
            s = self.skip_id(item_id, reason, authorized=True)
            if s is not None:
                skipped.append(s)
        return skipped

    def ready(self) -> list[TodoItem]:
        """Return TODOs that are runnable right now: pending with no
        pending or failed dependencies.
        """
        ids = {i.id for i in self.items}
        out: list[TodoItem] = []
        for item in self.items:
            if item.status != TodoStatus.PENDING:
                continue
            ok = True
            for dep_id in item.dependencies:
                if dep_id not in ids:
                    continue  # unknown id treated as satisfied
                dep = self.get(dep_id)
                if dep is None:
                    continue
                if dep.status in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS, TodoStatus.FAILED):
                    ok = False
                    break
            if ok:
                out.append(item)
        return out

    def first_unresolved(self) -> TodoItem | None:
        """Return the first PENDING/IN_PROGRESS TODO whose deps are satisfied.

        Used by resume to pick up from the right place.
        """
        for item in self.items:
            if item.status not in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS):
                continue
            if self._deps_satisfied(item):
                return item
        return None

    def _deps_satisfied(self, item: TodoItem) -> bool:
        for dep_id in item.dependencies:
            dep = self.get(dep_id)
            if dep is None:
                continue
            if dep.status in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS, TodoStatus.FAILED):
                return False
        return True

    @staticmethod
    def _mark_started(item: TodoItem) -> None:
        item.status = TodoStatus.IN_PROGRESS
        if item.started_at is None:
            item.started_at = time.time()

    @staticmethod
    def _mark_completed(item: TodoItem, evidence: dict[str, Any] | None) -> None:
        item.status = TodoStatus.COMPLETED
        item.completed_at = time.time()
        if item.started_at is None:
            item.started_at = item.completed_at
        if evidence:
            item.evidence = evidence
        item.error = None

    @staticmethod
    def _mark_failed(item: TodoItem, error: str | None) -> None:
        item.status = TodoStatus.FAILED
        item.completed_at = time.time()
        if error:
            item.error = error

    def display(self) -> list[str]:
        return [item.display() for item in self.items]

    def to_event_items(self) -> list[dict[str, Any]]:
        return [
            {
                "id": i.id,
                "title": i.description,
                "status": i.status.value,
                "evidence": i.evidence,
                "error": i.error,
                "required": i.required,
                "category": i.category,
                "expected_operations": list(i.expected_operations),
            }
            for i in self.items
        ]

    @property
    def completed_count(self) -> int:
        return sum(1 for i in self.items if i.status == TodoStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for i in self.items if i.status == TodoStatus.FAILED)

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
    models_used: list[str] = field(default_factory=list)
    model_fallbacks: int = 0
    tests_run: int = 0
    tests_passed: int | None = None
    git_commit: str | None = None
    git_push: str | None = None
    paused_reason: str | None = None
    # Typed failure category for clean UX (see FailureReason enum)
    failure_reason: str | None = None
    # Operations that produced real evidence (for the duplicate-op guard)
    completed_operations: list[str] = field(default_factory=list)
    # Harness 2.0: set when budget-driven compaction trimmed tool history.
    compaction_note: str | None = None
    execution_timeline: TaskExecutionTimeline = field(default_factory=TaskExecutionTimeline)


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
    COMMITTING = "committing"
    PUSHING = "pushing"
    COMPLETE = "complete"
    COMPLETED = "complete"


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
    # Hard ceiling on the estimated prompt tokens the agent may send in one
    # request. The loop trims history until the assembled message list fits,
    # so this is an enforced limit rather than a hint. Keep it comfortably
    # below the smallest context window the router may select.
    context_token_budget: int = 120_000


# ── Canonical ToolResult constructors ─────────────────────────────────────────────────────
# Use these instead of constructing ToolResult(...) directly in tool implementations.
# Every tool in the runtime must use these factories so that the contract is uniform
# and the runtime can reason about results without inspecting unstructured error strings.


class ToolResults:
    """Canonical constructors for ToolResult.

    Use these in every tool implementation so that the runtime gets consistent,
    structured results regardless of which tool produced them.  These factories
    document the expected shape of each result type and keep all the defaults
    in one place.

    ToolResult fields
    ──────────────────
    status       : ToolResultStatus (required)
    output       : str  — stdout / content; empty string when there is none
    error        : str | None  — human-readable explanation of what went wrong
    metadata     : dict  — structured operation-specific data
    retryable    : bool  — True = transient; runtime should retry
    exit_code    : int | None  — raw exit code from a subprocess; None for tools
                                 that do not spawn subprocesses
    stderr       : str | None  — raw stderr from a subprocess; None otherwise
    """

    # ── Success ───────────────────────────────────────────────────────────────────────

    @staticmethod
    def success(
        output: str,
        *,
        metadata: dict[str, Any] | None = None,
        exit_code: int | None = None,
        stderr: str | None = None,
    ) -> ToolResult:
        """Command/tool completed successfully."""
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=output,
            metadata=metadata or {},
            exit_code=exit_code,
            stderr=stderr,
        )

    # ── Errors ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def error(
        error: str,
        *,
        output: str = "",
        metadata: dict[str, Any] | None = None,
        exit_code: int | None = None,
        stderr: str | None = None,
        retryable: bool = True,
    ) -> ToolResult:
        """Command or tool failed (non-zero exit, missing resource, etc.).

        retryable=True  → transient (network glitch, flaky command). Runtime may retry.
        retryable=False → permanent (bad path, bad arguments, logic error). Do not retry.
        """
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=output,
            error=error,
            metadata=metadata or {},
            exit_code=exit_code,
            stderr=stderr,
            retryable=retryable,
        )

    @staticmethod
    def permission_denied(
        error: str = "Permission denied",
        *,
        output: str = "",
    ) -> ToolResult:
        """Call was blocked by a permission policy or workspace confinement.

        These are not execution failures — the tool never ran.
        """
        return ToolResult(
            status=ToolResultStatus.PERMISSION_DENIED,
            output=output,
            error=error,
            retryable=False,
        )

    @staticmethod
    def timeout(
        error: str,
        *,
        timeout_seconds: float | None = None,
        output: str = "",
        stderr: str | None = None,
    ) -> ToolResult:
        """Operation exceeded its allocated time budget.

        timeouts are transient — retrying with a higher timeout is reasonable.
        """
        meta: dict[str, Any] = {}
        if timeout_seconds is not None:
            meta["timeout_seconds"] = timeout_seconds
        return ToolResult(
            status=ToolResultStatus.TIMEOUT,
            output=output,
            error=error,
            metadata=meta,
            retryable=True,
            stderr=stderr,
        )

    @staticmethod
    def network_failure(
        error: str,
        *,
        output: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Network-level failure (DNS, connection refused, etc.).

        These are always transient — retry after a short back-off.
        """
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=output,
            error=error,
            metadata=metadata or {},
            retryable=True,
        )

    @staticmethod
    def git_failure(
        operation: str,
        error: str,
        *,
        output: str = "",
        exit_code: int | None = None,
        stderr: str | None = None,
        retryable: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Git operation failed (bad identity, auth, no remote, etc.).

        Git failures are almost always permanent — the user must fix the
        underlying configuration before retrying will succeed.
        """
        meta: dict[str, Any] = {"operation": operation}
        if metadata:
            meta.update(metadata)
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=output,
            error=error,
            metadata=meta,
            exit_code=exit_code,
            stderr=stderr,
            retryable=retryable,
        )

    @staticmethod
    def from_exception(
        exc: BaseException,
        *,
        retryable: bool = True,
    ) -> ToolResult:
        """Tool raised an exception during execution.

        retryable=True  → unexpected / transient (OOM, race condition).
        retryable=False → programming error, re-running will fail the same way.
        """
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output="",
            error=str(exc),
            retryable=retryable,
        )

    @staticmethod
    def unknown_tool(tool_name: str) -> ToolResult:
        """Requested tool does not exist in the runtime."""
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output="",
            error=f"Unknown tool: {tool_name}",
            retryable=False,
        )

    @staticmethod
    def missing_argument(tool_name: str, args: list[str]) -> ToolResult:
        """Required arguments were not supplied."""
        missing = ", ".join(args)
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output="",
            error=f"{tool_name}: missing required argument(s): {missing}",
            retryable=False,
        )


# ── Failure classification enums ─────────────────────────────────────────────────────
# Used for canonical ToolResult fields.


class Retryability(str, Enum):
    """Retryability classification for a tool execution."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    PAYMENT_REQUIRED = "payment_required"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class FailureType(str, Enum):
    """Failure classification for a tool execution."""

    EXECUTION_ERROR = "execution_error"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    NETWORK_FAILURE = "network_failure"
    UNKNOWN = "unknown"
    TEST_FAILURE = "test_failure"
    GIT_FAILURE = "git_failure"
    COMMAND_SYNTAX = "command_syntax"
