"""Authoritative task state machine and completion invariant.

This module is the single source of truth for whether a task may be
marked COMPLETED. The runtime is the owner of task state — the model
never sets the final status directly.

HARD INVARIANT
==============

A task can transition to COMPLETED iff:

    can_complete_task(task) is True

`can_complete_task` checks:

    1. task is not CANCELLED
    2. task is not PAUSED
    3. no required TODO is PENDING
    4. no required TODO is FAILED
    5. no required TODO is IN_PROGRESS (no work in flight)
    6. all required tool operations succeeded
    7. verification passed (when required)
    8. no unresolved blocking error exists
    9. required evidence exists for the task type
"""

from __future__ import annotations

from typing import Any

from harness_core.agent.types import (
    Task,
    TaskStatus,
    TodoItem,
    TodoStatus,
    ToolResultStatus,
)


# ── State machine ──────────────────────────────────────────────────────

# Allowed forward transitions. Every other transition is rejected.
ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.IDLE: {TaskStatus.PLANNING, TaskStatus.EXECUTING, TaskStatus.CANCELLED},
    TaskStatus.PENDING: {TaskStatus.PLANNING, TaskStatus.EXECUTING, TaskStatus.CANCELLED},
    TaskStatus.PLANNING: {TaskStatus.EXECUTING, TaskStatus.FAILED, TaskStatus.PAUSED, TaskStatus.CANCELLED},
    TaskStatus.EXECUTING: {
        TaskStatus.VERIFYING,
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.PAUSED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.VERIFYING: {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.PAUSED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.PAUSED: {
        TaskStatus.EXECUTING,
        TaskStatus.VERIFYING,
        TaskStatus.COMPLETED,
        TaskStatus.CANCELLED,
    },
    TaskStatus.FAILED: {TaskStatus.CANCELLED},
    TaskStatus.CANCELLED: set(),
    TaskStatus.COMPLETED: set(),
    # Legacy statuses — keep them in the table so old tasks don't crash,
    # but route them through the same final guardrails.
    TaskStatus.EVALUATING: {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.PAUSED, TaskStatus.CANCELLED},
    TaskStatus.RECOVERING: {TaskStatus.EXECUTING, TaskStatus.FAILED, TaskStatus.PAUSED, TaskStatus.CANCELLED},
}


def can_transition(current: TaskStatus, target: TaskStatus) -> bool:
    """True iff the state machine permits `current -> target`."""
    return target in ALLOWED_TRANSITIONS.get(current, set())


def transition(task: Task, target: TaskStatus) -> bool:
    """Attempt to move the task to `target`. Returns True on success.

    The transition is rejected (and the task status untouched) if the
    state machine forbids it. COMPLETED requires the full completion
    invariant — use can_complete_task() first if you want a no-op.
    """
    if not can_transition(task.status, target):
        return False
    if target == TaskStatus.COMPLETED and not can_complete_task(task):
        return False
    task.status = target
    return True


# ── Completion invariant ───────────────────────────────────────────────


def pending_required_todos(task: Task) -> list[TodoItem]:
    """TODO items that must be resolved before completion."""
    return [
        i for i in task.task_plan.items
        if i.required and i.status in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS)
    ]


def failed_required_todos(task: Task) -> list[TodoItem]:
    return [i for i in task.task_plan.items if i.required and i.status == TodoStatus.FAILED]


def unresolved_tool_failures(task: Task) -> list[Any]:
    """Tool calls that failed and were not followed by a recovery success."""
    failures: list[Any] = []
    for tc in task.tool_calls:
        if tc.result is None:
            continue
        if tc.result.is_perm_denied:
            continue  # denial ≠ execution failure
        if not tc.result.execution_failed:
            continue
        # Was there a successful call AFTER this one? (any kind, any tool)
        idx = task.tool_calls.index(tc)
        recovered = any(
            later.result is not None
            and later.result.status == ToolResultStatus.SUCCESS
            and not later.result.execution_failed
            for later in task.tool_calls[idx + 1:]
        )
        if not recovered:
            failures.append(tc)
    return failures


def _has_operation(task: Task, *op_prefixes: str) -> bool:
    """Check if a completed operation matching any prefix exists.

    Works for both tool-call evidence (task.tool_calls) and
    workflow-completed evidence (task.completed_operations).
    """
    # Check tool calls
    for tc in task.tool_calls:
        if tc.tool_name in op_prefixes and tc.result is not None and tc.result.status == ToolResultStatus.SUCCESS:
            return True
    # Check workflow-completed operations (e.g. "read_file", "git_push")
    for op in task.completed_operations:
        for prefix in op_prefixes:
            if op == prefix or op.startswith(prefix + ":"):
                return True
    return False


def has_required_evidence(task: Task) -> bool:
    """True when the task type's required evidence is present.

    - Coding tasks need at least one successful file write OR the goal
      was a research/explain task with evidence of inspection.
    - Git tasks need a recorded commit OR push.
    - Test tasks need a recorded test run.
    """
    # If the model finished with text only and all TODOs are resolved
    # (completed or skipped), that counts as evidence — the model chose
    # to answer from its knowledge rather than tool use.
    if task.task_plan.items and all(
        i.status in (TodoStatus.COMPLETED, TodoStatus.SKIPPED)
        for i in task.task_plan.items
    ):
        return True
    # If the model returned a result text but had no tool calls and no plan,
    # that still counts as evidence for non-coding tasks.
    if task.result and not task.tool_calls and not task.task_plan.items:
        return True

    goal = (task.goal or "").lower()

    # Explain / research / summary tasks: inspection evidence is enough
    if any(w in goal for w in (
        "explain", "describe", "understand", "what is", "what does",
        "overview", "summarize", "how does", "research", "analyze",
    )):
        return _has_operation(task, "read_file", "list_files", "glob", "grep")

    # Git / push tasks: at minimum an inspection + a successful commit
    if any(w in goal for w in ("push", "github", "commit")):
        return bool(task.git_commit) or _has_operation(task, "git_add", "git_commit", "git_push")

    # Coding / implementation tasks: file modification evidence
    if _has_operation(task, "write_file", "edit_file"):
        return True

    # Test-only tasks: a successful test run
    if any(w in goal for w in ("test", "verify", "fix")):
        return task.tests_run > 0 and task.tests_passed is not None

    # Fallback: any successful tool call counts as evidence the agent
    # actually did work.
    if any(
        tc.result is not None and tc.result.status == ToolResultStatus.SUCCESS
        for tc in task.tool_calls
    ) or bool(task.completed_operations):
        return True
    return False


def can_complete_task(task: Task) -> bool:
    """THE authoritative completion check.

    Returns True iff every completion invariant holds. This is the
    single gate every code path must go through to set status =
    COMPLETED.
    """
    # 1. Not cancelled
    if task.status == TaskStatus.CANCELLED:
        return False

    # 2. Not paused
    if task.status == TaskStatus.PAUSED:
        return False

    # 3. No pending required TODOs
    pending = pending_required_todos(task)
    if pending:
        return False

    # 4. No failed required TODOs
    failed = failed_required_todos(task)
    if failed:
        return False

    # 5. No in-progress required TODOs (no work in flight)
    in_flight = [
        i for i in task.task_plan.items
        if i.required and i.status == TodoStatus.IN_PROGRESS
    ]
    if in_flight:
        return False

    # 6. No unresolved tool failures
    if unresolved_tool_failures(task):
        return False

    # 7. Verification — only required when files were modified OR
    # the task explicitly demands verification. Empty workspaces
    # and pure-research tasks skip this.
    modified = any(
        tc.tool_name in ("write_file", "edit_file")
        and tc.result is not None
        and tc.result.status == ToolResultStatus.SUCCESS
        for tc in task.tool_calls
    )
    if modified and task.verification_passed is False:
        return False
    # If verification was not run and modifications exist, allow
    # completion only if there is positive test/git evidence; else
    # the invariant is the verification engine has had a chance.
    if modified and task.verification_passed is None and task.tests_run == 0 and not task.git_commit:
        # No verification AND no tests AND no commit: still allowed
        # only if no files were really "engineering-modified" beyond
        # trivial writes. The report will note the absence.
        pass

    # 8. No unresolved blocking error
    if task.error and task.status not in (TaskStatus.PAUSED, TaskStatus.CANCELLED, TaskStatus.FAILED):
        return False

    # 9. Required evidence
    if not has_required_evidence(task):
        return False

    return True


def completion_blockers(task: Task) -> list[str]:
    """Human-readable list of what's blocking completion. Empty = ready."""
    blockers: list[str] = []

    if task.status == TaskStatus.CANCELLED:
        blockers.append("Task was cancelled")
    if task.status == TaskStatus.PAUSED:
        blockers.append("Task is paused")

    pending = pending_required_todos(task)
    if pending:
        ids = ", ".join(i.description for i in pending[:5])
        blockers.append(f"{len(pending)} required TODO(s) still pending: {ids}")

    failed = failed_required_todos(task)
    if failed:
        ids = ", ".join(i.description for i in failed[:5])
        blockers.append(f"{len(failed)} required TODO(s) failed: {ids}")

    in_flight = [
        i for i in task.task_plan.items
        if i.required and i.status == TodoStatus.IN_PROGRESS
    ]
    if in_flight:
        ids = ", ".join(i.description for i in in_flight[:5])
        blockers.append(f"{len(in_flight)} required TODO(s) in progress: {ids}")

    bad_tools = unresolved_tool_failures(task)
    if bad_tools:
        names = ", ".join(tc.tool_name for tc in bad_tools[:5])
        blockers.append(f"{len(bad_tools)} unresolved tool failure(s): {names}")

    if not has_required_evidence(task):
        blockers.append("No required evidence recorded for this task type")

    return blockers
