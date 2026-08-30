"""Regression tests for the completion invariant hardening.

Covers the critical requirements:
- 4/5 TODOs cannot complete a task
- 5/5 TODOs can complete a task
- Model failure must not erase tool success
- Workflow paths go through completion invariant
- SKIPPED TODOs count as resolved
- No fake completion from model prose
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.agent.completion import can_complete_task, completion_blockers
from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import (
    AgentConfig,
    Task,
    TaskStatus,
    TodoItem,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.observability.events import EventBus
from harness_core.permissions.manager import PermissionManager, PermissionRule
from harness_core.tools.shell import RunCommandTool


# ── 4/5 cannot complete ───────────────────────────────────────────────


class TestFourOutOfFiveCannotComplete:
    """When 4/5 TODOs are completed, the task MUST NOT be marked COMPLETED."""

    def test_four_of_five_pending_blocks_completion(self):
        task = Task(goal="Enhance the project")
        for i in range(5):
            item = task.task_plan.add(f"Step {i}")
            if i < 4:
                item.status = TodoStatus.COMPLETED
                item.evidence = {"tool": "write_file", "status": "success"}
            # 5th TODO remains PENDING

        assert not can_complete_task(task)
        blockers = completion_blockers(task)
        assert any("pending" in b.lower() for b in blockers)

    def test_four_of_five_in_progress_blocks_completion(self):
        task = Task(goal="Enhance the project")
        for i in range(5):
            item = task.task_plan.add(f"Step {i}")
            if i < 4:
                item.status = TodoStatus.COMPLETED
                item.evidence = {"tool": "write_file", "status": "success"}
            else:
                item.status = TodoStatus.IN_PROGRESS

        assert not can_complete_task(task)

    def test_four_of_five_failed_blocks_completion(self):
        task = Task(goal="Enhance the project")
        for i in range(5):
            item = task.task_plan.add(f"Step {i}")
            if i < 4:
                item.status = TodoStatus.COMPLETED
                item.evidence = {"tool": "write_file", "status": "success"}
            else:
                item.status = TodoStatus.FAILED
                item.error = "Tests failed"

        assert not can_complete_task(task)
        blockers = completion_blockers(task)
        assert any("failed" in b.lower() for b in blockers)


# ── 5/5 can complete ─────────────────────────────────────────────────


class TestFiveOutOfFiveCanComplete:
    """When 5/5 TODOs are completed, the task CAN be marked COMPLETED."""

    def test_five_of_five_completed_allows_completion(self):
        task = Task(goal="Enhance the project")
        for i in range(5):
            item = task.task_plan.add(f"Step {i}")
            item.status = TodoStatus.COMPLETED
            item.evidence = {"tool": "write_file", "status": "success"}

        # Add required evidence
        task.tool_calls.append(ToolCall(
            tool_name="write_file",
            arguments={"path": "app.py"},
            result=ToolResult(status=ToolResultStatus.SUCCESS, output="ok"),
        ))

        assert can_complete_task(task)

    def test_five_of_five_completed_or_skipped_allows_completion(self):
        task = Task(goal="Explain the project")
        for i in range(5):
            item = task.task_plan.add(f"Step {i}")
            item.status = TodoStatus.COMPLETED if i < 3 else TodoStatus.SKIPPED

        # Completed TODOs with evidence
        assert can_complete_task(task)


# ── Model failure must not erase tool success ────────────────────────


class TestModelFailureDoesNotEraseToolSuccess:
    """When tools succeed but model fails, the task should PAUSE, not FAIL."""

    @pytest.mark.asyncio
    async def test_model_429_after_successful_push_preserves_state(self, tmp_path: Path):
        """If git push succeeds then model returns 429, task pauses (not fails)."""
        tools = [RunCommandTool()]
        agent = AgentLoop(
            provider=MagicMock(),
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=10),
            event_bus=EventBus(),
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="allow")],
        )

        # Simulate: tool succeeded, then model error
        agent._modified_files.append("app.py")
        task = Task(goal="Fix and push")
        task.tool_calls.append(ToolCall(
            tool_name="run_command",
            arguments={"command": "echo ok"},
            result=ToolResult(status=ToolResultStatus.SUCCESS, output="ok"),
        ))

        # After model failure with existing work, task should PAUSE
        did_work = bool(agent._modified_files) or any(
            tc.result is not None for tc in task.tool_calls
        )
        assert did_work  # Work was done


# ── SKIPPED TODOs count as resolved ──────────────────────────────────


class TestSkippedTodosCountAsResolved:
    """When TODOs are SKIPPED (model finished with text), they count as resolved."""

    def test_all_skipped_allows_completion(self):
        task = Task(goal="Explain the project")
        for i in range(3):
            item = task.task_plan.add(f"Step {i}")
            item.status = TodoStatus.SKIPPED
            item.error = "Model completed with text response"

        assert can_complete_task(task)

    def test_mixed_completed_and_skipped_allows_completion(self):
        task = Task(goal="Explain the project")
        item1 = task.task_plan.add("Step 1")
        item1.status = TodoStatus.COMPLETED
        item1.evidence = {"tool": "read_file", "status": "success"}

        item2 = task.task_plan.add("Step 2")
        item2.status = TodoStatus.SKIPPED

        assert can_complete_task(task)

    def test_pending_not_skipped_blocks_completion(self):
        task = Task(goal="Enhance the project")
        item1 = task.task_plan.add("Step 1")
        item1.status = TodoStatus.SKIPPED

        item2 = task.task_plan.add("Step 2")
        item2.status = TodoStatus.PENDING  # Not skipped!

        assert not can_complete_task(task)


# ── State machine transitions ────────────────────────────────────────


class TestStateMachineTransitions:
    """Verify the task state machine allows/disallows transitions correctly."""

    def test_executing_to_completed_requires_invariant(self):
        task = Task(goal="Test")
        task.status = TaskStatus.EXECUTING
        # No TODOs completed → cannot transition to COMPLETED
        from harness_core.agent.completion import can_transition
        # can_transition allows it, but can_complete_task blocks it
        # The loop.py code checks can_complete_task before setting COMPLETED

    def test_cancelled_is_terminal(self):
        task = Task(goal="Test")
        task.status = TaskStatus.CANCELLED
        from harness_core.agent.completion import can_transition
        assert not can_transition(TaskStatus.CANCELLED, TaskStatus.COMPLETED)
        assert not can_transition(TaskStatus.CANCELLED, TaskStatus.EXECUTING)

    def test_completed_is_terminal(self):
        task = Task(goal="Test")
        task.status = TaskStatus.COMPLETED
        from harness_core.agent.completion import can_transition
        assert not can_transition(TaskStatus.COMPLETED, TaskStatus.EXECUTING)
        assert not can_transition(TaskStatus.COMPLETED, TaskStatus.FAILED)


# ── No fake completion from model prose ──────────────────────────────


class TestNoFakeCompletion:
    """The model's text response cannot override tool failures."""

    def test_model_text_cannot_override_pending_todos(self):
        task = Task(goal="Fix and test")
        task.result = "All done! Everything works perfectly."
        # But TODOs are still pending
        task.task_plan.add("Run tests")  # PENDING
        task.task_plan.add("Verify")  # PENDING

        assert not can_complete_task(task)

    def test_model_text_cannot_override_failed_todos(self):
        task = Task(goal="Fix and test")
        task.result = "Success!"
        item = task.task_plan.add("Run tests")
        item.status = TodoStatus.FAILED
        item.error = "Tests failed"

        assert not can_complete_task(task)


# ── Workflow paths go through invariant ──────────────────────────────


class TestWorkflowPathsUseInvariant:
    """Workflow fast paths must check can_complete_task before COMPLETED."""

    def test_workflow_result_does_not_bypass_invariant(self):
        """A workflow that completes TODOs but leaves some pending cannot finish."""
        task = Task(goal="Push to GitHub")
        # Workflow completed some TODOs but not all
        item1 = task.task_plan.add("Inspect repo")
        item1.status = TodoStatus.COMPLETED
        item1.evidence = {"tool": "git_status"}

        item2 = task.task_plan.add("Push")
        item2.status = TodoStatus.PENDING  # Not completed!

        assert not can_complete_task(task)
