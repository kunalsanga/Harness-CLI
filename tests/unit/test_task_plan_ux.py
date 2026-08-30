"""Tests for TaskPlan, TodoItem, thinking events, and todo updates."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.agent.types import (
    Task,
    TaskPlan,
    TaskStatus,
    TodoItem,
    TodoStatus,
)


# ─── TodoItem Tests ────────────────────────────────────────────────────────


class TestTodoItem:
    """Test individual TODO item behavior."""

    def test_pending_symbol(self):
        item = TodoItem(description="Inspect code", status=TodoStatus.PENDING)
        assert item.symbol == "☐"
        assert item.display() == "☐ Inspect code"

    def test_active_symbol(self):
        item = TodoItem(description="Run tests", status=TodoStatus.ACTIVE)
        assert item.symbol == "◐"
        assert item.display() == "◐ Run tests"

    def test_completed_symbol(self):
        item = TodoItem(description="Edit file", status=TodoStatus.COMPLETED)
        assert item.symbol == "✓"
        assert item.display() == "✓ Edit file"

    def test_failed_symbol(self):
        item = TodoItem(description="Fix bug", status=TodoStatus.FAILED)
        assert item.symbol == "✗"
        assert item.display() == "✗ Fix bug"

    def test_skipped_symbol(self):
        item = TodoItem(description="Skip this", status=TodoStatus.SKIPPED)
        assert item.symbol == "—"
        assert item.display() == "— Skip this"


# ─── TaskPlan Tests ────────────────────────────────────────────────────────


class TestTaskPlan:
    """Test dynamic task plan management."""

    def test_add_items(self):
        plan = TaskPlan()
        plan.add("Inspect project")
        plan.add("Implement changes")
        plan.add("Run tests")
        assert len(plan.items) == 3
        assert plan.total_count == 3
        assert plan.completed_count == 0

    def test_complete_item(self):
        plan = TaskPlan()
        plan.add("Inspect project")
        plan.add("Run tests")
        plan.complete("Inspect project")
        assert plan.items[0].status == TodoStatus.COMPLETED
        assert plan.items[1].status == TodoStatus.PENDING
        assert plan.completed_count == 1

    def test_activate_item(self):
        plan = TaskPlan()
        plan.add("Inspect project")
        plan.activate("Inspect project")
        assert plan.items[0].status == TodoStatus.ACTIVE

    def test_fail_item(self):
        plan = TaskPlan()
        plan.add("Run tests")
        plan.fail("Run tests")
        assert plan.items[0].status == TodoStatus.FAILED

    def test_is_complete(self):
        plan = TaskPlan()
        plan.add("Step 1")
        plan.add("Step 2")
        assert not plan.is_complete
        plan.complete("Step 1")
        plan.complete("Step 2")
        assert plan.is_complete

    def test_is_complete_with_skipped(self):
        plan = TaskPlan()
        plan.add("Step 1")
        plan.add("Step 2")
        plan.complete("Step 1")
        plan.items[1].status = TodoStatus.SKIPPED
        assert plan.is_complete

    def test_display(self):
        plan = TaskPlan()
        plan.add("Inspect")
        plan.add("Implement")
        plan.complete("Inspect")
        display = plan.display()
        assert len(display) == 2
        assert "✓ Inspect" in display[0]
        assert "☐ Implement" in display[1]

    def test_complete_nonexistent_item(self):
        plan = TaskPlan()
        plan.add("Step 1")
        plan.complete("Nonexistent")  # Should not raise
        assert plan.items[0].status == TodoStatus.PENDING


# ─── Task Integration Tests ────────────────────────────────────────────────


class TestTaskWithPlan:
    """Test Task with integrated TaskPlan."""

    def test_task_has_task_plan(self):
        task = Task(goal="Enhance calculator")
        assert isinstance(task.task_plan, TaskPlan)
        assert len(task.task_plan.items) == 0

    def test_task_has_thinking(self):
        task = Task(goal="Enhance calculator")
        assert task.thinking == ""

    def test_task_plan_adds_items(self):
        task = Task(goal="Enhance calculator")
        task.task_plan.add("Inspect project")
        task.task_plan.add("Implement changes")
        assert task.task_plan.total_count == 2


# ─── Agent Loop Plan Integration Tests ─────────────────────────────────────


class TestAgentLoopPlanIntegration:
    """Test that agent loop creates and manages the task plan."""

    @pytest.mark.asyncio
    async def test_plan_created_event_emitted(self):
        """Agent loop should emit plan.created and todo.updated events."""
        from harness_core.agent.loop import AgentLoop
        from harness_core.agent.types import AgentConfig
        from harness_core.observability.events import EventBus
        from harness_core.providers.base import CompletionRequest, CompletionResponse
        from harness_core.routing.health import ModelHealthTracker
        from harness_core.routing.fallback import FallbackEngine, FallbackConfig, RetryConfig

        bus = EventBus()
        events_received = []

        async def capture_event(event):
            events_received.append(event.type)

        bus.on("plan.created", capture_event)
        bus.on("todo.updated", capture_event)

        # Mock provider that returns a plan then a text response
        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # Planning response
                return CompletionResponse(
                    content="1. Inspect project\n2. Implement changes\n3. Run tests",
                    model="mock", provider="mock",
                )
            else:
                return CompletionResponse(
                    content="Done!",
                    model="mock", provider="mock",
                )

        provider = MagicMock()
        provider.generate = mock_generate

        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=MagicMock(),
            config=AgentConfig(max_iterations=5),
            event_bus=bus,
        )

        task = await loop.run("Enhance this calculator")

        assert "plan.created" in events_received
        assert "todo.updated" in events_received
        assert task.task_plan.total_count == 3

    @pytest.mark.asyncio
    async def test_thinking_event_emitted(self):
        """Agent loop should emit thinking.status events."""
        from harness_core.agent.loop import AgentLoop
        from harness_core.agent.types import AgentConfig
        from harness_core.observability.events import EventBus
        from harness_core.providers.base import CompletionResponse

        bus = EventBus()
        thinking_messages = []

        async def capture_thinking(event):
            thinking_messages.append(event.data.get("message", ""))

        bus.on("thinking.status", capture_thinking)

        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return CompletionResponse(
                    content="1. Inspect\n2. Implement",
                    model="mock", provider="mock",
                )
            return CompletionResponse(
                content="Done!",
                model="mock", provider="mock",
            )

        provider = MagicMock()
        provider.generate = mock_generate

        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=MagicMock(),
            config=AgentConfig(max_iterations=5),
            event_bus=bus,
        )

        await loop.run("Test task")

        assert len(thinking_messages) >= 1
        assert any("task" in m.lower() or "execution" in m.lower() for m in thinking_messages)
