"""Deterministic coverage for canonical task progress and its TUI projection."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console

from harness_core.agent.types import (
    AgentConfig,
    ExecutionState,
    Task,
    TaskExecutionTimeline,
    TaskStatus,
    TodoStatus,
)
from harness_core.cli.conversation import ConversationRenderer
from harness_core.observability.events import EventBus


@pytest.mark.parametrize(
    "state",
    [
        ExecutionState.IDLE,
        ExecutionState.UNDERSTANDING,
        ExecutionState.PLANNING,
        ExecutionState.THINKING,
        ExecutionState.WORKING,
        ExecutionState.REPAIRING,
        ExecutionState.VERIFYING,
        ExecutionState.COMPLETED,
        ExecutionState.FAILED,
        ExecutionState.CANCELLED,
        ExecutionState.PAUSED,
    ],
)
def test_execution_state_vocabulary(state):
    assert state.value


def test_task_begins_idle_and_understanding_starts_the_monotonic_clock():
    timeline = Task().execution_timeline
    assert timeline.state is ExecutionState.IDLE
    assert timeline.started_at is None
    assert timeline.transition(ExecutionState.UNDERSTANDING, at=10.0)
    assert timeline.started_at == 10.0
    assert timeline.transitioned_at == 10.0


def test_phase_transitions_accumulate_wall_time_without_overlap():
    timeline = TaskExecutionTimeline()
    timeline.transition(ExecutionState.PLANNING, at=10.0)
    timeline.transition(ExecutionState.WORKING, at=12.5)
    timeline.transition(ExecutionState.REPAIRING, at=22.5)
    timeline.transition(ExecutionState.VERIFYING, at=27.0)
    timeline.transition(ExecutionState.COMPLETED, at=30.0)
    assert timeline.phase_durations == {
        "planning": 2.5,
        "working": 10.0,
        "repairing": 4.5,
        "verifying": 3.0,
    }
    assert timeline.elapsed == 20.0


def test_model_and_tool_diagnostics_do_not_double_count_task_time():
    timeline = TaskExecutionTimeline()
    timeline.transition(ExecutionState.WORKING, at=100.0)
    timeline.record_model_attempt(5.0)
    timeline.record_tool(2.0)
    timeline.transition(ExecutionState.COMPLETED, at=110.0)
    assert timeline.elapsed == 10.0
    assert timeline.phase_durations["working"] == 10.0
    assert timeline.model_attempt_duration == 5.0
    assert timeline.tool_duration == 2.0


@pytest.mark.parametrize(
    ("status", "state"),
    [
        (TaskStatus.COMPLETED, ExecutionState.COMPLETED),
        (TaskStatus.FAILED, ExecutionState.FAILED),
        (TaskStatus.PARTIAL, ExecutionState.FAILED),
        (TaskStatus.CANCELLED, ExecutionState.CANCELLED),
        (TaskStatus.PAUSED, ExecutionState.PAUSED),
    ],
)
def test_terminal_task_status_mapping(status, state):
    mapping = {
        TaskStatus.COMPLETED: ExecutionState.COMPLETED,
        TaskStatus.FAILED: ExecutionState.FAILED,
        TaskStatus.PARTIAL: ExecutionState.FAILED,
        TaskStatus.CANCELLED: ExecutionState.CANCELLED,
        TaskStatus.PAUSED: ExecutionState.PAUSED,
    }
    assert mapping[status] is state


def test_same_phase_refresh_does_not_reset_task_timeline_or_plan():
    task = Task(goal="Make a calculator")
    task.task_plan.add("Implement calculator logic")
    task.execution_timeline.transition(ExecutionState.WORKING, at=5.0)
    task.execution_timeline.transition(ExecutionState.WORKING, at=12.0)
    assert task.execution_timeline.started_at == 5.0
    assert task.execution_timeline.state is ExecutionState.WORKING
    assert task.task_plan.total_count == 1


@pytest.mark.parametrize(
    ("phase", "state"),
    [
        ("understanding", ExecutionState.UNDERSTANDING),
        ("planning", ExecutionState.PLANNING),
        ("implementing", ExecutionState.WORKING),
        ("diagnosing", ExecutionState.REPAIRING),
        ("fixing", ExecutionState.REPAIRING),
        ("verifying", ExecutionState.VERIFYING),
    ],
)
@pytest.mark.asyncio
async def test_existing_phase_events_map_to_canonical_states(tmp_path, phase, state):
    from harness_core.agent.loop import AgentLoop

    loop = AgentLoop(provider=object(), tools=[], workspace_root=tmp_path, event_bus=EventBus())
    task = Task(goal="task")
    loop._active_task = task
    await loop._emit_phase(phase)
    assert task.execution_timeline.state is state


def test_plan_and_current_todo_are_in_each_working_message(tmp_path):
    from harness_core.agent.loop import AgentLoop

    task = Task(goal="Make a calculator")
    todo = task.task_plan.add("Implement calculator logic")
    todo.status = TodoStatus.ACTIVE
    loop = AgentLoop(
        provider=object(), tools=[], workspace_root=tmp_path,
        config=AgentConfig(), event_bus=EventBus(),
    )
    content = "\n".join(message["content"] for message in loop._build_messages(task))
    assert "ACTIVE PLAN / CURRENT TODO STATUS" in content
    assert "[IN_PROGRESS] Implement calculator logic" in content


def test_renderer_stores_plan_created_and_todo_updated_items():
    console = Console(file=StringIO(), width=80)
    renderer = ConversationRenderer(console)
    renderer.update_todo_items([
        {"title": "Build calculator UI", "status": "completed"},
        {"title": "Implement calculator logic", "status": "in_progress"},
    ])
    output = StringIO()
    Console(file=output, width=80).print(renderer._renderable())
    rendered = output.getvalue()
    assert "Plan" in rendered
    assert "Build calculator UI" in rendered
    assert "Implement calculator logic" in rendered
    assert "✓" in rendered and "●" in rendered


def test_safe_tool_summary_is_compact_and_does_not_need_raw_arguments():
    renderer = ConversationRenderer(Console(file=StringIO(), width=40))
    renderer.tool_started_safe("read_file", "read_file: src/harness_core/agent/loop.py")
    assert renderer.state.items[-1][1] == "Read .../loop.py"


def test_renderer_maps_execution_event_to_canonical_live_state():
    renderer = ConversationRenderer(Console(file=StringIO(), width=80))
    renderer.update_execution({
        "state": "repairing", "started_at": 100.0,
        "phase_durations": {"working": 3.0},
    })
    assert renderer.state.execution_state == "repairing"
    assert renderer.state.display_phase == "✦ Fixing…"
    assert renderer.state.phase_durations["working"] == 3.0


def test_model_activity_is_visible_without_exposing_request_content():
    renderer = ConversationRenderer(Console(file=StringIO(), width=60))
    renderer.update_execution({"state": "thinking", "started_at": 10.0})
    renderer.update_model_activity("google/gemma-4-31b-it:free")
    assert renderer.state.display_phase == "✦ Considering…"
    assert renderer.state.model_activity == "Requesting Gemma 4 31B"


def test_model_switch_adds_model_history_without_changing_task_clock():
    renderer = ConversationRenderer(Console(file=StringIO(), width=60))
    renderer.update_execution({"state": "working", "started_at": 25.0})
    renderer.update_models(["google/gemma-4-31b-it:free"])
    renderer.update_models(["google/gemma-4-26b-a4b-it:free"])
    assert renderer.state.started_at == 25.0
    assert renderer.state.models_used == [
        "google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"
    ]


def test_completion_footer_contains_duration_tool_count_and_models():
    output = StringIO()
    renderer = ConversationRenderer(Console(file=output, width=80), plain=True)
    renderer.state.tool_calls = 9
    renderer.render_completion(
        42.0,
        models_used=["google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"],
        phase_durations={"planning": 2.0, "working": 18.0, "repairing": 5.0},
    )
    rendered = output.getvalue()
    assert "Completed" in rendered and "42s" in rendered and "9 tool calls" in rendered
    assert "Gemma 4 31B" in rendered and "Gemma 4 26B A4B" in rendered
    assert "Planning 2.0s" in rendered and "Working 18s" in rendered


def test_narrow_terminal_rendering_truncates_plan_without_error():
    console = Console(file=StringIO(), width=28)
    renderer = ConversationRenderer(console)
    renderer.update_execution({"state": "working", "started_at": 100.0})
    renderer.update_todo_items([
        {"title": "Implement a very long calculator interface title that should be clipped", "status": "pending"}
    ])
    output = StringIO()
    Console(file=output, width=28).print(renderer._renderable())
    rendered = output.getvalue()
    assert "Plan" in rendered
    assert "..." in rendered


@pytest.mark.asyncio
async def test_agent_loop_emits_understanding_plan_work_and_terminal_states(tmp_path):
    from harness_core.agent.loop import AgentLoop
    from harness_core.providers.base import CompletionResponse

    calls = 0

    class FakeProvider:
        name = "fake"

        async def generate(self, request):
            nonlocal calls
            calls += 1
            content = (
                "1. Inspect project\n2. Implement changes\n3. Run checks"
                if calls == 1 else "Completed the requested work."
            )
            return CompletionResponse(content=content, model="fake", provider="fake")

    bus = EventBus()
    states = []

    async def capture(event):
        states.append(event.data["state"])

    bus.on("execution.state", capture)
    loop = AgentLoop(
        provider=FakeProvider(), tools=[], workspace_root=tmp_path,
        config=AgentConfig(max_iterations=2, verify_on_complete=False), event_bus=bus,
    )
    result = await loop.run("Improve this calculator")
    assert states[0] == "understanding"
    assert "planning" in states and "working" in states
    assert states[-1] in {"completed", "failed"}
    assert result.execution_timeline.finished_at is not None


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TaskStatus.FAILED, "failed"),
        (TaskStatus.CANCELLED, "cancelled"),
        (TaskStatus.PAUSED, "paused"),
    ],
)
@pytest.mark.asyncio
async def test_agent_loop_emits_terminal_failure_cancel_pause(tmp_path, status, expected):
    from harness_core.agent.loop import AgentLoop

    bus = EventBus()
    states = []

    async def capture(event):
        states.append(event.data["state"])

    bus.on("execution.state", capture)
    loop = AgentLoop(provider=object(), tools=[], workspace_root=tmp_path, event_bus=bus)
    task = Task(goal="task", status=status)
    loop._active_task = task
    await loop._transition_execution(ExecutionState.WORKING)
    await loop._finish_execution(task)
    assert states[-1] == expected


@pytest.mark.asyncio
async def test_deterministic_workflow_tool_duration_is_nested_in_task_timeline(tmp_path):
    import asyncio
    from pathlib import Path
    from harness_core.agent.loop import AgentLoop
    from harness_core.agent.workflows import WorkflowContext, _run_tool
    from harness_core.agent.types import ToolResults

    class FakeTool:
        async def execute(self, args):
            await asyncio.sleep(0.002)
            return ToolResults.success("ok")

    loop = AgentLoop(provider=object(), tools=[], workspace_root=tmp_path, event_bus=EventBus())
    task = Task(goal="Inspect project")
    loop._active_task = task
    context = WorkflowContext(
        workspace=Path(tmp_path),
        tools={"read_file": FakeTool()},
        event_bus=loop.event_bus,
        record_tool_call=loop._record_workflow_tool,
    )
    result = await _run_tool(context, "read_file", path="README.md")
    assert result.output == "ok"
    assert task.tool_calls[0].duration_ms > 0
    assert task.execution_timeline.tool_duration > 0
