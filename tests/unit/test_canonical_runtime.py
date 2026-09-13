"""Stage 1 — Canonical Runtime regression tests.

Proves that:
1. InteractiveShell._execute_task() routes through EngineeringRuntime
2. EngineeringRuntime.execute_interactive() creates a single-task graph
3. WorkerAgent receives the interactive AgentConfig from the shell
4. The canonical path is the single authoritative execution path
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.observability.events import EventBus


# ── Helpers ────────────────────────────────────────────────────────────────


class _FakeProvider:
    """Minimal provider stub for testing."""

    name = "fake"

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def generate(self, request):
        from harness_core.providers.base import CompletionResponse

        return CompletionResponse(content="Task completed successfully", model="fake")

    async def list_models(self):
        return []


def _make_shell(workspace: Path):
    """Create an InteractiveShell with minimal setup for testing."""
    from harness_core.cli.interactive import InteractiveShell

    shell = InteractiveShell.__new__(InteractiveShell)
    shell.model = None
    shell.mode = "auto"
    shell.free = False
    shell.local = False
    shell.plain = True
    shell.max_iterations = 5
    shell.max_cost = None
    shell.workspace = str(workspace)
    shell.max_parallel = 3
    shell.verbose = False
    shell.console = MagicMock()
    shell.session_id = None
    shell.session_manager = None
    shell.current_model = ""
    shell.current_provider = ""
    shell.total_tool_calls = 0
    shell.total_iterations = 0
    shell.session_start = 0.0
    shell.task_start = 0.0
    shell.running = False
    shell.cancel_event = asyncio.Event()
    shell._event_bus = EventBus()
    shell._agent_loop = None
    shell._provider = _FakeProvider()
    shell._router = MagicMock()
    shell._task_aware = None
    shell._tools = []
    shell._last_stats = {}
    shell._active_runtime = None
    shell._view_model = None
    shell._conv = None
    shell._prompt_session = None
    shell._pt_history = None
    shell._pt_completer = None
    shell._pt_history_items = []
    return shell


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_interactive_creates_single_task_graph(tmp_path):
    """EngineeringRuntime.execute_interactive() creates a 1-task graph without planner."""
    from harness_core.runtime.runtime import EngineeringRuntime
    from harness_core.runtime.state import RuntimeStatus

    event_bus = EventBus()
    provider = _FakeProvider()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=provider,
        plan_provider=provider,
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
    )

    outcome = await runtime.execute_interactive("Test goal")

    assert outcome.status == RuntimeStatus.SUCCESS
    graph = outcome.graph
    assert graph is not None
    assert graph.get_total_count() == 1
    task = graph.get_task("interactive_task_1")
    assert task is not None
    assert task.description == "Test goal"


@pytest.mark.asyncio
async def test_execute_interactive_emits_canonical_events(tmp_path):
    """execute_interactive emits the canonical event sequence."""
    from harness_core.runtime.runtime import EngineeringRuntime

    event_bus = EventBus()
    provider = _FakeProvider()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=provider,
        plan_provider=provider,
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
    )

    await runtime.execute_interactive("Test goal")

    types = {e.type for e in event_bus.get_history()}
    # Canonical events that must be present
    assert "runtime_started" in types
    assert "plan_created" in types
    assert "task.started" in types
    assert "task.completed" in types
    assert "verification_completed" in types
    assert "runtime_completed" in types


@pytest.mark.asyncio
async def test_execute_interactive_uses_same_event_bus(tmp_path):
    """Events from WorkerAgent/AgentLoop flow through the same EventBus."""
    from harness_core.runtime.runtime import EngineeringRuntime

    event_bus = EventBus()
    provider = _FakeProvider()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=provider,
        plan_provider=provider,
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
    )

    received_events = []

    async def capture(event):
        received_events.append(event.type)

    event_bus.on("*", capture)

    await runtime.execute_interactive("Test goal")

    # The shell's event handlers would receive these same events
    # (tool events only appear when the model makes tool calls)
    assert "task.started" in received_events
    assert "runtime_completed" in received_events


@pytest.mark.asyncio
async def test_worker_agent_receives_agent_config(tmp_path):
    """WorkerAgent uses the AgentConfig passed from the shell."""
    from harness_core.agents.worker import WorkerAgent
    from harness_core.agents.domain import AgentContract, AgentRole, WorkspaceScope
    from harness_core.agent.types import AgentConfig, AgentRole as LoopRole

    config = AgentConfig(
        role=LoopRole.BUILD,
        max_iterations=10,
        model_preference="test-model",
        routing_mode="free",
    )

    contract = AgentContract(
        role=AgentRole.CODER,
        task_id="test_task",
        objective="Test",
        allowed_tools=[],
        workspace_scope=WorkspaceScope.PROJECT,
    )

    worker = WorkerAgent(
        contract=contract,
        provider=_FakeProvider(),
        tools=[],
        event_bus=EventBus(),
        workspace_path=str(tmp_path),
        agent_config=config,
    )

    assert worker._agent_config is config
    assert worker._agent_config.max_iterations == 10
    assert worker._agent_config.model_preference == "test-model"


@pytest.mark.asyncio
async def test_scheduler_passes_interactive_config_to_worker(tmp_path):
    """Scheduler forwards interactive_config to WorkerAgent."""
    from harness_core.agents.scheduler import Scheduler
    from harness_core.agents.registry import AgentRegistry
    from harness_core.agent.types import AgentConfig, AgentRole as LoopRole

    config = AgentConfig(
        role=LoopRole.BUILD,
        max_iterations=7,
        model_preference="custom-model",
    )

    scheduler = Scheduler(
        event_bus=EventBus(),
        registry=AgentRegistry(),
        provider=_FakeProvider(),
        tools=[],
        workspace_path=str(tmp_path),
        interactive_config=config,
    )

    assert scheduler._interactive_config is config


@pytest.mark.asyncio
async def test_interactive_shell_routes_through_engineering_runtime(tmp_path):
    """InteractiveShell._execute_task() uses EngineeringRuntime, not AgentLoop directly."""
    shell = _make_shell(tmp_path)
    shell._setup_event_handlers()

    with patch(
        "harness_core.runtime.runtime.EngineeringRuntime.execute_interactive",
        new_callable=AsyncMock,
    ) as mock_exec:
        from harness_core.runtime.runtime import RuntimeOutcome
        from harness_core.runtime.state import RuntimeStatus, ProjectState

        state = ProjectState(project_id="test", workspace=str(tmp_path))
        state.status = RuntimeStatus.SUCCESS
        mock_exec.return_value = RuntimeOutcome(
            state=state,
            status=RuntimeStatus.SUCCESS,
            duration_ms=100.0,
        )

        result = await shell._execute_task("test goal")

        # The runtime was used, not AgentLoop directly
        mock_exec.assert_called_once_with("test goal")


@pytest.mark.asyncio
async def test_no_competing_runtimes(tmp_path):
    """There is only ONE runtime path: EngineeringRuntime.execute_interactive()."""
    # Verify that the InteractiveShell code path does NOT call AgentLoop.run()
    shell = _make_shell(tmp_path)
    shell._setup_event_handlers()

    with patch(
        "harness_core.runtime.runtime.EngineeringRuntime.execute_interactive",
        new_callable=AsyncMock,
    ) as mock_runtime_exec, patch(
        "harness_core.agent.loop.AgentLoop.run",
        new_callable=AsyncMock,
    ) as mock_agent_run:
        from harness_core.runtime.runtime import RuntimeOutcome
        from harness_core.runtime.state import RuntimeStatus, ProjectState

        state = ProjectState(project_id="test", workspace=str(tmp_path))
        state.status = RuntimeStatus.SUCCESS
        mock_runtime_exec.return_value = RuntimeOutcome(
            state=state,
            status=RuntimeStatus.SUCCESS,
            duration_ms=100.0,
        )

        await shell._execute_task("test goal")

        # EngineeringRuntime was used
        mock_runtime_exec.assert_called_once()
        # AgentLoop.run was NOT called directly (it's called inside WorkerAgent)
        mock_agent_run.assert_not_called()
