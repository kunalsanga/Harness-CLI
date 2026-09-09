"""Phase 10 — live UI E2E (Part 27).

Proves that *real* EventBus events emitted by the real EngineeringRuntime /
Scheduler / WorkerAgent chain update the RuntimeViewModel state:

    agent_started            → UI agent becomes active
    tool events (stamped)    → current operation updates
    file events              → file list updates
    test events              → test state updates
    recovery_started         → recovery state updates
    agent_completed          → agent becomes completed
    verification_completed   → verification state updates
    runtime_completed        → final state renders

No event is synthesized by the view model; the runtime drives everything.
Only the model/provider boundary and WorkerAgent.run are stubbed, exactly
like the existing unified-runtime E2E.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import AgentConfig as LoopAgentConfig
from harness_core.agent.types import AgentRole as LoopRole
from harness_core.agents.domain import (
    AgentResult,
    AgentRole,
    AgentStatus,
)
from harness_core.agents.worker import WorkerAgent
from harness_core.cli.runtime_dashboard import RuntimeViewModel
from harness_core.observability.events import EventBus
from harness_core.providers.base import CompletionRequest, CompletionResponse
from harness_core.runtime.requirements import Requirements
from harness_core.runtime.runtime import EngineeringRuntime
from harness_core.runtime.state import RuntimeStatus, VerificationStatus

PLAN_JSON = """{
  "summary": "Build a small task application",
  "tasks": [
    {
      "task_id": "architect_1",
      "title": "Design architecture",
      "objective": "Design the architecture",
      "role": "architect",
      "dependencies": [],
      "success_criteria": ["architecture documented"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [],
      "traceable_to": ["REQ-001"]
    },
    {
      "task_id": "tester_1",
      "title": "Run tests",
      "objective": "Execute the test suite and report results",
      "role": "tester",
      "dependencies": ["architect_1"],
      "success_criteria": ["tests reported"],
      "workspace_scope": "project",
      "priority": 3,
      "resources": [],
      "traceable_to": ["REQ-002"]
    }
  ]
}"""


def _plan_provider() -> MagicMock:
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=CompletionResponse(content=PLAN_JSON, model="fake-planner")
    )
    return provider


def _runtime_provider() -> MagicMock:
    provider = MagicMock()

    async def mock_gen(request, *args, **kwargs):
        await asyncio.sleep(0.005)
        content = ""
        if request and hasattr(request, "messages") and request.messages:
            content = str(request.messages[0].content)
        if "autonomous engineering recovery coordinator" in content:
            return CompletionResponse(
                content=(
                    '{"strategy": "debug_and_fix", "specialist_role": "debugger", '
                    '"objective": "Fix the failing tests", '
                    '"success_criteria": ["tests pass"]}'
                ),
                model="fake-recovery",
            )
        return CompletionResponse(content="Done", model="fake-scheduler")

    provider.generate = AsyncMock(side_effect=mock_gen)
    return provider


class _Scenario:
    def __init__(self) -> None:
        self.ran: list[str] = []


def _make_worker_fake(scenario: _Scenario):
    """Tester fails on first run; recovery retest passes."""

    async def fake_run(self):
        contract = self.contract
        scenario.ran.append(contract.task_id)
        await asyncio.sleep(0.02)
        if contract.role == AgentRole.TESTER and "_retest_" not in contract.task_id:
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                tests_passed=1,
                tests_total=3,
                summary="2 tests failed",
                findings=[{"type": "test_failure", "test_name": "x"}],
                files_changed=["tests/results.json"],
            )
        if contract.role == AgentRole.DEBUGGER:
            return AgentResult(
                agent_id=contract.agent_id,
                role=contract.role,
                status=AgentStatus.COMPLETED,
                summary="Fixed the bug",
                files_changed=["src/app.py"],
            )
        return AgentResult(
            agent_id=contract.agent_id,
            role=contract.role,
            status=AgentStatus.COMPLETED,
            summary=f"{contract.role.value} done",
            files_changed=["docs/architecture.md"] if contract.task_id == "architect_1" else [],
        )

    return fake_run


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_ui_e2e_real_events_drive_view_model(tmp_path):
    """Real runtime events must drive the view model end to end."""
    scenario = _Scenario()
    event_bus = EventBus()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=_runtime_provider(),
        plan_provider=_plan_provider(),
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
        max_concurrency=2,
    )

    vm = RuntimeViewModel()
    vm.attach(event_bus)

    original_run = WorkerAgent.run
    WorkerAgent.run = _make_worker_fake(scenario)
    try:
        outcome = await asyncio.wait_for(
            runtime.run(
                "Build a small task application",
                requirements=Requirements.from_entries(
                    objective="Build a small task application",
                    functional=["Architecture documented", "Tests pass"],
                ),
            ),
            timeout=60.0,
        )
    finally:
        WorkerAgent.run = original_run

    # The CLI finalizes the view model with the authoritative outcome; the
    # dashboard does the same before rendering its final summary.
    vm.finalize(outcome)
    snap = vm.snapshot()

    # Terminal state rendered from real events.
    assert snap["terminal"] is True
    assert snap["status"] == "success"

    # Agent lifecycle visible: architect ran; tester ran and failed once;
    # recovery agents (debugger + retest) were spawned by the real scheduler.
    agent_roles = {a["role"] for a in snap["agents"]}
    assert "architect" in agent_roles
    assert "tester" in agent_roles
    assert "debugger" in agent_roles

    # Recovery state came from real recovery_started events.
    assert snap["recovery"]["attempts"] >= 1
    assert snap["recovery"]["exhausted"] is False

    # Verification completed on the real runtime.
    assert snap["verification"]["status"] == "passed"

    # Files merged from the authoritative outcome (real task files_changed).
    paths = {f["path"] for f in snap["files"]}
    assert "docs/architecture.md" in paths
    assert "src/app.py" in paths

    # Cross-check against the authoritative runtime state.
    assert outcome.status == RuntimeStatus.SUCCESS
    assert outcome.state.verification_status == VerificationStatus.PASSED


@pytest.mark.asyncio
async def test_agent_loop_stamps_identity_on_tool_events(tmp_path):
    """Real AgentLoop events carry task/agent identity for attribution.

    The WorkerAgent passes its contract identity into the loop, and the loop
    stamps it on tool events so the dashboard never has to guess which agent
    performed an operation.
    """
    from harness_core.agent.types import ToolResult, ToolResultStatus
    from harness_core.tools.base import Tool, ToolSchema

    class _EchoTool(Tool):
        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(
                name="echo_tool",
                description="Echo a value",
                parameters={"type": "object", "properties": {}},
            )

        async def execute(self, arguments: dict) -> ToolResult:
            return ToolResult(output="ok", status=ToolResultStatus.SUCCESS)

    class _LoopProvider:
        name = "loop-fake"

        def __init__(self) -> None:
            self.calls: list[CompletionResponse] = []

        async def generate(self, request: CompletionRequest):
            # Call 1: the loop's planning request — return a plain plan.
            if len(self.calls) == 0:
                self.calls.append(
                    CompletionResponse(
                        content="1. Use the echo tool\n2. Verify it worked", model="fake"
                    )
                )
                return self.calls[-1]
            # Call 2: the execution loop — request the echo tool.
            if len(self.calls) == 1:
                self.calls.append(
                    CompletionResponse(
                        content="",
                        model="fake",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "echo_tool",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    )
                )
                return self.calls[-1]
            return CompletionResponse(content="done", model="fake")

        async def stream(self, request):
            yield CompletionResponse(content="done", model="fake")

    event_bus = EventBus()
    vm = RuntimeViewModel()
    vm.attach(event_bus)

    loop = AgentLoop(
        provider=_LoopProvider(),
        tools=[_EchoTool()],
        workspace_root=tmp_path,
        config=LoopAgentConfig(role=LoopRole.BUILD, max_iterations=2),
        event_bus=event_bus,
        agent_id="agent-xyz",
        task_id="task-abc",
    )
    task = await loop.run("Use the echo tool")

    # The real loop executed the tool call...
    assert any(tc.tool_name == "echo_tool" for tc in task.tool_calls)

    # ...and every tool event it emitted carried its identity, which the
    # view model used to attribute the operation truthfully.
    assert vm.agents["task-abc"].name == "agent-xyz"
    assert vm.agents["task-abc"].operation == "echo_tool"
    assert vm.tool_calls >= 1

    # The stamped events also reached the bus with identity intact.
    tool_events = [e for e in event_bus.get_history("tool.call")]
    assert tool_events
    assert tool_events[0].data.get("task_id") == "task-abc"
    assert tool_events[0].data.get("agent_id") == "agent-xyz"
