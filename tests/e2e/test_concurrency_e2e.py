"""Phase 10 — parallel execution proof (Parts 9, 26, 29).

Drives the REAL EngineeringRuntime -> Scheduler with three independent root
tasks and one dependent task.  The scheduler is configured with
max_concurrency=3 and must demonstrably run the roots at the same time:

    * interval overlap between the independent roots (real concurrency)
    * wall-clock duration well below the serial sum (measurable speedup)

Only the model/provider boundary and WorkerAgent.run are stubbed; the task
graph, scheduler dispatch, readiness and dependency logic are production code.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.agents.domain import AgentResult, AgentStatus
from harness_core.agents.worker import WorkerAgent
from harness_core.observability.events import EventBus
from harness_core.providers.base import CompletionResponse
from harness_core.runtime.runtime import EngineeringRuntime
from harness_core.runtime.state import RuntimeStatus

PLAN_JSON = """{
  "summary": "Parallel build",
  "tasks": [
    {
      "task_id": "root_a",
      "title": "Component A",
      "objective": "Implement component A",
      "role": "backend",
      "dependencies": [],
      "success_criteria": ["a done"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [],
      "traceable_to": []
    },
    {
      "task_id": "root_b",
      "title": "Component B",
      "objective": "Implement component B",
      "role": "frontend",
      "dependencies": [],
      "success_criteria": ["b done"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [],
      "traceable_to": []
    },
    {
      "task_id": "root_c",
      "title": "Component C",
      "objective": "Implement component C",
      "role": "database",
      "dependencies": [],
      "success_criteria": ["c done"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [],
      "traceable_to": []
    },
    {
      "task_id": "join_1",
      "title": "Integration",
      "objective": "Integrate A, B and C",
      "role": "integration",
      "dependencies": ["root_a", "root_b", "root_c"],
      "success_criteria": ["joined"],
      "workspace_scope": "project",
      "priority": 4,
      "resources": [],
      "traceable_to": []
    }
  ]
}"""

ROOT_SLEEP = 0.25  # seconds each independent root blocks in the worker


def _plan_provider() -> MagicMock:
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=CompletionResponse(content=PLAN_JSON, model="fake-planner")
    )
    return provider


def _runtime_provider() -> MagicMock:
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=CompletionResponse(content="Done", model="fake-scheduler")
    )
    return provider


class _Timings:
    """Start/end timestamps recorded by the (stubbed) worker runs."""

    def __init__(self) -> None:
        self.runs: dict[str, tuple[float, float]] = {}


def _make_worker_fake(timings: _Timings):
    async def fake_run(self):
        contract = self.contract
        start = time.monotonic()
        # Independent roots block long enough to be measurably parallel.
        if contract.task_id in ("root_a", "root_b", "root_c"):
            await asyncio.sleep(ROOT_SLEEP)
        else:
            await asyncio.sleep(0.02)
        end = time.monotonic()
        timings.runs[contract.task_id] = (start, end)
        return AgentResult(
            agent_id=contract.agent_id,
            role=contract.role,
            status=AgentStatus.COMPLETED,
            summary=f"{contract.task_id} done",
        )

    return fake_run


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_independent_tasks_run_concurrently(tmp_path):
    timings = _Timings()
    event_bus = EventBus()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=_runtime_provider(),
        plan_provider=_plan_provider(),
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
        max_concurrency=3,
    )

    original_run = WorkerAgent.run
    WorkerAgent.run = _make_worker_fake(timings)
    started = time.monotonic()
    try:
        outcome = await asyncio.wait_for(
            runtime.run("Build three independent components and integrate them"),
            timeout=60.0,
        )
    finally:
        WorkerAgent.run = original_run
    wall = time.monotonic() - started

    assert outcome.status == RuntimeStatus.SUCCESS

    # 1. All three roots demonstrably overlapped in time — the scheduler
    #    dispatched them together instead of serializing them.
    for a in ("root_a", "root_b", "root_c"):
        for b in ("root_a", "root_b", "root_c"):
            if a < b:
                assert _intervals_overlap(timings.runs[a], timings.runs[b]), (
                    f"{a} and {b} did not overlap — execution was serialized"
                )

    # 2. Wall clock is well below the serial sum (3 × ROOT_SLEEP ≈ 0.75s).
    serial_baseline = 3 * ROOT_SLEEP
    assert wall < serial_baseline - 0.1, (
        f"parallel wall time {wall:.2f}s should be well under serial "
        f"baseline {serial_baseline:.2f}s"
    )

    # 3. The dependent task really waited for all three roots.
    join_start = timings.runs["join_1"][0]
    for root in ("root_a", "root_b", "root_c"):
        assert timings.runs[root][1] <= join_start, (
            f"join_1 started before {root} finished"
        )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_concurrency_events_visible_on_bus(tmp_path):
    """The scheduler's real dispatch events prove concurrent activity."""
    timings = _Timings()
    event_bus = EventBus()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=_runtime_provider(),
        plan_provider=_plan_provider(),
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
        max_concurrency=3,
    )
    original_run = WorkerAgent.run
    WorkerAgent.run = _make_worker_fake(timings)
    try:
        await runtime.run("Build three independent components and integrate them")
    finally:
        WorkerAgent.run = original_run

    started = [e for e in event_bus.get_history("task.started")]
    task_ids = {e.data.get("task_id") for e in started}
    assert {"root_a", "root_b", "root_c"} <= task_ids

    # task.started events now carry the scheduler-owned role (Phase 10).
    roles = {e.data.get("role") for e in started if e.data.get("task_id") in ("root_a", "root_b")}
    assert roles == {"backend", "frontend"}
