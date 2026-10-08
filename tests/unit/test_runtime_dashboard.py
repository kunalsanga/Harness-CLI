"""Phase 10 — RuntimeViewModel / LiveTerminalUI unit tests.

The dashboard must be a *pure adapter* over real EventBus events: every state
field derives from an emitted event (or the authoritative RuntimeOutcome via
``finalize``).  These tests prove the mapping and that nothing is fabricated.
"""

from __future__ import annotations

from types import SimpleNamespace

from harness_core.cli.runtime_dashboard import (
    LiveTerminalUI,
    RuntimeViewModel,
    format_elapsed,
    tool_display_name,
)
from harness_core.observability.events import Event, EventBus


async def _emit(bus: EventBus, event_type: str, data: dict | None = None) -> None:
    await bus.emit(Event(type=event_type, source="test", data=data or {}))


# ── RuntimeViewModel: event → state mapping ────────────────────────────────


async def test_vm_maps_runtime_lifecycle_events():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)

    await _emit(bus, "runtime_started", {"project_id": "p1"})
    await _emit(bus, "runtime_stage_changed", {"stage": "plan", "detail": ""})
    await _emit(bus, "plan_created", {"total_tasks": 5})
    await _emit(bus, "runtime_completed", {"status": "success"})

    snap = vm.snapshot()
    assert snap["status"] == "success"
    assert snap["terminal"] is True
    assert snap["stage"] == "plan"
    assert snap["metrics"]["plan_tasks"] == 5


async def test_vm_maps_task_and_tool_events_to_agents():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)

    await _emit(bus, "task.started", {"task_id": "t1", "agent_id": "a1", "role": "backend"})
    # Real AgentLoop events are stamped with task identity (Phase 10).
    await _emit(
        bus, "tool.call", {"tool": "write_file", "args": {"path": "src/api.py"}, "task_id": "t1"}
    )
    await _emit(bus, "tool.result", {"tool": "write_file", "status": "success", "task_id": "t1"})

    agents = vm.snapshot()["agents"]
    assert len(agents) == 1
    agent = agents[0]
    assert agent["task_id"] == "t1"
    assert agent["status"] == "running"
    assert "api.py" in agent["detail"]
    # File evidence came from the real tool event, not from fabrication.
    assert vm.files.get("src/api.py") is not None
    assert vm.tool_calls == 1


async def test_vm_completed_agent_is_marked_completed():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)

    await _emit(bus, "task.started", {"task_id": "t1", "agent_id": "a1"})
    await _emit(bus, "task.completed", {"task_id": "t1", "agent_id": "a1"})
    await _emit(bus, "task.started", {"task_id": "t2", "agent_id": "a2"})
    await _emit(bus, "task.failed", {"task_id": "t2", "agent_id": "a2", "error": "boom"})

    by_id = {a["task_id"]: a for a in vm.snapshot()["agents"]}
    assert by_id["t1"]["status"] == "completed"
    assert by_id["t2"]["status"] == "failed"


async def test_vm_distinguishes_waiting_and_blocked():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)

    await _emit(bus, "task.queued", {"task_id": "t1", "agent_id": "a1"})
    await _emit(bus, "task.started", {"task_id": "t2", "agent_id": "a2"})
    await _emit(bus, "resource_lock_waiting", {"task_id": "t3"})

    # A waiting agent is visible but clearly not running.
    agents = vm.snapshot()["agents"]
    statuses = {a["task_id"]: a["status"] for a in agents}
    assert statuses.get("t1") == "queued"
    assert statuses.get("t2") == "running"


async def test_vm_maps_recovery_events():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)

    await _emit(
        bus, "recovery_started", {"task_id": "t1", "category": "TEST_FAILURE", "attempt": 1}
    )
    assert vm.snapshot()["recovery"]["attempts"] == 1
    assert vm.snapshot()["recovery"]["active"] is True

    await _emit(bus, "recovery_exhausted", {"task_id": "t1"})
    snap = vm.snapshot()
    assert snap["recovery"]["exhausted"] is True
    assert snap["recovery"]["active"] is False


async def test_vm_maps_test_and_verification_events():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)

    await _emit(bus, "test.completed", {"passed": 31, "total": 34, "success": False})
    await _emit(bus, "verification.started", {})
    await _emit(bus, "verification.completed", {"passed": True, "summary": "all verified"})

    snap = vm.snapshot()
    assert snap["tests"]["passed"] == 31
    assert snap["tests"]["total"] == 34
    assert snap["verification"]["status"] == "passed"


async def test_vm_never_fabricates_activity():
    """An empty VM has no agents, no files, no tests, no activity."""
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)
    await _emit(bus, "runtime_started", {})
    snap = vm.snapshot()
    assert snap["agents"] == []
    assert snap["files"] == []
    assert snap["activity"] == []
    assert snap["tests"]["has_evidence"] is False


def test_vm_finalize_merges_authoritative_outcome():
    vm = RuntimeViewModel()
    outcome = SimpleNamespace(
        status="success",
        state=SimpleNamespace(
            status=SimpleNamespace(value="success"),
            stage=SimpleNamespace(value="succeeded"),
            completed_at=1.0,
            original_request="Build app",
            workspace="/tmp/ws",
            verification_status=SimpleNamespace(value="passed"),
            verification_summary="all verified",
            recovery_attempts=1,
            recovery_exhausted=False,
        ),
        graph=SimpleNamespace(
            tasks={
                "t1": SimpleNamespace(
                    files_changed=["src/app.py", "tests/test_app.py"]
                )
            }
        ),
    )
    vm.finalize(outcome)
    snap = vm.snapshot()
    assert snap["status"] == "success"
    assert snap["verification"]["status"] == "passed"
    assert {f["path"] for f in snap["files"]} == {"src/app.py", "tests/test_app.py"}


def test_vm_detach_stops_consuming_events():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)
    vm.detach(bus)

    import asyncio

    async def _after():
        await _emit(bus, "runtime_stage_changed", {"stage": "execute"})

    asyncio.run(_after())
    assert vm.snapshot()["stage"] == "unknown"


# ── LiveTerminalUI: non-blocking rendering ────────────────────────────────


def test_live_ui_plain_mode_renders_without_blocking():
    from rich.console import Console

    console = Console(record=True)
    vm = RuntimeViewModel()
    vm.stage = "execute"
    vm.tool_calls = 3

    ui = LiveTerminalUI(console, plain=True)
    ui.start(vm)
    ui.update(vm)  # must not raise and must not block
    ui.update(vm)
    ui.stop(vm)

    text = console.export_text()
    assert "execute" in text
    assert "3 tools" in text


def test_live_ui_render_contains_real_state_only():
    from rich.console import Console

    from harness_core.cli.runtime_dashboard import AgentView, FileChange

    console = Console(record=True)
    vm = RuntimeViewModel()
    vm.request = "Build a task manager"
    vm.stage = "execute"
    vm.agents["t1"] = AgentView(
        task_id="t1", name="a1", role="backend", status="running",
        operation="write src/api.py", detail="src/api.py", latest_at=1.0,
    )
    vm.files["src/api.py"] = FileChange(path="src/api.py", status="M")
    vm.tests.passed = 31
    vm.tests.total = 34

    # The Rich dashboard is a pure function of the view-model state.
    ui = LiveTerminalUI(console, plain=True)
    console.print(ui._renderable(vm))
    text = console.export_text()
    assert "Considering" in text
    assert "Build a task manager" in text


def test_format_elapsed():
    assert format_elapsed(5) == "5.0s"
    assert format_elapsed(65) == "1m05s"
    assert "h" in format_elapsed(3700)


def test_tool_display_name_concise():
    assert tool_display_name("read_file", {"path": "a.py"}) == "read a.py"
    assert tool_display_name("run_command", {"command": "pytest x"}) == "run pytest x"
    assert tool_display_name("git_status", {}) == "git status"
