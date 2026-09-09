"""Phase 10.5 — validation semantics + git evidence unit tests (Parts 11-13, 30).

A previous successful suite must never mask a later failing command, and git
state must derive from real git tool events — never from model prose.
"""

from __future__ import annotations

from harness_core.cli.runtime_dashboard import RuntimeViewModel
from harness_core.observability.events import Event, EventBus


async def _emit(bus: EventBus, event_type: str, data: dict | None = None) -> None:
    await bus.emit(Event(type=event_type, source="test", data=data or {}))


# ── Validation semantics (Part 11) ─────────────────────────────────────────


async def test_green_suite_then_failing_command_keeps_latest_failure():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)

    # Baseline suite passes.
    await _emit(
        bus, "test.completed",
        {"passed": 32, "total": 32, "success": True, "command": "pytest -q"},
    )
    assert vm.baseline_validation() is not None
    assert vm.latest_validation() is not None
    assert vm.latest_validation().is_failure is False

    # A later targeted command fails (workflow test / node test.js).
    await _emit(
        bus, "test.completed",
        {"passed": 0, "total": 1, "success": False, "command": "node test.js"},
    )
    latest = vm.latest_validation()
    assert latest is not None
    assert latest.is_failure is True
    # The green baseline is still remembered separately, but 'latest' is the
    # failing run — the stale success cannot mask it.
    baseline = vm.baseline_validation()
    assert baseline is not None
    assert baseline.is_failure is False


async def test_latest_validation_exposed_in_snapshot():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)
    await _emit(bus, "test.completed", {"passed": 10, "total": 10, "success": True})
    await _emit(bus, "test.completed", {"passed": 2, "total": 3, "success": False})
    snap = vm.snapshot()
    assert snap["validation"]["latest"]["success"] is False
    assert snap["validation"]["baseline"]["success"] is True
    assert len(snap["validation"]["runs"]) == 2


async def test_test_command_tool_failure_records_validation_run():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)
    # A shell command that looks like a test command fails.
    await _emit(
        bus, "tool.result",
        {"tool": "run_command", "status": "error", "exit_code": 1,
         "command": "pytest -q tests/api", "task_id": "t1"},
    )
    latest = vm.latest_validation()
    assert latest is not None
    assert latest.is_failure is True
    assert latest.exit_code == 1


# ── Git evidence (Parts 13, 30) ────────────────────────────────────────────


async def test_git_state_derives_from_tool_results_only():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)

    # No git events → no git state claimed.
    assert vm.git["commit"] == ""
    assert vm.git["push"] == ""
    assert vm.git["repo_detected"] is False

    await _emit(bus, "tool.result", {"tool": "git_status", "status": "success",
                                     "metadata": {"operation": "git_status", "clean": False},
                                     "task_id": "t1"})
    await _emit(bus, "tool.result", {"tool": "git_remote", "status": "success",
                                     "metadata": {"operation": "git_remote", "remotes": {"origin": "url"}},
                                     "task_id": "t1"})
    await _emit(bus, "tool.result", {"tool": "git_commit", "status": "success",
                                     "metadata": {"operation": "git_commit", "commit_hash": "06a6ae3"},
                                     "task_id": "t1"})
    await _emit(bus, "tool.result", {"tool": "git_push", "status": "success",
                                     "metadata": {"operation": "git_push", "remote": "origin", "branch": "main"},
                                     "task_id": "t1"})

    assert vm.git["repo_detected"] is True
    assert vm.git["remote"] == "origin"
    assert vm.git["commit"] == "06a6ae3"
    assert vm.git["push"] == "origin/main"
    assert "git_commit" in vm.git["operations"]
    assert "git_push" in vm.git["operations"]


async def test_git_evidence_requires_success_result():
    vm = RuntimeViewModel()
    bus = EventBus()
    vm.attach(bus)
    # A FAILED git commit must not populate commit evidence.
    await _emit(bus, "tool.result", {"tool": "git_commit", "status": "error",
                                     "metadata": {"commit_hash": "deadbeef"},
                                     "error": "commit rejected", "task_id": "t1"})
    assert vm.git["commit"] == ""
    assert "git_commit" not in vm.git["operations"]


# ── Evidence-based finalize (Part 32) ──────────────────────────────────────


async def test_finalize_adopts_authoritative_outcome():
    from types import SimpleNamespace

    vm = RuntimeViewModel()
    state = SimpleNamespace(
        status=SimpleNamespace(value="success"),
        stage=SimpleNamespace(value="succeeded"),
        completed_at=1.0,
        original_request="build a todo app",
        workspace="/tmp/w",
        verification_status=SimpleNamespace(value="passed"),
        verification_summary="all checks passed",
        recovery_attempts=0,
        recovery_exhausted=False,
    )
    task = SimpleNamespace(files_changed=["src/app.py"])
    graph = SimpleNamespace(tasks={"t1": task})
    outcome = SimpleNamespace(state=state, graph=graph)
    vm.finalize(outcome)
    assert vm.status == "success"
    assert vm.request == "build a todo app"
    assert "src/app.py" in vm.files
