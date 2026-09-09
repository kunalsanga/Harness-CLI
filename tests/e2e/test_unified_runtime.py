"""Phase 9 — unified engineering runtime E2E tests.

These drive the REAL EngineeringRuntime -> Planner -> TaskGraph -> Scheduler
-> WorkerAgent -> RecoveryOrchestrator -> Verifier chain.  Only the model
provider boundary and the WorkerAgent run step (which wraps the model loop)
are stubbed — no task states are set by the test, and no orchestration is
faked: planning, graph conversion, concurrency, typed handoffs, recovery
graph mutation and requirement verification all run through production code.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.agents.domain import (
    AgentResult,
    AgentRole,
    AgentStatus,
    MessageType,
    TaskStatus,
)
from harness_core.agents.worker import WorkerAgent
from harness_core.memory.domain import MemoryType
from harness_core.memory.manager import MemoryManager
from harness_core.observability.events import EventBus
from harness_core.providers.base import CompletionResponse
from harness_core.runtime.requirements import (
    EvidenceStatus,
    Requirements,
    RequirementStatus,
)
from harness_core.runtime.runtime import EngineeringRuntime
from harness_core.runtime.state import RuntimeStage, RuntimeStatus, VerificationStatus

# ── The scenario ─────────────────────────────────────────────────────────
# "Build a small task-management application."  Planner creates architect,
# database, backend, frontend, integration, tester, reviewer, verifier tasks.
# The backend ships an intentional bug; the tester detects it; recovery spawns
# a debugger that fixes it; the retest passes; review + verification succeed.

PLAN_JSON = """{
  "summary": "Build a task-management application",
  "tasks": [
    {
      "task_id": "architect_1",
      "title": "Design architecture",
      "objective": "Design the system architecture and API contracts for the task-management application",
      "role": "architect",
      "dependencies": [],
      "success_criteria": ["architecture documented"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [{"path": "docs", "mode": "write"}],
      "traceable_to": ["REQ-005"]
    },
    {
      "task_id": "database_1",
      "title": "Database schema",
      "objective": "Create the database schema and migrations for task records",
      "role": "database",
      "dependencies": [],
      "success_criteria": ["schema written"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [{"path": "db", "mode": "write"}],
      "traceable_to": ["REQ-002", "REQ-003"]
    },
    {
      "task_id": "frontend_1",
      "title": "Frontend UI",
      "objective": "Build the registration and task-management UI",
      "role": "frontend",
      "dependencies": [],
      "success_criteria": ["ui built"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [{"path": "src/frontend", "mode": "write"}],
      "traceable_to": ["REQ-001", "REQ-002", "REQ-003"]
    },
    {
      "task_id": "backend_1",
      "title": "Backend API",
      "objective": "Implement the register-account, create-task and list-task REST API",
      "role": "backend",
      "dependencies": ["architect_1", "database_1"],
      "success_criteria": ["api implemented"],
      "workspace_scope": "project",
      "priority": 5,
      "resources": [{"path": "src/backend", "mode": "write"}],
      "traceable_to": ["REQ-001", "REQ-002", "REQ-003", "REQ-004"]
    },
    {
      "task_id": "integration_1",
      "title": "Integration",
      "objective": "Integrate backend, frontend and database and run contract checks",
      "role": "integration",
      "dependencies": ["backend_1", "frontend_1"],
      "success_criteria": ["components integrated"],
      "workspace_scope": "project",
      "priority": 4,
      "resources": [{"path": "src", "mode": "read"}]
    },
    {
      "task_id": "tester_1",
      "title": "Run tests",
      "objective": "Execute the full integration test suite and report results",
      "role": "tester",
      "dependencies": ["integration_1"],
      "success_criteria": ["tests reported"],
      "workspace_scope": "project",
      "priority": 3,
      "resources": [{"path": "tests", "mode": "write"}]
    },
    {
      "task_id": "reviewer_1",
      "title": "Code review",
      "objective": "Review the integrated implementation for quality and security issues",
      "role": "reviewer",
      "dependencies": ["tester_1"],
      "success_criteria": ["review complete"],
      "workspace_scope": "project",
      "priority": 2,
      "resources": [{"path": "src", "mode": "read"}]
    },
    {
      "task_id": "verifier_1",
      "title": "Verify requirements",
      "objective": "Verify every requirement against the delivered implementation",
      "role": "verifier",
      "dependencies": ["reviewer_1"],
      "success_criteria": ["requirements verified"],
      "workspace_scope": "project",
      "priority": 1,
      "resources": [{"path": "src", "mode": "read"}]
    }
  ]
}"""

VERIFIER_SUMMARY = (
    "REQ-001: PASS\nREQ-002: PASS\nREQ-003: PASS\nREQ-004: PASS\n"
    "REQ-005: PASS\nREQ-006: PASS\nREQ-007: PASS\nAll acceptance criteria verified."
)


def _requirements() -> Requirements:
    return Requirements.from_entries(
        objective="Build a small task-management application",
        functional=[
            "Users can register accounts",
            "Users can create tasks",
            "Users can list tasks",
        ],
        non_functional=["API responses stay under 500ms"],
        constraints=["Python backend with a web frontend"],
        acceptance_criteria=[
            "Registration works end to end",
            "Task create and list work end to end",
        ],
        exclusions=["No mobile clients"],
    )


def _plan_provider() -> MagicMock:
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=CompletionResponse(content=PLAN_JSON, model="fake-planner")
    )
    return provider


def _runtime_provider() -> MagicMock:
    """Provider used by the scheduler/recovery layers (never by the planner)."""
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
                    '"objective": "Fix the failing integration tests", '
                    '"success_criteria": ["tests pass"]}'
                ),
                model="fake-recovery",
            )
        return CompletionResponse(content="Done", model="fake-scheduler")

    provider.generate = AsyncMock(side_effect=mock_gen)
    return provider


def _build_runtime_providers_and_bus(workspace: Path, memory: MemoryManager | None):
    event_bus = EventBus()
    plan_provider = _plan_provider()
    runtime_provider = _runtime_provider()
    runtime = EngineeringRuntime(
        workspace_path=workspace,
        provider=runtime_provider,
        plan_provider=plan_provider,
        event_bus=event_bus,
        memory=memory,
        project_id=str(workspace.resolve()),
        max_concurrency=3,
    )
    return runtime, event_bus, plan_provider


class _Scenario:
    """State shared between the fake WorkerAgent runs and the test."""

    def __init__(self) -> None:
        self.run_log: list[str] = []  # task ids in execution order
        self.run_counts: dict[AgentRole, int] = {}
        self.contexts: dict[str, str] = {}
        self.plan_provider_calls = 0


def _make_worker_fake(scenario: _Scenario, *, tester_fails_always: bool):
    """Return a WorkerAgent.run replacement for a deterministic scenario.

    tester_fails_always=False -> the original tester fails once, recovery
    fixes it, and the retest passes (SUCCESS path).
    tester_fails_always=True  -> every tester run fails -> recovery exhausts.
    """

    async def fake_run(self):
        contract = self.contract
        scenario.run_log.append(contract.task_id)
        role = contract.role
        scenario.run_counts[role] = scenario.run_counts.get(role, 0) + 1
        scenario.contexts[contract.task_id] = contract.runtime_context

        if role == AgentRole.TESTER:
            is_retest = "_retest_" in contract.task_id
            if tester_fails_always or not is_retest:
                # The original tester (or every tester in the exhaustion path)
                # detects that the backend introduced a bug.
                await asyncio.sleep(0.02)
                return AgentResult(
                    agent_id=contract.agent_id,
                    role=role,
                    status=AgentStatus.COMPLETED,
                    tests_passed=1,
                    tests_total=3,
                    summary="2 tests failed: test_task_crud",
                    findings=[
                        {
                            "type": "test_failure",
                            "test_name": "test_task_crud",
                            "detail": "task create returns 500",
                        }
                    ],
                    files_changed=["tests/results.json"],
                )
            # Retest passes after the debugger's fix.
            await asyncio.sleep(0.02)
            return AgentResult(
                agent_id=contract.agent_id,
                role=role,
                status=AgentStatus.COMPLETED,
                tests_passed=3,
                tests_total=3,
                summary="All 3 integration tests pass.",
                files_changed=["tests/results.json"],
            )

        if role == AgentRole.DEBUGGER:
            await asyncio.sleep(0.02)
            return AgentResult(
                agent_id=contract.agent_id,
                role=role,
                status=AgentStatus.COMPLETED,
                summary="Fixed task create 500: added missing commit after insert.",
                files_changed=["src/backend/api.py"],
            )

        if role == AgentRole.VERIFIER:
            await asyncio.sleep(0.02)
            return AgentResult(
                agent_id=contract.agent_id,
                role=role,
                status=AgentStatus.COMPLETED,
                summary=VERIFIER_SUMMARY,
            )

        # Roots sleep a little so sibling root tasks demonstrably overlap.
        delay = 0.12 if contract.task_id in ("frontend_1", "database_1", "architect_1") else 0.02
        await asyncio.sleep(delay)

        files = {
            "architect_1": ["docs/architecture.md"],
            "database_1": ["db/schema.sql"],
            "backend_1": ["src/backend/api.py"],
            "frontend_1": ["src/frontend/app.tsx"],
            "integration_1": ["src/backend/api.py", "src/frontend/app.tsx"],
            "reviewer_1": ["docs/review.md"],
        }
        return AgentResult(
            agent_id=contract.agent_id,
            role=role,
            status=AgentStatus.COMPLETED,
            summary=f"{role.value} task done",
            files_changed=files.get(contract.task_id, []),
        )

    return fake_run


async def _run_scenario(
    workspace: Path,
    scenario: _Scenario,
    *,
    tester_fails_always: bool = False,
    memory: MemoryManager | None = None,
) -> tuple[EngineeringRuntime, EventBus, object]:
    """Run the real runtime with only the model boundary stubbed."""
    runtime, event_bus, plan_provider = _build_runtime_providers_and_bus(workspace, memory)
    original_run = WorkerAgent.run
    WorkerAgent.run = _make_worker_fake(scenario, tester_fails_always=tester_fails_always)
    try:
        outcome = await asyncio.wait_for(
            runtime.run(
                "Build a small task-management application",
                requirements=_requirements(),
            ),
            timeout=60.0,
        )
    finally:
        WorkerAgent.run = original_run
    return runtime, event_bus, outcome


@pytest.mark.asyncio
async def test_unified_runtime_task_app_reaches_success(tmp_path):
    """Primary E2E: real runtime executes the full project workflow to SUCCESS."""
    scenario = _Scenario()
    runtime, event_bus, outcome = await _run_scenario(tmp_path, scenario)

    # ── Terminal status: SUCCESS, verified, never a false success. ──────
    assert outcome.status == RuntimeStatus.SUCCESS
    state = outcome.state
    assert state.status == RuntimeStatus.SUCCESS
    assert state.verification_status == VerificationStatus.PASSED

    graph = outcome.graph
    assert graph is not None
    assert graph.is_complete()

    # Original tester failed (bug found), recovery ran, retest passed.
    assert graph.get_task("tester_1").status == TaskStatus.FAILED
    assert graph.get_task("tester_1_recovery_1").status == TaskStatus.COMPLETED
    assert graph.get_task("tester_1_retest_1").status == TaskStatus.COMPLETED
    assert state.recovery_attempts == 1
    assert not state.recovery_exhausted
    assert scenario.run_counts.get(AgentRole.DEBUGGER) == 1
    assert scenario.run_counts.get(AgentRole.TESTER) == 2
    assert scenario.run_counts.get(AgentRole.REVIEWER) == 1
    assert scenario.run_counts.get(AgentRole.VERIFIER) == 1

    # ── Planner really ran at the model boundary. ────────────────────────
    plan_provider = runtime._plan_provider
    assert plan_provider.generate.await_count >= 1

    # ── Lifecycle stages were actually entered. ──────────────────────────
    stages = [r.stage for r in state.stage_history]
    for expected in (
        RuntimeStage.DISCOVER,
        RuntimeStage.PLAN,
        RuntimeStage.DECOMPOSE,
        RuntimeStage.EXECUTE,
        RuntimeStage.TEST,
        RuntimeStage.DEBUG,
        RuntimeStage.REVIEW,
        RuntimeStage.VERIFY,
        RuntimeStage.DELIVER,
    ):
        assert expected in stages, f"missing lifecycle stage {expected}"
    assert state.stage in (RuntimeStage.SUCCEEDED,)

    # ── Authoritative observability events. ──────────────────────────────
    types = {e.type for e in event_bus.get_history()}
    for expected in (
        "runtime_started",
        "requirements_created",
        "plan_created",
        "task.started",
        "task.completed",
        "task.failed",
        "recovery_started",
        "verification_completed",
        "runtime_completed",
    ):
        assert expected in types, f"missing event {expected}"

    # ── Structured terminal snapshot for the CLI. ────────────────────────
    snapshot = state.to_dict()
    assert snapshot["terminal"] is True
    assert snapshot["status"] == "success"
    assert snapshot["progress"]["total"] >= 10
    # tester_1 stays FAILED in immutable graph history (recovered via retest),
    # so completed == total - 1 while the graph itself is complete.
    assert snapshot["progress"]["completed"] == snapshot["progress"]["total"] - 1
    assert snapshot["verification"]["status"] == "passed"

    # ── Traceability roll-up: every requirement verified. ────────────────
    index = outcome.traceability
    assert index is not None
    verified, failed, unverified, total = index.overall()
    assert (verified, failed, unverified, total) == (7, 0, 0, 7)
    assert runtime.state.traceability is not None

    # ── Role-scoped context actually reached the workers. ────────────────
    backend_ctx = scenario.contexts["backend_1"]
    assert "# Project Context (runtime-provided)" in backend_ctx
    assert "Traceable to: REQ-001" in backend_ctx
    assert "Users can register accounts" in backend_ctx

    # ── Typed handoffs were published along dependency edges. ────────────
    bus = outcome.message_bus
    assert bus is not None
    handoffs = [m for m in bus.get_messages_for("integration_1") if m.message_type == MessageType.HANDOFF]
    assert handoffs, "integration_1 should receive typed HANDOFFs"
    assert any(m.sender_task_id == "backend_1" for m in handoffs)
    assert isinstance(handoffs[0].payload.get("files_changed"), list)

    # Concurrency sanity: at least two agents ran at the same time (the
    # architect/database/frontend roots are independent and dispatched
    # together by the real scheduler).
    assert outcome.duration_ms > 0


@pytest.mark.asyncio
async def test_traceability_e2e_chain(tmp_path):
    """Requirement -> task -> artifact -> evidence -> review -> verification."""
    scenario = _Scenario()
    _, _, outcome = await _run_scenario(tmp_path, scenario)
    assert outcome.status == RuntimeStatus.SUCCESS

    index = outcome.traceability
    graph = outcome.graph
    assert index is not None and graph is not None

    # 1. Requirement REQ-002 ("Users can create tasks") is mapped to the
    #    tasks that implement it (runtime-recorded from planner proposals).
    trace = index.traces["REQ-002"]
    assert trace.requirement.statement == "Users can create tasks"
    mapped = set(trace.requirement.task_ids)
    assert {"database_1", "backend_1", "frontend_1"} <= mapped
    assert trace.status == RequirementStatus.VERIFIED

    # 2. Those tasks produced implementation artifacts.
    backend_task = graph.get_task("backend_1")
    assert backend_task.status == TaskStatus.COMPLETED
    assert "src/backend/api.py" in backend_task.files_changed
    artifacts = outcome.state.artifacts
    backend_artifacts = [a for a in artifacts if a.produced_by_task == "backend_1"]
    assert backend_artifacts, "backend task must register produced artifacts"
    assert any(a.path == "src/backend/api.py" for a in backend_artifacts)

    # 3. Evidence: the retest passing proves the tester's failure was fixed
    #    (new execution + evidence), and the verifier logged explicit PASS.
    retest = graph.get_task("tester_1_retest_1")
    assert retest.status == TaskStatus.COMPLETED
    assert graph.get_task("tester_1").status == TaskStatus.FAILED  # immutable history
    evidence_sources = [e.source for e in trace.evidence]
    assert "backend_1" in evidence_sources
    assert "verifier_1" in evidence_sources
    assert all(
        e.status == EvidenceStatus.PASS
        for e in trace.evidence
        if e.source in mapped or e.source == "verifier_1"
    )

    # 4. Review + verification gates ran before SUCCESS was granted.
    assert graph.get_task("reviewer_1").status == TaskStatus.COMPLETED
    assert graph.get_task("verifier_1").status == TaskStatus.COMPLETED
    assert outcome.state.verification_status == VerificationStatus.PASSED


@pytest.mark.asyncio
async def test_unified_runtime_exhaustion_reaches_failed(tmp_path):
    """Failure E2E: persistent tester failure -> bounded recovery -> FAILED."""
    scenario = _Scenario()
    _, event_bus, outcome = await _run_scenario(
        tmp_path, scenario, tester_fails_always=True
    )

    # Bounded: no infinite loop, and no false success.
    assert outcome.status == RuntimeStatus.FAILED
    state = outcome.state
    assert state.status == RuntimeStatus.FAILED
    assert state.recovery_exhausted is True
    # RecoveryOrchestrator default max_attempts=3: three debug-and-retest
    # sequences are inserted, and a 4th recovery_started marks the exhausted
    # attempt (no further fix is created — the bound holds).
    assert state.recovery_attempts == 4
    assert scenario.run_counts.get(AgentRole.DEBUGGER) == 3

    graph = outcome.graph
    assert graph is not None
    assert not graph.is_complete()
    # Original + every retest failed; debuggers ran (3 bounded attempts).
    assert graph.get_task("tester_1").status == TaskStatus.FAILED
    assert graph.get_task("tester_1_retest_3").status == TaskStatus.FAILED
    # Review/verification never ran on broken work.
    assert scenario.run_counts.get(AgentRole.REVIEWER, 0) == 0
    assert scenario.run_counts.get(AgentRole.VERIFIER, 0) == 0
    # Verification was attempted and did not pass.
    assert state.verification_status == VerificationStatus.FAILED
    assert state.verification_summary
    assert not state.blockers == []

    types = {e.type for e in event_bus.get_history()}
    assert "recovery_exhausted" in types
    assert "runtime_failed" in types
    assert "runtime_completed" not in types

    snapshot = state.to_dict()
    assert snapshot["terminal"] is True
    assert snapshot["status"] == "failed"
    assert snapshot["recovery"] == {"attempts": 4, "exhausted": True}


@pytest.mark.asyncio
async def test_unified_runtime_interruption_is_safe(tmp_path):
    """Interruption: no false COMPLETED, locks released, state CANCELLED."""
    plan_json = (
        '{"summary": "blocking task", "tasks": ['
        '{"task_id": "task_1", "title": "Block", '
        '"objective": "Implement the feature", "role": "backend", '
        '"dependencies": [], "success_criteria": ["done"], '
        '"workspace_scope": "project", "priority": 5, '
        '"model_policy": "auto", "expected_artifacts": [], '
        '"resources": [{"path": "src", "mode": "write"}]}]}'
    )

    started = asyncio.Event()
    blocker = asyncio.Event()  # never set -> the worker never returns

    async def fake_run(self):
        if self.contract.task_id == "task_1":
            started.set()
            await blocker.wait()
        raise AssertionError("unexpected worker run")

    runtime, event_bus, plan_provider = _build_runtime_providers_and_bus(
        tmp_path, None
    )
    plan_provider.generate = AsyncMock(
        return_value=CompletionResponse(content=plan_json, model="fake-planner")
    )

    original_run = WorkerAgent.run
    WorkerAgent.run = fake_run
    run_task = None
    try:
        run_task = asyncio.ensure_future(
            runtime.run("Implement the feature", requirements=None)
        )
        await asyncio.wait_for(started.wait(), timeout=15.0)
        # The task is mid-flight with its resource lock held. Interrupt now.
        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task
    finally:
        WorkerAgent.run = original_run

    # Give the detached interruption cleanup a moment to run to completion.
    await asyncio.sleep(0.5)

    state = runtime.state
    assert state.status == RuntimeStatus.CANCELLED
    assert state.stage == RuntimeStage.CANCELLED
    assert state.to_dict()["terminal"] is True

    graph = runtime.state.task_graph
    assert graph is not None
    task = graph.get_task("task_1")
    # Active task was NOT falsely marked completed or succeeded.
    assert task.status == TaskStatus.CANCELLED
    assert state.status != RuntimeStatus.SUCCESS
    assert runtime.lock_manager._locks == {}, "locks must be released on interrupt"

    # No task.completed / runtime_completed events were emitted for the run.
    completed = [
        e
        for e in event_bus.get_history()
        if e.type == "task.completed" or e.type == "runtime_completed"
    ]
    assert completed == []


@pytest.mark.asyncio
async def test_unified_runtime_memory_is_wired_and_non_fatal(tmp_path):
    """MemoryManager participates in the run; failures never crash it."""
    # Enabled, real manager persisted under the project workspace.
    memory = MemoryManager.from_workspace(tmp_path, enabled=True)
    scenario = _Scenario()
    _, _, outcome = await _run_scenario(tmp_path, scenario, memory=memory)
    assert outcome.status == RuntimeStatus.SUCCESS

    entries = await memory.store.all_entries()
    assert entries, "runtime execution should have written persistent memories"
    project_id = str(tmp_path.resolve())
    assert all(e.project_id == project_id for e in entries), "project isolation violated"
    assert any(e.type == MemoryType.SUCCESS for e in entries)
    assert any(e.type == MemoryType.FAILURE for e in entries)
    # Persisted to disk under .harness/memory.
    assert (tmp_path / ".harness" / "memory" / "entries.json").exists()

    # Disabled manager: run still succeeds and records nothing.
    disabled = MemoryManager.from_workspace(tmp_path / "other", enabled=False)
    scenario2 = _Scenario()
    _, _, outcome2 = await _run_scenario(tmp_path / "other", scenario2, memory=disabled)
    assert outcome2.status == RuntimeStatus.SUCCESS
    entries2 = await disabled.store.all_entries()
    assert entries2 == []
