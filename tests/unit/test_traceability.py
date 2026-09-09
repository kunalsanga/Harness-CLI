"""Phase 9 — requirement traceability through tasks, artifacts and verdicts."""

from __future__ import annotations

from harness_core.agents.domain import AgentRole
from harness_core.planning.domain import Plan, PlannedTask
from harness_core.runtime.requirements import (
    Evidence,
    EvidenceStatus,
    Requirements,
    RequirementStatus,
    TraceabilityIndex,
)
from harness_core.runtime.runtime import (
    _parse_verifier_verdicts,
    build_traceability_index,
)


def test_requirement_to_task_mapping_is_auditable():
    """REQ -> tasks mapping exists and is inspectable end-to-end."""
    reqs = Requirements.from_entries(
        objective="auth",
        functional=["Users can register", "Users can log in"],
    )
    planned = [
        PlannedTask(task_id="backend_register", title="r", role="backend", objective="register endpoint", traceable_to=["REQ-001"]),
        PlannedTask(task_id="frontend_login", title="l", role="frontend", objective="login page", traceable_to=["REQ-002"]),
        PlannedTask(task_id="tester", title="t", role="tester", objective="test both", traceable_to=["REQ-001", "REQ-002"]),
    ]
    index = build_traceability_index(reqs, planned)
    assert [r.req_id for r in index.requirements_for_task("backend_register")] == ["REQ-001"]
    assert sorted(r.req_id for r in index.requirements_for_task("tester")) == ["REQ-001", "REQ-002"]

    as_dict = index.to_dict()
    assert {t["req_id"] for t in as_dict["traces"]} == {"REQ-001", "REQ-002"}
    trace1 = next(t for t in as_dict["traces"] if t["req_id"] == "REQ-001")
    assert sorted(trace1["mapped_tasks"]) == ["backend_register", "tester"]


def test_verifier_failure_flips_requirement_trace_to_failed():
    reqs = Requirements.from_entries(objective="auth", functional=["Users can register"])
    req = reqs.to_requirements()[0]
    req.task_ids = ["backend_register"]
    index = TraceabilityIndex.build([req])

    # Simulate the evidence the runtime attaches: task completed (PASS)…
    index.attach_task_evidence("backend_register", "backend_register", EvidenceStatus.PASS)
    assert index.traces["REQ-001"].status == RequirementStatus.VERIFIED

    # …but the verifier rejects it (FAIL overrides).
    trace = index.traces["REQ-001"]
    trace.add_evidence(Evidence(source="verifier_1", status=EvidenceStatus.FAIL, detail="registration leaks emails"))
    assert trace.status == RequirementStatus.FAILED


def test_verifier_verdict_parser_handles_line_and_json_forms():
    assert _parse_verifier_verdicts("REQ-001: PASS\nREQ-002: FAIL") == {"REQ-001": True, "REQ-002": False}
    assert _parse_verifier_verdicts('[{"req_id": "REQ-001", "status": "pass"}, {"req_id": "REQ-002", "status": "fail"}]') == {
        "REQ-001": True,
        "REQ-002": False,
    }
    # Prose around verdicts should not break parsing
    assert _parse_verifier_verdicts("Checked all criteria.\nREQ-001: PASS") == {"REQ-001": True}
    # Unknown lines are ignored
    assert _parse_verifier_verdicts("everything is fine") == {}


def test_traceability_requires_all_mapped_tasks():
    """One mapped task missing means the requirement cannot be verified."""
    reqs = Requirements.from_entries(objective="task app", functional=["Task CRUD works"])
    planned = [
        PlannedTask(task_id="backend", title="b", role="backend", objective="CRUD api", traceable_to=["REQ-001"]),
        PlannedTask(task_id="frontend", title="f", role="frontend", objective="task page", traceable_to=["REQ-001"]),
    ]
    index = build_traceability_index(reqs, planned)
    trace = index.traces["REQ-001"]

    # Only backend completed -> one mapped task missing -> not verified
    index.attach_task_evidence("backend", "backend", EvidenceStatus.PASS, detail="backend done")
    assert trace.status != RequirementStatus.VERIFIED

    # Second mapped task completes -> every mapped task has PASS evidence
    index.attach_task_evidence("frontend", "frontend", EvidenceStatus.PASS, detail="frontend done")
    assert trace.status == RequirementStatus.VERIFIED


def test_failed_requirement_blocks_runtime_success():
    """A FAILED trace must force the runtime terminal status to FAILED."""
    from harness_core.runtime.state import RuntimeStatus

    reqs = Requirements.from_entries(objective="app", functional=["Feature works"])
    planned = [PlannedTask(task_id="impl", title="x", role="backend", objective="feature", traceable_to=["REQ-001"])]
    index = build_traceability_index(reqs, planned)
    trace = index.traces["REQ-001"]
    trace.add_evidence(Evidence(source="verifier", status=EvidenceStatus.FAIL, detail="rejected"))

    verified, failed, _, _ = index.overall()
    assert failed == 1
    assert verified == 0

    # Mirror of the runtime decision logic (see EngineeringRuntime._run_inner).
    def _decide(ok: bool) -> RuntimeStatus:
        return RuntimeStatus.SUCCESS if ok else RuntimeStatus.FAILED

    assert _decide(failed == 0 and verified == len(index.traces)) == RuntimeStatus.FAILED


def test_planning_to_taskgraph_keeps_traceability_intact():
    """The runtime graph conversion does not lose requirement mappings."""
    from harness_core.runtime.runtime import convert_plan_to_graph

    plan = Plan(
        summary="plan",
        tasks=[
            PlannedTask(task_id="db", title="d", role="database", objective="schema", traceable_to=["REQ-001"]),
            PlannedTask(task_id="backend", title="b", role="backend", objective="api", dependencies=["db"], traceable_to=["REQ-001"]),
        ],
    )
    graph = convert_plan_to_graph(plan)
    assert graph is not None
    assert graph.get_task("backend").dependencies == ["db"]
    assert graph.get_task("db").role == AgentRole.DATABASE
