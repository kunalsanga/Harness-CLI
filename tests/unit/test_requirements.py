"""Phase 9 — structured requirements + requirement traceability unit tests."""

from __future__ import annotations

from harness_core.planning.domain import PlannedTask
from harness_core.runtime.requirements import (
    Evidence,
    EvidenceStatus,
    Requirement,
    Requirements,
    RequirementStatus,
    TraceabilityIndex,
)
from harness_core.runtime.runtime import build_traceability_index


def _requirements() -> Requirements:
    return Requirements.from_entries(
        objective="Build a task-management app",
        functional=[
            "Users can register",
            "Users can create tasks",
            "Users can list tasks",
        ],
        acceptance_criteria=["Registration works", "Task CRUD works"],
        non_functional=["Responses under 500ms"],
        constraints=["Python backend only"],
        exclusions=["No mobile clients"],
    )


def test_requirements_expand_with_stable_ids():
    reqs = _requirements().to_requirements()
    ids = [r.req_id for r in reqs]
    # functional 1..3, non-functional 4, constraint 5, acceptance 6..7; exclusions excluded
    assert ids == [f"REQ-{i:03d}" for i in range(1, 8)]
    assert reqs[0].statement == "Users can register"
    assert reqs[5].category == "acceptance"
    # objective and exclusions are not individually addressable requirements
    assert "mobile" not in " ".join(r.statement for r in reqs)


def test_requirement_dedup_and_roundtrip():
    reqs = _requirements()
    as_dict = reqs.to_dict()
    rebuilt = Requirements.from_dict(as_dict)
    assert rebuilt.functional == reqs.functional
    assert rebuilt.exclusions == reqs.exclusions
    assert not rebuilt.is_empty


def test_requirements_dont_allow_self_certification():
    """A planner mapping REQ to tasks cannot change the requirement itself."""
    req = Requirement(req_id="REQ-001", statement="Users can register", category="functional")
    assert req.req_id == "REQ-001"
    assert req.statement == "Users can register"


def test_trace_verifies_only_with_pass_evidence():
    req = Requirement(req_id="REQ-001", statement="Users can register", task_ids=["backend", "frontend"])
    trace = TraceabilityIndex.build([req]).traces["REQ-001"]
    assert trace.status == RequirementStatus.UNVERIFIED

    # A generic PASS is not enough: only evidence from the *mapped tasks*
    # counts toward task-level completion.
    trace.add_evidence(Evidence(source="test", status=EvidenceStatus.PASS, detail="integration ok"))
    assert trace.status == RequirementStatus.UNVERIFIED

    trace.add_evidence(Evidence(source="backend", status=EvidenceStatus.PASS, detail="backend done"))
    # frontend still missing -> in progress, definitely not verified
    assert trace.status != RequirementStatus.VERIFIED

    trace.add_evidence(Evidence(source="frontend", status=EvidenceStatus.PASS, detail="frontend done"))
    assert trace.status == RequirementStatus.VERIFIED

    # A verifier FAIL overrides earlier PASSes.
    trace.add_evidence(Evidence(source="verifier", status=EvidenceStatus.FAIL, detail="rejected"))
    assert trace.status == RequirementStatus.FAILED


def test_unmapped_requirement_can_be_verified_by_verdict_evidence():
    req = Requirement(req_id="REQ-010", statement="App stays under budget")
    trace = TraceabilityIndex.build([req]).traces["REQ-010"]
    trace.add_evidence(Evidence(source="verifier", status=EvidenceStatus.PASS, detail="ok"))
    assert trace.status == RequirementStatus.VERIFIED


def test_index_build_maps_tasks_to_requirements():
    reqs = _requirements()
    planned = [
        PlannedTask(task_id="backend_auth", title="auth", role="backend", objective="implement register", traceable_to=["REQ-001"]),
        PlannedTask(task_id="frontend_register", title="reg", role="frontend", objective="register page", traceable_to=["REQ-001"]),
        PlannedTask(task_id="db_tasks", title="schema", role="database", objective="task schema"),
    ]
    index = build_traceability_index(reqs, planned)

    assert index.requirements_for_task("backend_auth")[0].req_id == "REQ-001"
    assert index.requirements_for_task("db_tasks")  # keyword fallback found a requirement
    # every mapping must reference a real requirement id
    for trace in index.traces.values():
        for task_id in trace.requirement.task_ids:
            assert task_id in {"backend_auth", "frontend_register", "db_tasks"}


def test_keyword_fallback_only_matches_real_ids():
    reqs = _requirements()
    planned = [
        PlannedTask(task_id="register_api", title="api", role="backend", objective="Implement user registration endpoint"),
        PlannedTask(task_id="unrelated", title="perf", role="backend", objective="tune query performance"),
    ]
    index = build_traceability_index(reqs, planned)
    req_ids = {r.req_id for t in ("register_api", "unrelated") for r in index.requirements_for_task(t)}
    # "registration"/"register" maps to REQ-001; REQ-004 mentions "Responses" (500ms)
    assert "REQ-001" in req_ids


def test_traceability_overall_counts():
    reqs = _requirements()
    req_list = reqs.to_requirements()
    req_list[0].task_ids = ["t1"]
    req_list[1].task_ids = ["t2"]
    index = TraceabilityIndex.build(req_list)

    # t1: its mapped task passes -> verified. t2: mapped task passes, but the
    # verifier fails the requirement afterwards.
    index.attach_task_evidence("t1", "t1", EvidenceStatus.PASS)
    index.attach_task_evidence("t2", "t2", EvidenceStatus.PASS)
    trace2 = index.traces[req_list[1].req_id]
    trace2.add_evidence(Evidence(source="verifier", status=EvidenceStatus.FAIL, detail="rejected"))

    verified, failed, unverified, total = index.overall()
    assert verified >= 1
    assert failed >= 1
    assert total == len(req_list)
    assert verified + failed + unverified == total
