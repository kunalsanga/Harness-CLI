"""Phase 9 — ProjectState: lifecycle, snapshots, derived TaskGraph views.

ProjectState must NOT be a second task database: every task number it reports
is derived from the authoritative TaskGraph it references.
"""

from __future__ import annotations

from harness_core.agents.domain import AgentRole, SubTask, TaskGraph, TaskStatus
from harness_core.runtime.state import (
    ArtifactRef,
    ProjectState,
    RuntimeStage,
    RuntimeStatus,
    VerificationStatus,
)


def test_stage_transitions_are_recorded():
    state = ProjectState(project_id="p1", workspace="/w")
    assert state.stage == RuntimeStage.UNKNOWN

    state.set_stage(RuntimeStage.PLAN)
    state.set_stage(RuntimeStage.EXECUTE, detail="go")
    state.set_stage(RuntimeStage.EXECUTE)  # duplicate ignored

    assert state.stage == RuntimeStage.EXECUTE
    assert [r.stage for r in state.stage_history] == [
        RuntimeStage.PLAN,
        RuntimeStage.EXECUTE,
    ]


def test_progress_is_derived_from_task_graph_not_duplicated():
    state = ProjectState()
    graph = TaskGraph()
    a = SubTask(task_id="a", role=AgentRole.CODER)
    b = SubTask(task_id="b", role=AgentRole.TESTER, dependencies=["a"])
    graph.add_task(a)
    graph.add_task(b)
    state.task_graph = graph

    assert state.total_tasks == 2
    assert state.completed_tasks == 0
    assert state.task_progress == (0, 2)

    # The *graph* is authoritative: mark b completed via the graph.
    graph.update_task_status("a", TaskStatus.COMPLETED)
    graph.update_task_status("b", TaskStatus.COMPLETED)

    assert state.completed_tasks == 2
    assert state.task_progress == (2, 2)
    # Active task ids come from graph status, not from any runtime mirror.
    assert state.active_task_ids == []
    graph.update_task_status("b", TaskStatus.RUNNING)
    assert state.active_task_ids == ["b"]


def test_terminal_status_fields():
    state = ProjectState(project_id="p", workspace="/w", original_request="build x")
    state.status = RuntimeStatus.SUCCESS
    state.set_stage(RuntimeStage.SUCCEEDED)
    state.completed_at = 1.0
    state.verification_status = VerificationStatus.PASSED
    state.register_artifact(ArtifactRef(kind="api_contract", path="/w/api.md", produced_by_task="t1"))
    state.register_warning("watch out")
    state.register_blocker("nope")

    snapshot = state.to_dict()
    assert snapshot["status"] == "success"
    assert snapshot["terminal"] is True
    assert snapshot["verification"]["status"] == "passed"
    assert snapshot["artifacts"][0]["kind"] == "api_contract"
    assert "watch out" in snapshot["warnings"]
    assert "nope" in snapshot["blockers"]
    assert snapshot["progress"] == {"completed": 0, "total": 0}


def test_non_terminal_stage_is_not_terminal():
    state = ProjectState()
    state.set_stage(RuntimeStage.EXECUTE)
    assert state.to_dict()["terminal"] is False


def test_failed_task_ids_tracked_separately():
    state = ProjectState()
    state.failed_task_ids.append("t1")
    assert state.failed_task_ids == ["t1"]
    snapshot = state.to_dict()
    assert snapshot["failed_tasks"] == ["t1"]
