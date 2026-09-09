"""Phase 9 — per-role ProjectContext assembler tests.

The assembler must never dump the whole project into a prompt: each role
receives only requirements/artifacts/handoffs relevant to it.
"""

from __future__ import annotations

from harness_core.agents.domain import AgentMessage, AgentRole, MessageType
from harness_core.agents.message_bus import AgentMessageBus
from harness_core.observability.events import EventBus
from harness_core.runtime.context import assemble_role_context
from harness_core.runtime.requirements import Requirements, TraceabilityIndex
from harness_core.runtime.state import ArtifactRef, ProjectState


def _fixture():
    reqs = Requirements.from_entries(
        objective="task app",
        functional=["Users can register", "Users can create tasks"],
        acceptance_criteria=["Registration works"],
        constraints=["Python only"],
        exclusions=["No mobile"],
    )
    index = TraceabilityIndex.build(reqs.to_requirements())
    index.requirements_for_task  # noqa: B018 (guard import)
    return reqs, index


def test_verifier_sees_acceptance_criteria():
    reqs, index = _fixture()
    ctx = assemble_role_context(role=AgentRole.VERIFIER, objective="verify", requirements=reqs)
    statements = " ".join(r["statement"] for r in ctx.relevant_requirements)
    assert "Registration works" in statements  # acceptance criterion visible
    assert "Users can register" in statements
    # constraints (not requirements) stay out of the acceptance list
    assert not any(r["category"] == "constraint" for r in ctx.relevant_requirements)


def test_backend_does_not_get_everything():
    reqs, index = _fixture()
    ctx = assemble_role_context(
        role=AgentRole.BACKEND, objective="api", requirements=reqs, traceability=index, task_id="unmapped-task"
    )
    # No requirements map to this task id -> no per-task mapping
    assert ctx.mapped_requirement_ids == []
    # Context does include exclusion + authority constraint though
    assert any("No mobile" in c for c in ctx.constraints)


def test_integrator_sees_contract_artifacts_only():
    reqs, index = _fixture()
    state = ProjectState()
    state.register_artifact(ArtifactRef(kind="api_contract", path="api.md", produced_by_task="backend_1"))
    state.register_artifact(ArtifactRef(kind="source", path="main.py", produced_by_task="backend_1"))

    ctx = assemble_role_context(
        role=AgentRole.FRONTEND, objective="ui", requirements=reqs, project_state=state, task_id="fe"
    )
    kinds = [a["kind"] for a in ctx.artifacts]
    assert "api_contract" in kinds
    assert "source" not in kinds  # integrators do not get implementation noise


def test_verifier_gets_review_reports():
    reqs, _ = _fixture()
    state = ProjectState()
    state.register_artifact(ArtifactRef(kind="review_report", path="review.md", produced_by_task="reviewer_1"))

    verifier_ctx = assemble_role_context(
        role=AgentRole.VERIFIER, objective="verify", requirements=reqs, project_state=state
    )
    assert any(a["kind"] == "review_report" for a in verifier_ctx.artifacts)


def test_context_includes_messages_from_bus_and_memory():
    reqs, _ = _fixture()
    from harness_core.agents.domain import TaskGraph

    graph = TaskGraph()
    task = type("T", (), {"task_id": "frontend_1"})()
    graph.tasks = {"backend_1": type("B", (), {"task_id": "backend_1"})(), "frontend_1": task}
    bus = AgentMessageBus(EventBus(), graph, None)  # type: ignore[arg-type]

    # Build a valid message referencing real tasks in the graph.
    msg = AgentMessage(
        sender_task_id="backend_1",
        recipient_task_id="frontend_1",
        message_type=MessageType.HANDOFF,
        payload={"contract": {"path": "api.md"}, "summary": "api ready"},
    )
    bus._inboxes["frontend_1"] = [msg]  # seed directly (no registry needed)

    ctx = assemble_role_context(
        role=AgentRole.FRONTEND, objective="ui", requirements=reqs, task_id="frontend_1", message_bus=bus
    )
    assert len(ctx.handoffs) == 1
    block = ctx.to_prompt_block()
    assert "api.md" in block  # structured payload is carried
    assert "Handoffs addressed to this task" in block


def test_memory_context_injected_verbatim():
    reqs, _ = _fixture()
    ctx = assemble_role_context(
        role=AgentRole.BACKEND,
        objective="api",
        requirements=reqs,
        memory_context="Prior JWT work failed because refresh tokens were never rotated.",
    )
    assert "refresh tokens were never rotated" in ctx.to_prompt_block()
