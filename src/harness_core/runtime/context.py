"""Targeted per-role project context (Phase 9 §8).

Never dump the whole project into a prompt.  Each role receives only:

    Requirements relevant to it
    + its task's traceability mapping
    + artifacts it is allowed to see / may need
    + prior handoffs addressed to it (from the message bus)
    + persistent-memory context relevant to its work
    + hard constraints (exclusions, workspace scope)

The assembler is a pure function of runtime-owned state so it stays
deterministic and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from harness_core.agents.domain import AgentRole

if TYPE_CHECKING:
    from harness_core.agents.message_bus import AgentMessageBus
    from harness_core.runtime.requirements import Requirements, TraceabilityIndex
    from harness_core.runtime.state import ArtifactRef, ProjectState

# Roles whose work is almost entirely verification rather than implementation.
_VERIFIER_ROLES = {AgentRole.VERIFIER, AgentRole.REVIEWER, AgentRole.SECURITY_REVIEWER}

# Artifact kinds handed to integrators (contracts/specs, not implementation noise).
_INTEGRATOR_ARTIFACT_KINDS = (
    "api_contract",
    "database_schema",
    "architecture_decision",
    "ui_spec",
    "review_report",
)
# Roles that primarily consume implementation artifacts of others.
_INTEGRATOR_ROLES = {AgentRole.INTEGRATION, AgentRole.FRONTEND}


@dataclass
class RoleContext:
    """What one agent task is allowed to know before it starts."""

    role: str
    objective: str
    relevant_requirements: list[dict[str, Any]] = field(default_factory=list)
    mapped_requirement_ids: list[str] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    memory_context: str = ""
    constraints: list[str] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Render this context into a compact prompt section."""
        lines = ["# Project Context (runtime-provided)"]
        if self.relevant_requirements:
            lines.append("\n## Requirements for this work")
            for req in self.relevant_requirements:
                lines.append(f"- [{req['req_id']}][{req['category']}] {req['statement']}")
        if self.mapped_requirement_ids:
            lines.append(f"Traceable to: {', '.join(self.mapped_requirement_ids)}")
        if self.artifacts:
            lines.append("\n## Available artifacts (reference paths, read as needed)")
            for art in self.artifacts:
                if art.get("produced_by_task"):
                    path_note = f"{art['path']} (by {art['produced_by_task']})"
                else:
                    path_note = art["path"]
                lines.append(f"- [{art['kind']}] {path_note}")
                if art.get("summary"):
                    lines.append(f"    {art['summary']}")
        if self.handoffs:
            lines.append("\n## Handoffs addressed to this task")
            for msg in self.handoffs:
                sender = msg.get("sender_task_id", "?")
                lines.append(f"- from {sender} type={msg.get('message_type', '?')}")
                payload = msg.get("payload")
                if isinstance(payload, dict) and payload:
                    lines.append(f"    payload: {_summarize(payload)}")
        if self.memory_context:
            lines.append(f"\n## Relevant historical context\n{self.memory_context}")
        if self.constraints:
            lines.append("\n## Constraints")
            for c in self.constraints:
                lines.append(f"- {c}")
        return "\n".join(lines)


def _summarize(payload: dict[str, Any], max_len: int = 300) -> str:
    """Compact, bounded representation of a structured handoff payload."""
    import json

    text = json.dumps(payload, default=str)
    return text[:max_len] + ("…" if len(text) > max_len else "")


def _requirements_for_role(
    requirements: Requirements | None, role: AgentRole
) -> list[dict[str, Any]]:
    """Pick the requirement statements relevant to a role.

    Implementation roles see their mapped requirements; verifier-type roles
    see every acceptance criterion (they must judge ALL of them); integrator
    roles see functional + non-functional requirements.
    """
    if requirements is None:
        return []
    if role in _VERIFIER_ROLES:
        return [
            {"req_id": r.req_id, "category": r.category, "statement": r.statement}
            for r in requirements.to_requirements()
            if r.category in ("acceptance", "functional", "non_functional")
        ]
    if role in _INTEGRATOR_ROLES:
        return [
            {"req_id": r.req_id, "category": r.category, "statement": r.statement}
            for r in requirements.to_requirements()
            if r.category in ("functional", "non_functional")
        ]
    # Default: functional + constraint requirements most plausibly map to an
    # implementation task; verifier-facing filters handled above.
    return [
        {"req_id": r.req_id, "category": r.category, "statement": r.statement}
        for r in requirements.to_requirements()
        if r.category in ("functional", "constraint", "acceptance")
    ]


def assemble_role_context(
    *,
    role: AgentRole,
    objective: str,
    requirements: Requirements | None = None,
    traceability: TraceabilityIndex | None = None,
    project_state: ProjectState | None = None,
    task_id: str = "",
    message_bus: AgentMessageBus | None = None,
    memory_context: str = "",
) -> RoleContext:
    """Build the context block for one task.

    Pure and dependency-light by design: callers (the runtime, tests) pass in
    only what they have.  The message bus is consulted only for handoffs whose
    recipient matches ``task_id``.
    """
    ctx = RoleContext(role=role.value, objective=objective or "")

    ctx.relevant_requirements = _requirements_for_role(requirements, role)

    if traceability is not None and task_id:
        mapped = traceability.requirements_for_task(task_id)
        ctx.mapped_requirement_ids = [r.req_id for r in mapped]

    artifacts: list[ArtifactRef] = list(project_state.artifacts) if project_state else []
    if role in _VERIFIER_ROLES:
        # Verifier sees everything produced.
        allowed = artifacts
    elif role in _INTEGRATOR_ROLES:
        # Integrators need contracts and schemas, not implementation noise.
        allowed = [a for a in artifacts if a.kind in _INTEGRATOR_ARTIFACT_KINDS]
    else:
        allowed = [a for a in artifacts if a.kind in _INTEGRATOR_ARTIFACT_KINDS]
    ctx.artifacts = [a.to_dict() for a in allowed]

    if message_bus is not None and task_id:
        for msg in message_bus.get_messages_for(task_id):
            ctx.handoffs.append(
                {
                    "sender_task_id": msg.sender_task_id,
                    "sender_agent_id": msg.sender_agent_id,
                    "message_type": msg.message_type.value,
                    "payload": msg.payload,
                }
            )

    ctx.memory_context = memory_context

    if requirements is not None:
        ctx.constraints = list(requirements.exclusions)
        ctx.constraints.append(
            "Runtime state is authoritative. Model output is never authoritative state."
        )

    return ctx
