"""
Recovery validator for validating recovery plans.
"""

from __future__ import annotations

from harness_core.agents.registry import AgentRegistry, AgentRole
from harness_core.recovery.planner import RecoveryPlan


class RecoveryValidator:
    """Validates proposed recovery plans."""

    def __init__(self, registry: AgentRegistry | None = None):
        self.registry = registry or AgentRegistry()

    def validate(self, plan: RecoveryPlan) -> list[str]:
        """Validate a recovery plan. Returns list of errors, empty if valid."""
        errors = []

        if not plan.target_task_id:
            errors.append("Recovery plan must specify a target_task_id.")

        if not plan.specialist_role:
            errors.append("Recovery plan must specify a specialist_role.")
            
        if not isinstance(plan.specialist_role, AgentRole):
            errors.append(f"Invalid role type for specialist_role: {type(plan.specialist_role)}")

        # Check if the role is registered and enabled
        if isinstance(plan.specialist_role, AgentRole):
            agent_profile = self.registry.get_default_for_role(plan.specialist_role)
            if not agent_profile:
                errors.append(f"No registered agent found for role: {plan.specialist_role.value}")
            elif not agent_profile.enabled:
                errors.append(f"Agent profile for role {plan.specialist_role.value} is disabled.")

        if not plan.objective:
            errors.append("Recovery plan must have a clear objective.")

        return errors
