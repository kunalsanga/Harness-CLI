"""
Recovery planner to propose recovery strategies using the model provider.
"""

from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any

from harness_core.providers.base import ModelProvider, CompletionRequest
from harness_core.agents.registry import AgentRegistry, AgentRole
from harness_core.agents.domain import TaskGraph
from harness_core.recovery.classifier import FailureClassification


class RecoveryStrategy(enum.Enum):
    """Strategies for recovering from a failure."""
    DEBUG_AND_FIX = "debug_and_fix"
    REWRITE_COMPONENT = "rewrite_component"
    REVERT_AND_RETRY = "revert_and_retry"
    INVESTIGATE_ENVIRONMENT = "investigate_environment"
    ESCALATE = "escalate"


@dataclass
class RecoveryPlan:
    """A proposed recovery plan."""
    strategy: RecoveryStrategy
    specialist_role: AgentRole
    target_task_id: str
    objective: str
    success_criteria: list[str] = field(default_factory=list)


RECOVERY_SYSTEM_PROMPT = """You are an autonomous engineering recovery coordinator.
A task in the engineering pipeline has failed. Your job is to propose a Recovery Plan.

You must output a structured JSON response matching this schema EXACTLY:
{{
    "strategy": "debug_and_fix",
    "specialist_role": "one of the available roles",
    "objective": "Detailed instruction of what must be fixed",
    "success_criteria": ["criterion 1"]
}}

Available Roles:
{roles}

Rules:
1. "specialist_role" MUST be one of the available roles (e.g. "debugger", "backend", "security_reviewer").
2. "strategy" MUST be one of: "debug_and_fix", "rewrite_component", "revert_and_retry", "investigate_environment", "escalate".
3. For TEST_FAILURE, usually "debugger" or the original component specialist is best.
4. Provide a clear, actionable objective.
5. Output MUST be purely valid JSON. No markdown fences.
"""


class RecoveryPlanner:
    """Proposes recovery plans based on failure classification."""

    def __init__(self, provider: ModelProvider, registry: AgentRegistry | None = None):
        self.provider = provider
        self.registry = registry or AgentRegistry()

    async def plan(self, failed_task_id: str, classification: FailureClassification, context: dict[str, Any] | None = None) -> RecoveryPlan | None:
        """Generate a recovery plan for a failed task."""
        roles = [r.name.lower() for r in self.registry._agents.values() if r.enabled]
        roles = sorted(list(set(roles)))
        
        system_prompt = RECOVERY_SYSTEM_PROMPT.format(roles=", ".join(roles))
        
        user_prompt = f"Failed Task ID: {failed_task_id}\n"
        user_prompt += f"Failure Classification:\n{json.dumps(classification.to_dict(), indent=2)}\n"
        if context:
            user_prompt += f"\nContext:\n{json.dumps(context, indent=2)}"

        req = CompletionRequest(
            model="", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        try:
            response = await self.provider.generate(req)
            content = response.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)
            
            # Use fallback for role if it doesn't match
            try:
                role_enum = AgentRole(data.get("specialist_role", "debugger"))
            except ValueError:
                role_enum = AgentRole.DEBUGGER
                
            try:
                strategy = RecoveryStrategy(data.get("strategy", "debug_and_fix"))
            except ValueError:
                strategy = RecoveryStrategy.DEBUG_AND_FIX

            return RecoveryPlan(
                strategy=strategy,
                specialist_role=role_enum,
                target_task_id=failed_task_id,
                objective=data.get("objective", "Fix the failure."),
                success_criteria=data.get("success_criteria", ["Tests pass"]),
            )

        except Exception as e:
            # Deterministic fallback on model parsing error
            return RecoveryPlan(
                strategy=RecoveryStrategy.DEBUG_AND_FIX,
                specialist_role=AgentRole.DEBUGGER,
                target_task_id=failed_task_id,
                objective=f"Diagnose and fix {classification.category.value}",
                success_criteria=["Tests pass without errors"],
            )
