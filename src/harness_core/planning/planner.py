"""Planner for generating a TaskGraph from a user request."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from harness_core.providers.base import ModelProvider, CompletionRequest
from harness_core.planning.domain import Plan, PlanningConstraint, PlanningResult
from harness_core.planning.validator import PlanValidator
from harness_core.agents.registry import AgentRegistry

if TYPE_CHECKING:
    from harness_core.memory.manager import MemoryManager


PLANNER_SYSTEM_PROMPT = """You are an expert software engineering architect and task planner.
Your goal is to decompose a high-level user request into a dependency graph of specialized tasks.

You must output a structured JSON response matching this schema EXACTLY:
{{
    "summary": "Brief description of the overall plan.",
    "tasks": [
        {{
            "task_id": "unique_string_id",
            "title": "Short title",
            "objective": "Detailed instruction of what must be done",
            "role": "one of the available roles",
            "dependencies": ["task_id_of_dependency1"],
            "success_criteria": ["criterion 1"],
            "workspace_scope": "project",
            "priority": 5,
            "model_policy": "auto",
            "expected_artifacts": [],
            "resources": [
                {{
                    "path": "src/frontend/**",
                    "mode": "write"
                }}
            ]
        }}
    ]
}}

Available Roles:
{roles}

Rules:
1. Do not invent new roles. Use only the exact strings provided above.
2. Ensure task_ids are unique and contain only alphanumeric characters and underscores.
3. dependencies must reference task_ids that exist in the plan. No circular dependencies.
4. If tasks can be done in parallel, they should NOT depend on each other.
5. Workspace scope should usually be "project" unless you specifically want to restrict the agent (e.g., "frontend", "backend").
6. The plan should be comprehensive and culminate in verification/testing if appropriate.
7. You may declare "resources" the task requires. "path" can be exact or glob. "mode" must be "read" or "write". This prevents concurrent agents from corrupting the workspace.
8. Output MUST be purely valid JSON. No markdown fences.
"""

class Planner:
    """Uses a ModelProvider to autonomously plan execution graphs."""

    def __init__(
        self,
        provider: ModelProvider,
        registry: AgentRegistry | None = None,
        memory: "MemoryManager | None" = None,
        project_id: str = "",
    ):
        self.provider = provider
        self.registry = registry or AgentRegistry()
        self.validator = PlanValidator(registry=self.registry)
        # Phase 8F: Retrieval-augmented planning
        self.memory = memory
        self.project_id = project_id

    async def plan(self, request: str, context: dict[str, Any] | None = None) -> PlanningResult:
        """Generate a plan from a high-level request."""
        context = context or {}

        # Build constraints/roles
        roles = [r.name.lower() for r in self.registry._agents.values() if r.enabled]
        # Keep unique values and sort
        roles = sorted(list(set(roles)))

        system_prompt = PLANNER_SYSTEM_PROMPT.format(roles=", ".join(roles))

        # Phase 8F: Inject retrieval-augmented context from prior projects
        if self.memory and getattr(self.memory, "enabled", False):
            try:
                prior_context = await self.memory.get_context_for_planner(
                    user_request=request, project_id=self.project_id
                )
                if prior_context:
                    system_prompt += f"\n\n# Prior Project Context\n{prior_context}\n"
            except Exception:
                # Memory failures must not break planning
                pass

        user_prompt = f"User Request: {request}\n"
        if context:
            user_prompt += f"\nProject Context:\n{json.dumps(context, indent=2)}"

        req = CompletionRequest(
            model="",  # Uses default provider model or router
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2, # Low temp for structured output
        )

        try:
            # For simplicity, we just use the provider directly. 
            # In real system, might route through ModelRouter
            response = await self.provider.generate(req)
            content = response.content.strip()
            
            # Remove markdown JSON fences if the model still outputs them
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            plan = Plan.from_json(content)
            
            # Validate
            errors = self.validator.validate(plan)
            if errors:
                return PlanningResult(
                    plan=plan,
                    success=False,
                    errors=errors,
                    raw_response=response.content,
                )
                
            return PlanningResult(
                plan=plan,
                success=True,
                raw_response=response.content,
            )

        except json.JSONDecodeError as e:
            return PlanningResult(
                success=False,
                errors=[f"Failed to parse model output as JSON: {e}"],
            )
        except Exception as e:
            return PlanningResult(
                success=False,
                errors=[f"Planning failed: {e}"],
            )
