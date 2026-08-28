"""
Task requirement profiles — what capabilities a task needs.

These feed into the ModelRouter for task-aware model selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskRequirementProfile:
    """What a specific task requires from a model."""

    task_type: str = "unknown"

    # Required capability levels (0.0-1.0). None = don't care.
    coding: Optional[float] = None
    tool_use: Optional[float] = None
    reasoning: Optional[float] = None
    planning: Optional[float] = None
    repository_navigation: Optional[float] = None
    context_handling: Optional[float] = None
    error_recovery: Optional[float] = None
    instruction_following: Optional[float] = None
    verification: Optional[float] = None

    # Operational requirements
    requires_tools: bool = True
    requires_vision: bool = False
    requires_streaming: bool = False
    requires_structured_output: bool = False

    def get_requirements(self) -> dict[str, Optional[float]]:
        """Get all capability requirements as a dict."""
        return {
            "coding": self.coding,
            "tool_use": self.tool_use,
            "reasoning": self.reasoning,
            "planning": self.planning,
            "repository_navigation": self.repository_navigation,
            "context_handling": self.context_handling,
            "error_recovery": self.error_recovery,
            "instruction_following": self.instruction_following,
            "verification": self.verification,
        }

    def compute_fit(self, model_capabilities: dict[str, Optional[float]]) -> float:
        """Compute how well a model's capabilities match this task's requirements.

        Returns 0.0-1.0 where 1.0 is perfect fit.
        """
        requirements = self.get_requirements()
        scores: list[float] = []

        for cap_name, required_level in requirements.items():
            if required_level is None:
                continue  # don't care about this capability
            model_score = model_capabilities.get(cap_name)
            if model_score is None:
                # Unknown capability — slight penalty but don't disqualify
                scores.append(0.5)
            else:
                # How well does the model meet the requirement?
                if model_score >= required_level:
                    scores.append(1.0)
                else:
                    # Proportional shortfall
                    scores.append(model_score / required_level)

        if not scores:
            return 0.5  # no requirements = neutral
        return sum(scores) / len(scores)


# ── Pre-built requirement profiles ────────────────────────────────────────

def bug_fix_profile() -> TaskRequirementProfile:
    return TaskRequirementProfile(
        task_type="bug_fix",
        coding=0.90,
        tool_use=0.90,
        reasoning=0.80,
        planning=0.50,
        repository_navigation=0.80,
        context_handling=0.80,
        error_recovery=0.80,
        instruction_following=0.80,
        verification=0.90,
    )


def implementation_profile() -> TaskRequirementProfile:
    return TaskRequirementProfile(
        task_type="implementation",
        coding=0.95,
        tool_use=0.90,
        reasoning=0.70,
        planning=0.70,
        repository_navigation=0.70,
        context_handling=0.70,
        error_recovery=0.60,
        instruction_following=0.80,
        verification=0.80,
    )


def debugging_profile() -> TaskRequirementProfile:
    return TaskRequirementProfile(
        task_type="debugging",
        coding=0.70,
        tool_use=0.85,
        reasoning=0.95,
        planning=0.50,
        repository_navigation=0.80,
        context_handling=0.70,
        error_recovery=0.90,
        instruction_following=0.70,
        verification=0.85,
    )


def refactoring_profile() -> TaskRequirementProfile:
    return TaskRequirementProfile(
        task_type="refactoring",
        coding=0.90,
        tool_use=0.85,
        reasoning=0.70,
        planning=0.80,
        repository_navigation=0.80,
        context_handling=0.70,
        error_recovery=0.60,
        instruction_following=0.80,
        verification=0.90,
    )


def testing_profile() -> TaskRequirementProfile:
    return TaskRequirementProfile(
        task_type="testing",
        coding=0.85,
        tool_use=0.90,
        reasoning=0.60,
        planning=0.50,
        repository_navigation=0.70,
        context_handling=0.60,
        error_recovery=0.50,
        instruction_following=0.80,
        verification=0.95,
    )


def research_profile() -> TaskRequirementProfile:
    return TaskRequirementProfile(
        task_type="research",
        coding=0.30,
        tool_use=0.60,
        reasoning=0.90,
        planning=0.70,
        repository_navigation=0.80,
        context_handling=0.90,
        error_recovery=0.30,
        instruction_following=0.80,
        verification=0.50,
        requires_tools=True,
    )


def repository_analysis_profile() -> TaskRequirementProfile:
    return TaskRequirementProfile(
        task_type="repository_analysis",
        coding=0.20,
        tool_use=0.80,
        reasoning=0.70,
        planning=0.50,
        repository_navigation=0.95,
        context_handling=0.80,
        error_recovery=0.30,
        instruction_following=0.80,
        verification=0.40,
    )


def documentation_profile() -> TaskRequirementProfile:
    return TaskRequirementProfile(
        task_type="documentation",
        coding=0.40,
        tool_use=0.60,
        reasoning=0.50,
        planning=0.50,
        repository_navigation=0.70,
        context_handling=0.70,
        error_recovery=0.20,
        instruction_following=0.90,
        verification=0.50,
    )


TASK_PROFILES: dict[str, callable] = {
    "bug_fix": bug_fix_profile,
    "implementation": implementation_profile,
    "debugging": debugging_profile,
    "refactoring": refactoring_profile,
    "testing": testing_profile,
    "research": research_profile,
    "repository_analysis": repository_analysis_profile,
    "documentation": documentation_profile,
}
