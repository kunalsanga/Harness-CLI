"""
Capability scoring weights and defaults.

Configurable weights for evaluating model capabilities against task requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CapabilityWeights:
    """Weights for capability scoring. Must sum to 1.0."""

    coding: float = 0.25
    tool_use: float = 0.15
    reasoning: float = 0.15
    planning: float = 0.10
    repository_navigation: float = 0.10
    context_handling: float = 0.10
    error_recovery: float = 0.10
    instruction_following: float = 0.03
    verification: float = 0.02

    def normalized(self) -> CapabilityWeights:
        """Return weights normalized to sum to 1.0."""
        total = (
            self.coding + self.tool_use + self.reasoning + self.planning
            + self.repository_navigation + self.context_handling
            + self.error_recovery + self.instruction_following + self.verification
        )
        if total == 0:
            return CapabilityWeights()
        return CapabilityWeights(
            coding=self.coding / total,
            tool_use=self.tool_use / total,
            reasoning=self.reasoning / total,
            planning=self.planning / total,
            repository_navigation=self.repository_navigation / total,
            context_handling=self.context_handling / total,
            error_recovery=self.error_recovery / total,
            instruction_following=self.instruction_following / total,
            verification=self.verification / total,
        )

    def to_dict(self) -> dict[str, float]:
        w = self.normalized()
        return {
            "coding": w.coding,
            "tool_use": w.tool_use,
            "reasoning": w.reasoning,
            "planning": w.planning,
            "repository_navigation": w.repository_navigation,
            "context_handling": w.context_handling,
            "error_recovery": w.error_recovery,
            "instruction_following": w.instruction_following,
            "verification": w.verification,
        }


# Pre-built weight profiles for different routing modes

CODING_WEIGHTS = CapabilityWeights(
    coding=0.30,
    tool_use=0.20,
    reasoning=0.10,
    planning=0.05,
    repository_navigation=0.15,
    context_handling=0.05,
    error_recovery=0.10,
    instruction_following=0.03,
    verification=0.02,
)

DEBUGGING_WEIGHTS = CapabilityWeights(
    coding=0.15,
    tool_use=0.15,
    reasoning=0.25,
    planning=0.05,
    repository_navigation=0.15,
    context_handling=0.05,
    error_recovery=0.15,
    instruction_following=0.03,
    verification=0.02,
)

REASONING_WEIGHTS = CapabilityWeights(
    coding=0.10,
    tool_use=0.10,
    reasoning=0.30,
    planning=0.15,
    repository_navigation=0.05,
    context_handling=0.10,
    error_recovery=0.10,
    instruction_following=0.05,
    verification=0.05,
)

FAST_WEIGHTS = CapabilityWeights(
    coding=0.15,
    tool_use=0.20,
    reasoning=0.10,
    planning=0.05,
    repository_navigation=0.10,
    context_handling=0.05,
    error_recovery=0.10,
    instruction_following=0.10,
    verification=0.15,
)
