"""Model scoring for intelligent routing.

Each scoring function returns a value in [0.0, 1.0] where 1.0 is best.
Weights are configurable and sum to 1.0 by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness_core.providers.base import ModelInfo


@dataclass
class ScoringWeights:
    """Configurable weights for model scoring. Must sum to 1.0."""

    capability: float = 0.20
    task_fit: float = 0.15
    tool_support: float = 0.15
    context_fit: float = 0.10
    cost: float = 0.10
    reliability: float = 0.15
    latency: float = 0.05
    free_bonus: float = 0.10

    def normalized(self) -> ScoringWeights:
        """Return weights normalized to sum to 1.0."""
        total = (
            self.capability
            + self.task_fit
            + self.tool_support
            + self.context_fit
            + self.cost
            + self.reliability
            + self.latency
            + self.free_bonus
        )
        if total == 0:
            return ScoringWeights()
        return ScoringWeights(
            capability=self.capability / total,
            task_fit=self.task_fit / total,
            tool_support=self.tool_support / total,
            context_fit=self.context_fit / total,
            cost=self.cost / total,
            reliability=self.reliability / total,
            latency=self.latency / total,
            free_bonus=self.free_bonus / total,
        )


@dataclass
class ScoringContext:
    """Context for scoring a model against a specific task."""

    task_description: str = ""
    requires_tools: bool = True
    requires_vision: bool = False
    requires_structured_output: bool = False
    estimated_context_tokens: int = 4096
    prefer_free: bool = False
    routing_mode: str = "auto"
    # Tag hints from the task (e.g. "coding", "reasoning", "research")
    task_tags: list[str] = field(default_factory=list)


def score_capability(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on raw model capability (context window as proxy).

    Larger context windows and free-tier availability indicate stronger models.
    """
    # Map context window to a 0-1 score using log scale
    if model.context_window <= 0:
        return 0.0
    import math
    # 8K = 0.2, 32K = 0.4, 128K = 0.6, 512K = 0.8, 1M+ = 1.0
    log_window = math.log2(model.context_window)
    # log2(8192) = 13, log2(1048576) = 20
    score = max(0.0, min(1.0, (log_window - 12) / 8))
    return score


def score_task_fit(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score how well the model fits the specific task.

    Uses model tags, id patterns, and task hints.
    """
    score = 0.5  # baseline

    model_id_lower = model.id.lower()

    # Coding task indicators
    coding_keywords = ["code", "coding", "coder", "program", "develop", "debug", "fix"]
    reasoning_keywords = ["reason", "think", "logic", "analysis", "plan"]
    research_keywords = ["research", "search", "find", "analyze"]

    task_lower = ctx.task_description.lower()
    task_tags_lower = [t.lower() for t in ctx.task_tags]

    # Check for coding alignment
    is_coding_task = any(kw in task_lower or kw in task_tags_lower for kw in coding_keywords)
    is_coding_model = any(kw in model_id_lower for kw in ["code", "coder", "program", "deepseek"])
    if is_coding_task and is_coding_model:
        score += 0.3
    elif is_coding_task and not is_coding_model:
        score -= 0.1

    # Check for reasoning alignment
    is_reasoning_task = any(kw in task_lower or kw in task_tags_lower for kw in reasoning_keywords)
    is_reasoning_model = any(kw in model_id_lower for kw in ["reason", "think", "o1", "o3", "opus"])
    if is_reasoning_task and is_reasoning_model:
        score += 0.2

    # Check for research alignment
    is_research_task = any(kw in task_lower or kw in task_tags_lower for kw in research_keywords)
    if is_research_task:
        score += 0.1  # most models can research

    # Cap to [0, 1]
    return max(0.0, min(1.0, score))


def score_tool_support(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on tool/function-calling support."""
    if not ctx.requires_tools:
        return 1.0  # doesn't matter
    return 1.0 if model.supports_tools else 0.0


def score_context_fit(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on whether the model's context window fits the task."""
    if model.context_window <= 0:
        return 0.0
    ratio = ctx.estimated_context_tokens / model.context_window
    if ratio <= 0.5:
        return 1.0  # plenty of room
    elif ratio <= 0.8:
        return 0.7  # fits but getting tight
    elif ratio <= 1.0:
        return 0.3  # barely fits
    else:
        return 0.0  # won't fit


def score_cost(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on cost — lower is better.

    Free models score 1.0. Expensive models score lower.
    Maps $0.001/1k → 0.9, $0.01/1k → 0.7, $0.1/1k → 0.4, $1.0/1k → 0.1.
    """
    if model.is_free:
        return 1.0

    total_cost = model.cost_per_1k_input + model.cost_per_1k_output
    if total_cost <= 0:
        return 1.0  # unknown/free

    import math
    # Use sigmoid-like mapping: cheap → high score, expensive → low score
    # $0.001 → ~0.9, $0.01 → ~0.7, $0.1 → ~0.4, $1.0 → ~0.1, $10+ → ~0.0
    log_cost = math.log10(total_cost)
    # log10(0.001)=-3, log10(0.01)=-2, log10(0.1)=-1, log10(1)=0, log10(10)=1
    # Map: -3→0.9, -2→0.7, -1→0.4, 0→0.1, 1→0.0
    score = max(0.0, min(1.0, 0.1 - log_cost * 0.3))
    return score


def score_reliability(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on tracked reliability (0.0 = never works, 1.0 = always works)."""
    return max(0.0, min(1.0, model.reliability))


def score_latency(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on latency — lower is better.

    Unknown latency (0) gets neutral score.
    """
    if model.latency_ms <= 0:
        return 0.5  # unknown, neutral
    # <500ms = 1.0, 1s = 0.8, 2s = 0.6, 5s = 0.3, 10s+ = 0.1
    score = max(0.0, min(1.0, 1.0 - (model.latency_ms / 10000)))
    return score


def score_free_bonus(model: ModelInfo, ctx: ScoringContext) -> float:
    """Bonus score for free models when prefer_free is enabled."""
    if not ctx.prefer_free:
        return 0.5  # neutral
    return 1.0 if model.is_free else 0.0


def compute_model_score(
    model: ModelInfo,
    ctx: ScoringContext,
    weights: ScoringWeights | None = None,
) -> float:
    """Compute the total weighted score for a model.

    Returns a value in [0.0, 1.0] where 1.0 is the best fit.
    """
    if weights is None:
        weights = ScoringWeights()
    w = weights.normalized()

    scores = {
        "capability": score_capability(model, ctx),
        "task_fit": score_task_fit(model, ctx),
        "tool_support": score_tool_support(model, ctx),
        "context_fit": score_context_fit(model, ctx),
        "cost": score_cost(model, ctx),
        "reliability": score_reliability(model, ctx),
        "latency": score_latency(model, ctx),
        "free_bonus": score_free_bonus(model, ctx),
    }

    total = (
        w.capability * scores["capability"]
        + w.task_fit * scores["task_fit"]
        + w.tool_support * scores["tool_support"]
        + w.context_fit * scores["context_fit"]
        + w.cost * scores["cost"]
        + w.reliability * scores["reliability"]
        + w.latency * scores["latency"]
        + w.free_bonus * scores["free_bonus"]
    )

    return max(0.0, min(1.0, total))


def rank_models(
    models: list[ModelInfo],
    ctx: ScoringContext,
    weights: ScoringWeights | None = None,
) -> list[tuple[ModelInfo, float]]:
    """Rank models by score. Returns list of (model, score) sorted descending."""
    scored = [(m, compute_model_score(m, ctx, weights)) for m in models]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
