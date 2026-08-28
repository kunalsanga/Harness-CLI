"""Model scoring for intelligent routing.

14-dimension scoring system:
 1. capability       — context window as proxy for raw capability
 2. task_fit         — model naming alignment with task type
 3. tool_support     — whether model supports function calling
 4. context_fit      — whether context window fits the task
 5. cost             — cost per token (lower is better)
 6. reliability      — historical success rate
 7. latency          — response latency
 8. free_bonus       — bonus for free models when prefer_free
 9. task_type_fit    — how well model matches task classification
10. capability_fit   — empirical capability from ModelRegistry
11. history_success  — historical success rate from PerformanceHistory
12. history_latency  — historical latency from PerformanceHistory
13. tool_efficiency  — historical tool-call efficiency
14. user_preference  — bonus for user-selected model

Each scoring function returns [0.0, 1.0] where 1.0 is best.
Weights are configurable and sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from harness_core.providers.base import ModelInfo


@dataclass
class ScoringWeights:
    """Configurable weights for 14-dimension model scoring. Must sum to 1.0."""

    # Original 8 dimensions
    capability: float = 0.10
    task_fit: float = 0.08
    tool_support: float = 0.10
    context_fit: float = 0.07
    cost: float = 0.07
    reliability: float = 0.10
    latency: float = 0.05
    free_bonus: float = 0.05

    # New 6 dimensions (M3.6)
    task_type_fit: float = 0.10      # TaskClassifier alignment
    capability_fit: float = 0.08     # Empirical capability from registry
    history_success: float = 0.05    # Historical success rate
    history_latency: float = 0.03    # Historical latency
    tool_efficiency: float = 0.03    # Tool-call efficiency
    user_preference: float = 0.01    # User-selected model bonus

    def normalized(self) -> ScoringWeights:
        """Return weights normalized to sum to 1.0."""
        total = (
            self.capability + self.task_fit + self.tool_support
            + self.context_fit + self.cost + self.reliability
            + self.latency + self.free_bonus
            + self.task_type_fit + self.capability_fit
            + self.history_success + self.history_latency
            + self.tool_efficiency + self.user_preference
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
            task_type_fit=self.task_type_fit / total,
            capability_fit=self.capability_fit / total,
            history_success=self.history_success / total,
            history_latency=self.history_latency / total,
            tool_efficiency=self.tool_efficiency / total,
            user_preference=self.user_preference / total,
        )


@dataclass
class ScoringContext:
    """Context for scoring a model against a specific task.

    Includes all 14 dimensions of routing context.
    """

    task_description: str = ""
    requires_tools: bool = True
    requires_vision: bool = False
    requires_structured_output: bool = False
    estimated_context_tokens: int = 4096
    prefer_free: bool = False
    routing_mode: str = "auto"
    # Tag hints from the task (e.g. "coding", "reasoning", "research")
    task_tags: list[str] = field(default_factory=list)

    # New: task classification (from TaskClassifier)
    task_type: str = ""              # e.g. "bug_fix", "implementation"
    classification_confidence: float = 0.0

    # New: empirical capability (from ModelRegistry)
    model_capability_scores: dict[str, float] = field(default_factory=dict)  # cap_name -> score

    # New: historical performance (from PerformanceHistory)
    historical_success_rate: float = 0.0
    historical_avg_latency_ms: float = 0.0
    historical_tool_efficiency: float = 0.0  # tool_calls / iterations (lower = more efficient)
    historical_recovery_rate: float = 0.0

    # New: user-selected model
    user_selected_model: str = ""


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


def score_task_type_fit(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on task type classification alignment.

    Uses TaskClassifier output to match model strengths to task type.
    Returns 0.0–1.0.
    """
    if not ctx.task_type:
        return 0.5  # no classification, neutral

    model_id_lower = model.id.lower()

    # Task type → model alignment scoring
    task_alignments: dict[str, list[tuple[list[str], float]]] = {
        "bug_fix": [
            ("coder", 0.8), ("deepseek", 0.7), ("debug", 0.6),
            ("claude", 0.7), ("gpt-4", 0.7),
        ],
        "implementation": [
            ("coder", 0.8), ("deepseek-coder", 0.9), ("qwen-coder", 0.8),
            ("starcoder", 0.8), ("codestral", 0.8),
        ],
        "debugging": [
            ("deepseek", 0.8), ("claude", 0.7), ("gpt-4", 0.7),
            ("reason", 0.6), ("think", 0.6),
        ],
        "refactoring": [
            ("coder", 0.7), ("deepseek", 0.7), ("claude", 0.8),
            ("gpt-4", 0.7),
        ],
        "testing": [
            ("coder", 0.7), ("deepseek", 0.7), ("claude", 0.8),
        ],
        "research": [
            ("claude", 0.8), ("gpt-4", 0.8), ("gemini", 0.7),
            ("llama", 0.6),
        ],
        "repository_analysis": [
            ("coder", 0.7), ("deepseek", 0.7), ("claude", 0.8),
        ],
        "documentation": [
            ("claude", 0.8), ("gpt-4", 0.7), ("gemini", 0.7),
        ],
    }

    alignments = task_alignments.get(ctx.task_type, [])
    if not alignments:
        return 0.5  # no alignment data

    best_score = 0.0
    for keywords, score in alignments:
        for kw in keywords:
            if kw in model_id_lower:
                best_score = max(best_score, score)
                break

    return best_score if best_score > 0 else 0.4  # slight penalty for unknown alignment


def score_capability_fit(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on empirical capability from ModelRegistry.

    Uses actual measured capability scores, not just naming heuristics.
    Returns 0.0–1.0. 0.5 = no data (neutral).
    """
    if not ctx.model_capability_scores:
        return 0.5  # no data, neutral

    # Average available capability scores
    scores = [v for v in ctx.model_capability_scores.values() if v is not None]
    if not scores:
        return 0.5
    return sum(scores) / len(scores)


def score_history_success(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on historical success rate.

    Returns 0.0–1.0. 0.5 = no history (neutral).
    """
    if ctx.historical_success_rate <= 0:
        return 0.5  # no history, neutral
    return max(0.0, min(1.0, ctx.historical_success_rate))


def score_history_latency(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on historical latency.

    Lower latency = higher score. 0 = unknown (neutral).
    Returns 0.0–1.0.
    """
    if ctx.historical_avg_latency_ms <= 0:
        return 0.5  # unknown, neutral
    # <500ms = 1.0, 1s = 0.8, 2s = 0.6, 5s = 0.3, 10s+ = 0.1
    return max(0.0, min(1.0, 1.0 - (ctx.historical_avg_latency_ms / 10000)))


def score_tool_efficiency(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score based on tool-call efficiency (lower ratio = more efficient).

    Efficiency = iterations / tool_calls. Higher is better.
    Returns 0.0–1.0. 0.5 = no data (neutral).
    """
    if ctx.historical_tool_efficiency <= 0:
        return 0.5  # no data, neutral
    # efficiency > 1.0 means more iterations per tool call (bad)
    # efficiency < 1.0 means more tool calls per iteration (good)
    # Map: 0.2 → 0.9, 0.5 → 0.7, 1.0 → 0.5, 2.0 → 0.3, 5.0 → 0.1
    return max(0.0, min(1.0, 0.6 - (ctx.historical_tool_efficiency - 0.5) * 0.4))


def score_user_preference(model: ModelInfo, ctx: ScoringContext) -> float:
    """Score bonus for user-selected model.

    Returns 0.0–1.0. 0.5 = no preference (neutral).
    """
    if ctx.user_selected_model and model.id == ctx.user_selected_model:
        return 1.0  # user explicitly selected this model
    return 0.5  # neutral


def compute_model_score(
    model: ModelInfo,
    ctx: ScoringContext,
    weights: ScoringWeights | None = None,
) -> float:
    """Compute the total weighted score for a model using 14 dimensions.

    Returns a value in [0.0, 1.0] where 1.0 is the best fit.
    """
    if weights is None:
        weights = ScoringWeights()
    w = weights.normalized()

    scores = {
        # Original 8 dimensions
        "capability": score_capability(model, ctx),
        "task_fit": score_task_fit(model, ctx),
        "tool_support": score_tool_support(model, ctx),
        "context_fit": score_context_fit(model, ctx),
        "cost": score_cost(model, ctx),
        "reliability": score_reliability(model, ctx),
        "latency": score_latency(model, ctx),
        "free_bonus": score_free_bonus(model, ctx),
        # New 6 dimensions
        "task_type_fit": score_task_type_fit(model, ctx),
        "capability_fit": score_capability_fit(model, ctx),
        "history_success": score_history_success(model, ctx),
        "history_latency": score_history_latency(model, ctx),
        "tool_efficiency": score_tool_efficiency(model, ctx),
        "user_preference": score_user_preference(model, ctx),
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
        + w.task_type_fit * scores["task_type_fit"]
        + w.capability_fit * scores["capability_fit"]
        + w.history_success * scores["history_success"]
        + w.history_latency * scores["history_latency"]
        + w.tool_efficiency * scores["tool_efficiency"]
        + w.user_preference * scores["user_preference"]
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
