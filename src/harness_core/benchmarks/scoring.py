"""
Benchmark scoring — deterministic scoring with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from harness_core.benchmarks.types import BenchmarkResult, BenchmarkSuiteResult


@dataclass
class BenchmarkScoringWeights:
    """Weights for combining benchmark results into an overall score."""

    coding: float = 0.25
    tool_use: float = 0.15
    navigation: float = 0.10
    recovery: float = 0.10
    context: float = 0.10
    verification: float = 0.10
    planning: float = 0.10
    success_bonus: float = 0.10

    def normalized(self) -> BenchmarkScoringWeights:
        total = (
            self.coding + self.tool_use + self.navigation + self.recovery
            + self.context + self.verification + self.planning + self.success_bonus
        )
        if total == 0:
            return BenchmarkScoringWeights()
        return BenchmarkScoringWeights(
            coding=self.coding / total,
            tool_use=self.tool_use / total,
            navigation=self.navigation / total,
            recovery=self.recovery / total,
            context=self.context / total,
            verification=self.verification / total,
            planning=self.planning / total,
            success_bonus=self.success_bonus / total,
        )


def aggregate_results(
    results: list[BenchmarkResult],
    weights: BenchmarkScoringWeights | None = None,
) -> BenchmarkSuiteResult:
    """Aggregate multiple benchmark results into a suite result."""
    if not results:
        return BenchmarkSuiteResult()

    w = (weights or BenchmarkScoringWeights()).normalized()

    # Collect per-dimension scores
    coding_scores = [r.coding_score for r in results if r.coding_score is not None]
    tool_scores = [r.tool_use_score for r in results if r.tool_use_score is not None]
    nav_scores = [r.navigation_score for r in results if r.navigation_score is not None]
    recovery_scores = [r.recovery_score for r in results if r.recovery_score is not None]
    context_scores = [r.context_score for r in results if r.context_score is not None]
    verification_scores = [r.verification_score for r in results if r.verification_score is not None]
    planning_scores = [r.planning_score for r in results if r.planning_score is not None]

    def avg_or_none(scores: list[float]) -> Optional[float]:
        return sum(scores) / len(scores) if scores else None

    coding_avg = avg_or_none(coding_scores)
    tool_avg = avg_or_none(tool_scores)
    nav_avg = avg_or_none(nav_scores)
    recovery_avg = avg_or_none(recovery_scores)
    context_avg = avg_or_none(context_scores)
    verification_avg = avg_or_none(verification_scores)
    planning_avg = avg_or_none(planning_scores)

    # Compute overall score
    dimensions = [
        (coding_avg, w.coding),
        (tool_avg, w.tool_use),
        (nav_avg, w.navigation),
        (recovery_avg, w.recovery),
        (context_avg, w.context),
        (verification_avg, w.verification),
        (planning_avg, w.planning),
    ]

    total_weight = 0.0
    weighted_sum = 0.0
    for score, weight in dimensions:
        if score is not None:
            weighted_sum += score * weight
            total_weight += weight

    # Success bonus
    success_count = sum(1 for r in results if r.success)
    success_rate = success_count / len(results)
    weighted_sum += success_rate * w.success_bonus
    total_weight += w.success_bonus

    overall = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Aggregate metrics
    total_latency = sum(r.latency_ms for r in results)
    total_tokens = sum(r.tokens_used for r in results)
    total_tool_calls = sum(r.tool_calls for r in results)

    return BenchmarkSuiteResult(
        model_id=results[0].model_id,
        provider=results[0].provider,
        results=results,
        overall_score=overall,
        coding_avg=coding_avg,
        tool_use_avg=tool_avg,
        navigation_avg=nav_avg,
        recovery_avg=recovery_avg,
        context_avg=context_avg,
        verification_avg=verification_avg,
        planning_avg=planning_avg,
        avg_latency_ms=total_latency / len(results),
        avg_tokens=total_tokens / len(results),
        avg_tool_calls=total_tool_calls / len(results),
        success_rate=success_rate,
        total_tasks=len(results),
    )
