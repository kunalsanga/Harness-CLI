"""
Task-aware routing bridge.

Connects TaskClassifier, TaskRequirementProfile, ModelRegistry,
PerformanceHistory, and EmpiricalHistory into the ModelRouter's
scoring pipeline with full evidence provenance.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from harness_core.classifier.classifier import TaskClassifier, TaskType
from harness_core.classifier.types import TaskRequirementProfile
from harness_core.models.capabilities import CapabilityWeights
from harness_core.models.empirical import (
    ConfidenceCalculator,
    EmpiricalHistory,
    EvidenceSource,
    ModelEmpiricalProfile,
    SampleConfidence,
    TaskOutcome,
)
from harness_core.models.history import PerformanceHistory, PerformanceRecord
from harness_core.models.registry import ModelRegistry
from harness_core.models.types import CapabilityConfidence, ModelProfile
from harness_core.providers.base import CompletionRequest, ModelInfo


class RoutingExplanation:
    """Detailed explanation of why a model was selected."""

    def __init__(self, model_id: str = "") -> None:
        self.model_id: str = model_id
        self.static_capability: float = 0.0
        self.empirical_task_success: float = 0.0
        self.empirical_samples: int = 0
        self.empirical_confidence: SampleConfidence = SampleConfidence.UNKNOWN
        self.recent_success_rate: float = 0.0
        self.latency_ms: float = 0.0
        self.final_score: float = 0.0
        self.reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "static_capability": round(self.static_capability, 3),
            "empirical_task_success": round(self.empirical_task_success, 3),
            "empirical_samples": self.empirical_samples,
            "empirical_confidence": self.empirical_confidence.value,
            "recent_success_rate": round(self.recent_success_rate, 3),
            "latency_ms": round(self.latency_ms, 1),
            "final_score": round(self.final_score, 3),
            "reason": self.reason,
        }


class TaskAwareRouter:
    """Bridges task classification, model registry, performance history,
    and empirical execution history into a unified scoring pipeline.

    This replaces the ad-hoc keyword detection in ModelRouter._build_scoring_context
    with the full TaskClassifier + ModelRegistry + EmpiricalHistory intelligence.
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        history: PerformanceHistory | None = None,
        empirical: EmpiricalHistory | None = None,
        classifier: TaskClassifier | None = None,
    ) -> None:
        self.registry = registry or ModelRegistry()
        self.history = history
        self.empirical = empirical
        self.classifier = classifier or TaskClassifier()

    def classify_task(self, request: CompletionRequest) -> tuple[TaskType, TaskRequirementProfile, float]:
        """Classify the task from a completion request.

        Returns (task_type, requirement_profile, confidence).
        """
        # Extract task description from the last user message
        task_desc = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                task_desc = msg.get("content", "")
                break

        if not task_desc:
            return TaskType.UNKNOWN, TaskRequirementProfile(), 0.0

        task_type, confidence = self.classifier.classify_with_confidence(task_desc)
        profile = self.classifier.get_profile(task_desc)
        return task_type, profile, confidence

    def score_model_for_task(
        self,
        model_id: str,
        task_type: TaskType,
        requirement_profile: TaskRequirementProfile,
        weights: CapabilityWeights | None = None,
    ) -> tuple[float, RoutingExplanation]:
        """Score a model against a specific task using registry + empirical data.

        Returns (score, explanation) where score is 0.0–1.0.
        """
        explanation = RoutingExplanation(model_id=model_id)

        profile = self.registry.get(model_id)
        if profile is None:
            return 0.0, explanation

        # Get model capabilities as a dict
        caps: dict[str, Optional[float]] = {}
        for attr in [
            "coding", "tool_use", "reasoning", "planning",
            "repository_navigation", "context_handling",
            "error_recovery", "instruction_following", "verification",
        ]:
            cap = getattr(profile.capabilities, attr)
            caps[attr] = cap.score

        # Use TaskRequirementProfile.compute_fit for task-aware scoring
        capability_fit = requirement_profile.compute_fit(caps)
        explanation.static_capability = capability_fit

        # Empirical performance (from EmpiricalHistory)
        empirical_bonus = 0.0
        if self.empirical is not None:
            emp_profile = self.empirical.get_profile(model_id)
            task_type_str = task_type.value

            if task_type_str in emp_profile.by_task_type:
                task_perf = emp_profile.by_task_type[task_type_str]
                confidence_weight = ConfidenceCalculator.confidence_weight(
                    task_perf.sample_confidence
                )

                # Empirical success rate weighted by confidence
                empirical_success = task_perf.success_rate * confidence_weight
                empirical_bonus = (empirical_success - 0.5) * 0.3  # -0.15 to +0.15

                explanation.empirical_task_success = task_perf.success_rate
                explanation.empirical_samples = task_perf.total_tasks
                explanation.empirical_confidence = task_perf.sample_confidence
                explanation.recent_success_rate = task_perf.recent_success_rate
                explanation.latency_ms = task_perf.avg_latency_ms
            elif emp_profile.overall.total_tasks > 0:
                # Use overall if no task-specific data
                overall = emp_profile.overall
                confidence_weight = ConfidenceCalculator.confidence_weight(
                    overall.sample_confidence
                )
                empirical_success = overall.success_rate * confidence_weight
                empirical_bonus = (empirical_success - 0.5) * 0.2

                explanation.empirical_task_success = overall.success_rate
                explanation.empirical_samples = overall.total_tasks
                explanation.empirical_confidence = overall.sample_confidence

        # Legacy history bonus (from PerformanceHistory)
        history_bonus = 0.0
        if self.history is not None:
            perf = self.history.get_performance(model_id)
            if perf.total_tasks > 0:
                history_bonus = (perf.success_rate - 0.5) * 0.1
                history_bonus += (perf.recovery_rate - 0.5) * 0.05

        # Combine: capability fit + empirical + history
        score = max(0.0, min(1.0, capability_fit + empirical_bonus + history_bonus))
        explanation.final_score = score

        # Generate explanation reason
        if explanation.empirical_samples > 0:
            explanation.reason = (
                f"Static fit: {capability_fit:.3f}, "
                f"Empirical {task_type.value} success: "
                f"{explanation.empirical_task_success:.1%} "
                f"({explanation.empirical_samples} samples, "
                f"{explanation.empirical_confidence.value})"
            )
        else:
            explanation.reason = (
                f"Static fit: {capability_fit:.3f}, "
                f"No empirical history for {task_type.value} tasks"
            )

        return score, explanation

    def rank_models_for_task(
        self,
        model_ids: list[str],
        task_type: TaskType,
        requirement_profile: TaskRequirementProfile,
        weights: CapabilityWeights | None = None,
    ) -> list[tuple[str, float]]:
        """Rank models by their fit for a specific task.

        Returns list of (model_id, score) sorted descending.
        """
        scored = [
            (mid, self.score_model_for_task(mid, task_type, requirement_profile, weights)[0])
            for mid in model_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def rank_with_explanation(
        self,
        model_ids: list[str],
        task_type: TaskType,
        requirement_profile: TaskRequirementProfile,
        weights: CapabilityWeights | None = None,
    ) -> list[tuple[str, float, RoutingExplanation]]:
        """Rank models with detailed explanations.

        Returns list of (model_id, score, explanation) sorted descending.
        """
        scored = []
        for mid in model_ids:
            score, explanation = self.score_model_for_task(
                mid, task_type, requirement_profile, weights
            )
            scored.append((mid, score, explanation))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def record_task_result(
        self,
        model_id: str,
        provider: str,
        task_type: str,
        success: bool,
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        tool_calls: int = 0,
        iterations: int = 0,
        recovered: bool = False,
    ) -> None:
        """Record a task result for future routing decisions."""
        if self.history is not None:
            self.history.record(PerformanceRecord(
                model_id=model_id,
                provider=provider,
                task_type=task_type,
                success=success,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                tool_calls=tool_calls,
                iterations=iterations,
                recovered=recovered,
            ))

    def record_execution(self, record: "ModelExecutionRecord") -> None:
        """Record a full execution result into empirical history."""
        if self.empirical is not None:
            self.empirical.record(record)

    def get_model_summary(self, model_id: str) -> dict[str, Any]:
        """Get a complete summary of a model from registry + history."""
        profile = self.registry.get(model_id)
        result: dict[str, Any] = {"model_id": model_id, "found": profile is not None}

        if profile is None:
            return result

        # Capabilities
        caps = {}
        for attr in [
            "coding", "tool_use", "reasoning", "planning",
            "repository_navigation", "context_handling",
            "error_recovery", "instruction_following", "verification",
        ]:
            cap = getattr(profile.capabilities, attr)
            caps[attr] = {
                "score": cap.score,
                "confidence": cap.confidence.value,
                "source": cap.source.value,
            }
        result["capabilities"] = caps
        result["average"] = profile.capabilities.get_average()

        # Provider metadata
        result["provider"] = profile.provider
        result["is_free"] = profile.is_free
        result["supports_tools"] = profile.supports_tools
        result["context_window"] = profile.context_window

        # Historical performance
        if self.history is not None:
            perf = self.history.get_performance(model_id)
            result["performance"] = perf.to_dict()

        return result
