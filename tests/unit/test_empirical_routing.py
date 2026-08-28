"""Tests for empirical model routing — M3.8 integration."""

import time
import pytest
from pathlib import Path

from harness_core.classifier.classifier import TaskClassifier, TaskType
from harness_core.classifier.types import TaskRequirementProfile
from harness_core.models.empirical import (
    ConfidenceCalculator,
    EmpiricalHistory,
    EvidenceSource,
    ModelExecutionRecord,
    ModelPerformanceAggregator,
    SampleConfidence,
    TaskOutcome,
)
from harness_core.models.registry import ModelRegistry
from harness_core.models.types import CapabilityConfidence, CapabilityScore, CapabilitySource, ModelProfile
from harness_core.routing.task_aware import RoutingExplanation, TaskAwareRouter


def _register_test_models(registry: ModelRegistry):
    """Register models with different capability profiles."""
    strong = ModelProfile(model_id="strong-coder", provider="test")
    strong.capabilities.coding = CapabilityScore(score=0.95, confidence=CapabilityConfidence.BENCHMARKED, source=CapabilitySource.HARNESS_BENCHMARK)
    strong.capabilities.tool_use = CapabilityScore(score=0.90, confidence=CapabilityConfidence.BENCHMARKED, source=CapabilitySource.HARNESS_BENCHMARK)
    strong.capabilities.debugging = CapabilityScore(score=0.85, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    strong.capabilities.verification = CapabilityScore(score=0.88, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    strong.supports_tools = True
    registry.register(strong)

    weak = ModelProfile(model_id="weak-coder", provider="test")
    weak.capabilities.coding = CapabilityScore(score=0.40, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    weak.capabilities.tool_use = CapabilityScore(score=0.35, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    weak.capabilities.debugging = CapabilityScore(score=0.50, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    weak.capabilities.verification = CapabilityScore(score=0.30, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    weak.supports_tools = True
    registry.register(weak)

    medium = ModelProfile(model_id="medium-coder", provider="test")
    medium.capabilities.coding = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    medium.capabilities.tool_use = CapabilityScore(score=0.65, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    medium.capabilities.debugging = CapabilityScore(score=0.60, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    medium.capabilities.verification = CapabilityScore(score=0.65, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
    medium.supports_tools = True
    registry.register(medium)


class TestTaskAwareRouterWithEmpiricalData:
    """Test TaskAwareRouter when empirical history exists."""

    def test_empirical_data_influences_ranking(self, tmp_path):
        """Models with good empirical history should rank higher."""
        registry = ModelRegistry()
        _register_test_models(registry)

        empirical = EmpiricalHistory(tmp_path / "emp.db")

        # Record 50 successful bug-fix runs for weak-coder
        now = time.time()
        for i in range(50):
            empirical.record(ModelExecutionRecord(
                model_id="weak-coder",
                provider="test",
                task_type="bug_fix",
                outcome=TaskOutcome.SUCCESS,
                verification_passed=True,
                duration_ms=2000,
                tool_calls=5,
                iterations=3,
                timestamp=now - (50 - i) * 3600,
            ))

        # Record 50 failed bug-fix runs for strong-coder
        for i in range(50):
            empirical.record(ModelExecutionRecord(
                model_id="strong-coder",
                provider="test",
                task_type="bug_fix",
                outcome=TaskOutcome.FAILURE,
                verification_passed=False,
                duration_ms=5000,
                tool_calls=10,
                iterations=5,
                timestamp=now - (50 - i) * 3600,
            ))

        router = TaskAwareRouter(registry=registry, empirical=empirical)
        classifier = TaskClassifier()
        task_type, _ = classifier.classify_with_confidence("Fix authentication bug")
        profile = classifier.get_profile("Fix authentication bug")

        # weak-coder should rank higher due to empirical success
        ranked = router.rank_models_for_task(
            ["strong-coder", "weak-coder", "medium-coder"],
            task_type, profile,
        )

        # weak-coder has 100% empirical success, strong-coder has 0%
        # The empirical data should make weak-coder rank higher
        assert ranked[0][0] == "weak-coder", f"Expected weak-coder first, got {ranked}"

    def test_no_empirical_data_uses_static(self):
        """Without empirical data, routing uses static capabilities."""
        registry = ModelRegistry()
        _register_test_models(registry)

        router = TaskAwareRouter(registry=registry)
        classifier = TaskClassifier()
        task_type, _ = classifier.classify_with_confidence("Fix bug")
        profile = classifier.get_profile("Fix bug")

        ranked = router.rank_models_for_task(
            ["strong-coder", "weak-coder", "medium-coder"],
            task_type, profile,
        )

        # Strong coder should be first (highest static capabilities)
        assert ranked[0][0] == "strong-coder"

    def test_explanation_shows_empirical_data(self, tmp_path):
        """Routing explanation should include empirical details."""
        registry = ModelRegistry()
        _register_test_models(registry)
        empirical = EmpiricalHistory(tmp_path / "exp.db")

        # Add some history
        for i in range(10):
            empirical.record(ModelExecutionRecord(
                model_id="strong-coder",
                task_type="bug_fix",
                outcome=TaskOutcome.SUCCESS if i < 9 else TaskOutcome.FAILURE,
                verification_passed=i < 9,
                duration_ms=1500 + i * 100,
                timestamp=time.time() - (10 - i) * 3600,
            ))

        router = TaskAwareRouter(registry=registry, empirical=empirical)
        classifier = TaskClassifier()
        task_type, _ = classifier.classify_with_confidence("Fix the bug")
        profile = classifier.get_profile("Fix the bug")

        _, explanation = router.score_model_for_task("strong-coder", task_type, profile)

        assert explanation.empirical_samples == 10
        assert explanation.empirical_confidence == SampleConfidence.LOW
        assert explanation.empirical_task_success > 0
        assert "bug_fix" in explanation.reason

    def test_cold_start_behavior(self):
        """New models with no history should still be routed intelligently."""
        registry = ModelRegistry()
        _register_test_models(registry)

        # Add a new model with no history
        new_model = ModelProfile(model_id="new-model", provider="test")
        new_model.capabilities.coding = CapabilityScore(score=0.92, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        new_model.supports_tools = True
        registry.register(new_model)

        router = TaskAwareRouter(registry=registry)
        classifier = TaskClassifier()
        task_type, _ = classifier.classify_with_confidence("Implement feature")
        profile = classifier.get_profile("Implement feature")

        ranked = router.rank_models_for_task(
            ["new-model", "strong-coder", "weak-coder"],
            task_type, profile,
        )

        # New model should rank based on static capabilities
        # It should not be penalized for having no history
        assert ranked[0][0] in ("new-model", "strong-coder")  # Both have high coding

    def test_rank_with_explanation(self, tmp_path):
        """rank_with_explanation should return detailed explanations."""
        registry = ModelRegistry()
        _register_test_models(registry)
        empirical = EmpiricalHistory(tmp_path / "rank.db")

        for i in range(5):
            empirical.record(ModelExecutionRecord(
                model_id="strong-coder",
                task_type="bug_fix",
                outcome=TaskOutcome.SUCCESS,
                duration_ms=1500,
                timestamp=time.time() - (5 - i) * 3600,
            ))

        router = TaskAwareRouter(registry=registry, empirical=empirical)
        classifier = TaskClassifier()
        task_type, _ = classifier.classify_with_confidence("Fix tests")
        profile = classifier.get_profile("Fix tests")

        ranked = router.rank_with_explanation(
            ["strong-coder", "weak-coder"],
            task_type, profile,
        )

        assert len(ranked) == 2
        assert all(isinstance(r[2], RoutingExplanation) for r in ranked)

        # Strong coder should have empirical data
        strong_exp = next(r[2] for r in ranked if r[0] == "strong-coder")
        assert strong_exp.empirical_samples == 5
        assert strong_exp.empirical_confidence == SampleConfidence.LOW

    def test_routing_explanation_to_dict(self):
        """RoutingExplanation.to_dict should produce clean output."""
        exp = RoutingExplanation(model_id="test-model")
        exp.static_capability = 0.85
        exp.empirical_task_success = 0.92
        exp.empirical_samples = 42
        exp.empirical_confidence = SampleConfidence.MEDIUM
        exp.final_score = 0.87
        exp.reason = "Strong static fit + good empirical data"

        d = exp.to_dict()
        assert d["model_id"] == "test-model"
        assert d["static_capability"] == 0.85
        assert d["empirical_task_success"] == 0.92
        assert d["empirical_samples"] == 42
        assert d["empirical_confidence"] == "medium"
        assert d["final_score"] == 0.87


class TestModelPerformanceAggregatorAdvanced:
    """Advanced aggregation tests."""

    def test_time_decay_preference(self):
        """Recent successes should outweigh old failures."""
        now = time.time()
        records = []

        # Old failures (6 months ago)
        for i in range(20):
            records.append(ModelExecutionRecord(
                model_id="m",
                task_type="coding",
                outcome=TaskOutcome.FAILURE,
                timestamp=now - 180 * 86400,
            ))

        # Recent successes (last week)
        for i in range(10):
            records.append(ModelExecutionRecord(
                model_id="m",
                task_type="coding",
                outcome=TaskOutcome.SUCCESS,
                verification_passed=True,
                timestamp=now - i * 86400,
            ))

        agg = ModelPerformanceAggregator(decay_half_life_days=30)
        perf = agg.aggregate_task(records, "coding")

        # Recent successes should dominate due to time decay
        assert perf.success_rate > 0.5, f"Expected >50% with time decay, got {perf.success_rate:.1%}"

    def test_recent_window_tracking(self):
        """Recent window should track last N tasks independently."""
        now = time.time()
        records = []

        # 40 successes (old)
        for i in range(40):
            records.append(ModelExecutionRecord(
                model_id="m",
                task_type="debugging",
                outcome=TaskOutcome.SUCCESS,
                timestamp=now - 100 * 3600 - (40 - i),  # well before failures
            ))

        # 10 recent failures
        for i in range(10):
            records.append(ModelExecutionRecord(
                model_id="m",
                task_type="debugging",
                outcome=TaskOutcome.FAILURE,
                timestamp=now - (10 - i),  # last 10 seconds
            ))

        agg = ModelPerformanceAggregator(recent_window=10)
        perf = agg.aggregate_task(records, "debugging")

        assert perf.total_tasks == 50
        assert perf.recent_tasks == 10
        assert perf.recent_success_rate == 0.0  # All 10 recent are failures

    def test_profile_by_task_type(self):
        """Profile should segment performance by task type."""
        records = []
        now = time.time()

        for i in range(20):
            records.append(ModelExecutionRecord(
                model_id="m",
                task_type="bug_fix",
                outcome=TaskOutcome.SUCCESS if i < 18 else TaskOutcome.FAILURE,
                timestamp=now - (20 - i) * 3600,
            ))

        for i in range(15):
            records.append(ModelExecutionRecord(
                model_id="m",
                task_type="implementation",
                outcome=TaskOutcome.SUCCESS if i < 10 else TaskOutcome.FAILURE,
                timestamp=now - (15 - i) * 3600,
            ))

        agg = ModelPerformanceAggregator()
        profile = agg.build_profile(records)

        assert profile.overall.total_tasks == 35
        assert "bug_fix" in profile.by_task_type
        assert "implementation" in profile.by_task_type
        assert profile.by_task_type["bug_fix"].total_tasks == 20
        assert profile.by_task_type["implementation"].total_tasks == 15


class TestEmpiricalHistoryIntegration:
    """Integration tests for EmpiricalHistory with routing."""

    def test_record_and_route(self, tmp_path):
        """Full pipeline: record → aggregate → route."""
        db = tmp_path / "int.db"
        history = EmpiricalHistory(db)
        registry = ModelRegistry()
        _register_test_models(registry)

        # Record empirical data
        now = time.time()
        for i in range(25):
            history.record(ModelExecutionRecord(
                model_id="strong-coder",
                provider="test",
                task_type="bug_fix",
                outcome=TaskOutcome.SUCCESS if i < 23 else TaskOutcome.FAILURE,
                verification_passed=i < 23,
                duration_ms=1500,
                timestamp=now - (25 - i) * 3600,
            ))

        router = TaskAwareRouter(registry=registry, empirical=history)
        classifier = TaskClassifier()
        task_type, _ = classifier.classify_with_confidence("Fix the auth bug")
        profile = classifier.get_profile("Fix the auth bug")

        ranked = router.rank_models_for_task(
            ["strong-coder", "weak-coder"],
            task_type, profile,
        )

        # strong-coder has 92% empirical success → should rank first
        assert ranked[0][0] == "strong-coder"

        # Verify the profile is accessible
        emp_profile = history.get_profile("strong-coder")
        assert emp_profile.overall.total_tasks == 25
        assert emp_profile.overall.sample_confidence == SampleConfidence.MEDIUM

    def test_multiple_task_types_routing(self, tmp_path):
        """Different task types should use different empirical segments."""
        db = tmp_path / "multi.db"
        history = EmpiricalHistory(db)
        registry = ModelRegistry()
        _register_test_models(registry)

        now = time.time()

        # Model A: excellent at bug_fix, poor at implementation
        for i in range(30):
            history.record(ModelExecutionRecord(
                model_id="strong-coder",
                provider="test",
                task_type="bug_fix",
                outcome=TaskOutcome.SUCCESS,
                duration_ms=1500,
                timestamp=now - (30 - i) * 3600,
            ))
        for i in range(30):
            history.record(ModelExecutionRecord(
                model_id="strong-coder",
                provider="test",
                task_type="implementation",
                outcome=TaskOutcome.FAILURE,
                duration_ms=5000,
                timestamp=now - (30 - i) * 3600,
            ))

        # Model B: opposite pattern
        for i in range(30):
            history.record(ModelExecutionRecord(
                model_id="weak-coder",
                provider="test",
                task_type="bug_fix",
                outcome=TaskOutcome.FAILURE,
                duration_ms=5000,
                timestamp=now - (30 - i) * 3600,
            ))
        for i in range(30):
            history.record(ModelExecutionRecord(
                model_id="weak-coder",
                provider="test",
                task_type="implementation",
                outcome=TaskOutcome.SUCCESS,
                duration_ms=1500,
                timestamp=now - (30 - i) * 3600,
            ))

        # Register two models with similar static capabilities
        # so empirical data becomes the deciding factor
        equal_a = ModelProfile(model_id="equal-a", provider="test")
        equal_a.capabilities.coding = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        equal_a.capabilities.tool_use = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        equal_a.capabilities.debugging = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        equal_a.capabilities.verification = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        equal_a.supports_tools = True
        registry.register(equal_a)

        equal_b = ModelProfile(model_id="equal-b", provider="test")
        equal_b.capabilities.coding = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        equal_b.capabilities.tool_use = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        equal_b.capabilities.debugging = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        equal_b.capabilities.verification = CapabilityScore(score=0.70, confidence=CapabilityConfidence.DECLARED, source=CapabilitySource.PROVIDER_METADATA)
        equal_b.supports_tools = True
        registry.register(equal_b)

        # Record empirical data for equal-a and equal-b with opposite strengths
        for i in range(30):
            history.record(ModelExecutionRecord(
                model_id="equal-a",
                provider="test",
                task_type="bug_fix",
                outcome=TaskOutcome.SUCCESS,
                duration_ms=1500,
                timestamp=now - (30 - i) * 3600,
            ))
        for i in range(30):
            history.record(ModelExecutionRecord(
                model_id="equal-a",
                provider="test",
                task_type="implementation",
                outcome=TaskOutcome.FAILURE,
                duration_ms=5000,
                timestamp=now - (30 - i) * 3600,
            ))

        for i in range(30):
            history.record(ModelExecutionRecord(
                model_id="equal-b",
                provider="test",
                task_type="bug_fix",
                outcome=TaskOutcome.FAILURE,
                duration_ms=5000,
                timestamp=now - (30 - i) * 3600,
            ))
        for i in range(30):
            history.record(ModelExecutionRecord(
                model_id="equal-b",
                provider="test",
                task_type="implementation",
                outcome=TaskOutcome.SUCCESS,
                duration_ms=1500,
                timestamp=now - (30 - i) * 3600,
            ))

        classifier = TaskClassifier()
        router = TaskAwareRouter(registry=registry, empirical=history)

        # For bug_fix: equal-a should win (100% empirical success)
        bugfix_type, _ = classifier.classify_with_confidence("Fix the bug")
        bugfix_profile = classifier.get_profile("Fix the bug")
        ranked_bugfix = router.rank_models_for_task(
            ["equal-a", "equal-b"], bugfix_type, bugfix_profile,
        )
        assert ranked_bugfix[0][0] == "equal-a"

        # For implementation: equal-b should win (100% empirical success)
        impl_type, _ = classifier.classify_with_confidence("Implement new feature from scratch")
        impl_profile = classifier.get_profile("Implement new feature from scratch")
        ranked_impl = router.rank_models_for_task(
            ["equal-a", "equal-b"], impl_type, impl_profile,
        )
        assert ranked_impl[0][0] == "equal-b"
