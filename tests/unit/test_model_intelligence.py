"""Tests for Model Intelligence (M3.5) — ModelRegistry, capabilities, classifier, benchmarks."""

import time
import pytest
from pathlib import Path

from harness_core.models.types import (
    CapabilityConfidence,
    CapabilityScore,
    CapabilitySource,
    CapabilityProfile,
    ModelProfile,
)
from harness_core.models.registry import ModelRegistry
from harness_core.models.capabilities import CapabilityWeights, CODING_WEIGHTS
from harness_core.models.history import PerformanceHistory, PerformanceRecord
from harness_core.models.discovery import model_info_to_profile, estimate_capabilities
from harness_core.classifier.classifier import TaskClassifier, TaskType
from harness_core.classifier.types import TaskRequirementProfile
from harness_core.benchmarks.types import BenchmarkCategory, BenchmarkResult, BenchmarkTask
from harness_core.benchmarks.scoring import BenchmarkScoringWeights, aggregate_results


# ── ModelProfile Tests ─────────────────────────────────────────────────────

class TestModelProfile:
    def test_default_profile(self):
        p = ModelProfile(model_id="test-model", provider="test")
        assert p.model_id == "test-model"
        assert p.provider == "test"
        assert p.display_name == "test-model"
        assert p.supports_tools is False
        assert p.is_free is False

    def test_capability_profile_average(self):
        caps = CapabilityProfile()
        assert caps.get_average() is None  # nothing measured

        caps.coding.score = 0.8
        caps.tool_use.score = 0.9
        assert caps.get_average() == pytest.approx(0.85)

    def test_capability_set_get(self):
        caps = CapabilityProfile()
        caps.set(
            "coding",
            score=0.95,
            confidence=CapabilityConfidence.BENCHMARKED,
            source=CapabilitySource.HARNESS_BENCHMARK,
        )
        assert caps.coding.score == 0.95
        assert caps.coding.confidence == CapabilityConfidence.BENCHMARKED

    def test_capability_is_measured(self):
        cap = CapabilityScore()
        assert cap.is_measured is False

        cap.score = 0.5
        assert cap.is_measured is True

    def test_capability_unknown_not_zero(self):
        cap = CapabilityScore()
        assert cap.score is None
        assert cap.is_measured is False
        # None != 0
        assert cap.score != 0.0

    def test_profile_to_dict(self):
        p = ModelProfile(model_id="m1", provider="p1")
        d = p.to_dict()
        assert d["model_id"] == "m1"
        assert d["provider"] == "p1"
        assert "capabilities" in d


# ── ModelRegistry Tests ────────────────────────────────────────────────────

class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        p = ModelProfile(model_id="m1", provider="p1")
        reg.register(p)
        assert reg.get("m1") is not None
        assert reg.get("m1").model_id == "m1"

    def test_unregister(self):
        reg = ModelRegistry()
        reg.register(ModelProfile(model_id="m1", provider="p1"))
        assert reg.unregister("m1") is True
        assert reg.get("m1") is None

    def test_list_all(self):
        reg = ModelRegistry()
        reg.register(ModelProfile(model_id="m1", provider="p1"))
        reg.register(ModelProfile(model_id="m2", provider="p2"))
        assert reg.count() == 2

    def test_search(self):
        reg = ModelRegistry()
        reg.register(ModelProfile(model_id="m1", provider="p1", is_free=True))
        reg.register(ModelProfile(model_id="m2", provider="p2", is_free=False))
        reg.register(ModelProfile(model_id="m3", provider="p1", is_free=True))

        free = reg.search(is_free=True)
        assert len(free) == 2

        p1 = reg.search(provider="p1")
        assert len(p1) == 2

    def test_update_capabilities(self):
        reg = ModelRegistry()
        reg.register(ModelProfile(model_id="m1", provider="p1"))
        ok = reg.update_capabilities("m1", {"coding": 0.9, "tool_use": 0.8})
        assert ok is True
        p = reg.get("m1")
        assert p.capabilities.coding.score == 0.9

    def test_update_health(self):
        reg = ModelRegistry()
        reg.register(ModelProfile(model_id="m1", provider="p1"))
        ok = reg.update_health("m1", reliability=0.95, latency=0.8)
        assert ok is True
        p = reg.get("m1")
        assert p.reliability_score == 0.95

    def test_providers(self):
        reg = ModelRegistry()
        reg.register(ModelProfile(model_id="m1", provider="openrouter"))
        reg.register(ModelProfile(model_id="m2", provider="ollama"))
        reg.register(ModelProfile(model_id="m3", provider="openrouter"))
        assert set(reg.providers()) == {"openrouter", "ollama"}

    def test_summary(self):
        reg = ModelRegistry()
        reg.register(ModelProfile(model_id="m1", provider="p1", is_free=True, supports_tools=True))
        reg.register(ModelProfile(model_id="m2", provider="p2", is_free=False, supports_tools=True))
        s = reg.summary()
        assert s["total"] == 2
        assert s["free"] == 1
        assert s["supports_tools"] == 2


# ── Classifier Tests ───────────────────────────────────────────────────────

class TestTaskClassifier:
    def test_classify_bug_fix(self):
        c = TaskClassifier()
        assert c.classify("Fix the authentication bug") == TaskType.BUG_FIX

    def test_classify_implementation(self):
        c = TaskClassifier()
        assert c.classify("Implement a new login function") == TaskType.IMPLEMENTATION

    def test_classify_refactoring(self):
        c = TaskClassifier()
        assert c.classify("Refactor the database module") == TaskType.REFACTORING

    def test_classify_testing(self):
        c = TaskClassifier()
        assert c.classify("Write unit tests for the API") == TaskType.TESTING

    def test_classify_research(self):
        c = TaskClassifier()
        assert c.classify("Explain how the authentication works") == TaskType.RESEARCH

    def test_classify_with_confidence(self):
        c = TaskClassifier()
        task_type, confidence = c.classify_with_confidence("Fix the bug in the login function")
        assert task_type == TaskType.BUG_FIX
        assert confidence > 0.5

    def test_get_profile(self):
        c = TaskClassifier()
        profile = c.get_profile("Fix the authentication bug")
        assert profile.task_type == "bug_fix"
        assert profile.coding is not None
        assert profile.tool_use is not None

    def test_unknown_task(self):
        c = TaskClassifier()
        assert c.classify("") == TaskType.UNKNOWN


# ── TaskRequirementProfile Tests ───────────────────────────────────────────

class TestTaskRequirementProfile:
    def test_get_requirements(self):
        p = TaskRequirementProfile(task_type="bug_fix", coding=0.9, tool_use=0.8)
        reqs = p.get_requirements()
        assert reqs["coding"] == 0.9
        assert reqs["tool_use"] == 0.8
        assert reqs["reasoning"] is None  # don't care

    def test_compute_fit_perfect(self):
        p = TaskRequirementProfile(task_type="bug_fix", coding=0.8, tool_use=0.8)
        model_caps = {"coding": 0.9, "tool_use": 0.9}
        assert p.compute_fit(model_caps) == 1.0

    def test_compute_fit_partial(self):
        p = TaskRequirementProfile(task_type="bug_fix", coding=0.9)
        model_caps = {"coding": 0.5}
        fit = p.compute_fit(model_caps)
        assert 0.0 < fit < 1.0

    def test_compute_fit_no_requirements(self):
        p = TaskRequirementProfile(task_type="unknown")
        assert p.compute_fit({}) == 0.5  # neutral


# ── CapabilityWeights Tests ────────────────────────────────────────────────

class TestCapabilityWeights:
    def test_normalized(self):
        w = CapabilityWeights()
        n = w.normalized()
        total = (n.coding + n.tool_use + n.reasoning + n.planning
                + n.repository_navigation + n.context_handling
                + n.error_recovery + n.instruction_following + n.verification)
        assert abs(total - 1.0) < 0.001

    def test_custom_weights(self):
        w = CapabilityWeights(coding=1.0, tool_use=0.0)
        n = w.normalized()
        # Normalization distributes across all 9 dimensions
        assert n.tool_use == 0.0
        assert n.coding > 0.5  # coding dominates after normalization
        # Total should still sum to ~1.0
        total = (n.coding + n.tool_use + n.reasoning + n.planning
                + n.repository_navigation + n.context_handling
                + n.error_recovery + n.instruction_following + n.verification)
        assert abs(total - 1.0) < 0.001

    def test_to_dict(self):
        w = CODING_WEIGHTS
        d = w.to_dict()
        assert "coding" in d
        assert abs(sum(d.values()) - 1.0) < 0.001


# ── PerformanceHistory Tests ───────────────────────────────────────────────

class TestPerformanceHistory:
    def test_record_and_retrieve(self, tmp_path):
        db = tmp_path / "test_perf.db"
        history = PerformanceHistory(db)

        history.record(PerformanceRecord(
            model_id="m1", provider="p1", task_type="bug_fix",
            success=True, latency_ms=1000, tokens_used=500, tool_calls=5,
        ))

        perf = history.get_performance("m1")
        assert perf.total_tasks == 1
        assert perf.success_count == 1
        assert perf.avg_latency_ms > 0

    def test_time_decay(self, tmp_path):
        db = tmp_path / "test_decay.db"
        history = PerformanceHistory(db)

        # Old record
        history.record(PerformanceRecord(
            model_id="m1", success=True, timestamp=time.time() - 86400 * 60,
        ))
        # New record
        history.record(PerformanceRecord(
            model_id="m1", success=False, timestamp=time.time(),
        ))

        perf = history.get_performance("m1", decay_half_life_days=30)
        assert perf.total_tasks == 2

    def test_empty_history(self, tmp_path):
        db = tmp_path / "test_empty.db"
        history = PerformanceHistory(db)
        perf = history.get_performance("nonexistent")
        assert perf.total_tasks == 0

    def test_clear(self, tmp_path):
        db = tmp_path / "test_clear.db"
        history = PerformanceHistory(db)
        history.record(PerformanceRecord(model_id="m1", success=True))
        history.record(PerformanceRecord(model_id="m2", success=True))
        count = history.clear()
        assert count == 2


# ── Benchmark Scoring Tests ────────────────────────────────────────────────

class TestBenchmarkScoring:
    def test_aggregate_results(self):
        results = [
            BenchmarkResult(
                task_name="t1", model_id="m1", success=True,
                coding_score=0.9, tool_use_score=0.8, latency_ms=1000,
            ),
            BenchmarkResult(
                task_name="t2", model_id="m1", success=True,
                coding_score=0.7, tool_use_score=0.9, latency_ms=2000,
            ),
        ]
        suite = aggregate_results(results)
        assert suite.total_tasks == 2
        assert suite.success_rate == 1.0
        assert suite.coding_avg == pytest.approx(0.8)
        assert suite.tool_use_avg == pytest.approx(0.85)

    def test_aggregate_empty(self):
        suite = aggregate_results([])
        assert suite.total_tasks == 0

    def test_scoring_weights_normalized(self):
        w = BenchmarkScoringWeights()
        n = w.normalized()
        total = (n.coding + n.tool_use + n.navigation + n.recovery
                + n.context + n.verification + n.planning + n.success_bonus)
        assert abs(total - 1.0) < 0.001


# ── BenchmarkTask Tests ────────────────────────────────────────────────────

class TestBenchmarkTask:
    def test_task_creation(self):
        task = BenchmarkTask(
            name="test_task",
            category=BenchmarkCategory.TOOL_USE,
            description="Test reading a file",
            setup_files={"test.py": "print('hello')"},
        )
        assert task.name == "test_task"
        assert task.category == BenchmarkCategory.TOOL_USE
        assert "test.py" in task.setup_files

    def test_task_result(self):
        result = BenchmarkResult(
            task_name="test",
            model_id="m1",
            success=True,
            score=0.85,
            coding_score=0.9,
            tool_use_score=0.8,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["score"] == 0.85


# ── Discovery Tests ────────────────────────────────────────────────────────

class TestDiscovery:
    def test_model_info_to_profile(self):
        from harness_core.providers.base import ModelInfo
        info = ModelInfo(
            id="test/model",
            name="Test Model",
            provider="test",
            context_window=32000,
            supports_tools=True,
            is_free=True,
        )
        profile = model_info_to_profile(info)
        assert profile.model_id == "test/model"
        assert profile.supports_tools is True
        assert profile.is_free is True

    def test_estimate_capabilities(self):
        from harness_core.providers.base import ModelInfo
        info = ModelInfo(
            id="deepseek-coder-33b",
            name="DeepSeek Coder",
            provider="openrouter",
            context_window=32000,
            supports_tools=True,
        )
        profile = model_info_to_profile(info)
        profile = estimate_capabilities(profile)
        # Should have estimated coding capability
        assert profile.capabilities.coding.is_measured


# ── Integration: Classifier + Registry + Routing ───────────────────────────

class TestClassifierRegistryIntegration:
    def test_classify_and_match(self):
        classifier = TaskClassifier()
        registry = ModelRegistry()

        # Register a strong coding model
        p = ModelProfile(model_id="coder-33b", provider="openrouter")
        p.capabilities.coding.score = 0.95
        p.capabilities.coding.confidence = CapabilityConfidence.BENCHMARKED
        p.capabilities.tool_use.score = 0.90
        p.capabilities.tool_use.confidence = CapabilityConfidence.BENCHMARKED
        registry.register(p)

        # Register a weak model
        p2 = ModelProfile(model_id="small-8b", provider="openrouter")
        p2.capabilities.coding.score = 0.50
        p2.capabilities.coding.confidence = CapabilityConfidence.DECLARED
        registry.register(p2)

        # Classify a bug fix task
        task = "Fix the authentication bug and make tests pass"
        task_type = classifier.classify(task)
        profile = classifier.get_profile(task)

        assert task_type == TaskType.BUG_FIX
        assert profile.task_type == "bug_fix"

        # Both models should be registered
        assert registry.count() == 2
        assert registry.get("coder-33b") is not None
        assert registry.get("small-8b") is not None
