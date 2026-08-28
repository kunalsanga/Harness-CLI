"""Tests for empirical model intelligence — M3.8."""

import time
import pytest
from pathlib import Path

from harness_core.models.empirical import (
    ConfidenceCalculator,
    EvidenceSource,
    ModelEmpiricalProfile,
    ModelExecutionRecord,
    ModelPerformanceAggregator,
    EmpiricalHistory,
    SampleConfidence,
    TaskOutcome,
    TaskPerformance,
)


class TestTaskOutcome:
    """Test outcome taxonomy."""

    def test_all_outcomes_exist(self):
        outcomes = [
            TaskOutcome.SUCCESS,
            TaskOutcome.PARTIAL_SUCCESS,
            TaskOutcome.FAILURE,
            TaskOutcome.TIMEOUT,
            TaskOutcome.MODEL_ERROR,
            TaskOutcome.TOOL_ERROR,
            TaskOutcome.PERMISSION_DENIED,
            TaskOutcome.USER_ABORTED,
            TaskOutcome.UNKNOWN,
        ]
        assert len(outcomes) == 9

    def test_outcome_values(self):
        assert TaskOutcome.SUCCESS.value == "success"
        assert TaskOutcome.PARTIAL_SUCCESS.value == "partial_success"
        assert TaskOutcome.TIMEOUT.value == "timeout"


class TestConfidenceCalculator:
    """Test sample-based confidence."""

    def test_zero_samples(self):
        assert ConfidenceCalculator.from_sample_count(0) == SampleConfidence.UNKNOWN

    def test_very_low(self):
        assert ConfidenceCalculator.from_sample_count(1) == SampleConfidence.VERY_LOW
        assert ConfidenceCalculator.from_sample_count(4) == SampleConfidence.VERY_LOW

    def test_low(self):
        assert ConfidenceCalculator.from_sample_count(5) == SampleConfidence.LOW
        assert ConfidenceCalculator.from_sample_count(19) == SampleConfidence.LOW

    def test_medium(self):
        assert ConfidenceCalculator.from_sample_count(20) == SampleConfidence.MEDIUM
        assert ConfidenceCalculator.from_sample_count(49) == SampleConfidence.MEDIUM

    def test_high(self):
        assert ConfidenceCalculator.from_sample_count(50) == SampleConfidence.HIGH
        assert ConfidenceCalculator.from_sample_count(1000) == SampleConfidence.HIGH

    def test_confidence_weights(self):
        assert ConfidenceCalculator.confidence_weight(SampleConfidence.UNKNOWN) == 0.0
        assert ConfidenceCalculator.confidence_weight(SampleConfidence.VERY_LOW) == 0.1
        assert ConfidenceCalculator.confidence_weight(SampleConfidence.LOW) == 0.3
        assert ConfidenceCalculator.confidence_weight(SampleConfidence.MEDIUM) == 0.6
        assert ConfidenceCalculator.confidence_weight(SampleConfidence.HIGH) == 1.0


class TestModelExecutionRecord:
    """Test execution record creation."""

    def test_default_record(self):
        r = ModelExecutionRecord()
        assert len(r.record_id) > 0
        assert r.outcome == TaskOutcome.UNKNOWN
        assert r.source == EvidenceSource.REAL_WORLD_OBSERVED
        assert r.timestamp > 0

    def test_record_with_fields(self):
        r = ModelExecutionRecord(
            model_id="test-model",
            provider="test",
            task_type="bug_fix",
            outcome=TaskOutcome.SUCCESS,
            verification_passed=True,
            duration_ms=1500,
            tool_calls=5,
            iterations=3,
        )
        assert r.model_id == "test-model"
        assert r.outcome == TaskOutcome.SUCCESS
        assert r.verification_passed is True
        assert r.duration_ms == 1500


class TestAggregation:
    """Test performance aggregation."""

    def _make_records(
        self,
        model_id: str = "test-model",
        task_type: str = "bug_fix",
        count: int = 10,
        success_rate: float = 0.8,
    ) -> list[ModelExecutionRecord]:
        records = []
        now = time.time()
        for i in range(count):
            success = i < int(count * success_rate)
            records.append(ModelExecutionRecord(
                model_id=model_id,
                task_type=task_type,
                outcome=TaskOutcome.SUCCESS if success else TaskOutcome.FAILURE,
                verification_passed=success,
                duration_ms=1000 + i * 100,
                tool_calls=3 + i % 5,
                iterations=2 + i % 3,
                total_tokens=1000 + i * 100,
                timestamp=now - (count - i) * 3600,  # 1 hour apart
            ))
        return records

    def test_aggregate_empty(self):
        agg = ModelPerformanceAggregator()
        perf = agg.aggregate_task([], "bug_fix")
        assert perf.total_tasks == 0
        assert perf.success_rate == 0.0

    def test_aggregate_basic(self):
        records = self._make_records(count=10, success_rate=0.8)
        agg = ModelPerformanceAggregator()
        perf = agg.aggregate_task(records, "bug_fix")

        assert perf.total_tasks == 10
        assert perf.success_count == 8
        assert perf.failure_count == 2
        assert 0.7 <= perf.success_rate <= 0.9  # time-weighted, so slightly different
        assert perf.sample_confidence == SampleConfidence.LOW

    def test_aggregate_recent(self):
        records = self._make_records(count=30, success_rate=0.9)
        agg = ModelPerformanceAggregator(recent_window=10)
        perf = agg.aggregate_task(records, "bug_fix")

        assert perf.total_tasks == 30
        assert perf.recent_tasks == 10
        # Recent success rate should be reasonable
        assert 0.0 <= perf.recent_success_rate <= 1.0

    def test_build_profile(self):
        records = self._make_records(count=10, success_rate=0.8)
        agg = ModelPerformanceAggregator()
        profile = agg.build_profile(records)

        assert profile.model_id == "test-model"
        assert profile.overall.total_tasks == 10
        assert "bug_fix" in profile.by_task_type
        assert profile.total_records == 10
        assert profile.primary_source == EvidenceSource.REAL_WORLD_OBSERVED

    def test_build_profile_multiple_task_types(self):
        records = (
            self._make_records(task_type="bug_fix", count=10, success_rate=0.9)
            + self._make_records(task_type="implementation", count=8, success_rate=0.7)
        )
        agg = ModelPerformanceAggregator()
        profile = agg.build_profile(records)

        assert len(profile.by_task_type) == 2
        assert "bug_fix" in profile.by_task_type
        assert "implementation" in profile.by_task_type
        assert profile.overall.total_tasks == 18

    def test_percentile(self):
        agg = ModelPerformanceAggregator()
        assert agg._percentile([1, 2, 3, 4, 5], 50) == 3
        assert agg._percentile([1, 2, 3, 4, 5], 95) == 5
        assert agg._percentile([], 50) == 0.0


class TestEmpiricalHistory:
    """Test SQLite-backed empirical history."""

    def test_record_and_retrieve(self, tmp_path):
        db = tmp_path / "test.db"
        history = EmpiricalHistory(db)

        record = ModelExecutionRecord(
            model_id="model-a",
            provider="test",
            task_type="bug_fix",
            outcome=TaskOutcome.SUCCESS,
            verification_passed=True,
            duration_ms=2000,
            tool_calls=5,
            iterations=3,
        )
        history.record(record)

        records = history.get_records("model-a")
        assert len(records) == 1
        assert records[0].model_id == "model-a"
        assert records[0].outcome == TaskOutcome.SUCCESS

    def test_get_profile(self, tmp_path):
        db = tmp_path / "test.db"
        history = EmpiricalHistory(db)

        for i in range(10):
            history.record(ModelExecutionRecord(
                model_id="model-b",
                task_type="bug_fix",
                outcome=TaskOutcome.SUCCESS if i < 8 else TaskOutcome.FAILURE,
                duration_ms=1000 + i * 100,
                timestamp=time.time() - (10 - i) * 3600,
            ))

        profile = history.get_profile("model-b")
        assert profile.model_id == "model-b"
        assert profile.overall.total_tasks == 10
        assert "bug_fix" in profile.by_task_type

    def test_count(self, tmp_path):
        db = tmp_path / "test.db"
        history = EmpiricalHistory(db)

        assert history.count() == 0

        history.record(ModelExecutionRecord(model_id="a"))
        history.record(ModelExecutionRecord(model_id="b"))

        assert history.count() == 2
        assert history.count("a") == 1

    def test_clear(self, tmp_path):
        db = tmp_path / "test.db"
        history = EmpiricalHistory(db)

        history.record(ModelExecutionRecord(model_id="a"))
        history.record(ModelExecutionRecord(model_id="b"))

        deleted = history.clear("a")
        assert deleted == 1
        assert history.count() == 1

    def test_task_type_filter(self, tmp_path):
        db = tmp_path / "test.db"
        history = EmpiricalHistory(db)

        history.record(ModelExecutionRecord(model_id="m", task_type="bug_fix"))
        history.record(ModelExecutionRecord(model_id="m", task_type="implementation"))
        history.record(ModelExecutionRecord(model_id="m", task_type="bug_fix"))

        records = history.get_records("m", task_type="bug_fix")
        assert len(records) == 2

    def test_cache_invalidation(self, tmp_path):
        db = tmp_path / "test.db"
        history = EmpiricalHistory(db)

        history.record(ModelExecutionRecord(
            model_id="c", outcome=TaskOutcome.SUCCESS,
        ))
        p1 = history.get_profile("c")
        assert p1.overall.total_tasks == 1

        history.record(ModelExecutionRecord(
            model_id="c", outcome=TaskOutcome.FAILURE,
        ))
        p2 = history.get_profile("c")
        assert p2.overall.total_tasks == 2

    def test_get_all_profiles(self, tmp_path):
        db = tmp_path / "test.db"
        history = EmpiricalHistory(db)

        history.record(ModelExecutionRecord(model_id="x"))
        history.record(ModelExecutionRecord(model_id="y"))

        profiles = history.get_all_profiles()
        assert len(profiles) == 2
        assert "x" in profiles
        assert "y" in profiles


class TestEvidenceSource:
    """Test evidence provenance."""

    def test_all_sources(self):
        sources = [
            EvidenceSource.PROVIDER_DECLARED,
            EvidenceSource.HARNESS_STATIC,
            EvidenceSource.HARNESS_BENCHMARKED,
            EvidenceSource.REAL_WORLD_OBSERVED,
        ]
        assert len(sources) == 4

    def test_source_values(self):
        assert EvidenceSource.REAL_WORLD_OBSERVED.value == "real_world_observed"
        assert EvidenceSource.HARNESS_BENCHMARKED.value == "harness_benchmarked"


class TestTaskPerformance:
    """Test task performance dataclass."""

    def test_to_dict(self):
        perf = TaskPerformance(
            task_type="bug_fix",
            total_tasks=10,
            success_rate=0.9,
            sample_confidence=SampleConfidence.LOW,
        )
        d = perf.to_dict()
        assert d["task_type"] == "bug_fix"
        assert d["total_tasks"] == 10
        assert d["success_rate"] == 0.9
        assert d["sample_confidence"] == "low"


class TestModelEmpiricalProfile:
    """Test empirical profile dataclass."""

    def test_to_dict(self):
        profile = ModelEmpiricalProfile(
            model_id="test",
            total_records=50,
            primary_source=EvidenceSource.REAL_WORLD_OBSERVED,
        )
        d = profile.to_dict()
        assert d["model_id"] == "test"
        assert d["total_records"] == 50
        assert d["primary_source"] == "real_world_observed"
