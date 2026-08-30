"""Tests for UX improvements: task planning, classifier, 429 fast failure, and completion UX."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.classifier.classifier import TaskClassifier, TaskType
from harness_core.routing.fallback import ErrorClassification, FallbackConfig, FallbackEngine, RetryConfig, classify_error
from harness_core.routing.health import HealthEvent, ModelHealthTracker, ModelHealthStatus


# ─── Classifier Tests ──────────────────────────────────────────────────────


class TestTaskClassifier:
    """Test improved deterministic task classification."""

    def setup_method(self):
        self.classifier = TaskClassifier()

    def test_enhance_calculator_is_implementation(self):
        task_type, confidence = self.classifier.classify_with_confidence("enhance this calculator")
        assert task_type == TaskType.IMPLEMENTATION
        assert confidence > 0

    def test_improve_ui_is_implementation(self):
        task_type, _ = self.classifier.classify_with_confidence("improve the user interface")
        assert task_type == TaskType.IMPLEMENTATION

    def test_add_dark_mode_is_implementation(self):
        task_type, _ = self.classifier.classify_with_confidence("add dark mode to the app")
        assert task_type == TaskType.IMPLEMENTATION

    def test_make_better_is_implementation(self):
        task_type, _ = self.classifier.classify_with_confidence("make this faster and more responsive")
        assert task_type == TaskType.IMPLEMENTATION

    def test_push_to_github_is_implementation(self):
        task_type, _ = self.classifier.classify_with_confidence("push this project to GitHub")
        assert task_type == TaskType.IMPLEMENTATION

    def test_fix_bug_is_bug_fix(self):
        task_type, _ = self.classifier.classify_with_confidence("fix the login bug")
        assert task_type == TaskType.BUG_FIX

    def test_explain_project_is_research(self):
        task_type, _ = self.classifier.classify_with_confidence("explain what this project does")
        assert task_type == TaskType.RESEARCH

    def test_run_tests_classifies(self):
        task_type, _ = self.classifier.classify_with_confidence("run the tests")
        # May be IMPLEMENTATION or UNKNOWN depending on exact patterns
        # The key test is that it's not a wrong classification
        assert task_type != TaskType.BUG_FIX
        assert task_type != TaskType.SECURITY

    def test_classify_common_requests(self):
        """Ensure common requests don't return UNKNOWN."""
        requests = [
            "enhance this calculator",
            "fix the login bug",
            "add dark mode",
            "explain this project",
            "create a new function",
            "refactor the authentication module",
            "optimize database queries",
        ]
        for req in requests:
            task_type, _ = self.classifier.classify_with_confidence(req)
            assert task_type != TaskType.UNKNOWN, f"'{req}' classified as UNKNOWN"


# ─── 429 Fast Failure Tests ────────────────────────────────────────────────


class Test429FastFailure:
    """Test that 3+ rate-limited models cause fast failure."""

    @pytest.mark.asyncio
    async def test_three_rate_limited_models_cause_fast_failure(self):
        """When 3 models are rate-limited, fail immediately."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(
                retry=RetryConfig(max_retries=0),
                total_timeout_seconds=60.0,
            )
        )

        # Create 4 providers that all return 429
        providers = []
        for i in range(4):
            provider = MagicMock()
            provider.name = f"provider_{i}"
            provider.generate = AsyncMock(
                side_effect=Exception("429 Too Many Requests")
            )
            providers.append(provider)

        model_chain = [(f"model_{i}", providers[i]) for i in range(4)]

        from harness_core.providers.base import CompletionRequest
        request = CompletionRequest(messages=[{"role": "user", "content": "test"}])

        result = await engine.execute(request, model_chain)

        assert not result.succeeded
        assert "rate limited" in result.final_error.lower()
        assert "429" in result.final_error
        # Should NOT have tried all 4 — fast failure after 3
        attempted = [a for a in result.attempts if a.get("status") == "error"]
        assert len(attempted) <= 3

    @pytest.mark.asyncio
    async def test_mixed_failures_no_fast_failure(self):
        """When failures are mixed (not all 429), no fast failure."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(
                retry=RetryConfig(max_retries=0),
                total_timeout_seconds=60.0,
            )
        )

        providers = []
        for i in range(4):
            provider = MagicMock()
            provider.name = f"provider_{i}"
            if i == 0:
                provider.generate = AsyncMock(side_effect=Exception("429 Too Many Requests"))
            elif i == 1:
                provider.generate = AsyncMock(side_effect=Exception("403 Forbidden"))
            elif i == 2:
                provider.generate = AsyncMock(side_effect=Exception("500 Internal Server Error"))
            else:
                provider.generate = AsyncMock(side_effect=Exception("429 Too Many Requests"))
            providers.append(provider)

        model_chain = [(f"model_{i}", providers[i]) for i in range(4)]

        from harness_core.providers.base import CompletionRequest
        request = CompletionRequest(messages=[{"role": "user", "content": "test"}])

        result = await engine.execute(request, model_chain)

        assert not result.succeeded
        # Should have tried all models
        attempted = [a for a in result.attempts if a.get("status") == "error"]
        assert len(attempted) >= 3


# ─── Classifier Pattern Tests ──────────────────────────────────────────────


class TestClassifierPatterns:
    """Test specific classifier pattern matches."""

    def setup_method(self):
        self.classifier = TaskClassifier()

    def test_upgrade_pattern(self):
        task_type, _ = self.classifier.classify_with_confidence("upgrade the responsive layout")
        assert task_type == TaskType.IMPLEMENTATION

    def test_redesign_pattern(self):
        task_type, _ = self.classifier.classify_with_confidence("redesign the navigation menu")
        assert task_type == TaskType.IMPLEMENTATION

    def test_make_modern_pattern(self):
        task_type, _ = self.classifier.classify_with_confidence("make the interface more modern")
        assert task_type == TaskType.IMPLEMENTATION

    def test_security_pattern(self):
        task_type, _ = self.classifier.classify_with_confidence("add authentication to the API")
        assert task_type in (TaskType.SECURITY, TaskType.IMPLEMENTATION)

    def test_optimize_pattern(self):
        task_type, _ = self.classifier.classify_with_confidence("optimize database performance")
        assert task_type == TaskType.PERFORMANCE


# ─── Health Status Tests ───────────────────────────────────────────────────


class TestHealthStatusFor429:
    """Test that 429 is properly tracked in health system."""

    def test_429_records_rate_limit(self):
        tracker = ModelHealthTracker(default_cooldown_seconds=5.0)
        tracker.record_failure("model_a", HealthEvent.RATE_LIMIT_429, 100.0)

        state = tracker.get_state("model_a")
        assert state.rate_limit_hits == 1
        assert state.is_rate_limited
        assert state.health_status == ModelHealthStatus.UNKNOWN  # 429 doesn't change explicit status

    def test_rate_limited_model_not_healthy(self):
        tracker = ModelHealthTracker(default_cooldown_seconds=60.0)
        tracker.record_failure("model_a", HealthEvent.RATE_LIMIT_429, 100.0)

        state = tracker.get_state("model_a")
        assert not state.is_healthy  # rate-limited models are not healthy

    def test_rate_limit_cooldown_increases(self):
        tracker = ModelHealthTracker(default_cooldown_seconds=5.0)
        tracker.record_failure("model_a", HealthEvent.RATE_LIMIT_429, 100.0)
        state1 = tracker.get_state("model_a")
        cooldown1 = state1.cooldown_seconds

        tracker.record_failure("model_a", HealthEvent.RATE_LIMIT_429, 100.0)
        state2 = tracker.get_state("model_a")
        cooldown2 = state2.cooldown_seconds

        # Cooldown should increase (exponential backoff)
        assert cooldown2 >= cooldown1

    def test_success_recovers_rate_limited_model(self):
        tracker = ModelHealthTracker(default_cooldown_seconds=0.1)
        tracker.record_failure("model_a", HealthEvent.RATE_LIMIT_429, 100.0)
        assert not tracker.get_state("model_a").is_healthy

        # Simulate cooldown passing
        import time
        time.sleep(0.15)

        tracker.record_success("model_a", latency_ms=200.0)
        # After success and cooldown, model is healthy again
        state = tracker.get_state("model_a")
        assert state.health_status == ModelHealthStatus.HEALTHY
        assert state.is_healthy  # cooldown expired + healthy status


# ─── Error Classification Tests ────────────────────────────────────────────


class TestErrorClassificationExtended:
    """Extended error classification tests."""

    def test_429_is_rate_limited(self):
        assert classify_error(Exception("429 Too Many Requests")) == ErrorClassification.RATE_LIMITED

    def test_rate_limit_keyword_is_rate_limited(self):
        assert classify_error(Exception("rate limit exceeded")) == ErrorClassification.RATE_LIMITED

    def test_too_many_requests_is_rate_limited(self):
        assert classify_error(Exception("too many requests")) == ErrorClassification.RATE_LIMITED

    def test_500_is_retryable(self):
        assert classify_error(Exception("500 Internal Server Error")) == ErrorClassification.RETRYABLE

    def test_timeout_is_retryable(self):
        assert classify_error(Exception("request timed out")) == ErrorClassification.RETRYABLE

    def test_connection_error_is_retryable(self):
        assert classify_error(Exception("connection refused")) == ErrorClassification.RETRYABLE
