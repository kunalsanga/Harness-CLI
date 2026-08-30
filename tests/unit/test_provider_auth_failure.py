"""Tests for provider auth failure detection and fallback behavior."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from harness_core.routing.fallback import (
    ErrorClassification,
    FallbackConfig,
    FallbackEngine,
    FallbackResult,
    ProviderAuthFailure,
    RetryConfig,
    classify_error,
)
from harness_core.routing.health import HealthEvent, ModelHealthTracker
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
)


class FakeProvider(ModelProvider):
    """A fake provider for testing."""

    def __init__(self, name: str = "test_provider") -> None:
        self._name = name
        self._generate_fn = AsyncMock()
        self._health_ok = True

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        return await self._generate_fn(request)

    async def stream(self, request: CompletionRequest):
        yield ""

    async def list_models(self):
        return []

    async def health_check(self) -> bool:
        return self._health_ok


class TestErrorClassification:
    """Test error classification for retry decisions."""

    def test_rate_limit_is_rate_limited(self):
        err = Exception("429 Too Many Requests")
        assert classify_error(err) == ErrorClassification.RATE_LIMITED

    def test_401_is_permanent(self):
        err = Exception("401 Unauthorized")
        assert classify_error(err) == ErrorClassification.PERMANENT

    def test_403_is_permanent(self):
        err = Exception("403 Forbidden")
        assert classify_error(err) == ErrorClassification.PERMANENT

    def test_forbidden_keyword_is_permanent(self):
        err = Exception("Client error '403 Forbidden'")
        assert classify_error(err) == ErrorClassification.PERMANENT

    def test_timeout_is_retryable(self):
        err = Exception("Connection timed out")
        assert classify_error(err) == ErrorClassification.RETRYABLE

    def test_server_error_is_retryable(self):
        err = Exception("500 Internal Server Error")
        assert classify_error(err) == ErrorClassification.RETRYABLE

    def test_context_overflow_is_rate_limited(self):
        err = Exception("Context token limit exceeded")
        assert classify_error(err) == ErrorClassification.RATE_LIMITED


class TestModelLevelFailureDetection:
    """Test that model-level auth failures are handled correctly."""

    @pytest.mark.asyncio
    async def test_403_marks_only_that_model_unavailable(self):
        """When one model returns 403, only that model is marked unavailable."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )

        provider = FakeProvider("openrouter")
        # model-a fails with 403, model-b succeeds
        call_count = 0
        async def _generate(request):
            nonlocal call_count
            call_count += 1
            if request.model == "model-a":
                raise Exception("Client error '403 Forbidden'")
            return CompletionResponse(content="Hello from model-b!")

        provider._generate_fn = _generate

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("model-a", provider),
            ("model-b", provider),  # Same provider, should still be tried
        ]

        result = await engine.execute(request, chain)

        assert result.succeeded is True
        assert result.model_used == "model-b"
        # Both models were attempted
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_401_marks_only_that_model_unavailable(self):
        """401 Unauthorized marks only the specific model as unavailable."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )

        provider = FakeProvider("openrouter")
        async def _generate(request):
            if request.model == "model-a":
                raise Exception("401 Unauthorized")
            return CompletionResponse(content="Hello!")

        provider._generate_fn = _generate

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("model-a", provider),
            ("model-b", provider),
        ]

        result = await engine.execute(request, chain)

        assert result.succeeded is True
        assert result.model_used == "model-b"

    @pytest.mark.asyncio
    async def test_failed_model_excluded_from_subsequent_requests(self):
        """After model fails with 403, it's excluded from future fallback chains."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )

        provider = FakeProvider("openrouter")
        async def _generate(request):
            if request.model == "bad-model":
                raise Exception("Client error '403 Forbidden'")
            return CompletionResponse(content="Hello!")

        provider._generate_fn = _generate

        # First request: bad-model fails, good-model succeeds
        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain1 = [("bad-model", provider), ("good-model", provider)]
        result1 = await engine.execute(request, chain1)
        assert result1.succeeded is True

        # Second request: bad-model should be skipped automatically
        chain2 = [("bad-model", provider), ("good-model", provider)]
        result2 = await engine.execute(request, chain2)
        assert result2.succeeded is True
        assert result2.model_used == "good-model"
        # bad-model should have been skipped (status=skipped)
        bad_skipped = [a for a in result2.attempts if a.get("model") == "bad-model"]
        assert len(bad_skipped) == 1
        assert bad_skipped[0]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_same_provider_other_models_still_tried(self):
        """Auth failure on one OpenRouter model does NOT block other OpenRouter models."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )

        provider = FakeProvider("openrouter")
        call_count = 0

        async def _generate(request):
            nonlocal call_count
            call_count += 1
            if request.model == "meta/muse-spark":
                raise Exception("Client error '403 Forbidden'")
            return CompletionResponse(content="Hello from glm!")

        provider._generate_fn = _generate

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("meta/muse-spark", provider),
            ("z-ai/glm-5.3-flash", provider),  # Same provider, different model
        ]

        result = await engine.execute(request, chain)

        assert result.succeeded is True
        assert result.model_used == "z-ai/glm-5.3-flash"
        # Both models were attempted
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_does_not_mark_model_unavailable(self):
        """Rate limits (429) should NOT mark model as permanently unavailable."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )

        provider = FakeProvider("openrouter")
        async def _generate(request):
            if request.model == "model-a":
                raise Exception("429 Rate Limited")
            return CompletionResponse(content="Hello!")

        provider._generate_fn = _generate

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [("model-a", provider), ("model-b", provider)]
        result = await engine.execute(request, chain)
        assert result.succeeded is True

    @pytest.mark.asyncio
    async def test_server_errors_allow_other_models(self):
        """Server errors on one model allow other models to be tried."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )

        provider = FakeProvider("openrouter")
        async def _generate(request):
            if request.model == "model-a":
                raise Exception("500 Internal Server Error")
            return CompletionResponse(content="Hello!")

        provider._generate_fn = _generate

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [("model-a", provider), ("model-b", provider)]
        result = await engine.execute(request, chain)
        assert result.succeeded is True
        assert result.model_used == "model-b"


class TestModelHealthStatus:
    """Test model-level health status tracking."""

    def test_auth_failed_model_marked_unavailable(self):
        """Model that returns 403 is marked UNAVAILABLE."""
        from harness_core.routing.health import ModelHealthTracker, HealthEvent, ModelHealthStatus

        tracker = ModelHealthTracker()
        tracker.record_failure("bad-model", HealthEvent.AUTH_FAILED)

        state = tracker.get_state("bad-model")
        assert state.health_status == ModelHealthStatus.UNAVAILABLE
        assert state.is_unavailable is True
        assert state.is_healthy is False

    def test_successful_model_marked_healthy(self):
        """Model that succeeds is marked HEALTHY."""
        from harness_core.routing.health import ModelHealthTracker, ModelHealthStatus

        tracker = ModelHealthTracker()
        tracker.record_success("good-model")

        state = tracker.get_state("good-model")
        assert state.health_status == ModelHealthStatus.HEALTHY
        assert state.is_healthy is True
        assert state.is_unavailable is False

    def test_unknown_model_is_healthy(self):
        """Model with no data is UNKNOWN and considered healthy (neutral)."""
        from harness_core.routing.health import ModelHealthTracker, ModelHealthStatus

        tracker = ModelHealthTracker()
        state = tracker.get_state("new-model")
        assert state.health_status == ModelHealthStatus.UNKNOWN
        assert state.is_healthy is True

    def test_auth_failed_model_recovers_on_success(self):
        """Model that was marked unavailable can recover if it later succeeds."""
        from harness_core.routing.health import ModelHealthTracker, HealthEvent, ModelHealthStatus

        tracker = ModelHealthTracker()
        tracker.record_failure("model-x", HealthEvent.AUTH_FAILED)
        assert tracker.get_state("model-x").is_unavailable is True

        # If the model later succeeds (e.g., after provider fixes access)
        tracker.record_success("model-x")
        assert tracker.get_state("model-x").is_unavailable is False
        assert tracker.get_state("model-x").is_healthy is True

    def test_rate_limited_model_recovers_after_cooldown(self):
        """Rate-limited model recovers after cooldown period."""
        from harness_core.routing.health import ModelHealthTracker, HealthEvent

        tracker = ModelHealthTracker(default_cooldown_seconds=0.1)  # 100ms cooldown
        tracker.record_failure("model-y", HealthEvent.RATE_LIMIT_429)
        assert tracker.get_state("model-y").is_healthy is False

        # Wait for cooldown
        import time
        time.sleep(0.15)
        assert tracker.get_state("model-y").is_healthy is True


class TestProviderAuthFailureException:
    """Test the ProviderAuthFailure exception."""

    def test_exception_attributes(self):
        exc = ProviderAuthFailure("openrouter", 403, "Forbidden")
        assert exc.provider_name == "openrouter"
        assert exc.status_code == 403
        assert exc.detail == "Forbidden"
        assert "openrouter" in str(exc)
        assert "403" in str(exc)
