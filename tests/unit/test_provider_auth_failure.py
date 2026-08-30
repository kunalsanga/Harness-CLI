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


class TestProviderAuthFailureDetection:
    """Test that provider-level auth failures are detected and stop retry storms."""

    @pytest.mark.asyncio
    async def test_403_stops_all_models_from_same_provider(self):
        """When one model returns 403, all models from the same provider should be skipped."""
        engine = FallbackEngine()

        # Create two providers - one with auth failure, one healthy
        bad_provider = FakeProvider("openrouter")
        good_provider = FakeProvider("anthropic")

        bad_provider._generate_fn = AsyncMock(
            side_effect=Exception("Client error '403 Forbidden'")
        )
        good_provider._generate_fn = AsyncMock(
            return_value=CompletionResponse(content="Hello!")
        )

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("model-a", bad_provider),
            ("model-b", bad_provider),  # Same provider - should be skipped
            ("model-c", good_provider),  # Different provider - should be tried
        ]

        result = await engine.execute(request, chain)

        assert result.succeeded is True
        assert result.model_used == "model-c"
        assert result.provider_used == "anthropic"

        # bad_provider.generate was called only once (for model-a)
        assert bad_provider._generate_fn.call_count == 1
        # good_provider.generate was called once (for model-c)
        assert good_provider._generate_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_401_stops_all_models_from_same_provider(self):
        """401 Unauthorized also triggers provider-level auth failure."""
        engine = FallbackEngine()

        bad_provider = FakeProvider("openrouter")
        bad_provider._generate_fn = AsyncMock(
            side_effect=Exception("401 Unauthorized")
        )

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("model-a", bad_provider),
            ("model-b", bad_provider),
            ("model-c", bad_provider),
        ]

        result = await engine.execute(request, chain)

        assert result.succeeded is False
        # Only first model should have been attempted
        assert bad_provider._generate_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_different_providers_not_affected_by_auth_failure(self):
        """Auth failure on one provider doesn't affect other providers."""
        engine = FallbackEngine()

        bad_provider = FakeProvider("openrouter")
        good_provider = FakeProvider("anthropic")

        bad_provider._generate_fn = AsyncMock(
            side_effect=Exception("403 Forbidden")
        )
        good_provider._generate_fn = AsyncMock(
            return_value=CompletionResponse(content="Hello!")
        )

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("model-a", bad_provider),
            ("model-b", good_provider),
        ]

        result = await engine.execute(request, chain)

        assert result.succeeded is True
        assert result.provider_used == "anthropic"

    @pytest.mark.asyncio
    async def test_server_errors_dont_trigger_provider_auth_failure(self):
        """Server errors (500) should NOT mark provider as auth-failed."""
        # max_retries=0 so each model is tried exactly once
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )

        provider = FakeProvider("openrouter")
        provider._generate_fn = AsyncMock(
            side_effect=Exception("500 Internal Server Error")
        )

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("model-a", provider),
            ("model-b", provider),
        ]

        result = await engine.execute(request, chain)

        # Both models should have been attempted (one call each)
        assert provider._generate_fn.call_count == 2
        # Provider should NOT be in auth failures
        assert not engine.is_provider_failed("openrouter")

    @pytest.mark.asyncio
    async def test_rate_limit_doesnt_trigger_provider_auth_failure(self):
        """Rate limits (429) should NOT mark provider as auth-failed."""
        engine = FallbackEngine()

        provider = FakeProvider("openrouter")
        provider._generate_fn = AsyncMock(
            side_effect=Exception("429 Rate Limited")
        )

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("model-a", provider),
            ("model-b", provider),
        ]

        result = await engine.execute(request, chain)

        assert not engine.is_provider_failed("openrouter")

    @pytest.mark.asyncio
    async def test_reset_provider_failures_clears_state(self):
        """reset_provider_failures clears the auth failure tracking."""
        engine = FallbackEngine()

        provider = FakeProvider("openrouter")
        provider._generate_fn = AsyncMock(
            side_effect=Exception("403 Forbidden")
        )

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [
            ("model-a", provider),
            ("model-b", provider),
        ]

        await engine.execute(request, chain)
        assert engine.is_provider_failed("openrouter")

        # Reset
        engine.reset_provider_failures()
        assert not engine.is_provider_failed("openrouter")


class TestFallbackResult:
    """Test fallback result properties."""

    def test_auth_failure_error_message(self):
        """Auth failure produces informative error message."""
        result = FallbackResult()
        result.attempts = [
            {"model": "model-a", "status": "error", "error": "Client error '403 Forbidden'"},
            {"model": "model-b", "status": "error", "error": "Client error '403 Forbidden'"},
        ]
        result.final_error = (
            "Provider authentication failed. 2 model(s) returned 401/403. "
            "Check your API key and provider permissions. "
            "Models: model-a, model-b"
        )

        assert "403" in result.final_error
        assert "authentication failed" in result.final_error.lower()
        assert "model-a" in result.final_error


class TestProviderAuthFailureException:
    """Test the ProviderAuthFailure exception."""

    def test_exception_attributes(self):
        exc = ProviderAuthFailure("openrouter", 403, "Forbidden")
        assert exc.provider_name == "openrouter"
        assert exc.status_code == 403
        assert exc.detail == "Forbidden"
        assert "openrouter" in str(exc)
        assert "403" in str(exc)
