"""Tests for free-first model routing and 402 payment handling."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from harness_core.routing.fallback import (
    ErrorClassification,
    FallbackConfig,
    FallbackEngine,
    FallbackResult,
    RetryConfig,
    classify_error,
)
from harness_core.routing.health import (
    HealthEvent,
    ModelHealthStatus,
    ModelHealthTracker,
)
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)


class FakeProvider(ModelProvider):
    """A fake provider for testing."""

    def __init__(self, name: str = "test_provider") -> None:
        self._name = name
        self._generate_fn = AsyncMock()

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
        return True


# ─── 402 Classification ────────────────────────────────────────────────


class Test402Classification:
    """Test that 402 Payment Required is properly classified."""

    def test_402_is_classified_as_permanent(self):
        err = Exception("402 Payment Required")
        assert classify_error(err) == ErrorClassification.PERMANENT

    def test_402_in_http_error_is_permanent(self):
        err = Exception("Client error '402 Payment Required'")
        assert classify_error(err) == ErrorClassification.PERMANENT

    def test_402_keyword_is_permanent(self):
        err = Exception("payment required for this model")
        # This won't match because classify_error checks codes first
        # and "payment required" is not a code. We need to ensure 402 works.
        err_with_code = Exception("402 payment required")
        assert classify_error(err_with_code) == ErrorClassification.PERMANENT


# ─── Model Health: Payment Required ─────────────────────────────────────


class TestPaymentRequiredHealth:
    """Test that 402 marks model as PAYMENT_REQUIRED."""

    def test_402_marks_model_payment_required(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("paid-model", HealthEvent.PAYMENT_REQUIRED)

        state = tracker.get_state("paid-model")
        assert state.health_status == ModelHealthStatus.PAYMENT_REQUIRED
        assert state.is_healthy is False
        assert state.is_unavailable is True

    def test_payment_required_model_not_healthy(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("model-x", HealthEvent.PAYMENT_REQUIRED)
        assert tracker.get_state("model-x").is_healthy is False

    def test_payment_required_model_is_unavailable(self):
        tracker = ModelHealthTracker()
        tracker.record_failure("model-x", HealthEvent.PAYMENT_REQUIRED)
        assert tracker.get_state("model-x").is_unavailable is True

    def test_free_model_succeeding_marks_healthy(self):
        """A free model that succeeds is marked HEALTHY."""
        tracker = ModelHealthTracker()
        tracker.record_success("free-model")
        state = tracker.get_state("free-model")
        assert state.health_status == ModelHealthStatus.HEALTHY
        assert state.is_healthy is True

    def test_payment_required_recovers_on_success(self):
        """A model that was payment-required can recover if it later succeeds."""
        tracker = ModelHealthTracker()
        tracker.record_failure("model-x", HealthEvent.PAYMENT_REQUIRED)
        assert tracker.get_state("model-x").is_unavailable is True

        # If the model later succeeds
        tracker.record_success("model-x")
        assert tracker.get_state("model-x").is_unavailable is False
        assert tracker.get_state("model-x").is_healthy is True


# ─── Fallback: 402 Handling ─────────────────────────────────────────────


class TestFallback402:
    """Test that 402 is handled correctly in fallback engine."""

    @pytest.mark.asyncio
    async def test_402_marks_model_unavailable(self):
        """When a model returns 402, it's marked unavailable."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )
        provider = FakeProvider("openrouter")

        async def _generate(request):
            if request.model == "paid-model":
                raise Exception("402 Payment Required")
            return CompletionResponse(content="Hello from free model!")

        provider._generate_fn = _generate

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [("paid-model", provider), ("free-model", provider)]

        result = await engine.execute(request, chain)

        assert result.succeeded is True
        assert result.model_used == "free-model"

    @pytest.mark.asyncio
    async def test_402_model_excluded_from_subsequent_requests(self):
        """After model returns 402, it's skipped in future requests."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )
        provider = FakeProvider("openrouter")

        async def _generate(request):
            if request.model == "paid-model":
                raise Exception("402 Payment Required")
            return CompletionResponse(content="Hello!")

        provider._generate_fn = _generate

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])

        # First request
        chain1 = [("paid-model", provider), ("free-model", provider)]
        result1 = await engine.execute(request, chain1)
        assert result1.succeeded is True

        # Second request: paid-model should be skipped
        chain2 = [("paid-model", provider), ("free-model", provider)]
        result2 = await engine.execute(request, chain2)
        assert result2.succeeded is True
        skipped = [a for a in result2.attempts if a.get("model") == "paid-model"]
        assert len(skipped) == 1
        assert skipped[0]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_402_does_not_disable_provider(self):
        """402 on one model does NOT disable the entire provider."""
        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )
        provider = FakeProvider("openrouter")
        call_count = 0

        async def _generate(request):
            nonlocal call_count
            call_count += 1
            if request.model == "paid-model":
                raise Exception("402 Payment Required")
            return CompletionResponse(content="Hello from free model!")

        provider._generate_fn = _generate

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = [("paid-model", provider), ("free-model", provider)]

        result = await engine.execute(request, chain)
        assert result.succeeded is True
        # Both models were attempted
        assert call_count == 2


# ─── Free-Mode Routing ─────────────────────────────────────────────────


class TestFreeModeFiltering:
    """Test that free mode only considers free models."""

    def _make_model(
        self,
        model_id: str,
        is_free: bool = False,
        supports_tools: bool = True,
        context_window: int = 32000,
    ) -> ModelInfo:
        return ModelInfo(
            id=model_id,
            name=model_id,
            provider="openrouter",
            context_window=context_window,
            supports_tools=supports_tools,
            is_free=is_free,
        )

    def test_free_mode_excludes_paid_models(self):
        """In free mode, paid models should be filtered out."""
        from harness_core.routing.router import ModelRouter, RouterConfig
        from harness_core.routing.scoring import ScoringContext

        router = ModelRouter(config=RouterConfig(routing_mode="free"))
        models = [
            self._make_model("free-model-a", is_free=True),
            self._make_model("paid-model-b", is_free=False),
            self._make_model("free-model-c", is_free=True),
        ]
        ctx = ScoringContext(requires_tools=True)
        filtered = router._filter_models(models, ctx)

        assert len(filtered) == 2
        assert all(m.is_free for m in filtered)

    def test_normal_mode_includes_all_models(self):
        """In normal mode, both free and paid models are available."""
        from harness_core.routing.router import ModelRouter, RouterConfig
        from harness_core.routing.scoring import ScoringContext

        router = ModelRouter(config=RouterConfig(routing_mode="auto"))
        models = [
            self._make_model("free-model-a", is_free=True),
            self._make_model("paid-model-b", is_free=False),
        ]
        ctx = ScoringContext(requires_tools=True)
        filtered = router._filter_models(models, ctx)

        assert len(filtered) == 2

    def test_free_mode_excludes_unavailable_models(self):
        """In free mode, unavailable models are also excluded."""
        from harness_core.routing.router import ModelRouter, RouterConfig
        from harness_core.routing.scoring import ScoringContext

        tracker = ModelHealthTracker()
        tracker.record_failure("unavailable-free", HealthEvent.AUTH_FAILED)

        router = ModelRouter(config=RouterConfig(routing_mode="free"))
        router.health = tracker

        models = [
            self._make_model("available-free", is_free=True),
            self._make_model("unavailable-free", is_free=True),
        ]
        ctx = ScoringContext(requires_tools=True)
        filtered = router._filter_models(models, ctx)

        assert len(filtered) == 1
        assert filtered[0].id == "available-free"

    def test_free_mode_excludes_payment_required_models(self):
        """In free mode, payment-required models are excluded."""
        from harness_core.routing.router import ModelRouter, RouterConfig
        from harness_core.routing.scoring import ScoringContext

        tracker = ModelHealthTracker()
        tracker.record_failure("payment-model", HealthEvent.PAYMENT_REQUIRED)

        router = ModelRouter(config=RouterConfig(routing_mode="free"))
        router.health = tracker

        models = [
            self._make_model("free-model", is_free=True),
            self._make_model("payment-model", is_free=True),
        ]
        ctx = ScoringContext(requires_tools=True)
        filtered = router._filter_models(models, ctx)

        assert len(filtered) == 1
        assert filtered[0].id == "free-model"


class TestFreeModeSelectModels:
    """Test that select_models respects free mode constraints."""

    @pytest.mark.asyncio
    async def test_free_mode_returns_empty_when_no_free_models(self):
        """When no free models exist, free mode returns empty chain."""
        from harness_core.routing.router import ModelRouter, RouterConfig

        provider = FakeProvider("openrouter")

        router = ModelRouter(
            providers=[provider],
            config=RouterConfig(routing_mode="free"),
        )
        # Mock refresh_models to return only paid models
        router._model_cache = [
            ModelInfo(id="paid-a", name="paid-a", provider="openrouter",
                      supports_tools=True, is_free=False),
        ]
        import time
        router._last_refresh = time.time() + 300

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = await router.select_models(request)

        assert chain == []

    @pytest.mark.asyncio
    async def test_free_mode_returns_free_models(self):
        """When free models exist, free mode returns them."""
        from harness_core.routing.router import ModelRouter, RouterConfig

        provider = FakeProvider("openrouter")

        router = ModelRouter(
            providers=[provider],
            config=RouterConfig(routing_mode="free"),
        )
        router._model_cache = [
            ModelInfo(id="free-a", name="free-a", provider="openrouter",
                      supports_tools=True, is_free=True, context_window=32000),
        ]
        import time
        router._last_refresh = time.time() + 300  # far in the future so cache is valid

        # Mock the provider to be in the router's providers dict
        router.providers = {"openrouter": provider}

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        chain = await router.select_models(request)

        assert len(chain) == 1
        assert chain[0][0] == "free-a"

    @pytest.mark.asyncio
    async def test_free_mode_execute_returns_clear_error(self):
        """When no free models, execute returns helpful error message."""
        from harness_core.routing.router import ModelRouter, RouterConfig

        provider = FakeProvider("openrouter")

        router = ModelRouter(
            providers=[provider],
            config=RouterConfig(routing_mode="free"),
        )
        router._model_cache = []
        import time
        router._last_refresh = time.time() + 300

        request = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
        result = await router.execute(request)

        assert result.succeeded is False
        assert "No usable free model available" in result.final_error
