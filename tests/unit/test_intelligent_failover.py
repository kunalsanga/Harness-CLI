"""Intelligent Multi-Provider Failover Tests.

Simulates and verifies:
1. OpenRouter succeeds (happy path).
2. OpenRouter returns HTTP 429 rate limit.
3. OpenRouter credits are exhausted (402 Payment Required).
4. OpenRouter authentication fails (401 Unauthorized).
5. OpenAI succeeds after OpenRouter fails.
6. OpenAI fails (e.g. quota exhausted) and NVIDIA NIM succeeds.
7. NVIDIA NIM fails (e.g. timeout) and FreeLLMAPI succeeds.
8. All configured providers fail -> informative exhaustion error.
9. A previously failed provider recovers after cooldown.
10. A disabled provider is skipped in the failover sequence.
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from harness_core.observability.events import EventBus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.routing.fallback import (
    FallbackConfig,
    FallbackEngine,
    FallbackResult,
)
from harness_core.routing.health import ModelHealthTracker
from harness_core.routing.router import ModelRouter, RouterConfig


class MockProvider(ModelProvider):
    """Mock provider for deterministic failover testing."""

    def __init__(self, name: str, models: list[str], failure_mode: str | None = None) -> None:
        self._name = name
        self._models = models
        self.failure_mode = failure_mode  # None, "429", "401", "402", "timeout", "500"
        self.generate_calls: list[str] = []
        self.is_healthy_flag = True

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or self._models[0]
        self.generate_calls.append(model)

        if self.failure_mode == "429":
            raise ProviderRequestError(
                ProviderErrorCategory.RATE_LIMIT,
                f"Rate limit exceeded on {self.name} (free model limit reached)",
                provider=self.name,
                model=model,
                status_code=429,
                retry_after=60.0,
            )
        elif self.failure_mode == "402":
            raise ProviderRequestError(
                ProviderErrorCategory.PAYMENT_REQUIRED,
                f"Credits exhausted on {self.name} (insufficient credits)",
                provider=self.name,
                model=model,
                status_code=402,
            )
        elif self.failure_mode == "401":
            raise ProviderRequestError(
                ProviderErrorCategory.AUTHENTICATION,
                f"Invalid API key on {self.name} (unauthorized)",
                provider=self.name,
                model=model,
                status_code=401,
            )
        elif self.failure_mode == "timeout":
            raise ProviderRequestError(
                ProviderErrorCategory.TIMEOUT,
                f"Request timed out on {self.name}",
                provider=self.name,
                model=model,
            )
        elif self.failure_mode == "500":
            raise ProviderRequestError(
                ProviderErrorCategory.PROVIDER,
                f"Internal Server Error on {self.name}",
                provider=self.name,
                model=model,
                status_code=500,
            )

        return CompletionResponse(
            content=f"Response from {self.name} using {model}",
            model=model,
            provider=self.name,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    async def stream(self, request: CompletionRequest):
        res = await self.generate(request)
        yield res.content

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=m,
                name=m,
                provider=self.name,
                context_window=32768,
                supports_tools=True,
                is_free=self.name in ("openrouter", "freellmapi"),
            )
            for m in self._models
        ]

    async def health_check(self) -> bool:
        return self.is_healthy_flag and self.failure_mode != "401"


# ─── 1. OpenRouter succeeds (Happy Path) ────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_1_openrouter_succeeds():
    p_openrouter = MockProvider("openrouter", ["openrouter/free"])
    p_openai = MockProvider("openai", ["gpt-4o-mini"])

    chain = [("openrouter/free", p_openrouter), ("gpt-4o-mini", p_openai)]
    engine = FallbackEngine(health_tracker=ModelHealthTracker(), event_bus=EventBus())

    req = CompletionRequest(messages=[{"role": "user", "content": "Hi"}])
    res = await engine.execute(req, chain)

    assert res.succeeded is True
    assert res.provider_used == "openrouter"
    assert res.model_used == "openrouter/free"
    assert p_openrouter.generate_calls == ["openrouter/free"]
    assert len(p_openai.generate_calls) == 0


# ─── 2. OpenRouter returns 429 Rate Limit -> OpenAI succeeds ─────────────────────

@pytest.mark.asyncio
async def test_scenario_2_openrouter_429_failover_to_openai():
    p_openrouter = MockProvider("openrouter", ["openrouter/free"], failure_mode="429")
    p_openai = MockProvider("openai", ["gpt-4o-mini"])

    chain = [("openrouter/free", p_openrouter), ("gpt-4o-mini", p_openai)]
    engine = FallbackEngine(health_tracker=ModelHealthTracker(), event_bus=EventBus())

    req = CompletionRequest(messages=[{"role": "user", "content": "Hi"}])
    res = await engine.execute(req, chain)

    assert res.succeeded is True
    assert res.provider_used == "openai"
    assert res.model_used == "gpt-4o-mini"
    assert "Response from openai" in res.response.content


# ─── 3. OpenRouter Credits Exhausted (402) -> OpenAI succeeds ────────────────────

@pytest.mark.asyncio
async def test_scenario_3_openrouter_credits_exhausted_failover():
    p_openrouter = MockProvider("openrouter", ["openrouter/free"], failure_mode="402")
    p_openai = MockProvider("openai", ["gpt-4o-mini"])

    chain = [("openrouter/free", p_openrouter), ("gpt-4o-mini", p_openai)]
    engine = FallbackEngine(health_tracker=ModelHealthTracker(), event_bus=EventBus())

    req = CompletionRequest(messages=[{"role": "user", "content": "Build feature"}])
    res = await engine.execute(req, chain)

    assert res.succeeded is True
    assert res.provider_used == "openai"
    assert res.model_used == "gpt-4o-mini"


# ─── 4. OpenRouter Auth Fails (401) -> OpenAI succeeds ───────────────────────────

@pytest.mark.asyncio
async def test_scenario_4_openrouter_auth_fails_failover_to_openai():
    p_openrouter = MockProvider("openrouter", ["openrouter/free"], failure_mode="401")
    p_openai = MockProvider("openai", ["gpt-4o-mini"])

    chain = [("openrouter/free", p_openrouter), ("gpt-4o-mini", p_openai)]
    events: list[dict] = []
    bus = EventBus()
    async def on_failover(e):
        events.append(e.data)
    bus.on("model.failover", on_failover)
    engine = FallbackEngine(health_tracker=ModelHealthTracker(), event_bus=bus)

    req = CompletionRequest(messages=[{"role": "user", "content": "Implement auth"}])
    res = await engine.execute(req, chain)

    assert res.succeeded is True
    assert res.provider_used == "openai"
    assert res.model_used == "gpt-4o-mini"
    # Verify failover event was emitted
    assert any(e.get("to_provider") == "openai" for e in events)


# ─── 5. Full Chain: OpenRouter -> OpenAI -> NVIDIA NIM -> FreeLLMAPI ─────────────

@pytest.mark.asyncio
async def test_scenario_5_openrouter_and_openai_fail_nvidia_succeeds():
    p_openrouter = MockProvider("openrouter", ["openrouter/free"], failure_mode="429")
    p_openai = MockProvider("openai", ["gpt-4o-mini"], failure_mode="402")
    p_nvidia = MockProvider("nvidia", ["meta/llama-3.1-70b-instruct"])
    p_freellm = MockProvider("freellmapi", ["auto"])

    chain = [
        ("openrouter/free", p_openrouter),
        ("gpt-4o-mini", p_openai),
        ("meta/llama-3.1-70b-instruct", p_nvidia),
        ("auto", p_freellm),
    ]
    engine = FallbackEngine(health_tracker=ModelHealthTracker(), event_bus=EventBus())

    req = CompletionRequest(messages=[{"role": "user", "content": "Continue work"}])
    res = await engine.execute(req, chain)

    assert res.succeeded is True
    assert res.provider_used == "nvidia"
    assert res.model_used == "meta/llama-3.1-70b-instruct"


@pytest.mark.asyncio
async def test_scenario_6_openrouter_openai_nvidia_fail_freellmapi_succeeds():
    p_openrouter = MockProvider("openrouter", ["openrouter/free"], failure_mode="429")
    p_openai = MockProvider("openai", ["gpt-4o-mini"], failure_mode="401")
    p_nvidia = MockProvider("nvidia", ["meta/llama-3.1-70b-instruct"], failure_mode="timeout")
    p_freellm = MockProvider("freellmapi", ["auto"])

    chain = [
        ("openrouter/free", p_openrouter),
        ("gpt-4o-mini", p_openai),
        ("meta/llama-3.1-70b-instruct", p_nvidia),
        ("auto", p_freellm),
    ]
    engine = FallbackEngine(health_tracker=ModelHealthTracker(), event_bus=EventBus())

    req = CompletionRequest(messages=[{"role": "user", "content": "Local fallback"}])
    res = await engine.execute(req, chain)

    assert res.succeeded is True
    assert res.provider_used == "freellmapi"
    assert res.model_used == "auto"
    assert "Response from freellmapi" in res.response.content


# ─── 7. All Providers Fail ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_7_all_providers_fail():
    p_openrouter = MockProvider("openrouter", ["openrouter/free"], failure_mode="429")
    p_openai = MockProvider("openai", ["gpt-4o-mini"], failure_mode="402")
    p_nvidia = MockProvider("nvidia", ["meta/llama-3.1-70b-instruct"], failure_mode="500")
    p_freellm = MockProvider("freellmapi", ["auto"], failure_mode="timeout")

    chain = [
        ("openrouter/free", p_openrouter),
        ("gpt-4o-mini", p_openai),
        ("meta/llama-3.1-70b-instruct", p_nvidia),
        ("auto", p_freellm),
    ]
    cfg = FallbackConfig(retry=MagicMock(max_retries=0))
    engine = FallbackEngine(health_tracker=ModelHealthTracker(), fallback_config=cfg, event_bus=EventBus())

    req = CompletionRequest(messages=[{"role": "user", "content": "Fail"}])
    res = await engine.execute(req, chain)

    assert res.succeeded is False
    assert res.final_error != ""


# ─── 8. Provider Recovery After Cooldown ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_8_provider_recovery():
    health = ModelHealthTracker()
    p_openrouter = MockProvider("openrouter", ["openrouter/free"])
    chain = [("openrouter/free", p_openrouter)]

    # Initial success
    engine = FallbackEngine(health_tracker=health, event_bus=EventBus())
    res1 = await engine.execute(CompletionRequest(messages=[{"role": "user", "content": "1"}]), chain)
    assert res1.succeeded is True

    # Temporary failure
    p_openrouter.failure_mode = "500"
    cfg = FallbackConfig(retry=MagicMock(max_retries=0))
    engine_fail = FallbackEngine(health_tracker=health, fallback_config=cfg, event_bus=EventBus())
    res2 = await engine_fail.execute(CompletionRequest(messages=[{"role": "user", "content": "2"}]), chain)
    assert res2.succeeded is False

    # Simulate recovery & cooldown expiry
    health.reset("openrouter/free")
    p_openrouter.failure_mode = None
    res3 = await engine.execute(CompletionRequest(messages=[{"role": "user", "content": "3"}]), chain)
    assert res3.succeeded is True
    assert res3.provider_used == "openrouter"


# ─── 9. Disabled Provider Skipped ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_scenario_9_disabled_provider_skipped_in_factory():
    from unittest.mock import patch
    from harness_core.providers.factory import create_providers

    config = {
        "providers": {
            "openrouter": {"enabled": False},
            "openai": {"enabled": True},
        }
    }
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        providers = create_providers(config)
        names = [p.name for p in providers]
        assert "openrouter" not in names
        assert "openai" in names
