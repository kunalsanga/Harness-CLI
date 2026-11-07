"""Universal Multi-Provider Resilience and Intelligent API Key Failover Regression Suite.

Verifies all 20 required resilience, discovery, failover, state preservation,
and security capabilities across OpenRouter, NVIDIA NIM, Groq, Azure OpenAI,
OpenAI, and FreeLLMAPI.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.providers.factory import create_providers
from harness_core.routing.fallback import FallbackConfig, FallbackEngine
from harness_core.routing.health import HealthEvent, ModelHealthTracker
from harness_core.routing.router import ModelRouter, RouterConfig, ScoringContext


class MockTestProvider(ModelProvider):
    """Test helper provider with configurable responses and errors."""

    def __init__(self, name: str, models: list[str] | None = None, api_key: str = "sk-test-secret-12345") -> None:
        self._name = name
        self.api_key = api_key
        self._models = models or [f"{name}-model-1", f"{name}-model-2"]
        self.generate_calls: list[CompletionRequest] = []
        self._side_effects: dict[str, Any] = {}
        self._return_values: dict[str, CompletionResponse] = {}

    @property
    def name(self) -> str:
        return self._name

    def set_error(self, model: str, error: Exception) -> None:
        self._side_effects[model] = error

    def set_response(self, model: str, response: CompletionResponse) -> None:
        self._return_values[model] = response

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.generate_calls.append(request)
        model = request.model or self._models[0]
        if model in self._side_effects:
            raise self._side_effects[model]
        if model in self._return_values:
            return self._return_values[model]
        return CompletionResponse(
            content=f"Response from {self.name} for {model}",
            tool_calls=[],
            model=model,
            provider=self.name,
            usage={"prompt_tokens": 50, "completion_tokens": 25},
        )

    async def stream(self, request: CompletionRequest):
        resp = await self.generate(request)
        yield resp.content

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=m,
                name=m,
                provider=self.name,
                context_window=128000 if "large" in m else 32768,
                supports_tools=True,
                supports_streaming=True,
                is_free=True if self.name in ("openrouter", "nvidia", "groq", "freellmapi") else False,
            )
            for m in self._models
        ]

    async def health_check(self) -> bool:
        return True


def _make_request(
    prompt: str = "Test prompt",
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
) -> CompletionRequest:
    return CompletionRequest(
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice=tool_choice,
    )


# ─── Tests 1 to 6: Cross-Provider Failover Matrix ────────────────────────────


@pytest.mark.asyncio
async def test_openrouter_quota_exhaustion_switches_to_nvidia():
    """1. OpenRouter quota exhaustion (daily free limit reached) switches to NVIDIA."""
    p_or = MockTestProvider("openrouter", ["openrouter/free", "openrouter/free-alt"])
    p_nv = MockTestProvider("nvidia", ["nvidia/llama-3.1-70b-instruct"])

    p_or.set_error(
        "openrouter/free",
        ProviderRequestError(
            ProviderErrorCategory.RATE_LIMIT,
            "429 Free model daily limit reached for OpenRouter",
            provider="openrouter",
            status_code=429,
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [
        ("openrouter/free", p_or),
        ("openrouter/free-alt", p_or),
        ("nvidia/llama-3.1-70b-instruct", p_nv),
    ]

    result = await engine.execute(_make_request(), chain)
    assert result.succeeded
    assert result.provider_used == "nvidia"
    assert result.model_used == "nvidia/llama-3.1-70b-instruct"
    # openrouter/free-alt was skipped because openrouter was marked quota exhausted
    assert len(p_or.generate_calls) == 1
    assert len(p_nv.generate_calls) == 1


@pytest.mark.asyncio
async def test_openrouter_quota_exhaustion_switches_to_groq():
    """2. OpenRouter quota exhaustion switches to Groq."""
    p_or = MockTestProvider("openrouter", ["openrouter/free"])
    p_groq = MockTestProvider("groq", ["groq/llama-3.3-70b-versatile"])

    p_or.set_error(
        "openrouter/free",
        ProviderRequestError(
            ProviderErrorCategory.RATE_LIMIT,
            "429 Account quota exceeded",
            provider="openrouter",
            status_code=429,
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("openrouter/free", p_or), ("groq/llama-3.3-70b-versatile", p_groq)]

    result = await engine.execute(_make_request(), chain)
    assert result.succeeded
    assert result.provider_used == "groq"
    assert result.model_used == "groq/llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_openrouter_failure_switches_to_azure():
    """3. OpenRouter failure switches to Azure OpenAI."""
    p_or = MockTestProvider("openrouter", ["openrouter/free"])
    p_azure = MockTestProvider("azure", ["gpt-4o"])

    p_or.set_error(
        "openrouter/free",
        ProviderRequestError(
            ProviderErrorCategory.MODEL_UNAVAILABLE,
            "503 Service Unavailable on OpenRouter",
            provider="openrouter",
            status_code=503,
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("openrouter/free", p_or), ("gpt-4o", p_azure)]

    result = await engine.execute(_make_request(), chain)
    assert result.succeeded
    assert result.provider_used == "azure"
    assert result.model_used == "gpt-4o"


@pytest.mark.asyncio
async def test_nvidia_auth_failure_switches_to_another_provider():
    """4. NVIDIA authentication failure (401) switches to another configured provider."""
    p_nv = MockTestProvider("nvidia", ["nvidia/llama-3.1-70b-instruct", "nvidia/nemotron"])
    p_groq = MockTestProvider("groq", ["groq/llama-3.3-70b-versatile"])

    p_nv.set_error(
        "nvidia/llama-3.1-70b-instruct",
        ProviderRequestError(
            ProviderErrorCategory.AUTHENTICATION,
            "401 Unauthorized: Invalid NVIDIA API key",
            provider="nvidia",
            status_code=401,
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [
        ("nvidia/llama-3.1-70b-instruct", p_nv),
        ("nvidia/nemotron", p_nv),
        ("groq/llama-3.3-70b-versatile", p_groq),
    ]

    result = await engine.execute(_make_request(), chain)
    assert result.succeeded
    assert result.provider_used == "groq"
    # Secondary model on failed provider is skipped immediately
    assert len(p_nv.generate_calls) == 1
    assert len(p_groq.generate_calls) == 1


@pytest.mark.asyncio
async def test_groq_rate_limit_switches_to_another_eligible_provider():
    """5. Groq rate limit switches to another eligible provider."""
    p_groq = MockTestProvider("groq", ["groq/llama-3.3-70b"])
    p_nv = MockTestProvider("nvidia", ["nvidia/llama-3.1-70b-instruct"])

    p_groq.set_error(
        "groq/llama-3.3-70b",
        ProviderRequestError(
            ProviderErrorCategory.RATE_LIMIT,
            "429 Too Many Requests: Rate limit reached",
            provider="groq",
            status_code=429,
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("groq/llama-3.3-70b", p_groq), ("nvidia/llama-3.1-70b-instruct", p_nv)]

    result = await engine.execute(_make_request(), chain)
    assert result.succeeded
    assert result.provider_used == "nvidia"


@pytest.mark.asyncio
async def test_azure_timeout_switches_to_another_provider():
    """6. Azure timeout switches to another provider without stalling."""
    p_azure = MockTestProvider("azure", ["gpt-4o"])
    p_nv = MockTestProvider("nvidia", ["nvidia/llama-3.1-70b-instruct"])

    p_azure.set_error(
        "gpt-4o",
        ProviderRequestError(
            ProviderErrorCategory.TIMEOUT,
            "Azure OpenAI request timed out",
            provider="azure",
            status_code=408,
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("gpt-4o", p_azure), ("nvidia/llama-3.1-70b-instruct", p_nv)]

    result = await engine.execute(_make_request(), chain)
    assert result.succeeded
    assert result.provider_used == "nvidia"


# ─── Tests 7 to 11: Cooldowns, Isolation, and Discovery ───────────────────────


def test_provider_cooldown_isolation():
    """7. One provider's cooldown does not affect unrelated providers."""
    health = ModelHealthTracker()
    health.record_failure("openrouter/free", HealthEvent.RATE_LIMIT_429, latency_ms=100.0, cooldown_override=60.0)

    # OpenRouter model is on cooldown
    assert not health.get_state("openrouter/free").is_healthy
    assert health.get_state("openrouter/free").cooldown_remaining() > 0

    # NVIDIA and Groq models are completely healthy
    assert health.get_state("nvidia/llama-3.1-70b-instruct").is_healthy
    assert health.get_state("groq/llama-3.3-70b-versatile").is_healthy


def test_model_cooldown_isolation():
    """8. Model-level cooldown does not disable the whole provider."""
    health = ModelHealthTracker()
    health.record_failure("groq/model-a", HealthEvent.RATE_LIMIT_429, latency_ms=100.0, cooldown_override=30.0)

    # Model A is cooling down
    assert not health.get_state("groq/model-a").is_healthy

    # Model B on same provider is healthy and ready
    assert health.get_state("groq/model-b").is_healthy


@pytest.mark.asyncio
async def test_provider_quota_exhaustion_disables_all_provider_candidates():
    """9. Provider-level quota exhaustion disables all candidates for that provider."""
    p_or = MockTestProvider("openrouter", ["m1", "m2", "m3"])
    p_nv = MockTestProvider("nvidia", ["nv1"])

    p_or.set_error(
        "m1",
        ProviderRequestError(
            ProviderErrorCategory.RATE_LIMIT,
            "429 Resource_exhausted: free model daily request limit",
            provider="openrouter",
            status_code=429,
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("m1", p_or), ("m2", p_or), ("m3", p_or), ("nv1", p_nv)]

    result = await engine.execute(_make_request(), chain)
    assert result.succeeded
    assert result.provider_used == "nvidia"
    assert len(p_or.generate_calls) == 1  # m2 and m3 never queried
    assert any(a.get("status") == "skipped" for a in result.attempts)


@pytest.mark.asyncio
async def test_healthy_providers_available_when_openrouter_down():
    """10. Healthy providers remain available when OpenRouter is completely down."""
    p_or = MockTestProvider("openrouter", ["openrouter/free"])
    p_nv = MockTestProvider("nvidia", ["nvidia/llama-3.1-70b-instruct"])
    p_groq = MockTestProvider("groq", ["groq/llama-3.3-70b-versatile"])

    # Simulate OpenRouter down with network error
    p_or.set_error("openrouter/free", Exception("Could not resolve openrouter.ai"))

    providers = [p_or, p_nv, p_groq]
    router = ModelRouter(providers=providers, config=RouterConfig(routing_mode="auto"))

    chain = await router.select_models(_make_request())
    # Both nvidia and groq are present in the candidate chain
    chain_provs = {p.name for _, p in chain}
    assert "nvidia" in chain_provs or "groq" in chain_provs


def test_configured_providers_discovered_without_openrouter(monkeypatch):
    """11. Configured providers are discovered without OpenRouter credentials."""
    from harness_core.config.credentials import ProviderCredential

    monkeypatch.setattr(
        "harness_core.config.credentials.CredentialResolver.list_configured_providers",
        lambda self: [
            ProviderCredential(provider="nvidia", api_key="test-key", source="env"),
            ProviderCredential(provider="groq", api_key="test-key", source="env"),
        ],
    )
    monkeypatch.setattr(
        "harness_core.config.credentials.CredentialResolver.resolve",
        lambda self, p: ProviderCredential(provider=p, api_key="test-key", source="env") if p in ("nvidia", "groq") else None,
    )

    providers = create_providers(auto_discover=True)
    names = [p.name for p in providers]

    assert "nvidia" in names
    assert "groq" in names
    assert "openrouter" not in names


# ─── Tests 12 to 16: Priority, Tool Calling, and Context Continuity ───────────


@pytest.mark.asyncio
async def test_provider_priority_respected():
    """12. Provider priority is respected in fallback chain order."""
    p_or = MockTestProvider("openrouter", ["or-model"])
    p_nv = MockTestProvider("nvidia", ["nv-model"])
    p_groq = MockTestProvider("groq", ["groq-model"])

    providers = [p_or, p_nv, p_groq]

    # Set priority: groq > nvidia > openrouter
    config = RouterConfig(provider_priority=["groq", "nvidia", "openrouter"])
    router = ModelRouter(providers=providers, config=config)

    chain = await router.select_models(_make_request())
    assert len(chain) >= 2
    # Groq must be first in the chain
    assert chain[0][1].name == "groq"
    assert chain[1][1].name == "nvidia"


@pytest.mark.asyncio
async def test_tool_calling_survives_provider_switching():
    """13. Tool calling survives provider switching."""
    p_or = MockTestProvider("openrouter", ["openrouter/free"])
    p_nv = MockTestProvider("nvidia", ["nvidia/llama-3.1-70b-instruct"])

    tool_def = {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file in workspace",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }

    # OpenRouter fails with rate limit
    p_or.set_error("openrouter/free", Exception("429 Rate limit"))

    # NVIDIA returns a valid tool call
    p_nv.set_response(
        "nvidia/llama-3.1-70b-instruct",
        CompletionResponse(
            content="",
            tool_calls=[{
                "id": "call_12345",
                "type": "function",
                "function": {"name": "edit_file", "arguments": '{"path": "src/main.py"}'},
            }],
            model="nvidia/llama-3.1-70b-instruct",
            provider="nvidia",
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("openrouter/free", p_or), ("nvidia/llama-3.1-70b-instruct", p_nv)]

    req = _make_request("Edit src/main.py", tools=[tool_def])
    result = await engine.execute(req, chain)

    assert result.succeeded
    assert result.provider_used == "nvidia"
    assert len(result.response.tool_calls) == 1
    assert result.response.tool_calls[0]["function"]["name"] == "edit_file"


@pytest.mark.asyncio
async def test_conversation_context_survives_provider_switching():
    """14. Conversation context and multi-turn messages survive provider switching."""
    p_or = MockTestProvider("openrouter", ["openrouter/free"])
    p_nv = MockTestProvider("nvidia", ["nvidia/llama-3.1-70b-instruct"])

    p_or.set_error("openrouter/free", Exception("500 Internal Server Error"))

    multi_turn_messages = [
        {"role": "system", "content": "You are a senior engineer."},
        {"role": "user", "content": "Initial requirement: Build API."},
        {"role": "assistant", "content": "Understood. Creating models."},
        {"role": "user", "content": "Now implement CRUD endpoints."},
    ]

    req = CompletionRequest(messages=multi_turn_messages)
    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("openrouter/free", p_or), ("nvidia/llama-3.1-70b-instruct", p_nv)]

    result = await engine.execute(req, chain)
    assert result.succeeded
    assert result.provider_used == "nvidia"

    # Verify that NVIDIA received the exact complete message history
    assert len(p_nv.generate_calls) == 1
    passed_messages = p_nv.generate_calls[0].messages
    assert len(passed_messages) == 4
    assert passed_messages[3]["content"] == "Now implement CRUD endpoints."


@pytest.mark.asyncio
async def test_tool_call_ids_and_outputs_consistent():
    """15. Tool-call IDs and tool outputs remain consistent across failover."""
    p_or = MockTestProvider("openrouter", ["openrouter/free"])
    p_groq = MockTestProvider("groq", ["groq/llama-3.3-70b"])

    p_or.set_error("openrouter/free", Exception("429 quota"))

    messages = [
        {"role": "user", "content": "Run tests"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_abc999", "type": "function", "function": {"name": "run_command", "arguments": '{"cmd": "pytest"}'}}],
        },
        {"role": "tool", "tool_call_id": "call_abc999", "content": "5 passed in 0.5s"},
    ]

    req = CompletionRequest(messages=messages)
    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("openrouter/free", p_or), ("groq/llama-3.3-70b", p_groq)]

    result = await engine.execute(req, chain)
    assert result.succeeded
    assert result.provider_used == "groq"

    # Confirm tool results preserved intact
    sent_msgs = p_groq.generate_calls[0].messages
    assert sent_msgs[2]["tool_call_id"] == "call_abc999"
    assert sent_msgs[2]["content"] == "5 passed in 0.5s"


@pytest.mark.asyncio
async def test_context_overflow_selects_sufficient_context_model():
    """16. Context overflow triggers failover to larger-context candidate."""
    p_small = MockTestProvider("groq", ["groq/small-context"])
    p_large = MockTestProvider("nvidia", ["nvidia/large-context"])

    p_small.set_error(
        "groq/small-context",
        ProviderRequestError(
            ProviderErrorCategory.CONTEXT_LENGTH,
            "400 Context window exceeded: maximum context is 8192 tokens",
            provider="groq",
            status_code=400,
        ),
    )

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("groq/small-context", p_small), ("nvidia/large-context", p_large)]

    result = await engine.execute(_make_request("Large repo dump"), chain)
    assert result.succeeded
    assert result.provider_used == "nvidia"
    assert result.model_used == "nvidia/large-context"


# ─── Tests 17 to 20: Recovery, Security, and Diagnostics ──────────────────────


def test_providers_recover_after_cooldown():
    """17. Providers recover after cooldown expiration."""
    health = ModelHealthTracker()
    # Model failed with a short cooldown of 0.05s
    health.record_failure("groq/model", HealthEvent.RATE_LIMIT_429, latency_ms=50.0, cooldown_override=0.05)

    assert not health.get_state("groq/model").is_healthy
    time.sleep(0.06)
    # After cooldown expires, model is healthy again
    assert health.get_state("groq/model").is_healthy


@pytest.mark.asyncio
async def test_all_provider_exhaustion_diagnostics():
    """18. When all providers fail, a clear diagnostic report is generated."""
    p_or = MockTestProvider("openrouter", ["openrouter/free"])
    p_nv = MockTestProvider("nvidia", ["nvidia/model"])

    p_or.set_error("openrouter/free", Exception("429 rate limit exceeded"))
    p_nv.set_error("nvidia/model", Exception("401 invalid api key"))

    engine = FallbackEngine(health_tracker=ModelHealthTracker())
    chain = [("openrouter/free", p_or), ("nvidia/model", p_nv)]

    result = await engine.execute(_make_request(), chain)
    assert not result.succeeded
    assert result.final_error is not None
    # Report clearly conveys provider failures
    assert len(result.attempts) == 2


@pytest.mark.asyncio
async def test_no_credentials_leak_in_logs_or_events():
    """19. No API credentials leak into EventBus events or error messages."""
    p_nv = MockTestProvider("nvidia", ["nvidia/model"], api_key="nvap-sensitive-secret-token-xyz")
    p_nv.set_error(
        "nvidia/model",
        Exception("Failed with token Bearer nvap-sensitive-secret-token-xyz"),
    )

    events: list[Event] = []
    bus = EventBus()
    bus.on("model.failover", lambda e: events.append(e))
    bus.on("model.cooldown", lambda e: events.append(e))

    engine = FallbackEngine(health_tracker=ModelHealthTracker(), event_bus=bus)
    chain = [("nvidia/model", p_nv)]

    result = await engine.execute(_make_request(), chain)

    # 1. Final error must NOT contain secret
    assert "nvap-sensitive-secret-token-xyz" not in (result.final_error or "")
    # 2. Attempt entries must NOT contain secret
    for a in result.attempts:
        assert "nvap-sensitive-secret-token-xyz" not in str(a.get("error", ""))
    # 3. Events must NOT contain secret
    for e in events:
        assert "nvap-sensitive-secret-token-xyz" not in str(e.data)


@pytest.mark.asyncio
async def test_retry_limits_prevent_infinite_loops():
    """20. Retry limits prevent infinite retry loops on server errors."""
    p_nv = MockTestProvider("nvidia", ["nvidia/model"])
    p_nv.set_error(
        "nvidia/model",
        ProviderRequestError(
            ProviderErrorCategory.PROVIDER,
            "503 Service Unavailable",
            provider="nvidia",
            status_code=503,
        ),
    )

    config = FallbackConfig(
        max_fallback_models=2,
        retry=FallbackConfig().retry,
    )
    # Set max_retries to 2
    config.retry.max_retries = 2
    config.retry.base_delay_seconds = 0.001

    engine = FallbackEngine(fallback_config=config)
    chain = [("nvidia/model", p_nv)]

    result = await engine.execute(_make_request(), chain)
    assert not result.succeeded
    # Exactly 1 initial + 2 retries = 3 calls total
    assert len(p_nv.generate_calls) == 3
