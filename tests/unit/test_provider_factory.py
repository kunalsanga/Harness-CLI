"""Deterministic tests for provider factory and normalized contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
    TokenUsage,
)
from harness_core.providers.factory import create_providers, load_project_provider_config
from harness_core.providers.normalized import NormalizedModelProvider
from harness_core.providers.routed import RoutedModelProvider
from harness_core.routing.fallback import FallbackConfig, FallbackResult, RetryConfig, classify_error
from harness_core.routing.router import ModelRouter, RouterConfig
from harness_core.routing.scoring import ScoringContext, score_tool_support


class _StubProvider(ModelProvider):
    def __init__(self, name: str = "stub", *, fail: Exception | None = None) -> None:
        self._name = name
        self.fail = fail
        self.closed = False
        self.configured_models: list[ModelInfo] = []

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        if self.fail is not None:
            raise self.fail
        return CompletionResponse(
            content="ok",
            model=request.model or "stub-model",
            provider=self.name,
            usage={"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        )

    async def stream(self, request: CompletionRequest):
        yield "ok"

    async def list_models(self) -> list[ModelInfo]:
        if self.fail is not None:
            raise self.fail
        return [
            ModelInfo(
                id="stub-model",
                name="stub-model",
                provider=self.name,
                supports_tools=True,
                is_free=True,
                context_window=8000,
            )
        ]

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


def test_token_usage_from_provider_preserves_known_fields() -> None:
    usage = TokenUsage.from_provider({
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "prompt_tokens_details": {"cached_tokens": 2},
        "completion_tokens_details": {"reasoning_tokens": 1},
    })
    assert usage is not None
    assert usage.input_tokens == 10
    assert usage.output_tokens == 4
    assert usage.cached_tokens == 2
    assert usage.reasoning_tokens == 1


def test_unknown_tool_capability_is_neutral_not_false() -> None:
    model = ModelInfo(id="m", name="m", provider="p", supports_tools=None)
    assert score_tool_support(model, ScoringContext(requires_tools=True)) == 0.5
    model.supports_tools = False
    assert score_tool_support(model, ScoringContext(requires_tools=True)) == 0.0
    model.supports_tools = True
    assert score_tool_support(model, ScoringContext(requires_tools=True)) == 1.0


def test_create_providers_defaults_to_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    providers = create_providers({})
    assert len(providers) == 1
    assert providers[0].name == "openrouter"
    assert isinstance(providers[0], NormalizedModelProvider)


def test_create_providers_builds_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_LLM_KEY", "local-key")
    providers = create_providers({
        "providers": {
            "local": {
                "type": "openai-compatible",
                "endpoint": "http://127.0.0.1:8000/v1",
                "model": "qwen",
                "api_key_env": "LOCAL_LLM_KEY",
                "is_free": True,
                "supports_tools": True,
            }
        }
    })
    assert len(providers) == 1
    assert providers[0].name == "local"
    assert any(m.id == "qwen" for m in providers[0].configured_models)


def test_load_project_provider_config_reads_yaml(tmp_path: Path) -> None:
    harness = tmp_path / ".harness"
    harness.mkdir()
    (harness / "config.yaml").write_text(
        "providers:\n  openai:\n    enabled: true\n    model: gpt-4o-mini\n",
        encoding="utf-8",
    )
    data = load_project_provider_config(tmp_path)
    assert data["providers"]["openai"]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_normalized_provider_wraps_legacy_exceptions() -> None:
    stub = _StubProvider(fail=RuntimeError("boom"))
    provider = NormalizedModelProvider(stub)
    with pytest.raises(ProviderRequestError) as ei:
        await provider.generate(CompletionRequest(messages=[{"role": "user", "content": "hi"}]))
    assert ei.value.category == ProviderErrorCategory.UNKNOWN
    await provider.close()
    assert stub.closed is True


@pytest.mark.asyncio
async def test_discovery_failure_surfaces_when_catalog_empty() -> None:
    stub = _StubProvider(fail=ProviderRequestError(
        ProviderErrorCategory.AUTHENTICATION,
        "discovery auth failed",
        provider="stub",
        status_code=401,
    ))
    router = ModelRouter(
        providers=[stub],
        config=RouterConfig(routing_mode="free"),
    )
    result = await router.execute(CompletionRequest(messages=[{"role": "user", "content": "hi"}]))
    assert not result.succeeded
    assert "Model discovery failed" in (result.final_error or "")
    assert "discovery auth failed" in (result.final_error or "")


@pytest.mark.asyncio
async def test_routed_provider_uses_router_execute() -> None:
    stub = _StubProvider()
    router = ModelRouter(
        providers=[stub],
        config=RouterConfig(
            routing_mode="free",
            fallback=FallbackConfig(retry=RetryConfig(max_retries=0, jitter=False)),
        ),
    )
    routed = RoutedModelProvider(router)
    response = await routed.generate(CompletionRequest(messages=[{"role": "user", "content": "hi"}]))
    assert response.content == "ok"
    assert response.token_usage is not None
    assert response.token_usage.total_tokens == 8


def test_provider_request_error_classifies_without_string_parsing() -> None:
    err = ProviderRequestError(ProviderErrorCategory.RATE_LIMIT, "whatever", provider="p")
    assert classify_error(err).value == "rate_limited"
    auth = ProviderRequestError(ProviderErrorCategory.AUTHENTICATION, "whatever", provider="p")
    assert classify_error(auth).value == "permanent"
