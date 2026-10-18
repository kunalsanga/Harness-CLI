"""Comprehensive regression tests for OpenRouter and provider reliability audit.

Validates:
- Credential precedence, empty/whitespace handling, and secret masking
- Accurate HTTP status classification (400, 401, 402, 403, 404, 408, 413, 429, 500, 502, 503, 504)
- Model-specific 403 permission failover vs provider-wide 401 auth termination
- Dynamic free route inclusion and context window
- Model pinning preservation for openrouter/free
- Secret redaction in exceptions and fallback results
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from harness_core.config.credentials import CredentialResolver, mask_secret
from harness_core.config.dotenv import load_dotenv, _load_single_env
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.providers.http_adapters import _error_category
from harness_core.providers.openrouter import OpenRouterProvider, _openrouter_error_category
from harness_core.routing.fallback import (
    ERROR_CATEGORY_AUTH,
    ERROR_CATEGORY_PERMISSION,
    ERROR_CATEGORY_RATE_LIMIT,
    ERROR_CATEGORY_TIMEOUT,
    ErrorClassification,
    FallbackConfig,
    FallbackEngine,
    RetryConfig,
    classify_error,
    normalized_error_category,
)
from harness_core.routing.health import ModelHealthTracker
from harness_core.routing.router import ModelRouter, RouterConfig


# ─── 1. Configuration & Credential Resolution ────────────────────────────────

def test_secret_masking() -> None:
    key = "sk-or-v1-abcdef1234567890abcdef1234567890"
    assert mask_secret(key) == f"sk-o{'*' * (len(key) - 8)}7890"
    assert mask_secret("") == ""
    assert mask_secret("short") == "*****"


def test_dotenv_empty_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=sk-or-v1-loaded-from-file\n", encoding="utf-8")
    
    # Empty string in os.environ should not block loading valid key from .env
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    count = _load_single_env(env_file)
    assert count == 1
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-v1-loaded-from-file"


def test_dotenv_quoted_value_stripping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('OPENROUTER_API_KEY="  sk-or-v1-clean-key  "\n', encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    
    count = _load_single_env(env_file)
    assert count == 1
    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-v1-clean-key"


def test_credential_resolver_loads_dotenv_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    resolver = CredentialResolver()
    
    with patch("harness_core.config.dotenv.load_dotenv") as mock_load:
        def fake_load():
            os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-mocked"
            return 1
        mock_load.side_effect = fake_load
        
        cred = resolver.resolve("openrouter")
        assert cred is not None
        assert cred.api_key == "sk-or-v1-mocked"
        assert cred.source == "env"


# ─── 2. HTTP Error Classification ─────────────────────────────────────────────

@pytest.mark.parametrize(
    ("status", "message", "expected_cat"),
    [
        (400, "malformed request", ProviderErrorCategory.INVALID_REQUEST),
        (401, "unauthorized", ProviderErrorCategory.AUTHENTICATION),
        (402, "payment required", ProviderErrorCategory.PAYMENT_REQUIRED),
        (403, "forbidden: only available on agentic harnesses", ProviderErrorCategory.PERMISSION),
        (404, "model not found", ProviderErrorCategory.MODEL_UNAVAILABLE),
        (408, "request timeout", ProviderErrorCategory.TIMEOUT),
        (413, "context window token limit exceeded", ProviderErrorCategory.CONTEXT_LENGTH),
        (422, "unprocessable entity", ProviderErrorCategory.INVALID_REQUEST),
        (429, "rate limit reached", ProviderErrorCategory.RATE_LIMIT),
        (500, "internal server error", ProviderErrorCategory.PROVIDER),
        (502, "bad gateway", ProviderErrorCategory.MODEL_UNAVAILABLE),
        (503, "upstream unavailable", ProviderErrorCategory.MODEL_UNAVAILABLE),
        (504, "gateway timeout", ProviderErrorCategory.PROVIDER),
    ],
)
def test_error_category_mappings(status: int, message: str, expected_cat: ProviderErrorCategory) -> None:
    assert _error_category(status, message) == expected_cat


def test_openrouter_error_category_wrapped_types() -> None:
    assert _openrouter_error_category(502, "provider_unavailable", "upstream error") == ProviderErrorCategory.MODEL_UNAVAILABLE
    assert _openrouter_error_category(429, "provider_rate_limit", "rate limit") == ProviderErrorCategory.RATE_LIMIT
    assert _openrouter_error_category(403, "permission_denied", "restricted") == ProviderErrorCategory.PERMISSION


def test_classify_error_categories() -> None:
    # 401: Permanent (stops rotation)
    err_401 = ProviderRequestError(ProviderErrorCategory.AUTHENTICATION, "Invalid API key", status_code=401)
    assert classify_error(err_401) == ErrorClassification.PERMANENT

    # 403: Permission denied (eligible for failover)
    err_403 = ProviderRequestError(ProviderErrorCategory.PERMISSION, "Model restricted to agentic harnesses", status_code=403)
    assert classify_error(err_403) == ErrorClassification.PERMISSION_DENIED

    # 413: Context overflow
    err_413 = ProviderRequestError(ProviderErrorCategory.CONTEXT_LENGTH, "Token limit exceeded", status_code=413)
    assert classify_error(err_413) == ErrorClassification.CONTEXT_OVERFLOW


# ─── 3. Fallback Engine: Permission Failover vs Auth Termination ──────────────

class MockProvider(ModelProvider):
    def __init__(self, name: str = "openrouter") -> None:
        self._name = name
        self.api_key = "sk-or-v1-secret-key"
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


@pytest.mark.asyncio
async def test_403_model_permission_fails_over_to_next_candidate() -> None:
    """A model-specific 403 (e.g. ThinkingMachines app restriction) must NOT
    terminate the fallback chain; it should failover to the next healthy model."""
    engine = FallbackEngine(fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0)))
    provider = MockProvider()

    call_history = []
    async def fake_generate(req: CompletionRequest) -> CompletionResponse:
        call_history.append(req.model)
        if req.model == "restricted/model:free":
            raise ProviderRequestError(
                ProviderErrorCategory.PERMISSION,
                "restricted/model:free is only available on agentic harnesses",
                status_code=403,
                model=req.model,
            )
        return CompletionResponse(content="Success from openrouter/free", model=req.model)

    provider._generate_fn = fake_generate

    chain = [
        ("restricted/model:free", provider),
        ("openrouter/free", provider),
    ]

    result = await engine.execute(CompletionRequest(messages=[{"role": "user", "content": "hi"}]), chain)
    assert result.succeeded is True
    assert result.model_used == "openrouter/free"
    assert call_history == ["restricted/model:free", "openrouter/free"]


@pytest.mark.asyncio
async def test_401_auth_failure_aborts_immediately() -> None:
    """A provider-wide 401 auth failure must abort immediately without trying model 2."""
    engine = FallbackEngine(fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0)))
    provider = MockProvider()

    call_history = []
    async def fake_generate(req: CompletionRequest) -> CompletionResponse:
        call_history.append(req.model)
        raise ProviderRequestError(
            ProviderErrorCategory.AUTHENTICATION,
            "Invalid API key (401 Unauthorized)",
            status_code=401,
            model=req.model,
        )

    provider._generate_fn = fake_generate

    chain = [
        ("model-1:free", provider),
        ("openrouter/free", provider),
    ]

    result = await engine.execute(CompletionRequest(messages=[{"role": "user", "content": "hi"}]), chain)
    assert result.succeeded is False
    assert result.error_category == ERROR_CATEGORY_AUTH
    assert "OPENROUTER_API_KEY" in (result.final_error or "")
    assert call_history == ["model-1:free"]


# ─── 4. Free Chain Construction & Dynamic Route Inclusion ─────────────────────

@pytest.mark.asyncio
async def test_free_chain_includes_dynamic_route_as_safety_net() -> None:
    """When multiple concrete free models are discovered, the dynamic route
    (openrouter/free) must be included at the end of the chain rather than sliced off."""
    provider = MockProvider()
    provider.configured_models = [
        ModelInfo(id="concrete/m1:free", name="m1", provider="openrouter", is_free=True, supports_tools=True, context_window=100000),
        ModelInfo(id="concrete/m2:free", name="m2", provider="openrouter", is_free=True, supports_tools=True, context_window=90000),
        ModelInfo(id="concrete/m3:free", name="m3", provider="openrouter", is_free=True, supports_tools=True, context_window=80000),
        ModelInfo(id="concrete/m4:free", name="m4", provider="openrouter", is_free=True, supports_tools=True, context_window=70000),
        ModelInfo(id="concrete/m5:free", name="m5", provider="openrouter", is_free=True, supports_tools=True, context_window=60000),
        ModelInfo(id="openrouter/free", name="free router", provider="openrouter", is_free=True, supports_tools=True, context_window=131072, tags=["dynamic-route"]),
    ]

    router = ModelRouter(
        providers=[provider],
        config=RouterConfig(routing_mode="free", max_fallback_chain=4),
    )

    req = CompletionRequest(messages=[{"role": "user", "content": "test"}], tools=[{"type": "function"}])
    chain = await router.select_models(req)

    chain_models = [mid for mid, _ in chain]
    assert len(chain_models) == 4
    # The last candidate in the chain must be the dynamic route
    assert chain_models[-1] == "openrouter/free"
    assert "concrete/m1:free" in chain_models


@pytest.mark.asyncio
async def test_pinning_openrouter_free_preserves_full_id() -> None:
    """Pinning openrouter/free must not get stripped to 'free'."""
    provider = MockProvider()
    provider.configured_models = [
        ModelInfo(id="openrouter/free", name="free router", provider="openrouter", is_free=True, supports_tools=True, tags=["dynamic-route"]),
    ]
    router = ModelRouter(providers=[provider])

    req = CompletionRequest(messages=[{"role": "user", "content": "test"}], model="openrouter/free")
    chain = await router.select_models(req)
    assert len(chain) == 1
    assert chain[0][0] == "openrouter/free"


# ─── 5. Secret Redaction ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_secret_redaction_in_fallback_error() -> None:
    engine = FallbackEngine(fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0)))
    provider = MockProvider()
    secret = "sk-or-v1-secret-1234567890abcdef"
    provider.api_key = secret

    async def fake_fail(req):
        raise RuntimeError(f"Bad request with key {secret}")

    provider._generate_fn = fake_fail

    result = await engine.execute(CompletionRequest(messages=[{"role": "user", "content": "hi"}]), [("m1", provider)])
    assert secret not in str(result.attempts)
    assert secret not in (result.final_error or "")
    assert "[REDACTED]" in str(result.attempts)
