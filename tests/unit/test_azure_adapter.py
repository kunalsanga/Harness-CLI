"""Tests for Azure OpenAI provider adapter."""

from __future__ import annotations

import httpx
import pytest

from harness_core.providers.azure import AzureOpenAIProvider
from harness_core.providers.base import CompletionRequest, CompletionResponse, ProviderErrorCategory, ProviderRequestError


def test_azure_provider_initialization():
    """Verify Azure provider initializes with custom or default parameters."""
    provider = AzureOpenAIProvider(
        api_key="test-azure-key",
        base_url="https://my-resource.openai.azure.com",
        default_model="my-gpt4o-deployment",
        api_version="2024-10-21",
    )

    assert provider.name == "azure"
    assert provider.api_key == "test-azure-key"
    assert provider.base_url == "https://my-resource.openai.azure.com"
    assert provider.deployment == "my-gpt4o-deployment"
    assert provider.api_version == "2024-10-21"


def test_azure_url_construction():
    """Verify correct URL construction for native Azure and OpenAI-compatible styles."""
    p_native = AzureOpenAIProvider(
        base_url="https://my-resource.openai.azure.com",
        default_model="gpt-4o",
        api_version="2024-08-01-preview",
    )
    assert p_native._build_completion_url("gpt-4o") == "/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"

    p_v1 = AzureOpenAIProvider(
        base_url="https://my-gateway.services.ai.azure.com/v1",
        default_model="gpt-4o",
    )
    assert p_v1._build_completion_url("gpt-4o") == "/chat/completions"


def test_azure_headers():
    """Verify Azure authentication headers."""
    provider = AzureOpenAIProvider(api_key="secret-key-12345")
    headers = provider._headers()
    assert headers["api-key"] == "secret-key-12345"
    assert headers["Authorization"] == "Bearer secret-key-12345"


@pytest.mark.asyncio
async def test_azure_generate_success():
    """Test successful generation with mocked HTTP response."""
    provider = AzureOpenAIProvider(
        api_key="test-key",
        base_url="https://my-resource.openai.azure.com",
        default_model="gpt-4o",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api-version=2024-08-01-preview" in str(request.url)
        assert request.headers.get("api-key") == "test-key"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-azure-123",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from Azure OpenAI!",
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 15, "completion_tokens": 10},
            },
        )

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=provider.base_url)

    req = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
    resp = await provider.generate(req)

    assert isinstance(resp, CompletionResponse)
    assert resp.content == "Hello from Azure OpenAI!"
    assert resp.provider == "azure"
    assert resp.model == "gpt-4o"
    assert resp.usage["prompt_tokens"] == 15
    await provider.close()


@pytest.mark.asyncio
async def test_azure_generate_tool_call():
    """Test generation with tool calling."""
    provider = AzureOpenAIProvider(
        api_key="test-key",
        base_url="https://my-resource.openai.azure.com",
        default_model="gpt-4o",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-azure-456",
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "run_command", "arguments": '{"command": "pytest"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15},
            },
        )

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=provider.base_url)

    req = CompletionRequest(
        messages=[{"role": "user", "content": "Run tests"}],
        tools=[{"type": "function", "function": {"name": "run_command"}}],
    )
    resp = await provider.generate(req)

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0]["function"]["name"] == "run_command"
    await provider.close()


@pytest.mark.asyncio
async def test_azure_error_normalization():
    """Verify HTTP errors from Azure are normalized to ProviderRequestError."""
    provider = AzureOpenAIProvider(
        api_key="test-key",
        base_url="https://my-resource.openai.azure.com",
        default_model="gpt-4o",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"code": "InvalidApiKey", "message": "Access denied due to invalid subscription key"}},
        )

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=provider.base_url)

    req = CompletionRequest(messages=[{"role": "user", "content": "Hello"}])
    with pytest.raises(ProviderRequestError) as excinfo:
        await provider.generate(req)

    assert excinfo.value.category == ProviderErrorCategory.AUTHENTICATION
    assert excinfo.value.status_code == 401
    await provider.close()


@pytest.mark.asyncio
async def test_azure_health_check_and_model_listing():
    """Verify health check and model listing on Azure provider."""
    provider = AzureOpenAIProvider(
        api_key="test-key",
        base_url="https://my-resource.openai.azure.com",
        default_model="gpt-4o",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "/openai/deployments" in str(request.url):
            return httpx.Response(
                200,
                json={"data": [{"id": "gpt-4o"}, {"id": "o3-mini"}]},
            )
        return httpx.Response(404)

    provider._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=provider.base_url)

    healthy = await provider.health_check()
    assert healthy is True

    models = await provider.list_models()
    assert len(models) == 2
    ids = [m.id for m in models]
    assert "gpt-4o" in ids
    assert "o3-mini" in ids
    for m in models:
        assert m.supports_tools is True
        assert m.provider == "azure"
    await provider.close()

