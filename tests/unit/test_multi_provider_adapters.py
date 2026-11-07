"""Unit tests for the four core AI provider adapters:

1. OpenRouter
2. OpenAI
3. NVIDIA NIM
4. FreeLLMAPI
"""

from __future__ import annotations

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.providers.openrouter import OpenRouterProvider
from harness_core.providers.http_adapters import OpenAIProvider
from harness_core.providers.nvidia import NvidiaProvider
from harness_core.providers.freellmapi import FreeLLMAPIProvider


# ─── 1. OpenRouter Adapter Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openrouter_adapter_success():
    provider = OpenRouterProvider(api_key="sk-or-test", base_url="https://openrouter.ai/api/v1")
    assert provider.name == "openrouter"
    assert provider.base_url == "https://openrouter.ai/api/v1"

    mock_resp = httpx.Response(
        200,
        json={
            "model": "deepseek/deepseek-chat",
            "choices": [{"message": {"content": "Hello from OpenRouter"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
        },
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        req = CompletionRequest(messages=[{"role": "user", "content": "Hi"}], model="deepseek/deepseek-chat")
        res = await provider.generate(req)

        assert res.content == "Hello from OpenRouter"
        assert res.model == "deepseek/deepseek-chat"
        assert res.token_usage.input_tokens == 15
        assert res.token_usage.output_tokens == 5
        # Verify auth headers on client
        client = await provider._get_client()
        assert client.headers.get("Authorization") == "Bearer sk-or-test"
    await provider.close()


@pytest.mark.asyncio
async def test_openrouter_error_classification():
    provider = OpenRouterProvider(api_key="sk-or-test")

    # 429 Rate Limit
    rate_limit_resp = httpx.Response(
        429,
        json={"error": {"message": "Rate limit reached", "code": 429}},
        headers={"Retry-After": "15"},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.HTTPStatusError("Rate Limit", request=rate_limit_resp.request, response=rate_limit_resp)
        with pytest.raises(ProviderRequestError) as exc_info:
            await provider.generate(CompletionRequest(messages=[{"role": "user", "content": "Hi"}]))
        assert exc_info.value.category == ProviderErrorCategory.RATE_LIMIT
        assert exc_info.value.retry_after == 15.0
    await provider.close()


# ─── 2. OpenAI Adapter Tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openai_adapter_success_and_headers():
    provider = OpenAIProvider(api_key="sk-test-key", base_url="https://api.openai.com/v1", default_model="gpt-4o-mini")
    assert provider.name == "openai"
    assert provider.base_url == "https://api.openai.com/v1"

    mock_resp = httpx.Response(
        200,
        json={
            "id": "chatcmpl-123",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "Hello from OpenAI", "tool_calls": []}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        req = CompletionRequest(messages=[{"role": "user", "content": "Test"}])
        res = await provider.generate(req)

        assert res.content == "Hello from OpenAI"
        assert res.model == "gpt-4o-mini"
        assert res.token_usage.cached_tokens == 5
        assert provider._client.headers.get("Authorization") == "Bearer sk-test-key"
    await provider.close()


@pytest.mark.asyncio
async def test_openai_quota_exhausted_classification():
    provider = OpenAIProvider(api_key="sk-test-key")

    # OpenAI returns 429 with insufficient_quota when billing is exhausted
    quota_resp = httpx.Response(
        429,
        json={"error": {"message": "You exceeded your current quota, please check your plan and billing details.", "code": "insufficient_quota"}},
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = quota_resp
        with pytest.raises(ProviderRequestError) as exc_info:
            await provider.generate(CompletionRequest(messages=[{"role": "user", "content": "Hi"}]))
        assert exc_info.value.category == ProviderErrorCategory.PAYMENT_REQUIRED
    await provider.close()


# ─── 3. NVIDIA NIM Adapter Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_nvidia_adapter_success_and_tools():
    provider = NvidiaProvider(api_key="nvapi-test-key", base_url="https://integrate.api.nvidia.com/v1")
    assert provider.name == "nvidia"
    assert provider.base_url == "https://integrate.api.nvidia.com/v1"

    mock_resp = httpx.Response(
        200,
        json={
            "model": "meta/llama-3.1-70b-instruct",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "test.txt"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 15},
        },
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Read test.txt"}],
            tools=[{"type": "function", "function": {"name": "read_file"}}],
        )
        res = await provider.generate(req)

        assert res.content == ""
        assert len(res.tool_calls) == 1
        assert res.tool_calls[0]["function"]["name"] == "read_file"
    await provider.close()


@pytest.mark.asyncio
async def test_nvidia_discovery_augmentation():
    provider = NvidiaProvider(api_key="nvapi-test-key")
    mock_models_resp = httpx.Response(
        200,
        json={"data": [{"id": "meta/llama-3.1-70b-instruct"}, {"id": "google/deplot"}]},
        request=httpx.Request("GET", "https://integrate.api.nvidia.com/v1/models"),
    )
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_models_resp
        models = await provider.list_models()
        instruct_model = next(m for m in models if m.id == "meta/llama-3.1-70b-instruct")
        assert instruct_model.supports_tools is True
        assert instruct_model.context_window == 128000
    await provider.close()


# ─── 4. FreeLLMAPI Adapter Tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_freellmapi_adapter_success_and_defaults():
    provider = FreeLLMAPIProvider(base_url="http://127.0.0.1:31415/v1", default_model="auto")
    assert provider.name == "freellmapi"
    assert provider.base_url == "http://127.0.0.1:31415/v1"

    mock_resp = httpx.Response(
        200,
        json={
            "model": "auto",
            "choices": [{"message": {"content": "Hello from FreeLLMAPI"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        },
        request=httpx.Request("POST", "http://127.0.0.1:31415/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        req = CompletionRequest(messages=[{"role": "user", "content": "Hi"}])
        res = await provider.generate(req)

        assert res.content == "Hello from FreeLLMAPI"
        assert res.model == "auto"
    await provider.close()


@pytest.mark.asyncio
async def test_freellmapi_health_check():
    provider = FreeLLMAPIProvider(base_url="http://127.0.0.1:31415/v1")

    # Reached models successfully
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = httpx.Response(200, json={"data": []}, request=httpx.Request("GET", "http://127.0.0.1:31415/v1/models"))
        assert await provider.health_check() is True

    # Gateway offline / connection error
    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        assert await provider.health_check() is False
    await provider.close()
