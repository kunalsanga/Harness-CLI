"""Unit tests for NVIDIA NIM provider."""

import os
import pytest
from unittest.mock import patch, MagicMock
from httpx import HTTPStatusError, Request, Response

from harness_core.providers.nvidia import NvidiaProvider
from harness_core.providers.base import CompletionRequest


@pytest.fixture
def clean_env():
    """Ensure clean environment for provider tests."""
    old_env = dict(os.environ)
    if "NVIDIA_API_KEY" in os.environ:
        del os.environ["NVIDIA_API_KEY"]
    if "NVIDIA_BASE_URL" in os.environ:
        del os.environ["NVIDIA_BASE_URL"]
    if "NVIDIA_MODEL" in os.environ:
        del os.environ["NVIDIA_MODEL"]
    
    yield
    
    os.environ.clear()
    os.environ.update(old_env)


def test_nvidia_provider_configuration(clean_env):
    """Test configuration with missing API key and custom base URL."""
    provider = NvidiaProvider()
    assert provider.api_key == ""
    assert provider.base_url == "https://integrate.api.nvidia.com/v1"

    os.environ["NVIDIA_API_KEY"] = "test_key"
    os.environ["NVIDIA_BASE_URL"] = "https://custom.nvidia.com/v1"
    os.environ["NVIDIA_MODEL"] = "test-model"

    provider2 = NvidiaProvider()
    assert provider2.api_key == "test_key"
    assert provider2.base_url == "https://custom.nvidia.com/v1"


@pytest.mark.asyncio
async def test_health_check_no_key(clean_env):
    provider = NvidiaProvider()
    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_successful_chat_normalization(clean_env):
    os.environ["NVIDIA_API_KEY"] = "test_key"
    provider = NvidiaProvider()
    
    mock_response = Response(
        200,
        json={
            "model": "deepseek-ai/deepseek-v4-flash-0731",
            "choices": [
                {
                    "message": {
                        "content": "Hello World",
                        "tool_calls": []
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2
            }
        },
        request=Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    )
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        request = CompletionRequest(messages=[{"role": "user", "content": "Hi"}])
        response = await provider.generate(request)
        
        assert response.content == "Hello World"
        assert response.model == "deepseek-ai/deepseek-v4-flash-0731"
        assert response.finish_reason == "stop"
        assert response.usage["prompt_tokens"] == 10
        assert len(response.tool_calls) == 0


@pytest.mark.asyncio
async def test_tool_call_normalization(clean_env):
    os.environ["NVIDIA_API_KEY"] = "test_key"
    provider = NvidiaProvider()
    
    mock_response = Response(
        200,
        json={
            "model": "deepseek-ai/deepseek-v4-flash-0731",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ]
        },
        request=Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    )
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        request = CompletionRequest(
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}]
        )
        response = await provider.generate(request)
        
        assert response.content == ""
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["function"]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_error_handling(clean_env):
    os.environ["NVIDIA_API_KEY"] = "test_key"
    provider = NvidiaProvider()
    
    mock_response = Response(
        401,
        json={"error": "Unauthorized"},
        request=Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    )
    
    with patch("httpx.AsyncClient.post", return_value=mock_response):
        request = CompletionRequest(messages=[{"role": "user", "content": "Hi"}])
        with pytest.raises(HTTPStatusError) as exc:
            await provider.generate(request)
        assert exc.value.response.status_code == 401


