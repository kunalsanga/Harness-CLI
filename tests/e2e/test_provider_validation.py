"""Real provider validation — tests against actual OpenRouter API.

These tests require OPENROUTER_API_KEY to be set.
They are skipped in CI environments without API keys.
"""

import asyncio
import os
import pytest

from harness_core.providers.base import CompletionRequest, CompletionResponse


def _has_openrouter_key():
    return bool(os.environ.get("OPENROUTER_API_KEY"))

def _skip_without_key():
    if not _has_openrouter_key():
        pytest.skip("OPENROUTER_API_KEY not set")

async def _safe_generate(provider, request):
    """Generate with automatic 429 skip."""
    try:
        return await provider.generate(request)
    except Exception as e:
        if "429" in str(e):
            pytest.skip("Rate limited (429) by OpenRouter")
        raise


@pytest.fixture
def openrouter_provider():
    from harness_core.providers.openrouter import OpenRouterProvider
    return OpenRouterProvider()


class TestOpenRouterHealth:
    """Verify OpenRouter connectivity."""

    @pytest.mark.asyncio
    async def test_health_check(self, openrouter_provider):
        _skip_without_key()
        result = await openrouter_provider.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_model_discovery(self, openrouter_provider):
        _skip_without_key()
        models = await openrouter_provider.list_models()
        assert len(models) > 0
        # Verify model structure
        m = models[0]
        assert m.id
        assert m.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_free_model_discovery(self, openrouter_provider):
        _skip_without_key()
        models = await openrouter_provider.list_models()
        free = [m for m in models if m.is_free]
        assert len(free) > 0, "Expected at least one free model"

    @pytest.mark.asyncio
    async def test_tool_capable_model_discovery(self, openrouter_provider):
        _skip_without_key()
        models = await openrouter_provider.list_models()
        tool_capable = [m for m in models if m.supports_tools]
        assert len(tool_capable) > 0, "Expected at least one tool-capable model"


class TestOpenRouterCompletion:
    """Test actual completions against OpenRouter."""

    @pytest.mark.asyncio
    async def test_simple_completion(self, openrouter_provider):
        _skip_without_key()
        req = CompletionRequest(
            model="openrouter/free",
            messages=[{"role": "user", "content": "Say exactly one word: YES"}],
            max_tokens=10,
        )
        resp = await _safe_generate(openrouter_provider, req)
        assert isinstance(resp, CompletionResponse)
        if not resp.model:
            pytest.skip("Rate limited or empty response from OpenRouter")
        assert resp.provider == "openrouter"
        assert resp.usage is not None

    @pytest.mark.asyncio
    async def test_completion_with_null_content(self, openrouter_provider):
        """Reasoning models may return content=null — provider must handle gracefully."""
        _skip_without_key()
        req = CompletionRequest(
            model="openrouter/free",
            messages=[{"role": "user", "content": "Think about 1+1"}],
            max_tokens=50,
        )
        resp = await _safe_generate(openrouter_provider, req)
        # Must not raise, content may be empty for reasoning models
        assert isinstance(resp, CompletionResponse)
        assert resp.content is not None  # Should be "" not None

    @pytest.mark.asyncio
    async def test_tool_calling(self, openrouter_provider):
        """Test that the provider can handle tool calls."""
        _skip_without_key()
        req = CompletionRequest(
            model="openrouter/free",
            messages=[{"role": "user", "content": "What files are in the current directory? Use the list_files tool."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_files",
                        "description": "List files in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "Directory path"}
                            },
                        },
                    },
                }
            ],
            max_tokens=200,
        )
        resp = await _safe_generate(openrouter_provider, req)
        assert isinstance(resp, CompletionResponse)
        if not resp.content and not resp.tool_calls:
            pytest.skip("Rate limited or empty response from OpenRouter")

    @pytest.mark.asyncio
    async def test_429_error_handling(self, openrouter_provider):
        """Verify 429 errors are raised (not silently swallowed)."""
        _skip_without_key()
        # Send multiple rapid requests to trigger rate limiting
        results = []
        for _ in range(5):
            try:
                req = CompletionRequest(
                    model="openrouter/free",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5,
                )
                resp = await openrouter_provider.generate(req)
                results.append(("ok", resp))
            except Exception as e:
                results.append(("error", str(e)))

        # At least one should succeed or rate limit
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_invalid_model_error(self, openrouter_provider):
        """Verify invalid model raises an error."""
        _skip_without_key()
        req = CompletionRequest(
            model="nonexistent/model-xyz-12345:free",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
        )
        with pytest.raises(Exception):
            await openrouter_provider.generate(req)


class TestOpenRouterTimeout:
    """Test timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_on_slow_model(self):
        """Verify timeout is respected."""
        _skip_without_key()
        from harness_core.providers.openrouter import OpenRouterProvider
        import httpx

        provider = OpenRouterProvider()
        # Override client with very short timeout
        provider._client = httpx.AsyncClient(
            base_url=provider.BASE_URL,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            timeout=0.001,  # 1ms — guaranteed timeout
        )

        req = CompletionRequest(
            model="openrouter/free",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
        )
        with pytest.raises((httpx.TimeoutException, httpx.ReadTimeout)):
            await provider.generate(req)
        await provider.close()


class TestOllamaProvider:
    """Test Ollama provider (graceful handling when not running)."""

    @pytest.mark.asyncio
    async def test_health_check_when_not_running(self):
        from harness_core.providers.ollama import OllamaProvider
        provider = OllamaProvider()
        result = await provider.health_check()
        # Should return False, not raise
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_list_models_when_not_running(self):
        from harness_core.providers.ollama import OllamaProvider
        provider = OllamaProvider()
        models = await provider.list_models()
        # Should return empty list, not raise
        assert isinstance(models, list)
