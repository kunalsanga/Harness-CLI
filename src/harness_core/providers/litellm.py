"""LiteLLM provider — unified interface to 100+ LLM providers.

LiteLLM acts as a proxy that translates OpenAI-compatible requests
to any supported provider (OpenAI, Anthropic, Azure, Bedrock, etc.).
"""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator

import httpx

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)


class LiteLLMProvider(ModelProvider):
    """LiteLLM unified provider.

    Requires LITELLM_API_KEY or OPENAI_API_KEY for the proxy,
    or can connect to a local LiteLLM proxy server.
    """

    DEFAULT_BASE_URL = "https://api.litellm.ai"
    LOCAL_BASE_URL = "http://localhost:4000"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY", "")
        self.base_url = (
            base_url
            or os.environ.get("LITELLM_API_BASE", "")
            or (self.LOCAL_BASE_URL if not self.api_key else self.DEFAULT_BASE_URL)
        )
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "litellm"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=120.0,
            )
        return self._client

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or "gpt-4o-mini",
            "messages": request.messages,
        }
        if request.tools:
            body["tools"] = request.tools
        if request.tool_choice:
            body["tool_choice"] = request.tool_choice
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature

        response = await client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        return CompletionResponse(
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls", []),
            model=data.get("model", ""),
            provider=self.name,
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or "gpt-4o-mini",
            "messages": request.messages,
            "stream": True,
        }
        if request.tools:
            body["tools"] = request.tools
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens

        async with client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

    async def list_models(self) -> list[ModelInfo]:
        """List models supported by LiteLLM.

        For the hosted proxy, this queries /model/info.
        For local proxies, returns a sensible default set.
        """
        try:
            client = await self._get_client()
            response = await client.get("/model/info")
            if response.status_code == 200:
                data = response.json()
                models = []
                for m in data.get("data", []):
                    model_info = m.get("model_info", {})
                    models.append(ModelInfo(
                        id=m.get("model_name", ""),
                        name=m.get("model_name", ""),
                        provider=self.name,
                        context_window=model_info.get("max_input_tokens", 0),
                        supports_tools=True,  # LiteLLM handles tool translation
                        is_free=False,
                        is_local="localhost" in self.base_url,
                    ))
                return models
        except Exception:
            pass

        # Fallback: return common models
        return [
            ModelInfo(
                id="gpt-4o-mini",
                name="GPT-4o Mini",
                provider=self.name,
                context_window=128000,
                supports_tools=True,
                is_local="localhost" in self.base_url,
            ),
            ModelInfo(
                id="gpt-4o",
                name="GPT-4o",
                provider=self.name,
                context_window=128000,
                supports_tools=True,
                is_local="localhost" in self.base_url,
            ),
            ModelInfo(
                id="claude-3-5-sonnet-20241022",
                name="Claude 3.5 Sonnet",
                provider=self.name,
                context_window=200000,
                supports_tools=True,
                is_local="localhost" in self.base_url,
            ),
        ]

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
