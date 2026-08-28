"""OpenRouter model provider."""

from __future__ import annotations

import os
from typing import Any, AsyncGenerator

import httpx

from harness_core.agent.types import ToolResult, ToolResultStatus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)


class OpenRouterProvider(ModelProvider):
    """OpenRouter multi-model gateway provider."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "openrouter"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/harness-engineering",
                    "X-Title": "Harness Engineering CLI",
                },
                timeout=120.0,
            )
        return self._client

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or "inclusionai/ling-3.0-flash-fin:free",
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
            "model": request.model or "meta-llama/llama-3.1-8b-instruct:free",
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
        client = await self._get_client()
        response = await client.get("/models")
        response.raise_for_status()
        data = response.json()

        models = []
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))

            models.append(
                ModelInfo(
                    id=m.get("id", ""),
                    name=m.get("name", ""),
                    provider=self.name,
                    context_window=m.get("context_length", 0),
                    supports_tools="tool" in str(m.get("supported_parameters", [])),
                    cost_per_1k_input=prompt_price * 1000,
                    cost_per_1k_output=completion_price * 1000,
                    is_free=prompt_price == 0 and completion_price == 0,
                )
            )
        return models

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
