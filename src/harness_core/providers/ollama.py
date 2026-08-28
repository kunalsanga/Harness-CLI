"""Ollama local model provider."""

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


class OllamaProvider(ModelProvider):
    """Ollama local inference provider."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "ollama"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=120.0,
            )
        return self._client

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or "llama3.1",
            "messages": request.messages,
            "stream": False,
        }
        if request.tools:
            body["tools"] = request.tools
        if request.max_tokens:
            body["options"] = {"num_predict": request.max_tokens}

        response = await client.post("/api/chat", json=body)
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        return CompletionResponse(
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls", []),
            model=data.get("model", ""),
            provider=self.name,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            finish_reason="stop" if not message.get("tool_calls") else "tool_calls",
        )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or "llama3.1",
            "messages": request.messages,
            "stream": True,
        }

        async with client.stream("POST", "/api/chat", json=body) as response:
            response.raise_for_status()
            import json
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue

    async def list_models(self) -> list[ModelInfo]:
        client = await self._get_client()
        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            models = []
            for m in data.get("models", []):
                models.append(
                    ModelInfo(
                        id=m.get("name", ""),
                        name=m.get("name", ""),
                        provider=self.name,
                        is_free=True,
                        is_local=True,
                    )
                )
            return models
        except Exception:
            return []

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
