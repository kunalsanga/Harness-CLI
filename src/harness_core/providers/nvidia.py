"""NVIDIA NIM model provider."""

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


class NvidiaProvider(ModelProvider):
    """NVIDIA NIM provider."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = base_url or os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "nvidia"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or os.environ.get("NVIDIA_MODEL", "moonshotai/kimi-k3"),
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

        # Handle content=None (reasoning models return content:null)
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", ""),
            provider=self.name,
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or os.environ.get("NVIDIA_MODEL", "moonshotai/kimi-k3"),
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
        """List available models. 
        NVIDIA NIM hosts many models. Here we expose configured models or fetch from /models.
        """
        try:
            client = await self._get_client()
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()

            models = []
            for m in data.get("data", []):
                model_id = m.get("id", "")
                
                # Assume standard NIM support tools unless it's a very specific old model.
                # In the absence of supported_parameters like OpenRouter, we default to True
                # for chat completion endpoints on modern LLMs.
                
                models.append(
                    ModelInfo(
                        id=model_id,
                        name=model_id,
                        provider=self.name,
                        context_window=128000, # default large
                        supports_tools=True,
                        cost_per_1k_input=0.0,
                        cost_per_1k_output=0.0,
                        is_free=False,
                    )
                )
            
            # If the /models endpoint doesn't return the configured model, inject it.
            configured_model = os.environ.get("NVIDIA_MODEL", "moonshotai/kimi-k3")
            if not any(m.id == configured_model for m in models):
                models.append(
                    ModelInfo(
                        id=configured_model,
                        name=configured_model,
                        provider=self.name,
                        context_window=128000,
                        supports_tools=True,
                    )
                )

            return models
        except Exception:
            # Fallback to a hardcoded list if /models endpoint is not available or fails
            configured_model = os.environ.get("NVIDIA_MODEL", "moonshotai/kimi-k3")
            return [
                ModelInfo(
                    id=configured_model,
                    name=configured_model,
                    provider=self.name,
                    context_window=128000,
                    supports_tools=True,
                ),
                ModelInfo(
                    id="nvidia/nemotron-3-super-120b-a12b",
                    name="nvidia/nemotron-3-super-120b-a12b",
                    provider=self.name,
                    context_window=128000,
                    supports_tools=True,
                )
            ]

    async def health_check(self) -> bool:
        """Check if provider is available by verifying auth and endpoint.
        
        Uses the configured model for a minimal chat completion request
        since NVIDIA NIM /models endpoint can be unreliable.
        """
        if not self.api_key:
            return False
        
        try:
            client = await self._get_client()
            # Try /models first (lighter)
            response = await client.get("/models")
            if response.status_code == 200:
                return True
        except Exception:
            pass
        
        # Fallback: try a minimal chat completion with the configured model
        try:
            client = await self._get_client()
            model = os.environ.get("NVIDIA_MODEL", "moonshotai/kimi-k3")
            body = {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
            response = await client.post("/chat/completions", json=body)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
