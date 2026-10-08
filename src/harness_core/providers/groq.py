"""Groq model provider."""

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


class GroqProvider(ModelProvider):
    """Groq API provider."""

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, default_model: str | None = None) -> None:
        # Use CredentialResolver if no explicit key provided
        if api_key:
            self.api_key = api_key
        else:
            try:
                from harness_core.config.credentials import CredentialResolver
                resolver = CredentialResolver()
                cred = resolver.resolve("groq")
                self.api_key = cred.api_key if cred else ""
            except Exception:
                self.api_key = os.environ.get("GROQ_API_KEY", "")
        self._client: httpx.AsyncClient | None = None
        self.base_url = base_url or self.BASE_URL
        self.default_model = default_model or "llama-3.1-8b-instant"

    @property
    def name(self) -> str:
        return "groq"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    @staticmethod
    def _extract_provider_detail(response) -> str:
        """Extract a safe detail from an error response without leaking secrets."""
        try:
            data = response.json()
            if isinstance(data, dict):
                err = data.get("error")
                if isinstance(err, dict):
                    msg = err.get("message") or err.get("code") or ""
                    if msg:
                        return str(msg)[:500]
                for k in ("message", "error", "detail"):
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()[:500]
                import json as _json
                return _json.dumps(data)[:500]
        except Exception:
            pass
        try:
            text = response.text or ""
            return text.strip()[:500]
        except Exception:
            return ""

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or self.default_model,
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

        try:
            response = await client.post("/chat/completions", json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if hasattr(e, "response") and e.response is not None else 0
            detail = self._extract_provider_detail(e.response) if hasattr(e, "response") and e.response is not None else str(e)
            model_id = request.model or body.get("model", "")
            raise RuntimeError(f"Groq {status} {detail} (model={model_id})" if detail else f"Groq {status} error (model={model_id})") from e
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Groq timeout after {client.timeout.read or 120}s (model={request.model or body.get('model','')})") from e
        except httpx.NetworkError as e:
            raise RuntimeError(f"Groq network error: {type(e).__name__} (model={request.model or body.get('model','')})") from e

        try:
            data = response.json()
        except Exception as e:
            raise RuntimeError(f"Groq invalid response (non-JSON) status={response.status_code} (model={request.model or body.get('model','')})") from e

        choice = data.get("choices", [{}])[0] if isinstance(data.get("choices"), list) and data.get("choices") else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        if not content and not tool_calls and isinstance(data.get("error"), dict):
            err_msg = data["error"].get("message", "")[:500]
            if err_msg:
                raise RuntimeError(f"Groq error: {err_msg} (model={request.model or body.get('model','')})")

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", ""),
            provider=self.name,
            usage=data.get("usage", {}) if isinstance(data.get("usage"), dict) else {},
            finish_reason=choice.get("finish_reason", "") if isinstance(choice, dict) else "",
        )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        client = await self._get_client()

        body: dict[str, Any] = {
            "model": request.model or self.default_model,
            "messages": request.messages,
            "stream": True,
        }
        if request.tools:
            body["tools"] = request.tools
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature

        try:
            async with client.stream("POST", "/chat/completions", json=body) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if hasattr(e, "response") and e.response is not None else 0
                    detail = self._extract_provider_detail(e.response) if hasattr(e, "response") and e.response is not None else str(e)
                    raise RuntimeError(f"Groq {status} {detail} (model={request.model or body.get('model','')})" if detail else f"Groq {status} error (model={request.model or body.get('model','')})") from e
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            if isinstance(chunk, dict) and chunk.get("error"):
                                err = chunk["error"]
                                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                                raise RuntimeError(f"Groq stream error: {str(msg)[:500]}")
                            delta = chunk.get("choices", [{}])[0].get("delta", {}) if isinstance(chunk.get("choices"), list) else {}
                            content = delta.get("content", "") if isinstance(delta, dict) else ""
                            if content:
                                yield content
                        except RuntimeError:
                            raise
                        except Exception:
                            continue
        except httpx.TimeoutException as e:
            raise RuntimeError(f"Groq stream timeout (model={request.model or body.get('model','')})") from e
        except httpx.NetworkError as e:
            raise RuntimeError(f"Groq stream network error: {type(e).__name__} (model={request.model or body.get('model','')})") from e

    async def list_models(self) -> list[ModelInfo]:
        client = await self._get_client()
        response = await client.get("/models")
        response.raise_for_status()
        data = response.json()

        models = []
        for m in data.get("data", []):
            models.append(
                ModelInfo(
                    id=m.get("id", ""),
                    name=m.get("id", ""),
                    provider=self.name,
                    context_window=m.get("context_window", 8192), # Groq doesn't send this in /models easily, default to 8k
                    supports_tools=True,
                    cost_per_1k_input=0,
                    cost_per_1k_output=0,
                    is_free=False,
                )
            )
        return models

    async def health_check(self) -> bool:
        """Check if the provider is available AND the API key is valid."""
        if not self.api_key:
            return False
        try:
            client = await self._get_client()
            response = await client.get("/models")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
