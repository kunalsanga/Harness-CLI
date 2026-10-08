"""9router provider — automatic model routing backend.

Architecture:
    Harness → NineRouterProvider → 9router gateway → automatic model selection

9router owns model selection. Harness delegates routing entirely to the
gateway. The provider uses the gateway's ``qd/auto`` routing model by
default. If the gateway returns an error, Harness reports it directly
rather than silently selecting an underlying model.

The response ``model`` field is captured as diagnostic metadata — the
user never needs to see or choose a model ID.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator

import httpx
import json

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)

logger = logging.getLogger(__name__)

# The 9router gateway's automatic routing entry point.
# Harness sends requests with this model identifier and lets
# 9router decide which underlying model to use.
_DEFAULT_ROUTING_MODEL = "qd/auto"


class NineRouterProvider(ModelProvider):
    """9router automatic routing backend.

    Connects to the local 9router gateway. All model selection is
    delegated to 9router — Harness never picks an underlying model.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:20128/v1",
    ) -> None:
        self._name = "9router"
        self.base_url = base_url.rstrip("/")
        # Last model reported by the gateway (diagnostic only)
        self.last_selected_model: str = ""
        if api_key:
            self.api_key = api_key
        else:
            try:
                from harness_core.config.credentials import CredentialResolver

                resolver = CredentialResolver()
                cred = resolver.resolve("9router")
                self.api_key = cred.api_key if cred else ""
            except Exception:
                self.api_key = os.environ.get("NINE_ROUTER_API_KEY", "")

    @property
    def name(self) -> str:
        return self._name

    def _get_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=180.0,
        )

    async def health_check(self) -> bool:
        """Check if the 9router gateway is reachable."""
        try:
            if not self.api_key:
                return False
            async with self._get_client() as client:
                resp = await client.get("/models")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List models exposed by the 9router gateway.

        This is informational/diagnostic. Normal operation uses
        automatic routing via ``qd/auto`` — users never need to
        choose from this list.
        """
        try:
            async with self._get_client() as client:
                resp = await client.get("/models")
                if resp.status_code != 200:
                    return []
                data = resp.json()
                models = []
                for m in data.get("data", []):
                    model_id = m.get("id", "")
                    if not model_id:
                        continue

                    capabilities = m.get("capabilities", {})
                    supports_tools = capabilities.get("tools", False)
                    context_length = m.get(
                        "context_length", m.get("context_window", 8192)
                    )

                    models.append(
                        ModelInfo(
                            id=model_id,
                            name=model_id.split("/")[-1]
                            if "/" in model_id
                            else model_id,
                            provider=self.name,
                            context_window=context_length,
                            supports_tools=supports_tools,
                            cost_per_1k_input=0.0,
                            cost_per_1k_output=0.0,
                            is_free=False,
                        )
                    )
                return models
        except Exception as e:
            logger.warning("9router list_models failed: %s", e)
            return []

    # ── Request helpers ────────────────────────────────────────────────

    def _resolve_model(self, req_model: str | None) -> str:
        """Determine the model identifier to send to the gateway.

        If the request has an explicit model (advanced/diagnostic use),
        honour it. Otherwise use the automatic routing entry point.
        """
        if req_model and req_model.strip():
            return req_model
        return _DEFAULT_ROUTING_MODEL

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format messages for the OpenAI-compatible gateway."""
        formatted = []
        for msg in messages:
            entry: dict[str, Any] = {"role": msg["role"]}
            if msg["role"] == "assistant" and "tool_calls" in msg:
                entry["content"] = msg.get("content", "")
                entry["tool_calls"] = msg["tool_calls"]
            elif msg["role"] == "tool":
                entry["content"] = msg.get("content", "")
                entry["tool_call_id"] = msg.get("tool_call_id", "")
            else:
                entry["content"] = msg.get("content", "")
            formatted.append(entry)
        return formatted

    # ── Core API methods ───────────────────────────────────────────────

    async def complete(self, req: CompletionRequest) -> dict[str, Any]:
        """Send a non-streaming completion request to 9router."""
        model = self._resolve_model(req.model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(req.messages),
            "stream": False,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens:
            payload["max_tokens"] = req.max_tokens
        if req.tools:
            payload["tools"] = req.tools
            payload["tool_choice"] = req.tool_choice or "auto"

        async with self._get_client() as client:
            resp = await client.post("/chat/completions", json=payload)
            if resp.status_code != 200:
                raise Exception(
                    f"9router error {resp.status_code}: {resp.text}"
                )
            result = resp.json()
            # Capture which model 9router actually selected (diagnostic)
            self.last_selected_model = result.get("model", "")
            return result

    async def generate(self, req: CompletionRequest) -> CompletionResponse:
        """Generate a completion, returning a structured CompletionResponse."""
        raw = await self.complete(req)

        choice = raw.get("choices", [{}])[0]
        message = choice.get("message", {})

        return CompletionResponse(
            content=message.get("content", ""),
            tool_calls=message.get("tool_calls", []),
            model=raw.get("model", self._resolve_model(req.model)),
            provider=self.name,
            usage=raw.get("usage", {}),
            finish_reason=choice.get("finish_reason", ""),
            metadata={"routing_model": self._resolve_model(req.model)},
        )

    async def stream(
        self, req: CompletionRequest
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a completion from 9router."""
        model = self._resolve_model(req.model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": self._format_messages(req.messages),
            "stream": True,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.max_tokens:
            payload["max_tokens"] = req.max_tokens
        if req.tools:
            payload["tools"] = req.tools
            payload["tool_choice"] = req.tool_choice or "auto"

        async with self._get_client() as client:
            async with client.stream(
                "POST", "/chat/completions", json=payload
            ) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise Exception(
                        f"9router stream error {resp.status_code}: "
                        f"{text.decode('utf-8')}"
                    )

                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            # Capture selected model from stream
                            if data.get("model"):
                                self.last_selected_model = data["model"]
                            yield data
                        except Exception:
                            continue

