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
    ProviderErrorCategory,
    ProviderRequestError,
)


def _retry_after(response) -> float | None:
    try:
        return float(response.headers.get("Retry-After", ""))
    except (AttributeError, TypeError, ValueError):
        return None


def _error_code(response):
    try:
        error = response.json().get("error", {})
        return error.get("code") if isinstance(error, dict) else None
    except Exception:
        return None


def _error_retry_after(error: dict) -> float | None:
    metadata = error.get("metadata", {}) if isinstance(error, dict) else {}
    if not isinstance(metadata, dict):
        return None
    try:
        return float(metadata.get("retry_after"))
    except (TypeError, ValueError):
        return None


def _openrouter_error_category(code, error_type: str, message: str) -> ProviderErrorCategory:
    from harness_core.providers.http_adapters import _error_category
    try:
        status = int(code)
    except (TypeError, ValueError):
        status = 0
    if error_type in {"provider_unavailable", "model_not_found"}:
        return ProviderErrorCategory.MODEL_UNAVAILABLE
    if error_type in {"rate_limit", "rate_limited"}:
        return ProviderErrorCategory.RATE_LIMIT
    return _error_category(status, message)


def _normalize_openrouter_usage(raw) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    normalized = dict(raw)
    details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    if isinstance(details, dict) and "cached_tokens" in details:
        normalized["cached_tokens"] = details["cached_tokens"]
    if isinstance(completion_details, dict) and "reasoning_tokens" in completion_details:
        normalized["reasoning_tokens"] = completion_details["reasoning_tokens"]
    return normalized


class OpenRouterProvider(ModelProvider):
    """OpenRouter multi-model gateway provider."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, default_model: str | None = None) -> None:
        # Use CredentialResolver if no explicit key provided
        if api_key:
            self.api_key = api_key
        else:
            try:
                from harness_core.config.credentials import CredentialResolver
                resolver = CredentialResolver()
                cred = resolver.resolve("openrouter")
                self.api_key = cred.api_key if cred else ""
                if base_url is None and cred and cred.base_url:
                    base_url = cred.base_url
                if default_model is None and cred and cred.default_model:
                    default_model = cred.default_model
            except Exception:
                self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._client: httpx.AsyncClient | None = None
        self.base_url = (base_url or self.BASE_URL).rstrip("/")
        self.default_model = default_model or "openrouter/free"

    @property
    def name(self) -> str:
        return "openrouter"

    async def routing_hints(self, mode: str) -> list[ModelInfo]:
        """Expose OpenRouter's dynamic free route as a normalized catalog entry.

        Lives at the provider boundary — core routing never hardcodes this ID.
        """
        return [ModelInfo(
            id="openrouter/free",
            name="OpenRouter free router",
            provider=self.name,
            supports_tools=True,
            supports_streaming=True,
            is_free=True,
            tags=["dynamic-route"],
        )]

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/harness-engineering",
                    "X-Title": "Harness Engineering CLI",
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
            from harness_core.providers.http_adapters import _error_category
            raise ProviderRequestError(_error_category(status, detail), detail or f"OpenRouter HTTP {status}", provider=self.name, model=model_id, status_code=status, retry_after=_retry_after(e.response), provider_code=_error_code(e.response)) from e
        except httpx.TimeoutException as e:
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, "OpenRouter request timed out", provider=self.name, model=request.model or body.get("model", "")) from e
        except httpx.NetworkError as e:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, "Could not connect to OpenRouter", provider=self.name, model=request.model or body.get("model", "")) from e

        try:
            data = response.json()
        except Exception as e:
            raise ProviderRequestError(
                ProviderErrorCategory.PROVIDER,
                f"OpenRouter invalid response (non-JSON) status={response.status_code}",
                provider=self.name,
                model=request.model or body.get("model", ""),
                status_code=response.status_code,
            ) from e

        choice = data.get("choices", [{}])[0] if isinstance(data.get("choices"), list) and data.get("choices") else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content") or ""
        tool_calls = message.get("tool_calls") or []
        if not content and not tool_calls and isinstance(data.get("error"), dict):
            err_msg = data["error"].get("message", "")[:500]
            # OpenRouter wraps provider failures in HTTP 200. The structured
            # code and error_type are the only reliable failover signals —
            # dropping them makes the failure unclassifiable (UNKNOWN), which
            # terminated the whole fallback chain (observed live with
            # nvidia/nemotron-3-nano: code 502, error_type provider_unavailable).
            err_code = data["error"].get("code", "")
            err_meta = data["error"].get("metadata") if isinstance(data["error"].get("metadata"), dict) else {}
            err_type = err_meta.get("error_type", "")
            parts = [p for p in (f"code {err_code}" if err_code != "" else "", err_type) if p]
            prefix = f" ({', '.join(parts)})" if parts else ""
            if err_msg or prefix:
                category = _openrouter_error_category(err_code, err_type, err_msg)
                raise ProviderRequestError(category, err_msg or "OpenRouter returned a provider error", provider=self.name, model=request.model or body.get("model", ""), status_code=int(err_code) if str(err_code).isdigit() else None, provider_code=err_code or err_type, retry_after=_error_retry_after(data["error"]))

        return CompletionResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", ""),
            provider=self.name,
            usage=_normalize_openrouter_usage(data.get("usage")),
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
                    raise ProviderRequestError(_openrouter_error_category(status, "", detail), detail or f"OpenRouter HTTP {status}", provider=self.name, model=request.model or body.get("model", ""), status_code=status, retry_after=_retry_after(e.response)) from e
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
                                code = err.get("code", "") if isinstance(err, dict) else ""
                                metadata = err.get("metadata", {}) if isinstance(err, dict) else {}
                                error_type = metadata.get("error_type", "") if isinstance(metadata, dict) else ""
                                raise ProviderRequestError(_openrouter_error_category(code, error_type, str(msg)), str(msg)[:500], provider=self.name, model=request.model or body.get("model", ""), status_code=int(code) if str(code).isdigit() else None, provider_code=code or error_type)
                            delta = chunk.get("choices", [{}])[0].get("delta", {}) if isinstance(chunk.get("choices"), list) else {}
                            content = delta.get("content", "") if isinstance(delta, dict) else ""
                            if content:
                                yield content
                        except ProviderRequestError:
                            raise
                        except Exception:
                            continue
        except httpx.TimeoutException as e:
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, "OpenRouter stream timed out", provider=self.name, model=request.model or body.get("model", "")) from e
        except httpx.NetworkError as e:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, "Could not connect to OpenRouter", provider=self.name, model=request.model or body.get("model", "")) from e

    async def list_models(self) -> list[ModelInfo]:
        client = await self._get_client()
        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            detail = self._extract_provider_detail(e.response) if e.response is not None else ""
            from harness_core.providers.http_adapters import _error_category
            raise ProviderRequestError(
                _error_category(status, detail),
                f"OpenRouter model discovery failed: {detail or status}",
                provider=self.name,
                status_code=status,
                retry_after=_retry_after(e.response),
            ) from e
        except httpx.TimeoutException as e:
            raise ProviderRequestError(
                ProviderErrorCategory.TIMEOUT, "OpenRouter model discovery timed out",
                provider=self.name,
            ) from e
        except httpx.NetworkError as e:
            raise ProviderRequestError(
                ProviderErrorCategory.NETWORK, "Could not connect to OpenRouter for model discovery",
                provider=self.name,
            ) from e
        except ValueError as e:
            raise ProviderRequestError(
                ProviderErrorCategory.PROVIDER, "OpenRouter returned an invalid model catalog",
                provider=self.name,
            ) from e

        models = []
        for m in data.get("data", []):
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))
            supported = m.get("supported_parameters")

            models.append(
                ModelInfo(
                    id=m.get("id", ""),
                    name=m.get("name", ""),
                    provider=self.name,
                    context_window=m.get("context_length", 0),
                    supports_tools=(
                        any(p in supported for p in ("tools", "tool_choice"))
                        if isinstance(supported, (list, tuple, set)) else None
                    ),
                    supports_streaming=True,
                    supports_vision=None,
                    supports_structured_output=None,
                    cost_per_1k_input=prompt_price * 1000,
                    cost_per_1k_output=completion_price * 1000,
                    is_free=prompt_price == 0 and completion_price == 0,
                )
            )
        return models

    async def health_check(self) -> bool:
        """Check if the provider is available AND the API key is valid.

        First checks if the key is non-empty, then verifies it against
        the /auth/key endpoint which validates the actual credential.
        """
        if not self.api_key:
            return False
        try:
            client = await self._get_client()
            # /auth/key validates the actual API key (returns 401 if invalid)
            response = await client.get("/auth/key")
            return response.status_code == 200
        except Exception:
            # Fallback: check if models endpoint works (less reliable)
            try:
                client = await self._get_client()
                response = await client.get("/models")
                return response.status_code == 200
            except Exception:
                return False

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
