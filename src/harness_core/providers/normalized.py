"""Compatibility wrapper for adapters that still raise wire-level exceptions."""

from __future__ import annotations

from typing import Any, AsyncGenerator

import httpx

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)


class NormalizedModelProvider(ModelProvider):
    """Keep legacy provider exceptions behind the provider-neutral contract."""

    def __init__(self, provider: ModelProvider) -> None:
        self.delegate = provider

    @property
    def name(self) -> str:
        return self.delegate.name

    @staticmethod
    def _normalize(exc: Exception, provider: str, model: str = "") -> ProviderRequestError:
        if isinstance(exc, ProviderRequestError):
            return exc
        if isinstance(exc, httpx.HTTPStatusError):
            response = exc.response
            try:
                from harness_core.providers.http_adapters import _error_category
                try:
                    payload = response.json()
                except ValueError:
                    payload = {}
                error = payload.get("error", payload) if isinstance(payload, dict) else {}
                detail = error.get("message", "") if isinstance(error, dict) else str(error)
                category = _error_category(response.status_code, str(detail))
                try:
                    retry_after = float(response.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    retry_after = None
                return ProviderRequestError(
                    category, f"{provider} request failed: {str(detail)[:400]}",
                    provider=provider, model=model, status_code=response.status_code,
                    retry_after=retry_after,
                )
            except Exception:
                return ProviderRequestError(
                    ProviderErrorCategory.PROVIDER, f"{provider} request failed",
                    provider=provider, model=model, status_code=response.status_code,
                )
        if isinstance(exc, httpx.TimeoutException):
            category = ProviderErrorCategory.TIMEOUT
        elif isinstance(exc, httpx.NetworkError):
            category = ProviderErrorCategory.NETWORK
        else:
            category = ProviderErrorCategory.UNKNOWN
        return ProviderRequestError(
            category, f"{provider} request failed: {type(exc).__name__}",
            provider=provider, model=model,
        )

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        try:
            return await self.delegate.generate(request)
        except Exception as exc:
            raise self._normalize(exc, self.name, request.model or "") from exc

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        try:
            async for item in self.delegate.stream(request):
                yield item
        except Exception as exc:
            raise self._normalize(exc, self.name, request.model or "") from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            return await self.delegate.list_models()
        except Exception as exc:
            raise self._normalize(exc, self.name) from exc

    async def health_check(self) -> bool:
        return await self.delegate.health_check()

    async def routing_hints(self, mode: str) -> list[ModelInfo]:
        return await self.delegate.routing_hints(mode)

    async def close(self) -> None:
        await self.delegate.close()
