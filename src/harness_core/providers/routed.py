"""Provider interface facade that sends auxiliary model calls through routing."""

from __future__ import annotations

from typing import AsyncGenerator, TYPE_CHECKING

from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelInfo, ModelProvider, ProviderErrorCategory, ProviderRequestError

if TYPE_CHECKING:
    from harness_core.routing.router import ModelRouter


class RoutedModelProvider(ModelProvider):
    """Use the same router/fallback policy for components expecting a provider."""

    def __init__(self, router: ModelRouter, name: str = "routed") -> None:
        self.router = router
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        result = await self.router.execute(request)
        if result.succeeded and result.response:
            return result.response
        category = ProviderErrorCategory.UNKNOWN
        category_by_name = {
            "authentication": ProviderErrorCategory.AUTHENTICATION,
            "auth": ProviderErrorCategory.AUTHENTICATION,
            "rate_limit": ProviderErrorCategory.RATE_LIMIT,
            "model_unavailable": ProviderErrorCategory.MODEL_UNAVAILABLE,
            "timeout": ProviderErrorCategory.TIMEOUT,
            "network": ProviderErrorCategory.NETWORK,
            "invalid_request": ProviderErrorCategory.INVALID_REQUEST,
            "payment": ProviderErrorCategory.PAYMENT_REQUIRED,
            "payment_required": ProviderErrorCategory.PAYMENT_REQUIRED,
            "tool": ProviderErrorCategory.TOOL,
            "provider": ProviderErrorCategory.PROVIDER,
            "server": ProviderErrorCategory.PROVIDER,
        }
        category = category_by_name.get(result.error_category, category)
        raise ProviderRequestError(category, result.final_error or "No compatible model completed the request", provider=self.name, model=result.model_used)

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        response = await self.generate(request)
        if response.content:
            yield response.content

    async def list_models(self) -> list[ModelInfo]:
        return await self.router.refresh_models(force=True)

    async def health_check(self) -> bool:
        return bool(self.router.providers)
