"""FreeLLMAPI model provider adapter.

Provides an OpenAI-compatible interface to local or remote FreeLLMAPI gateways.
"""

from __future__ import annotations

import os
from typing import Any

from harness_core.config.credentials import CredentialResolver
from harness_core.providers.base import ModelInfo
from harness_core.providers.http_adapters import OpenAICompatibleProvider


class FreeLLMAPIProvider(OpenAICompatibleProvider):
    """Adapter for FreeLLMAPI gateway endpoints."""

    provider_name = "freellmapi"
    default_base_url = "http://127.0.0.1:31415/v1"
    default_model = "auto"
    api_key_env = "FREELLMAPI_API_KEY"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        resolver = CredentialResolver()
        cred = resolver.resolve("freellmapi")
        resolved_key = api_key or (cred.api_key if cred else "") or os.environ.get("FREELLMAPI_API_KEY", "")
        resolved_url = (
            base_url
            or (cred.base_url if cred else None)
            or os.environ.get("FREELLMAPI_BASE_URL")
            or self.default_base_url
        )
        resolved_model = (
            default_model
            or (cred.default_model if cred else None)
            or os.environ.get("FREELLMAPI_MODEL")
            or self.default_model
        )

        super().__init__(
            api_key=resolved_key,
            base_url=resolved_url,
            default_model=resolved_model,
            provider_name=self.provider_name,
            api_key_env=self.api_key_env,
            timeout=timeout,
        )

    async def health_check(self) -> bool:
        """Check if FreeLLMAPI gateway is reachable."""
        try:
            # Try /models or /health
            resp = await self._client.get("/models", timeout=5.0)
            if resp.status_code < 500:
                return True
        except Exception:
            pass
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """Discover models from FreeLLMAPI endpoint with sensible fallbacks."""
        try:
            models = await super().list_models()
            if models:
                # If models were returned from /models, ensure they default to tool capability if unknown
                augmented = []
                for m in models:
                    augmented.append(ModelInfo(
                        id=m.id,
                        name=m.name,
                        provider=self.name,
                        context_window=m.context_window or 32768,
                        supports_tools=True if m.supports_tools is not False else False,
                        supports_streaming=True,
                        supports_vision=m.supports_vision,
                        supports_structured_output=m.supports_structured_output,
                        is_free=True,
                        is_local=True,
                    ))
                return augmented
        except Exception:
            pass

        # If gateway doesn't provide /models, expose default model info
        return [
            ModelInfo(
                id=self.default_model_id,
                name=f"FreeLLMAPI ({self.default_model_id})",
                provider=self.name,
                context_window=32768,
                supports_tools=True,
                supports_streaming=True,
                is_free=True,
                is_local=True,
                tags=["freellmapi", "local"],
            )
        ]

