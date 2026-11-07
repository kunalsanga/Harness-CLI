"""Azure OpenAI model provider adapter.

Provides support for Azure OpenAI endpoints, deployments, and OpenAI-compatible
gateways hosted on Microsoft Azure.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator

import httpx

from harness_core.config.credentials import CredentialResolver
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.providers.http_adapters import _error_category, _provider_error


class AzureOpenAIProvider(ModelProvider):
    """Adapter for Azure OpenAI service deployments."""

    provider_name = "azure"
    default_api_version = "2024-08-01-preview"
    default_deployment = "gpt-4o"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        api_version: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        resolver = CredentialResolver()
        cred = resolver.resolve("azure")

        self.api_key = (
            api_key
            or (cred.api_key if cred else "")
            or os.environ.get("AZURE_OPENAI_API_KEY", "")
            or os.environ.get("AZURE_API_KEY", "")
        )
        raw_endpoint = (
            base_url
            or (cred.base_url if cred else None)
            or os.environ.get("AZURE_OPENAI_ENDPOINT")
            or os.environ.get("AZURE_OPENAI_BASE_URL")
            or os.environ.get("AZURE_ENDPOINT")
            or ""
        ).rstrip("/")
        self.base_url = raw_endpoint
        self.deployment = (
            default_model
            or (cred.default_model if cred else None)
            or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
            or os.environ.get("AZURE_DEPLOYMENT")
            or os.environ.get("AZURE_OPENAI_MODEL")
            or os.environ.get("AZURE_MODEL")
            or self.default_deployment
        )
        self.api_version = (
            api_version
            or os.environ.get("AZURE_OPENAI_API_VERSION")
            or self.default_api_version
        )
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self.provider_name

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        if self.api_key:
            headers["Authorization"] = (
                self.api_key if self.api_key.startswith("Bearer ") else f"Bearer {self.api_key}"
            )
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url if self.base_url else None,
                headers=self._headers(),
                timeout=self.timeout,
            )
        return self._client

    def _build_completion_url(self, deployment: str) -> str:
        """Construct the chat completions path / URL for Azure.

        Handles both standard Azure endpoints (https://resource.openai.azure.com)
        and OpenAI-compatible gateway endpoints (https://.../v1).
        """
        dep = deployment or self.deployment
        if self.base_url.endswith("/v1"):
            return "/chat/completions"
        if "/openai/deployments/" in self.base_url:
            return f"/chat/completions?api-version={self.api_version}"
        return f"/openai/deployments/{dep}/chat/completions?api-version={self.api_version}"

    def _request_body(self, request: CompletionRequest, deployment: str, *, stream: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": request.messages,
        }
        # In Azure native endpoints, the model is specified in the URL path.
        # However, OpenAI-compatible proxy endpoints also expect a model in the body.
        body["model"] = deployment
        if request.tools:
            body["tools"] = request.tools
        if request.tool_choice:
            body["tool_choice"] = request.tool_choice
        if request.max_tokens is not None:
            body["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if stream:
            body["stream"] = True
        return body

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        client = await self._get_client()
        deployment = request.model or self.deployment
        url = self._build_completion_url(deployment)
        body = self._request_body(request, deployment)

        try:
            response = await client.post(url, json=body, headers=self._headers())
            if response.is_error:
                raise _provider_error(self.name, deployment, response)
            data = response.json()
        except ProviderRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderRequestError(
                ProviderErrorCategory.TIMEOUT,
                f"Azure OpenAI request timed out (deployment={deployment})",
                provider=self.name,
                model=deployment,
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(
                ProviderErrorCategory.NETWORK,
                f"Could not connect to Azure OpenAI at {self.base_url}",
                provider=self.name,
                model=deployment,
            ) from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError(
                ProviderErrorCategory.PROVIDER,
                "Azure OpenAI returned an invalid completion response",
                provider=self.name,
                model=deployment,
            ) from exc

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return CompletionResponse(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or [],
            model=data.get("model") or deployment,
            provider=self.name,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
            finish_reason=choice.get("finish_reason") or "",
            metadata={"provider_response_id": data.get("id", "")},
        )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        client = await self._get_client()
        deployment = request.model or self.deployment
        url = self._build_completion_url(deployment)
        body = self._request_body(request, deployment, stream=True)

        try:
            async with client.stream("POST", url, json=body, headers=self._headers()) as response:
                if response.is_error:
                    await response.aread()
                    raise _provider_error(self.name, deployment, response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    choices = chunk.get("choices") or []
                    if choices:
                        content = (choices[0].get("delta") or {}).get("content")
                        if content:
                            yield str(content)
        except ProviderRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderRequestError(
                ProviderErrorCategory.TIMEOUT,
                f"Azure OpenAI stream timed out (deployment={deployment})",
                provider=self.name,
                model=deployment,
            ) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(
                ProviderErrorCategory.NETWORK,
                f"Could not connect to Azure OpenAI at {self.base_url}",
                provider=self.name,
                model=deployment,
            ) from exc

    async def health_check(self) -> bool:
        """Check if Azure OpenAI endpoint is configured and reachable."""
        if not self.api_key or not self.base_url:
            return False
        try:
            client = await self._get_client()
            # Try deployments endpoint first
            test_url = f"/openai/deployments?api-version={self.api_version}" if not self.base_url.endswith("/v1") else "/models"
            resp = await client.get(test_url, timeout=5.0)
            if resp.status_code < 400:
                return True
            if resp.status_code in (401, 403):
                return False
        except Exception:
            pass

        # Fallback: test minimal completion with configured deployment
        try:
            client = await self._get_client()
            url = self._build_completion_url(self.deployment)
            resp = await client.post(
                url,
                json={"messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
                timeout=5.0,
            )
            return resp.status_code < 400
        except Exception:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List configured deployments and discovered Azure models."""
        deployments: list[ModelInfo] = []

        # 1. Always include the primary configured deployment
        if self.deployment:
            deployments.append(
                ModelInfo(
                    id=self.deployment,
                    name=f"Azure ({self.deployment})",
                    provider=self.name,
                    context_window=128000,
                    supports_tools=True,
                    supports_streaming=True,
                    is_free=False,
                )
            )

        # 2. Try dynamic discovery if endpoint supports listing
        if self.api_key and self.base_url:
            try:
                client = await self._get_client()
                test_url = f"/openai/deployments?api-version={self.api_version}" if not self.base_url.endswith("/v1") else "/models"
                resp = await client.get(test_url, timeout=5.0)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data") or data.get("value") or []
                    for item in items:
                        dep_id = str(item.get("id") or item.get("deployment_name") or "")
                        if dep_id and not any(d.id == dep_id for d in deployments):
                            deployments.append(
                                ModelInfo(
                                    id=dep_id,
                                    name=f"Azure ({dep_id})",
                                    provider=self.name,
                                    context_window=128000,
                                    supports_tools=True,
                                    supports_streaming=True,
                                    is_free=False,
                                )
                            )
            except Exception:
                pass

        return deployments

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
