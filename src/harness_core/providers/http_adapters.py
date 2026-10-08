"""HTTP adapters for native and OpenAI-compatible chat APIs.

Wire formats, authentication headers, error responses, streaming and tool-call
translation live here. The rest of Harness sees only the base provider types.
"""

from __future__ import annotations

import json
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


def _error_category(status: int, message: str) -> ProviderErrorCategory:
    low = message.lower()
    if status == 401 or "unauthorized" in low or "invalid api key" in low:
        return ProviderErrorCategory.AUTHENTICATION
    if status == 403 or "forbidden" in low:
        return ProviderErrorCategory.AUTHENTICATION
    if status == 429 or "rate limit" in low or "too many requests" in low:
        return ProviderErrorCategory.RATE_LIMIT
    if status == 402 or "payment required" in low:
        return ProviderErrorCategory.PAYMENT_REQUIRED
    if status in (404, 410, 502, 503, 504) or any(
        phrase in low for phrase in ("model unavailable", "no endpoints", "not found")
    ):
        return ProviderErrorCategory.MODEL_UNAVAILABLE
    if status in (400, 413, 422):
        return ProviderErrorCategory.INVALID_REQUEST
    if status >= 500:
        return ProviderErrorCategory.PROVIDER
    return ProviderErrorCategory.UNKNOWN


def _provider_error(
    provider: str, model: str, response: httpx.Response
) -> ProviderRequestError:
    try:
        body = response.json()
    except ValueError:
        body = {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("detail") or error.get("code") or "Provider request failed")
        code = error.get("code")
    else:
        message, code = str(error or "Provider request failed"), None
    try:
        retry_after = float(response.headers.get("Retry-After", ""))
    except ValueError:
        retry_after = None
    return ProviderRequestError(
        _error_category(response.status_code, message),
        f"{provider} request failed: {message[:400]}",
        provider=provider,
        model=model,
        status_code=response.status_code,
        retry_after=retry_after,
        provider_code=code,
    )


class OpenAICompatibleProvider(ModelProvider):
    """Adapter for APIs implementing the OpenAI chat-completions contract."""

    provider_name = "openai-compatible"
    default_base_url = ""
    default_model = ""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        provider_name: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._name = provider_name or self.provider_name
        self.api_key_env = api_key_env
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.default_model_id = default_model or self.default_model
        credential = CredentialResolver().resolve(self._name)
        self.api_key = api_key or (credential.api_key if credential else "")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers(),
            timeout=timeout,
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def name(self) -> str:
        return self._name

    def _request_body(self, request: CompletionRequest, *, stream: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model or self.default_model_id,
            "messages": request.messages,
        }
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
        body = self._request_body(request)
        model = str(body.get("model") or "")
        try:
            response = await self._client.post("/chat/completions", json=body)
            if response.is_error:
                raise _provider_error(self.name, model, response)
            data = response.json()
        except ProviderRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, f"{self.name} request timed out", provider=self.name, model=model) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, f"Could not connect to {self.name}", provider=self.name, model=model) from exc
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderRequestError(ProviderErrorCategory.PROVIDER, f"{self.name} returned an invalid completion response", provider=self.name, model=model) from exc
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return CompletionResponse(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or [],
            model=data.get("model") or model,
            provider=self.name,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
            finish_reason=choice.get("finish_reason") or "",
            metadata={"provider_response_id": data.get("id", "")},
        )

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        body = self._request_body(request, stream=True)
        model = str(body.get("model") or "")
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.is_error:
                    raise _provider_error(self.name, model, response)
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
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, f"{self.name} stream timed out", provider=self.name, model=model) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, f"Could not connect to {self.name}", provider=self.name, model=model) from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.get("/models")
            if response.is_error:
                raise _provider_error(self.name, "", response)
            data = response.json()
        except ProviderRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, f"{self.name} model discovery timed out", provider=self.name) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, f"Could not connect to {self.name}", provider=self.name) from exc
        result: list[ModelInfo] = []
        for item in data.get("data", []):
            if not isinstance(item, dict) or not item.get("id"):
                continue
            supported = item.get("supported_parameters")
            tool_support = ("tools" in supported) if isinstance(supported, (list, tuple, set)) else None
            result.append(ModelInfo(
                id=str(item["id"]), name=str(item.get("name") or item["id"]), provider=self.name,
                context_window=int(item.get("context_length") or item.get("context_window") or 0),
                supports_tools=tool_support,
                supports_vision=None, supports_structured_output=None,
                supports_streaming=None, supports_reasoning=None,
                is_free=float((item.get("pricing") or {}).get("prompt", 1) or 0) == 0,
            ))
        return result

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/models")
            return response.status_code < 500
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()


class OpenAIProvider(OpenAICompatibleProvider):
    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    default_model = "gpt-4o-mini"
    api_key_env = "OPENAI_API_KEY"


class AnthropicProvider(ModelProvider):
    """Native Anthropic Messages API adapter."""

    def __init__(self, *, api_key: str | None = None, base_url: str = "https://api.anthropic.com", default_model: str = "claude-sonnet-4-20250514", timeout: float = 120.0) -> None:
        cred = CredentialResolver().resolve("anthropic")
        self.api_key = api_key or (cred.api_key if cred else "")
        self.base_url = base_url.rstrip("/")
        self.default_model_id = default_model
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})

    @property
    def name(self) -> str:
        return "anthropic"

    def _body(self, request: CompletionRequest) -> dict[str, Any]:
        system = "\n".join(str(m.get("content", "")) for m in request.messages if m.get("role") == "system")
        messages = []
        for message in request.messages:
            role = message.get("role", "user")
            if role == "system":
                continue
            if role == "tool":
                messages.append({"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": message.get("tool_call_id", ""),
                    "content": str(message.get("content", "")),
                }]})
                continue
            blocks: list[dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": str(message["content"])})
            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments", "{}"))
                except (ValueError, TypeError):
                    arguments = {}
                blocks.append({"type": "tool_use", "id": tool_call.get("id", ""), "name": function.get("name", ""), "input": arguments})
            messages.append({"role": "assistant" if role == "assistant" else "user", "content": blocks or ""})
        body: dict[str, Any] = {"model": request.model or self.default_model_id, "messages": messages, "max_tokens": request.max_tokens or 4096}
        if system:
            body["system"] = system
        if request.tools:
            body["tools"] = [{"name": t["function"]["name"], "description": t["function"].get("description", ""), "input_schema": t["function"].get("parameters", {})} for t in request.tools]
        if request.temperature is not None:
            body["temperature"] = request.temperature
        return body

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        body = self._body(request)
        try:
            response = await self._client.post("/v1/messages", json=body)
            if response.is_error:
                raise _provider_error(self.name, body["model"], response)
            data = response.json()
        except ProviderRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, "Anthropic request timed out", provider=self.name, model=body["model"]) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, "Could not connect to Anthropic", provider=self.name, model=body["model"]) from exc
        content, calls = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append({"id": block.get("id", ""), "type": "function", "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input", {}))}})
        usage = data.get("usage") or {}
        normalized_usage = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "total_tokens": (usage.get("input_tokens", 0) + usage.get("output_tokens", 0))}
        return CompletionResponse("\n".join(content), calls, data.get("model", body["model"]), self.name, normalized_usage, data.get("stop_reason", ""))

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        body = self._body(request)
        body["stream"] = True
        try:
            async with self._client.stream("POST", "/v1/messages", json=body) as response:
                if response.is_error:
                    raise _provider_error(self.name, body["model"], response)
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            event = json.loads(line[5:].strip())
                        except ValueError:
                            continue
                        if event.get("type") == "content_block_delta":
                            text = (event.get("delta") or {}).get("text")
                            if text:
                                yield text
        except ProviderRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, "Anthropic stream timed out", provider=self.name, model=body["model"]) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, "Could not connect to Anthropic", provider=self.name, model=body["model"]) from exc

    async def list_models(self) -> list[ModelInfo]:
        # Anthropic does not expose a generally available public model catalog.
        return []

    async def health_check(self) -> bool:
        return bool(self.api_key)

    async def close(self) -> None:
        await self._client.aclose()


class GeminiProvider(ModelProvider):
    """Native Google Gemini generateContent API adapter."""

    def __init__(self, *, api_key: str | None = None, base_url: str = "https://generativelanguage.googleapis.com/v1beta", default_model: str = "gemini-2.5-flash", timeout: float = 120.0) -> None:
        cred = CredentialResolver().resolve("gemini")
        self.api_key = api_key or (cred.api_key if cred else "")
        self.base_url = base_url.rstrip("/")
        self.default_model_id = default_model
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    @property
    def name(self) -> str:
        return "gemini"

    def _body(self, request: CompletionRequest) -> dict[str, Any]:
        contents = []
        system = []
        for message in request.messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if role == "system":
                system.append(content)
            elif role == "tool":
                try:
                    result = json.loads(content)
                except ValueError:
                    result = {"result": content}
                contents.append({"role": "user", "parts": [{"functionResponse": {"name": message.get("name", "tool"), "response": result}}]})
            else:
                parts: list[dict[str, Any]] = []
                if content:
                    parts.append({"text": content})
                for tool_call in message.get("tool_calls", []) or []:
                    function = tool_call.get("function", {})
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                    except (ValueError, TypeError):
                        arguments = {}
                    parts.append({"functionCall": {"name": function.get("name", ""), "args": arguments}})
                contents.append({"role": "model" if role == "assistant" else "user", "parts": parts or [{"text": ""}]})
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n".join(system)}]}
        generation: dict[str, Any] = {}
        if request.max_tokens is not None:
            generation["maxOutputTokens"] = request.max_tokens
        if request.temperature is not None:
            generation["temperature"] = request.temperature
        if generation:
            body["generationConfig"] = generation
        if request.tools:
            declarations = []
            for tool in request.tools:
                fn = tool.get("function", {})
                declarations.append({"name": fn.get("name", ""), "description": fn.get("description", ""), "parameters": fn.get("parameters", {})})
            body["tools"] = [{"functionDeclarations": declarations}]
        return body

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or self.default_model_id
        try:
            response = await self._client.post(f"/models/{model}:generateContent", params={"key": self.api_key}, json=self._body(request))
            if response.is_error:
                raise _provider_error(self.name, model, response)
            data = response.json()
        except ProviderRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, "Gemini request timed out", provider=self.name, model=model) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, "Could not connect to Gemini", provider=self.name, model=model) from exc
        candidate = (data.get("candidates") or [{}])[0]
        content, calls = [], []
        for part in (candidate.get("content") or {}).get("parts", []):
            if part.get("text"):
                content.append(part["text"])
            if part.get("functionCall"):
                call = part["functionCall"]
                calls.append({"id": call.get("id", call.get("name", "")), "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("args", {}))}})
        usage = data.get("usageMetadata") or {}
        normalized_usage = {"input_tokens": usage.get("promptTokenCount"), "output_tokens": usage.get("candidatesTokenCount"), "total_tokens": usage.get("totalTokenCount")}
        return CompletionResponse("\n".join(content), calls, model, self.name, normalized_usage, candidate.get("finishReason", ""))

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        model = request.model or self.default_model_id
        try:
            async with self._client.stream("POST", f"/models/{model}:streamGenerateContent", params={"key": self.api_key, "alt": "sse"}, json=self._body(request)) as response:
                if response.is_error:
                    raise _provider_error(self.name, model, response)
                async for line in response.aiter_lines():
                    payload = line[5:].strip() if line.startswith("data:") else ""
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                        for part in (data.get("candidates") or [{}])[0].get("content", {}).get("parts", []):
                            if part.get("text"):
                                yield part["text"]
                    except ValueError:
                        continue
        except ProviderRequestError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderRequestError(ProviderErrorCategory.TIMEOUT, "Gemini stream timed out", provider=self.name, model=model) from exc
        except httpx.NetworkError as exc:
            raise ProviderRequestError(ProviderErrorCategory.NETWORK, "Could not connect to Gemini", provider=self.name, model=model) from exc

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.get("/models", params={"key": self.api_key})
            if response.is_error:
                raise _provider_error(self.name, "", response)
            data = response.json()
        except ProviderRequestError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError):
            return []
        return [ModelInfo(id=m["name"].removeprefix("models/"), name=m.get("displayName", m["name"]), provider=self.name, supports_tools=("functionCalling" in m.get("supportedGenerationMethods", [])), supports_streaming=("streamGenerateContent" in m.get("supportedGenerationMethods", [])), context_window=int(m.get("inputTokenLimit", 0) or 0)) for m in data.get("models", []) if m.get("name") and "generateContent" in m.get("supportedGenerationMethods", [])]

    async def health_check(self) -> bool:
        return bool(self.api_key)

    async def close(self) -> None:
        await self._client.aclose()
