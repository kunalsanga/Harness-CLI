"""Base model provider abstraction."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProviderErrorCategory(str, Enum):
    """Provider-neutral failure categories used by routing and failover."""

    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    MODEL_UNAVAILABLE = "model_unavailable"
    NETWORK = "network"
    TIMEOUT = "timeout"
    PROVIDER = "provider"
    INVALID_REQUEST = "invalid_request"
    PAYMENT_REQUIRED = "payment_required"
    TOOL = "tool"
    UNKNOWN = "unknown"


class ProviderRequestError(Exception):
    """Normalized provider request error; vendor details stay in the adapter."""

    def __init__(
        self,
        category: ProviderErrorCategory,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        status_code: int | None = None,
        retry_after: float | None = None,
        provider_code: str | int | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.category = category
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retry_after = retry_after
        self.provider_code = provider_code
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class TokenUsage:
    """Provider-neutral token counts. Missing provider values remain unknown."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    estimated: bool = False

    @classmethod
    def from_provider(cls, raw: dict[str, Any] | None) -> TokenUsage | None:
        if not raw:
            return None
        input_tokens = raw.get("prompt_tokens", raw.get("input_tokens"))
        output_tokens = raw.get("completion_tokens", raw.get("output_tokens"))
        total_tokens = raw.get("total_tokens")
        prompt_details = raw.get("prompt_tokens_details") or raw.get("input_tokens_details") or {}
        completion_details = raw.get("completion_tokens_details") or {}
        cached = raw.get("cached_tokens", prompt_details.get("cached_tokens"))
        reasoning = raw.get("reasoning_tokens", completion_details.get("reasoning_tokens"))
        if all(value is None for value in (input_tokens, output_tokens, total_tokens, cached, reasoning)):
            return None
        return cls(input_tokens, output_tokens, total_tokens, cached, reasoning, False)


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    name: str
    provider: str
    context_window: int = 0
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_structured_output: bool | None = None
    supports_streaming: bool | None = None
    supports_reasoning: bool | None = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    is_free: bool = False
    is_local: bool = False
    latency_ms: float = 0.0
    reliability: float = 1.0
    tags: list[str] = field(default_factory=list)


@dataclass
class CompletionRequest:
    """Request to generate a completion."""

    messages: list[dict[str, Any]]
    model: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:
    """Response from a model completion."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_usage(self) -> TokenUsage | None:
        return TokenUsage.from_provider(self.usage)


class ModelProvider(abc.ABC):
    """Abstract base class for model providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Provider name."""

    @abc.abstractmethod
    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        """Generate a completion."""

    @abc.abstractmethod
    async def stream(self, request: CompletionRequest):
        """Stream a completion."""

    @abc.abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """List available models."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available."""

    async def close(self) -> None:
        """Release provider resources. Override in subclasses."""

    async def routing_hints(self, mode: str) -> list[ModelInfo]:
        """Return provider-specific virtual routes as normalized model entries.

        For example, a gateway may expose a dynamic free route not represented
        by a concrete row in its model catalog. The router consumes only this
        normalized contract and does not branch on provider names.
        """
        return []
