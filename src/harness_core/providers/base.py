"""Base model provider abstraction."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelInfo:
    """Information about a model."""

    id: str
    name: str
    provider: str
    context_window: int = 0
    supports_tools: bool = False
    supports_vision: bool = False
    supports_structured_output: bool = False
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
