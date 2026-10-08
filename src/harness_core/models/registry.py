"""
Centralized ModelRegistry — provider-agnostic model intelligence.

Supports OpenRouter, Ollama, LiteLLM, and future providers without redesign.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from harness_core.models.types import (
    CapabilityConfidence,
    CapabilitySource,
    CapabilityScore,
    ModelProfile,
)


class ModelRegistry:
    """Provider-agnostic model registry with capability tracking.

    Thread-safe. In-memory with optional persistence via history.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelProfile] = {}
        self._lock = threading.Lock()

    # ── CRUD ──────────────────────────────────────────────────────────────

    def register(self, profile: ModelProfile) -> None:
        """Register or update a model profile."""
        with self._lock:
            existing = self._models.get(profile.model_id)
            if existing is None:
                profile.discovered_at = profile.discovered_at or time.time()
                profile.last_seen = time.time()
                self._models[profile.model_id] = profile
            else:
                # Merge: keep existing capabilities, update metadata
                existing.context_window = profile.context_window or existing.context_window
                existing.supports_tools = profile.supports_tools or existing.supports_tools
                existing.supports_streaming = profile.supports_streaming or existing.supports_streaming
                existing.supports_reasoning = profile.supports_reasoning or existing.supports_reasoning
                existing.supports_structured_output = (
                    profile.supports_structured_output or existing.supports_structured_output
                )
                existing.supports_vision = profile.supports_vision or existing.supports_vision
                existing.is_free = profile.is_free if profile.is_free else existing.is_free
                existing.is_local = profile.is_local if profile.is_local else existing.is_local
                existing.input_cost_per_1k = profile.input_cost_per_1k or existing.input_cost_per_1k
                existing.output_cost_per_1k = profile.output_cost_per_1k or existing.output_cost_per_1k
                existing.tags = list(set(existing.tags + profile.tags))
                existing.last_seen = time.time()

    def unregister(self, model_id: str) -> bool:
        """Remove a model from the registry."""
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False

    def get(self, model_id: str) -> Optional[ModelProfile]:
        """Get a model profile by ID."""
        with self._lock:
            return self._models.get(model_id)

    def list_all(self) -> list[ModelProfile]:
        """List all registered models."""
        with self._lock:
            return list(self._models.values())

    def search(
        self,
        provider: str | None = None,
        is_free: bool | None = None,
        is_local: bool | None = None,
        supports_tools: bool | None = None,
        min_context: int | None = None,
        tags: list[str] | None = None,
    ) -> list[ModelProfile]:
        """Search models by criteria."""
        with self._lock:
            results = list(self._models.values())

        if provider is not None:
            results = [m for m in results if m.provider == provider]
        if is_free is not None:
            results = [m for m in results if m.is_free == is_free]
        if is_local is not None:
            results = [m for m in results if m.is_local == is_local]
        if supports_tools is not None:
            results = [m for m in results if m.supports_tools == supports_tools]
        if min_context is not None:
            results = [m for m in results if m.context_window >= min_context]
        if tags:
            results = [m for m in results if any(t in m.tags for t in tags)]

        return results

    # ── Capability Updates ────────────────────────────────────────────────

    def update_capabilities(
        self,
        model_id: str,
        capabilities: dict[str, float],
        confidence: CapabilityConfidence = CapabilityConfidence.OBSERVED,
        source: CapabilitySource = CapabilitySource.HARNESS_OBSERVED,
        benchmark_version: str = "",
    ) -> bool:
        """Update capability scores for a model."""
        with self._lock:
            profile = self._models.get(model_id)
            if profile is None:
                return False

        for cap_name, score in capabilities.items():
            profile.capabilities.set(
                cap_name,
                score=score,
                confidence=confidence,
                source=source,
                benchmark_version=benchmark_version,
            )
        return True

    def record_benchmark(
        self,
        model_id: str,
        benchmark_name: str,
        score: float,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Record a benchmark result."""
        return self.update_capabilities(
            model_id,
            {benchmark_name: score},
            confidence=CapabilityConfidence.BENCHMARKED,
            source=CapabilitySource.HARNESS_BENCHMARK,
        )

    # ── Health ────────────────────────────────────────────────────────────

    def update_health(
        self,
        model_id: str,
        reliability: float | None = None,
        latency: float | None = None,
        availability: float | None = None,
    ) -> bool:
        """Update operational health for a model."""
        with self._lock:
            profile = self._models.get(model_id)
            if profile is None:
                return False

            if reliability is not None:
                profile.reliability_score = reliability
            if latency is not None:
                profile.latency_score = latency
            if availability is not None:
                profile.availability_score = availability
            return True

    # ── Stats ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Number of registered models."""
        with self._lock:
            return len(self._models)

    def providers(self) -> list[str]:
        """List unique providers."""
        with self._lock:
            return list({m.provider for m in self._models.values()})

    def summary(self) -> dict[str, Any]:
        """Get a summary of the registry."""
        with self._lock:
            models = list(self._models.values())

        free_count = sum(1 for m in models if m.is_free)
        local_count = sum(1 for m in models if m.is_local)
        tool_count = sum(1 for m in models if m.supports_tools)
        benchmarked = sum(
            1 for m in models
            if m.capabilities.get_average() is not None
        )

        return {
            "total": len(models),
            "providers": self.providers(),
            "free": free_count,
            "local": local_count,
            "supports_tools": tool_count,
            "benchmarked": benchmarked,
        }
