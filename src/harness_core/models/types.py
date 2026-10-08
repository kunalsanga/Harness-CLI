"""
Model profile types and capability evidence system.

Separates four concepts:
  1. Provider metadata (what the provider says)
  2. Declared capabilities (explicitly configured)
  3. Observed capabilities (Harness has seen it work)
  4. Benchmarked capabilities (measured by Harness benchmarks)

Unknown must NOT equal zero. None means "not measured."
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


class CapabilityConfidence(enum.Enum):
    """How confident we are in a capability score."""

    UNKNOWN = "unknown"       # No data at all
    DECLARED = "declared"     # From provider metadata / user config
    OBSERVED = "observed"     # Harness has seen it succeed/fail in practice
    BENCHMARKED = "benchmarked"  # Measured by Harness benchmark engine


class CapabilitySource(enum.Enum):
    """Where capability data came from."""

    PROVIDER_METADATA = "provider_metadata"
    USER_DECLARED = "user_declared"
    HARNESS_OBSERVED = "harness_observed"
    HARNESS_BENCHMARK = "harness_benchmark"
    HISTORICAL = "historical"


@dataclass
class CapabilityScore:
    """A single measured capability with provenance."""

    score: Optional[float] = None  # None = not measured. NOT zero.
    confidence: CapabilityConfidence = CapabilityConfidence.UNKNOWN
    source: CapabilitySource = CapabilitySource.PROVIDER_METADATA
    sample_count: int = 0  # How many observations/benchmarks
    last_updated: float = 0.0  # Unix timestamp
    benchmark_version: str = ""  # For versioning benchmarks

    @property
    def is_measured(self) -> bool:
        return self.score is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "confidence": self.confidence.value,
            "source": self.source.value,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
            "benchmark_version": self.benchmark_version,
        }


@dataclass
class CapabilityProfile:
    """Complete capability evidence for a model."""

    # Agent capabilities (measured by benchmarks/observations)
    coding: CapabilityScore = field(default_factory=CapabilityScore)
    tool_use: CapabilityScore = field(default_factory=CapabilityScore)
    reasoning: CapabilityScore = field(default_factory=CapabilityScore)
    planning: CapabilityScore = field(default_factory=CapabilityScore)
    repository_navigation: CapabilityScore = field(default_factory=CapabilityScore)
    context_handling: CapabilityScore = field(default_factory=CapabilityScore)
    error_recovery: CapabilityScore = field(default_factory=CapabilityScore)
    instruction_following: CapabilityScore = field(default_factory=CapabilityScore)
    verification: CapabilityScore = field(default_factory=CapabilityScore)

    def get(self, name: str) -> CapabilityScore:
        """Get a capability by name."""
        return getattr(self, name, CapabilityScore())

    def set(
        self,
        name: str,
        score: float,
        confidence: CapabilityConfidence,
        source: CapabilitySource,
        sample_count: int = 1,
        benchmark_version: str = "",
    ) -> None:
        """Set a capability score with provenance."""
        import time

        cap = CapabilityScore(
            score=score,
            confidence=confidence,
            source=source,
            sample_count=sample_count,
            last_updated=time.time(),
            benchmark_version=benchmark_version,
        )
        setattr(self, name, cap)

    def get_average(self) -> Optional[float]:
        """Average of all measured capabilities. None if none measured."""
        scores = [
            getattr(self, attr).score
            for attr in [
                "coding", "tool_use", "reasoning", "planning",
                "repository_navigation", "context_handling",
                "error_recovery", "instruction_following", "verification",
            ]
            if getattr(self, attr).score is not None
        ]
        if not scores:
            return None
        return sum(scores) / len(scores)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coding": self.coding.to_dict(),
            "tool_use": self.tool_use.to_dict(),
            "reasoning": self.reasoning.to_dict(),
            "planning": self.planning.to_dict(),
            "repository_navigation": self.repository_navigation.to_dict(),
            "context_handling": self.context_handling.to_dict(),
            "error_recovery": self.error_recovery.to_dict(),
            "instruction_following": self.instruction_following.to_dict(),
            "verification": self.verification.to_dict(),
        }


@dataclass
class ModelProfile:
    """Full profile for a model in the registry.

    Combines provider metadata, capability evidence, and operational data.
    """

    # Identity
    model_id: str = ""
    provider: str = ""
    display_name: str = ""

    # Provider metadata (what the provider says)
    context_window: int = 0
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    currency: str = "USD"
    supports_tools: bool | None = None
    supports_streaming: bool | None = None
    supports_reasoning: bool | None = None
    supports_structured_output: bool | None = None
    supports_vision: bool | None = None
    is_free: bool = False
    is_local: bool = False

    # Capability evidence (measured by Harness)
    capabilities: CapabilityProfile = field(default_factory=CapabilityProfile)

    # Operational evidence
    reliability_score: Optional[float] = None  # None = unknown
    latency_score: Optional[float] = None
    availability_score: Optional[float] = None

    # Metadata
    tags: list[str] = field(default_factory=list)
    discovered_at: float = 0.0
    last_seen: float = 0.0

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = self.model_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "display_name": self.display_name,
            "context_window": self.context_window,
            "is_free": self.is_free,
            "is_local": self.is_local,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            "supports_reasoning": self.supports_reasoning,
            "supports_vision": self.supports_vision,
            "supports_structured_output": self.supports_structured_output,
            "input_cost_per_1k": self.input_cost_per_1k,
            "output_cost_per_1k": self.output_cost_per_1k,
            "capabilities": self.capabilities.to_dict(),
            "reliability_score": self.reliability_score,
            "latency_score": self.latency_score,
            "tags": self.tags,
        }
