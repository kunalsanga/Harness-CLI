"""Model health tracking.

Tracks per-model success/failure rates, latency, error types,
and provides reliability scores for routing decisions.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HealthEvent(Enum):
    """Types of health events for a model."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    RATE_LIMIT_429 = "429"
    CLIENT_ERROR_4XX = "4xx"
    SERVER_ERROR_5XX = "5xx"
    TOOL_CALL_FAILURE = "tool_call_failure"
    INVALID_RESPONSE = "invalid_response"
    NETWORK_ERROR = "network_error"
    AUTH_FAILED = "auth_failed"  # 401/403 — model-specific, not provider-wide
    PAYMENT_REQUIRED = "402"     # 402 — model requires payment/credits


class ModelHealthStatus(Enum):
    """Health status for a model."""

    UNKNOWN = "unknown"       # No data yet
    HEALTHY = "healthy"       # Working normally
    UNAVAILABLE = "unavailable" # Returned 401/403 — not available to this user
    PAYMENT_REQUIRED = "payment_required" # 402 — model requires payment
    RATE_LIMITED = "rate_limited" # 429 — temporarily unavailable
    ERROR = "error"           # Server errors, timeouts, etc.
    NO_TOOL_USE = "no_tool_use"  # Responded without using required tools — rotate away


@dataclass
class ModelHealthState:
    """Health state for a single model."""

    model_id: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    rate_limit_hits: int = 0
    timeouts: int = 0
    client_errors: int = 0
    server_errors: int = 0
    tool_call_failures: int = 0
    invalid_responses: int = 0
    network_errors: int = 0

    # Latency tracking (recent window)
    latencies_ms: list[float] = field(default_factory=list)
    max_latency_window: int = 50

    # Cost tracking
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost: float = 0.0

    # Timing
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    last_rate_limit_time: float = 0.0
    consecutive_failures: int = 0

    # Cooldown: if a model is rate-limited, don't use it for this many seconds
    cooldown_seconds: float = 0.0

    # Explicit health status (model-level, not provider-level)
    health_status: ModelHealthStatus = ModelHealthStatus.UNKNOWN
    last_auth_failure_time: float = 0.0
    # When set, the model is rotated away from until this unix time passes
    no_tool_cooldown_until: float = 0.0

    @property
    def reliability(self) -> float:
        """Reliability score: 0.0 = unreliable, 1.0 = always succeeds."""
        if self.total_calls == 0:
            return 0.5  # unknown, neutral
        return self.successes / self.total_calls

    @property
    def avg_latency_ms(self) -> float:
        """Average latency over recent window."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def is_rate_limited(self) -> bool:
        """Whether this model is currently in a rate-limit cooldown."""
        if self.cooldown_seconds <= 0:
            return False
        return (time.time() - self.last_rate_limit_time) < self.cooldown_seconds

    @property
    def is_healthy(self) -> bool:
        """Whether the model is considered usable right now."""
        # Explicit status overrides all other checks
        if self.health_status == ModelHealthStatus.UNAVAILABLE:
            return False
        if self.health_status == ModelHealthStatus.PAYMENT_REQUIRED:
            return False  # requires payment, not usable in free mode
        if self.health_status == ModelHealthStatus.RATE_LIMITED:
            return self.is_rate_limited  # may have recovered
        if self.health_status == ModelHealthStatus.NO_TOOL_USE:
            # Rotate away until the cooldown timestamp passes, then recover
            return time.time() >= self.no_tool_cooldown_until
        if self.health_status == ModelHealthStatus.ERROR:
            return self.consecutive_failures < 3  # recover after some time
        if self.is_rate_limited:
            return False
        if self.consecutive_failures >= 5:
            return False
        return True

    @property
    def is_unavailable(self) -> bool:
        """Whether this model is permanently unavailable (auth or payment failure)."""
        return self.health_status in (
            ModelHealthStatus.UNAVAILABLE,
            ModelHealthStatus.PAYMENT_REQUIRED,
        )

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.latencies_ms.append(latency_ms)
        if len(self.latencies_ms) > self.max_latency_window:
            self.latencies_ms = self.latencies_ms[-self.max_latency_window :]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for observability."""
        return {
            "model_id": self.model_id,
            "total_calls": self.total_calls,
            "successes": self.successes,
            "failures": self.failures,
            "rate_limit_hits": self.rate_limit_hits,
            "reliability": round(self.reliability, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "consecutive_failures": self.consecutive_failures,
            "is_rate_limited": self.is_rate_limited,
            "is_healthy": self.is_healthy,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost": round(self.estimated_cost, 6),
        }


class ModelHealthTracker:
    """Tracks health state for all models across all providers."""

    def __init__(self, default_cooldown_seconds: float = 60.0) -> None:
        self._states: dict[str, ModelHealthState] = {}
        self._default_cooldown = default_cooldown_seconds

    def get_state(self, model_id: str) -> ModelHealthState:
        """Get or create health state for a model."""
        if model_id not in self._states:
            self._states[model_id] = ModelHealthState(
                model_id=model_id,
                cooldown_seconds=self._default_cooldown,
            )
        return self._states[model_id]

    def record_success(
        self,
        model_id: str,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record a successful call."""
        state = self.get_state(model_id)
        state.total_calls += 1
        state.successes += 1
        state.consecutive_failures = 0
        state.last_success_time = time.time()
        state.total_input_tokens += input_tokens
        state.total_output_tokens += output_tokens
        state.estimated_cost += cost
        if latency_ms > 0:
            state.record_latency(latency_ms)
        # Mark as healthy on success (recovers from previous errors)
        state.health_status = ModelHealthStatus.HEALTHY

    def record_failure(
        self,
        model_id: str,
        event: HealthEvent,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a failure with its type."""
        state = self.get_state(model_id)
        state.total_calls += 1
        state.failures += 1
        state.consecutive_failures += 1
        state.last_failure_time = time.time()

        if latency_ms > 0:
            state.record_latency(latency_ms)

        if event == HealthEvent.RATE_LIMIT_429:
            state.rate_limit_hits += 1
            state.last_rate_limit_time = time.time()
            # Exponential cooldown: more hits = longer cooldown
            state.cooldown_seconds = min(
                300.0, self._default_cooldown * (2 ** min(state.rate_limit_hits - 1, 4))
            )
        elif event == HealthEvent.TIMEOUT:
            state.timeouts += 1
        elif event == HealthEvent.CLIENT_ERROR_4XX:
            state.client_errors += 1
        elif event == HealthEvent.SERVER_ERROR_5XX:
            state.server_errors += 1
        elif event == HealthEvent.TOOL_CALL_FAILURE:
            state.tool_call_failures += 1
        elif event == HealthEvent.INVALID_RESPONSE:
            state.invalid_responses += 1
        elif event == HealthEvent.NETWORK_ERROR:
            state.network_errors += 1
        elif event == HealthEvent.AUTH_FAILED:
            state.client_errors += 1
            state.health_status = ModelHealthStatus.UNAVAILABLE
            state.last_auth_failure_time = time.time()
        elif event == HealthEvent.PAYMENT_REQUIRED:
            state.client_errors += 1
            state.health_status = ModelHealthStatus.PAYMENT_REQUIRED

    def record_tool_call_failure(self, model_id: str) -> None:
        """Record a tool call failure (model called wrong tool / bad args)."""
        self.record_failure(model_id, HealthEvent.TOOL_CALL_FAILURE)

    def record_no_tool_usage(self, model_id: str, cooldown_seconds: float = 300.0) -> None:
        """Record that a model responded without using required tools.

        Marks the model so routing rotates to a different model for the
        cooldown window. Recovers automatically once the window passes or
        the model next succeeds.
        """
        state = self.get_state(model_id)
        state.tool_call_failures += 1
        state.health_status = ModelHealthStatus.NO_TOOL_USE
        state.no_tool_cooldown_until = time.time() + cooldown_seconds

    def get_healthy_models(
        self, model_ids: list[str] | None = None
    ) -> list[str]:
        """Return model IDs that are currently healthy (not rate-limited, not failing)."""
        ids = model_ids or list(self._states.keys())
        return [mid for mid in ids if self.get_state(mid).is_healthy]

    def get_reliability(self, model_id: str) -> float:
        """Get the reliability score for a model."""
        return self.get_state(model_id).reliability

    def get_all_states(self) -> dict[str, ModelHealthState]:
        """Get all tracked health states."""
        return dict(self._states)

    def reset(self, model_id: str | None = None) -> None:
        """Reset health state for a model or all models."""
        if model_id:
            self._states.pop(model_id, None)
        else:
            self._states.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize all states for observability."""
        return {mid: state.to_dict() for mid, state in self._states.items()}
