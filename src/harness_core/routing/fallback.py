"""Fallback engine with retry, exponential backoff, and error classification.

Handles the chain: primary → fallback → fallback → final failure.
Never retries permanent errors unnecessarily.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)
from harness_core.routing.health import HealthEvent, ModelHealthTracker


class ErrorClassification(Enum):
    """Classification of errors for retry decisions."""

    RETRYABLE = "retryable"  # transient, worth retrying
    PERMANENT = "permanent"  # don't retry (bad request, auth, etc.)
    RATE_LIMITED = "rate_limited"  # try another model, not retry same
    UNKNOWN = "unknown"


def classify_error(error: Exception) -> ErrorClassification:
    """Classify an error to determine retry strategy."""
    error_str = str(error).lower()

    # Rate limiting
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return ErrorClassification.RATE_LIMITED

    # Permanent client errors (don't retry)
    if any(code in error_str for code in ["400", "401", "403", "404", "422"]):
        return ErrorClassification.PERMANENT
    if any(kw in error_str for kw in ["unauthorized", "forbidden", "not found", "invalid"]):
        return ErrorClassification.PERMANENT

    # Retryable server errors
    if any(code in error_str for code in ["429", "500", "502", "503", "504"]):
        return ErrorClassification.RETRYABLE
    if any(kw in error_str for kw in ["timeout", "timed out", "connection", "network"]):
        return ErrorClassification.RETRYABLE
    if "overloaded" in error_str or "capacity" in error_str:
        return ErrorClassification.RETRYABLE

    # Context overflow — try a bigger model
    if any(kw in error_str for kw in ["context", "token limit", "too long", "maximum context"]):
        return ErrorClassification.RATE_LIMITED  # treat as "try another model"

    return ErrorClassification.UNKNOWN


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 2
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True

    def delay_for_attempt(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        delay = min(
            self.base_delay_seconds * (self.backoff_factor ** attempt),
            self.max_delay_seconds,
        )
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""

    max_fallback_models: int = 3
    retry: RetryConfig = field(default_factory=RetryConfig)
    # Maximum total time across all fallbacks for a single request
    total_timeout_seconds: float = 120.0


@dataclass
class FallbackResult:
    """Result of a fallback attempt chain."""

    response: CompletionResponse | None = None
    model_used: str = ""
    provider_used: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
    succeeded: bool = False
    final_error: str | None = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def fallback_count(self) -> int:
        return max(0, self.attempts - 1) if False else max(0, len(self.attempts) - 1)


class ProviderAuthFailure(Exception):
    """Raised when a provider-level auth failure is detected.

    Signals that all models from this provider will fail with the same
    auth error, so no further models from the same provider should be tried.
    """

    def __init__(self, provider_name: str, status_code: int, detail: str = "") -> None:
        self.provider_name = provider_name
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Provider {provider_name} auth failure: {status_code} {detail}")


class FallbackEngine:
    """Executes completion requests with fallback, retry, and error classification.

    Architecture:
        1. Try primary model
        2. If retryable error → retry with backoff (up to max_retries)
        3. If rate-limited or retries exhausted → try next fallback model
        4. Model-specific auth failure → mark that model unavailable
        5. If all models fail → return final error
    """

    def __init__(
        self,
        health_tracker: ModelHealthTracker | None = None,
        fallback_config: FallbackConfig | None = None,
    ) -> None:
        self.health = health_tracker or ModelHealthTracker()
        self.config = fallback_config or FallbackConfig()

    async def execute(
        self,
        request: CompletionRequest,
        model_chain: list[tuple[str, ModelProvider]],
    ) -> FallbackResult:
        """Execute a request through the fallback chain.

        Args:
            request: The completion request (model field is overridden per attempt).
            model_chain: Ordered list of (model_id, provider) to try.

        Returns:
            FallbackResult with the outcome.
        """
        overall_start = time.time()
        result = FallbackResult()

        for model_idx, (model_id, provider) in enumerate(model_chain):
            if time.time() - overall_start > self.config.total_timeout_seconds:
                result.final_error = "Total timeout exceeded across all fallback models"
                break

            # Check if model is healthy before trying
            health_state = self.health.get_state(model_id)
            if health_state.is_unavailable:
                # Skip models that are known to be unavailable (auth failure)
                result.attempts.append({
                    "model": model_id,
                    "status": "skipped",
                    "reason": "model unavailable (auth failed)",
                })
                continue
            if not health_state.is_healthy and model_idx > 0:
                # Skip unhealthy fallback models (but always try the primary)
                result.attempts.append({
                    "model": model_id,
                    "status": "skipped",
                    "reason": "unhealthy",
                })
                continue

            # Retry loop for this model
            attempt_request = CompletionRequest(
                messages=request.messages,
                model=model_id,
                tools=request.tools,
                tool_choice=request.tool_choice,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
                metadata=request.metadata,
            )

            for retry_attempt in range(self.config.retry.max_retries + 1):
                if time.time() - overall_start > self.config.total_timeout_seconds:
                    result.final_error = "Total timeout exceeded"
                    break

                start = time.time()
                try:
                    response = await provider.generate(attempt_request)
                    latency_ms = (time.time() - start) * 1000

                    # Record success
                    input_tokens = response.usage.get("prompt_tokens", 0)
                    output_tokens = response.usage.get("completion_tokens", 0)
                    self.health.record_success(
                        model_id,
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )

                    result.response = response
                    result.model_used = model_id
                    result.provider_used = provider.name
                    result.succeeded = True
                    result.total_latency_ms = (time.time() - overall_start) * 1000
                    result.attempts.append({
                        "model": model_id,
                        "status": "success",
                        "retry": retry_attempt,
                        "latency_ms": round(latency_ms, 1),
                    })
                    return result

                except Exception as e:
                    latency_ms = (time.time() - start) * 1000
                    classification = classify_error(e)

                    result.attempts.append({
                        "model": model_id,
                        "status": "error",
                        "retry": retry_attempt,
                        "classification": classification.value,
                        "error": str(e)[:200],
                        "latency_ms": round(latency_ms, 1),
                    })

                    if classification == ErrorClassification.RATE_LIMITED:
                        self.health.record_failure(model_id, HealthEvent.RATE_LIMIT_429, latency_ms)
                        # Don't retry rate-limited models; move to next fallback
                        break

                    elif classification == ErrorClassification.PERMANENT:
                        # Detect auth errors (401/403) — mark THIS MODEL as unavailable
                        # Do NOT disable the entire provider
                        error_str = str(e).lower()
                        is_auth_error = any(code in error_str for code in ["401", "403"])
                        is_auth_keyword = any(kw in error_str for kw in ["unauthorized", "forbidden"])
                        if is_auth_error or is_auth_keyword:
                            self.health.record_failure(model_id, HealthEvent.AUTH_FAILED, latency_ms)
                        else:
                            self.health.record_failure(model_id, HealthEvent.CLIENT_ERROR_4XX, latency_ms)
                        # Don't retry permanent errors
                        break

                    elif classification == ErrorClassification.RETRYABLE:
                        self.health.record_failure(model_id, HealthEvent.SERVER_ERROR_5XX, latency_ms)
                        # Retry with backoff if we have retries left
                        if retry_attempt < self.config.retry.max_retries:
                            delay = self.config.retry.delay_for_attempt(retry_attempt)
                            await asyncio.sleep(delay)
                            continue
                        # Retries exhausted, move to next fallback
                        break

                    else:
                        # Unknown error — record and move on
                        self.health.record_failure(model_id, HealthEvent.INVALID_RESPONSE, latency_ms)
                        break

        # All models failed
        result.total_latency_ms = (time.time() - overall_start) * 1000
        if not result.final_error:
            # Build a more informative error message
            auth_failures = []
            other_failures = []
            for attempt in result.attempts:
                if attempt.get("status") == "error":
                    error_str = attempt.get("error", "")
                    if "401" in error_str or "403" in error_str or "forbidden" in error_str.lower():
                        auth_failures.append(attempt.get("model", "unknown"))
                    else:
                        other_failures.append(attempt.get("model", "unknown"))

            if auth_failures and not other_failures:
                # All failures were auth errors on specific models
                result.final_error = (
                    f"{len(auth_failures)} model(s) returned 401/403 and are unavailable. "
                    f"Models: {', '.join(auth_failures[:3])}. "
                    f"Try other models or check your provider access."
                )
            elif auth_failures:
                result.final_error = (
                    f"{len(auth_failures)} model(s) unavailable (401/403), "
                    f"{len(other_failures)} model(s) failed for other reasons. "
                    f"Last error: {result.attempts[-1].get('error', 'unknown')}"
                )
            else:
                result.final_error = (
                    f"All {len(model_chain)} models failed. "
                    f"Last error: {result.attempts[-1].get('error', 'unknown') if result.attempts else 'no attempts'}"
                )
        return result
