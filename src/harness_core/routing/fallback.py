"""Fallback engine with retry, exponential backoff, and error classification.

Handles the chain: primary → fallback → fallback → final failure.
Never retries permanent errors unnecessarily.
"""

from __future__ import annotations

import asyncio
import logging
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
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.routing.health import HealthEvent, ModelHealthTracker
from harness_core.observability.events import Event, EventBus

logger = logging.getLogger(__name__)


class ErrorClassification(Enum):
    """Classification of errors for retry decisions."""

    RETRYABLE = "retryable"  # transient, worth retrying
    PERMANENT = "permanent"  # don't retry (bad request, auth, etc.)
    RATE_LIMITED = "rate_limited"  # try another model, not retry same
    MODEL_UNAVAILABLE = "model_unavailable"  # endpoint/model temporarily unavailable
    UNKNOWN = "unknown"


def classify_error(error: Exception) -> ErrorClassification:
    """Classify an error to determine retry strategy."""
    if isinstance(error, ProviderRequestError):
        return {
            ProviderErrorCategory.AUTHENTICATION: ErrorClassification.PERMANENT,
            ProviderErrorCategory.INVALID_REQUEST: ErrorClassification.PERMANENT,
            ProviderErrorCategory.PAYMENT_REQUIRED: ErrorClassification.PERMANENT,
            ProviderErrorCategory.TOOL: ErrorClassification.PERMANENT,
            ProviderErrorCategory.RATE_LIMIT: ErrorClassification.RATE_LIMITED,
            ProviderErrorCategory.MODEL_UNAVAILABLE: ErrorClassification.MODEL_UNAVAILABLE,
            ProviderErrorCategory.TIMEOUT: ErrorClassification.RETRYABLE,
            ProviderErrorCategory.NETWORK: ErrorClassification.RETRYABLE,
            ProviderErrorCategory.PROVIDER: ErrorClassification.RETRYABLE,
        }.get(error.category, ErrorClassification.UNKNOWN)
    error_str = str(error).lower()

    # Authentication and payment errors are configuration/account failures,
    # never reasons to rotate the shared OpenRouter credential across models.
    if any(code in error_str for code in ["401", "403", "unauthorized", "forbidden"]):
        return ErrorClassification.PERMANENT
    if "402" in error_str or "payment required" in error_str:
        return ErrorClassification.PERMANENT

    # Rate limiting
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return ErrorClassification.RATE_LIMITED

    # Explicit model/provider availability failures may be model-specific.
    # "provider_unavailable" is OpenRouter's structured error_type for wrapped
    # upstream provider failures (observed live: Nemotron free endpoint
    # exhausting its worker request limit); it is model-specific and transient.
    if any(phrase in error_str for phrase in (
        "model unavailable", "model is unavailable", "model not found",
        "no endpoints found", "provider unavailable", "provider_unavailable",
        "temporarily unavailable",
    )):
        return ErrorClassification.MODEL_UNAVAILABLE

    # Payment required (402) — model requires payment, don't retry
    if "402" in error_str or "payment required" in error_str:
        return ErrorClassification.PERMANENT

    # Permanent client errors (don't retry)
    if any(code in error_str for code in ["400", "401", "403", "404", "422"]):
        return ErrorClassification.PERMANENT
    if any(kw in error_str for kw in ["unauthorized", "forbidden", "not found", "invalid"]):
        return ErrorClassification.PERMANENT

    # Retryable server errors (429 already handled as RATE_LIMITED above — don't double-classify)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
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

    max_retries: int = 1
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
    # Bound on a single provider attempt. A hung request must not consume the
    # whole task budget; per-attempt timeouts let the chain move on quickly.
    model_attempt_timeout_seconds: float = 90.0


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
    # Normalized category of the final failure (rate_limit, model_unavailable,
    # timeout, auth, ...) so callers never parse provider error strings.
    error_category: str = ""

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


# Normalized error categories — the provider-specific strings are translated
# into these once, here, so the AgentLoop and UI never parse provider text.
ERROR_CATEGORY_AUTH = "auth"
ERROR_CATEGORY_INVALID_REQUEST = "invalid_request"
ERROR_CATEGORY_RATE_LIMIT = "rate_limit"
ERROR_CATEGORY_MODEL_UNAVAILABLE = "model_unavailable"
ERROR_CATEGORY_TIMEOUT = "timeout"
ERROR_CATEGORY_NETWORK = "network"
ERROR_CATEGORY_SERVER = "server"
ERROR_CATEGORY_PAYMENT = "payment"
ERROR_CATEGORY_UNKNOWN = "unknown"


def normalized_error_category(classification: "ErrorClassification", error_str: str | Exception) -> str:
    """Map an ErrorClassification + raw error text to a normalized category."""
    if isinstance(error_str, ProviderRequestError):
        return error_str.category.value
    low = str(error_str or "").lower()
    if classification == ErrorClassification.PERMANENT:
        if "402" in low or "payment required" in low:
            return ERROR_CATEGORY_PAYMENT
        if "404" in low or "not found" in low:
            return ERROR_CATEGORY_INVALID_REQUEST
        return ERROR_CATEGORY_AUTH
    if classification == ErrorClassification.RATE_LIMITED:
        return ERROR_CATEGORY_RATE_LIMIT
    if classification == ErrorClassification.MODEL_UNAVAILABLE:
        return ERROR_CATEGORY_MODEL_UNAVAILABLE
    if classification == ErrorClassification.RETRYABLE:
        if "timeout" in low or "timed out" in low:
            return ERROR_CATEGORY_TIMEOUT
        if "network" in low or "connection" in low:
            return ERROR_CATEGORY_NETWORK
        return ERROR_CATEGORY_SERVER
    return ERROR_CATEGORY_UNKNOWN


_RETRY_AFTER_RE = __import__("re").compile(
    r"retry[-_ ]?after[\s:=]*([0-9]+(?:\.[0-9]+)?)", __import__("re").IGNORECASE
)


def extract_retry_after(error: Exception) -> float | None:
    """Extract a Retry-After hint (seconds) from an error, if present.

    Providers and gateways commonly embed "Retry-After: 20" in the message or
    metadata; honoring it avoids pointless retries against a known-dead model.
    """
    if isinstance(error, ProviderRequestError) and error.retry_after is not None:
        return error.retry_after
    m = _RETRY_AFTER_RE.search(str(error))
    if m:
        try:
            value = float(m.group(1))
            if 0 < value <= 3600:
                return value
        except ValueError:
            pass
    return None


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
        event_bus: EventBus | None = None,
    ) -> None:
        self.health = health_tracker or ModelHealthTracker()
        self.config = fallback_config or FallbackConfig()
        self.event_bus = event_bus

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
        previous_reason = ""

        for model_idx, (model_id, provider) in enumerate(model_chain):
            if time.time() - overall_start > self.config.total_timeout_seconds:
                result.final_error = "Total timeout exceeded across all fallback models"
                break

            # Check if model is healthy before trying
            # Fail-fast rule: a model with an active cooldown or known-unavailable
            # status is skipped at ANY chain position — including primary. The
            # router already orders cooldown-expired models first; selecting a
            # model we know is cooling down would waste the task's time budget.
            health_state = self.health.get_state(model_id)
            if health_state.is_unavailable:
                # Skip models that are known to be unavailable (auth failure)
                result.attempts.append({
                    "model": model_id,
                    "status": "skipped",
                    "reason": "model unavailable (auth failed)",
                })
                continue
            if not health_state.is_healthy:
                # Skip models with an active temporary-failure cooldown.
                result.attempts.append({
                    "model": model_id,
                    "status": "skipped",
                    "reason": f"cooling down ({health_state.display_status.lower()})",
                })
                continue

            # Retry loop for this model
            logger.info("[model] %s", model_id)
            if model_idx:
                logger.info(
                    "[failover] switching to %s after %s",
                    model_id,
                    previous_reason or "model unavailable",
                )
            if self.event_bus is not None:
                await self.event_bus.emit(Event(
                    type="model.request_started",
                    source="model_fallback",
                    data={"model": model_id, "provider": provider.name},
                ))
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
                    try:
                        response = await asyncio.wait_for(
                            provider.generate(attempt_request),
                            timeout=self.config.model_attempt_timeout_seconds,
                        )
                    except asyncio.TimeoutError:
                        raise TimeoutError(
                            f"Model request timed out after "
                            f"{self.config.model_attempt_timeout_seconds:.0f}s (model={model_id})"
                        )
                    latency_ms = (time.time() - start) * 1000

                    # Record success
                    usage = response.token_usage
                    self.health.record_success(
                        model_id,
                        latency_ms=latency_ms,
                        input_tokens=(usage.input_tokens or 0) if usage else 0,
                        output_tokens=(usage.output_tokens or 0) if usage else 0,
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

                    raw = str(e).strip()
                    secret = str(getattr(provider, "api_key", "") or "")
                    if secret:
                        raw = raw.replace(secret, "[REDACTED]")
                    if not raw:
                        raw = f"{type(e).__name__}: {e.__class__.__name__} (empty message, model={model_id})"
                    category = normalized_error_category(classification, e)
                    retry_after = extract_retry_after(e)
                    result.attempts.append({
                        "model": model_id,
                        "status": "error",
                        "retry": retry_attempt,
                        "classification": classification.value,
                        "category": category,
                        "retry_after": retry_after,
                        "error": raw[:600],
                        "error_short": raw[:200],
                        "latency_ms": round(latency_ms, 1),
                    })

                    if classification == ErrorClassification.RATE_LIMITED:
                        self.health.record_failure(
                            model_id, HealthEvent.RATE_LIMIT_429, latency_ms,
                            cooldown_override=retry_after,
                        )
                        previous_reason = classification.value
                        remaining = self.health.get_state(model_id).cooldown_remaining()
                        if self.event_bus is not None:
                            await self.event_bus.emit(Event(
                                type="model.cooldown",
                                source="model_fallback",
                                data={
                                    "model": model_id,
                                    "category": category,
                                    "cooldown_seconds": round(remaining, 1),
                                },
                            ))
                        if self.event_bus is not None and model_idx + 1 < len(model_chain):
                            await self.event_bus.emit(Event(
                                type="model.failover",
                                source="model_fallback",
                                data={
                                    "from": model_id,
                                    "to": model_chain[model_idx + 1][0],
                                    "reason": classification.value,
                                },
                            ))
                        break

                    elif classification == ErrorClassification.MODEL_UNAVAILABLE:
                        self.health.record_failure(
                            model_id, HealthEvent.SERVER_ERROR_5XX, latency_ms,
                            cooldown_override=retry_after,
                        )
                        previous_reason = classification.value
                        remaining = self.health.get_state(model_id).cooldown_remaining()
                        if self.event_bus is not None:
                            await self.event_bus.emit(Event(
                                type="model.cooldown",
                                source="model_fallback",
                                data={
                                    "model": model_id,
                                    "category": category,
                                    "cooldown_seconds": round(remaining, 1),
                                },
                            ))
                        if self.event_bus is not None and model_idx + 1 < len(model_chain):
                            await self.event_bus.emit(Event(
                                type="model.failover",
                                source="model_fallback",
                                data={
                                    "from": model_id,
                                    "to": model_chain[model_idx + 1][0],
                                    "reason": classification.value,
                                },
                            ))
                        break

                    elif classification == ErrorClassification.PERMANENT:
                        error_str = str(e).lower()
                        # Detect specific failure types for model-level health tracking
                        normalized_category = e.category if isinstance(e, ProviderRequestError) else None
                        is_auth_error = normalized_category == ProviderErrorCategory.AUTHENTICATION or any(code in error_str for code in ["401", "403"])
                        is_auth_keyword = any(kw in error_str for kw in ["unauthorized", "forbidden"])
                        is_payment = (
                            normalized_category == ProviderErrorCategory.PAYMENT_REQUIRED
                            or "402" in error_str
                            or "payment required" in error_str
                        )
                        is_invalid_request = normalized_category == ProviderErrorCategory.INVALID_REQUEST
                        if is_payment:
                            self.health.record_failure(model_id, HealthEvent.PAYMENT_REQUIRED, latency_ms)
                        elif is_auth_error or is_auth_keyword:
                            self.health.record_failure(model_id, HealthEvent.AUTH_FAILED, latency_ms)
                        else:
                            self.health.record_failure(model_id, HealthEvent.CLIENT_ERROR_4XX, latency_ms)
                        # Shared credential and malformed-request failures affect
                        # the whole provider, so do not rotate through models.
                        if is_auth_error or is_auth_keyword:
                            # Status code first so UI truncation still preserves
                            # 401-vs-403 classification. Credential hint is
                            # provider-specific and never includes secret values.
                            status = None
                            if isinstance(e, ProviderRequestError):
                                status = e.status_code
                            if status is None:
                                for code in (401, 403):
                                    if str(code) in str(e):
                                        status = code
                                        break
                            credential_hints = {
                                "openrouter": "OPENROUTER_API_KEY",
                                "openai": "OPENAI_API_KEY",
                                "anthropic": "ANTHROPIC_API_KEY",
                                "gemini": "GEMINI_API_KEY",
                                "nvidia": "NVIDIA_API_KEY",
                                "groq": "GROQ_API_KEY",
                                "9router": "NINE_ROUTER_API_KEY",
                            }
                            key_name = credential_hints.get(provider.name, "provider credentials")
                            prefix = f"{status} " if status is not None else ""
                            result.final_error = (
                                f"{prefix}Authentication or access failed for provider {provider.name}. "
                                f"Check {key_name}."
                            )
                            result.error_category = ERROR_CATEGORY_AUTH
                            result.total_latency_ms = (time.time() - overall_start) * 1000
                            return result
                        if is_payment:
                            # Model-specific payment responses can rotate to the
                            # next free/local candidate; never invent a paid model.
                            previous_reason = "payment_required"
                            if self.event_bus is not None and model_idx + 1 < len(model_chain):
                                await self.event_bus.emit(Event(
                                    type="model.failover",
                                    source="model_fallback",
                                    data={"from": model_id, "to": model_chain[model_idx + 1][0], "reason": previous_reason},
                                ))
                            break
                        detail = str(e).strip()[:240]
                        status = e.status_code if isinstance(e, ProviderRequestError) else None
                        status_prefix = f"{status} " if status is not None else ""
                        result.final_error = (
                            f"{status_prefix}Invalid request rejected by provider {provider.name}: {detail}"
                            if is_invalid_request
                            else f"{status_prefix}Permanent request failure for provider {provider.name}: {detail}"
                        )
                        result.error_category = "invalid_request" if is_invalid_request else (
                            normalized_category.value if normalized_category else "provider"
                        )
                        result.total_latency_ms = (time.time() - overall_start) * 1000
                        return result

                    elif classification == ErrorClassification.RETRYABLE:
                        # Distinguish timeout/network from plain server errors so
                        # health state and cooldowns reflect the real failure mode.
                        if category == ERROR_CATEGORY_TIMEOUT:
                            health_event = HealthEvent.TIMEOUT
                        elif category == ERROR_CATEGORY_NETWORK:
                            health_event = HealthEvent.NETWORK_ERROR
                        else:
                            health_event = HealthEvent.SERVER_ERROR_5XX
                        self.health.record_failure(
                            model_id, health_event, latency_ms,
                            cooldown_override=retry_after,
                        )
                        # Retry with backoff if we have retries left
                        if retry_attempt < self.config.retry.max_retries:
                            delay = self.config.retry.delay_for_attempt(retry_attempt)
                            await asyncio.sleep(delay)
                            continue
                        # Retries exhausted, move to next fallback
                        previous_reason = classification.value
                        remaining = self.health.get_state(model_id).cooldown_remaining()
                        if self.event_bus is not None:
                            await self.event_bus.emit(Event(
                                type="model.cooldown",
                                source="model_fallback",
                                data={
                                    "model": model_id,
                                    "category": category,
                                    "cooldown_seconds": round(remaining, 1),
                                },
                            ))
                        if self.event_bus is not None and model_idx + 1 < len(model_chain):
                            await self.event_bus.emit(Event(
                                type="model.failover",
                                source="model_fallback",
                                data={
                                    "from": model_id,
                                    "to": model_chain[model_idx + 1][0],
                                    "reason": classification.value,
                                },
                            ))
                        break

                    else:
                        # Unknown/programming errors are not model availability
                        # signals; fail instead of hiding them with rotation.
                        self.health.record_failure(model_id, HealthEvent.INVALID_RESPONSE, latency_ms)
                        detail = str(e).strip()[:240] or "empty message"
                        result.final_error = (
                            f"Unclassified provider request failure for {provider.name}: {detail}"
                        )
                        result.error_category = ERROR_CATEGORY_UNKNOWN
                        result.total_latency_ms = (time.time() - overall_start) * 1000
                        return result

        # All models failed
        result.total_latency_ms = (time.time() - overall_start) * 1000
        # Surface the normalized category of the last real provider error.
        for attempt in reversed(result.attempts):
            if attempt.get("status") == "error":
                result.error_category = str(attempt.get("category", ERROR_CATEGORY_UNKNOWN))
                break
        if not result.final_error:
            auth_failures: list[str] = []
            payment_failures: list[str] = []
            other_failures: list[str] = []
            rate_failures: list[str] = []
            error_attempts = [a for a in result.attempts if a.get("status") == "error"]
            skipped = [a for a in result.attempts if a.get("status") == "skipped"]
            for attempt in error_attempts:
                error_str = attempt.get("error", "") or ""
                error_lower = error_str.lower()
                if "429" in error_str or "rate limit" in error_lower or "too many requests" in error_lower:
                    rate_failures.append(attempt.get("model", "unknown"))
                elif "401" in error_str or "403" in error_str or "forbidden" in error_lower or "unauthorized" in error_lower:
                    auth_failures.append(attempt.get("model", "unknown"))
                elif "402" in error_str or "payment required" in error_lower:
                    payment_failures.append(attempt.get("model", "unknown"))
                else:
                    other_failures.append(attempt.get("model", "unknown"))

            # Prefer the last *error* attempt's message, not the last attempt (which may be skipped)
            last_err = ""
            if error_attempts:
                last_err = (error_attempts[-1].get("error") or "").strip()
                if not last_err:
                    last_err = error_attempts[-1].get("error_short", "") or "empty provider error"
            elif skipped:
                # All models were skipped as unavailable/unhealthy before any
                # provider attempt — fail fast with per-model cooldown detail.
                attempted_errors = [a for a in result.attempts if a.get("status") == "error"]
                lines: list[str] = []
                for a in result.attempts:
                    mid = a.get("model", "?")
                    if a.get("status") == "skipped":
                        state = self.health.get_state(mid)
                        remaining = state.cooldown_remaining()
                        status = state.display_status
                        if remaining > 0:
                            lines.append(f"  • {mid} — {status} · cooldown {remaining:.0f}s")
                        else:
                            lines.append(f"  • {mid} — {status}")
                    elif a.get("status") == "error":
                        cat = a.get("category", "")
                        state = self.health.get_state(mid)
                        remaining = state.cooldown_remaining()
                        detail = f"{cat}" if cat else (a.get("error_short", "")[:80] or "failed")
                        if remaining > 0:
                            lines.append(f"  • {mid} — {detail} · cooldown {remaining:.0f}s")
                        else:
                            lines.append(f"  • {mid} — {detail}")
                result.final_error = (
                    "No configured free model is currently available. Attempted:\n"
                    + "\n".join(lines)
                    + "\nNo paid model was used. Retry after the cooldowns expire."
                )
                if attempted_errors:
                    last = attempted_errors[-1]
                    result.final_error += f"\nLast error: {last.get('error_short', '')[:160]}"
                return result

            # Structured per-model summary for diagnostics
            def _fmt_list(label: str, items: list[str]) -> str:
                return f"{label}: {', '.join(items[:3])}" if items else ""

            if rate_failures and not auth_failures and not payment_failures and not other_failures:
                result.final_error = (
                    f"All {len(rate_failures)} models are rate limited (429). "
                    f"Models: {', '.join(rate_failures)}. "
                    f"Last error: {last_err or 'rate limited'}"
                )
            elif auth_failures and not other_failures and not rate_failures:
                result.final_error = (
                    f"{len(auth_failures)} model(s) returned 401/403 and are unavailable. "
                    f"Models: {', '.join(auth_failures[:3])}. "
                    f"Last error: {last_err or 'auth failure'}"
                )
            elif payment_failures and not other_failures and not auth_failures and not rate_failures:
                result.final_error = (
                    f"{len(payment_failures)} model(s) require payment (402). "
                    f"Enable free mode (/free) or configure a provider with credits. "
                    f"Last error: {last_err or 'payment required'}"
                )
            else:
                parts: list[str] = []
                if rate_failures:
                    parts.append(f"{len(rate_failures)} rate limited (429)")
                if auth_failures:
                    parts.append(f"{len(auth_failures)} unavailable (401/403)")
                if payment_failures:
                    parts.append(f"{len(payment_failures)} require payment (402)")
                if other_failures:
                    parts.append(f"{len(other_failures)} failed")
                summary = ", ".join(parts) if parts else f"{len(model_chain)} models failed"
                # Never produce "Last error: unknown" — always surface something typed
                last_display = last_err or "no error details (check provider/attempts)"
                result.final_error = f"All {len(model_chain)} models failed ({summary}). Last error: {last_display}"
                # Attach per-attempt breakdown for verbose diagnostics (safe, no secrets)
                if error_attempts:
                    def _cls(v: str) -> str:
                        return "error" if v == "unknown" else v
                    breakdown = "; ".join(f"{a.get('model','?')} — {_cls(a.get('classification','?'))}: {a.get('error_short', a.get('error',''))[:120]}" for a in error_attempts[:8])
                    result.final_error += f" | Attempts: {breakdown}"
        return result
