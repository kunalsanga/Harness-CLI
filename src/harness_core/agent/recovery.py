"""
Agent recovery system — classifies errors and determines recovery strategy.

Handles: model timeout, provider timeout, HTTP 429, HTTP 500,
malformed response, invalid tool call, command failure, test failure,
context overflow, provider unavailable, permission denial.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class RecoveryStrategy(enum.Enum):
    """Recovery strategies for different error types."""

    RETRY = "retry"                    # Retry the same operation
    FALLBACK_MODEL = "fallback_model"  # Try a different model
    SKIP_TOOL = "skip_tool"            # Skip the failed tool call
    ABORT = "abort"                    # Stop execution
    USER_ACTION = "user_action"        # Requires user intervention


@dataclass
class RecoveryDecision:
    """Decision on how to handle an error."""

    strategy: RecoveryStrategy
    reason: str
    retry_delay_seconds: float = 0.0
    fallback_model: str = ""


def classify_error(error: Exception) -> RecoveryDecision:
    """Classify an error and determine the appropriate recovery strategy.

    Returns a RecoveryDecision with the strategy and reason.
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()

    # Rate limiting (429)
    if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
        return RecoveryDecision(
            strategy=RecoveryStrategy.FALLBACK_MODEL,
            reason="Rate limited — try another model",
            retry_delay_seconds=5.0,
        )

    # Timeout
    if "timeout" in error_str or "timed out" in error_str or "asyncio.timeout" in error_type:
        return RecoveryDecision(
            strategy=RecoveryStrategy.RETRY,
            reason="Timeout — retry with backoff",
            retry_delay_seconds=2.0,
        )

    # Context overflow
    if any(kw in error_str for kw in ["context", "token limit", "too long", "maximum context"]):
        return RecoveryDecision(
            strategy=RecoveryStrategy.FALLBACK_MODEL,
            reason="Context overflow — try a larger model",
        )

    # Provider unavailable (5xx)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return RecoveryDecision(
            strategy=RecoveryStrategy.RETRY,
            reason="Server error — retry",
            retry_delay_seconds=3.0,
        )

    # Authentication/authorization (401, 403)
    if any(code in error_str for code in ["401", "403"]) or \
       any(kw in error_str for kw in ["unauthorized", "forbidden"]):
        return RecoveryDecision(
            strategy=RecoveryStrategy.ABORT,
            reason="Authentication failed — cannot recover",
        )

    # Not found (404)
    if "404" in error_str or "not found" in error_str:
        return RecoveryDecision(
            strategy=RecoveryStrategy.ABORT,
            reason="Model or resource not found",
        )

    # Permission denied
    if "permission" in error_str or "denied" in error_str:
        return RecoveryDecision(
            strategy=RecoveryStrategy.SKIP_TOOL,
            reason="Permission denied — skip this tool call",
        )

    # Tool-related errors
    if "unknown tool" in error_str or "tool" in error_type:
        return RecoveryDecision(
            strategy=RecoveryStrategy.SKIP_TOOL,
            reason="Unknown tool — skip and continue",
        )

    # Network errors
    if any(kw in error_str for kw in ["connection", "network", "dns", "resolve"]):
        return RecoveryDecision(
            strategy=RecoveryStrategy.RETRY,
            reason="Network error — retry",
            retry_delay_seconds=5.0,
        )

    # Default: retry once, then abort
    return RecoveryDecision(
        strategy=RecoveryStrategy.RETRY,
        reason=f"Unknown error ({error_type}) — retry once",
        retry_delay_seconds=1.0,
    )
