"""Explicit context budgets with thresholds and usage accounting.

Distinguishes token allocations by message class so the runtime can answer
"where did the context go" and enforce compaction deterministically.

Budgets are configurable — the runtime ships sane defaults but never
hardcodes a single magic number as policy:

    user_critical   latest user request + steering messages (never dropped)
    assistant       assistant reasoning/output blocks
    tool_output     tool results / execution events
    file_content    file reads injected into context
    summaries       compaction summaries and task-state roll-ups
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BudgetClass(str, Enum):
    """Classes of context, each with its own allocation."""

    USER_CRITICAL = "user_critical"
    ASSISTANT = "assistant"
    TOOL_OUTPUT = "tool_output"
    FILE_CONTENT = "file_content"
    SUMMARIES = "summaries"


@dataclass
class ContextBudgetConfig:
    """Configurable budget limits.

    ``max_total_tokens`` is the hard ceiling on assembled context.
    ``compaction_threshold`` (fraction) triggers proactive compaction;
    ``emergency_threshold`` (fraction) forces aggressive deterministic
    truncation. Compaction must start strictly before the emergency
    threshold, which is validated in ``__post_init__``.
    """

    max_total_tokens: int = 100_000
    # Per-class allocation as fractions of max_total_tokens.
    class_allocation: dict[BudgetClass, float] = field(default_factory=lambda: {
        BudgetClass.USER_CRITICAL: 0.10,
        BudgetClass.ASSISTANT: 0.15,
        BudgetClass.TOOL_OUTPUT: 0.35,
        BudgetClass.FILE_CONTENT: 0.25,
        BudgetClass.SUMMARIES: 0.15,
    })
    compaction_threshold: float = 0.80   # start compacting at 80% usage
    emergency_threshold: float = 0.95    # force truncation at 95% usage

    def __post_init__(self) -> None:
        if self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if not 0.0 < self.compaction_threshold < self.emergency_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < compaction_threshold < emergency_threshold <= 1"
            )
        # Normalize allocations so they always sum to 1.0 regardless of
        # user customization — keeps per-class limits meaningful.
        total = sum(self.class_allocation.values())
        if total <= 0:
            raise ValueError("class_allocation must have positive mass")
        if abs(total - 1.0) > 1e-9:
            self.class_allocation = {
                k: v / total for k, v in self.class_allocation.items()
            }

    def limit_for(self, cls: BudgetClass) -> int:
        """Per-class token limit."""
        return int(self.max_total_tokens * self.class_allocation.get(cls, 0.0))


@dataclass
class _ClassUsage:
    tokens: int = 0
    messages: int = 0


class ContextBudgetManager:
    """Tracks context usage against an explicit, configurable budget.

    The manager is an accounting layer, not an enforcer: callers report
    usage per class and query state. Enforcement (dropping/compacting)
    lives in the compaction/pipeline layers that *own* those messages —
    this keeps single-responsibility and makes the manager trivially
    testable.
    """

    def __init__(self, config: ContextBudgetConfig | None = None) -> None:
        self.config = config or ContextBudgetConfig()
        self._usage: dict[BudgetClass, _ClassUsage] = {
            cls: _ClassUsage() for cls in BudgetClass
        }

    # ── accounting ────────────────────────────────────────────────────

    def record(self, cls: BudgetClass, tokens: int) -> None:
        """Record token usage for a message class."""
        if tokens <= 0:
            return
        u = self._usage[cls]
        u.tokens += tokens
        u.messages += 1

    def reset(self) -> None:
        """Reset all usage counters (e.g. at run start)."""
        for u in self._usage.values():
            u.tokens = 0
            u.messages = 0

    # ── queries ───────────────────────────────────────────────────────

    @property
    def current_usage(self) -> int:
        return sum(u.tokens for u in self._usage.values())

    @property
    def max_tokens(self) -> int:
        return self.config.max_total_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.current_usage)

    @property
    def usage_fraction(self) -> float:
        return self.current_usage / self.max_tokens if self.max_tokens else 0.0

    @property
    def compaction_recommended(self) -> bool:
        """True when usage crossed the proactive compaction threshold."""
        return self.usage_fraction >= self.config.compaction_threshold

    @property
    def emergency_truncation_required(self) -> bool:
        """True when usage crossed the emergency threshold."""
        return self.usage_fraction >= self.config.emergency_threshold

    def class_over_limit(self, cls: BudgetClass) -> bool:
        limit = self.config.limit_for(cls)
        return limit > 0 and self._usage[cls].tokens > limit

    def status(self) -> dict[str, Any]:
        """Structured status for events / dashboards."""
        return {
            "max_total_tokens": self.max_tokens,
            "current_usage": self.current_usage,
            "remaining": self.remaining,
            "usage_fraction": round(self.usage_fraction, 4),
            "compaction_threshold": self.config.compaction_threshold,
            "emergency_threshold": self.config.emergency_threshold,
            "compaction_recommended": self.compaction_recommended,
            "emergency_truncation_required": self.emergency_truncation_required,
            "per_class": {
                cls.value: {
                    "tokens": self._usage[cls].tokens,
                    "messages": self._usage[cls].messages,
                    "limit": self.config.limit_for(cls),
                }
                for cls in BudgetClass
            },
            "updated_at": time.time(),
        }
