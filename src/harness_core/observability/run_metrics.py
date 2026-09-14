"""Per-run observability counters (milestone Part 15).

Captures the structured data needed to answer, post-run:

    how many model calls?            model_calls
    how many tool calls?             tool_calls
    how much context?                context_tokens, file contents selected
    how often compacted?             compactions, tokens_saved
    which files were selected?       selected_files
    how often AI relevance failed?   ai_relevance_failures
    how often steering occurred?     steering_messages
    how many workers spawned?        workers_spawned
    how often recovery happened?     recovery_attempts
    why did the run fail?            failure_reason
    how long did each stage take?    stage_durations

No analytics dashboard — just structured counters, optionally attached to
``run.completed`` and persisted per run. Cheap: all mutations are dict
increments; no I/O.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Stages the runtime times by default.
DEFAULT_STAGES = (
    "discovery", "planning", "executing", "verification",
    "compaction", "relevance",
)


@dataclass
class RunMetrics:
    """Structured counters for one run."""

    run_id: str = ""
    started_at: float = field(default_factory=time.time)
    model_calls: int = 0
    tool_calls: int = 0
    context_tokens: int = 0
    compactions: int = 0
    tokens_saved_by_compaction: int = 0
    selected_files: list[str] = field(default_factory=list)
    ai_relevance_failures: int = 0
    ai_relevance_used: int = 0
    ai_relevance_skipped: int = 0
    steering_messages: int = 0
    workers_spawned: int = 0
    recovery_attempts: int = 0
    quiet_tool_calls: int = 0
    ask_user_questions: int = 0
    failure_reason: str = ""
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    stage_starts: dict[str, float] = field(default_factory=dict)

    # ── recording ─────────────────────────────────────────────────────

    def record_model_call(self) -> None:
        self.model_calls += 1

    def record_tool_call(self, quiet: bool = False) -> None:
        self.tool_calls += 1
        if quiet:
            self.quiet_tool_calls += 1

    def record_context(self, tokens: int, files: list[str] | None = None) -> None:
        self.context_tokens += tokens
        for f in files or []:
            if f not in self.selected_files:
                self.selected_files.append(f)

    def record_compaction(self, tokens_saved: int) -> None:
        self.compactions += 1
        self.tokens_saved_by_compaction += max(0, tokens_saved)

    def record_ai_relevance(self, *, used: bool = False, failed: bool = False, skipped: bool = False) -> None:
        """Record one AI relevance stage outcome (exactly one flag true)."""
        if used:
            self.ai_relevance_used += 1
        elif failed:
            self.ai_relevance_failures += 1
        elif skipped:
            self.ai_relevance_skipped += 1

    def record_steering(self) -> None:
        self.steering_messages += 1

    def record_worker_spawned(self) -> None:
        self.workers_spawned += 1

    def record_recovery(self) -> None:
        self.recovery_attempts += 1

    def record_ask_user(self) -> None:
        self.ask_user_questions += 1

    # ── stage timing ──────────────────────────────────────────────────

    def stage_start(self, stage: str) -> None:
        self.stage_starts[stage] = time.monotonic()

    def stage_end(self, stage: str) -> None:
        start = self.stage_starts.pop(stage, None)
        if start is not None:
            self.stage_durations_ms[stage] = round(
                (time.monotonic() - start) * 1000, 1
            )

    # ── snapshot ──────────────────────────────────────────────────────

    def finish(self, failure_reason: str = "") -> dict[str, Any]:
        """Finalize and return the structured summary for run.completed."""
        self.failure_reason = failure_reason
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "duration_ms": round((time.time() - self.started_at) * 1000, 1),
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "quiet_tool_calls": self.quiet_tool_calls,
            "context_tokens": self.context_tokens,
            "compactions": self.compactions,
            "tokens_saved_by_compaction": self.tokens_saved_by_compaction,
            "selected_files": self.selected_files[:50],
            "ai_relevance": {
                "used": self.ai_relevance_used,
                "failed": self.ai_relevance_failures,
                "skipped": self.ai_relevance_skipped,
            },
            "steering_messages": self.steering_messages,
            "workers_spawned": self.workers_spawned,
            "recovery_attempts": self.recovery_attempts,
            "ask_user_questions": self.ask_user_questions,
            "failure_reason": self.failure_reason,
            "stage_durations_ms": dict(self.stage_durations_ms),
        }
