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
    """Structured counters for one run.

    Expanded with latency distributions and structured telemetry for
    observability (Directive 16).
    """

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
    # Directive 16: structured latency distributions
    model_latencies_ms: list[float] = field(default_factory=list)
    provider_latencies_ms: list[float] = field(default_factory=list)
    tool_latencies_ms: dict[str, list[float]] = field(default_factory=dict)
    verification_duration_ms: float = 0.0
    total_tokens_used: int = 0
    # Directive 16: error/retry tracking
    provider_errors: dict[str, int] = field(default_factory=dict)  # error_type -> count
    retries: int = 0
    context_reuses: int = 0
    context_compactions: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0

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

    # ── Directive 16: structured telemetry ────────────────────────────

    def record_model_latency(self, latency_ms: float) -> None:
        """Record one model call latency."""
        self.model_latencies_ms.append(latency_ms)
        self.model_calls += 1

    def record_provider_latency(self, latency_ms: float) -> None:
        """Record one provider HTTP latency."""
        self.provider_latencies_ms.append(latency_ms)

    def record_tool_latency(self, tool_name: str, latency_ms: float) -> None:
        """Record latency for a specific tool."""
        if tool_name not in self.tool_latencies_ms:
            self.tool_latencies_ms[tool_name] = []
        self.tool_latencies_ms[tool_name].append(latency_ms)

    def record_provider_error(self, error_type: str) -> None:
        """Record a provider error by classification."""
        self.provider_errors[error_type] = self.provider_errors.get(error_type, 0) + 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retries += 1

    def record_context_reuse(self) -> None:
        """Record a context reuse hit (skipped re-read)."""
        self.context_reuses += 1

    def record_context_compaction(self, tokens_saved: int) -> None:
        """Record a context compaction event."""
        self.context_compactions += 1
        self.tokens_saved_by_compaction += max(0, tokens_saved)

    def record_tokens(self, tokens: int) -> None:
        """Record cumulative token usage."""
        self.total_tokens_used += tokens

    def record_task_completed(self) -> None:
        self.tasks_completed += 1

    def record_task_failed(self) -> None:
        self.tasks_failed += 1

    def record_verification_duration(self, duration_ms: float) -> None:
        self.verification_duration_ms = duration_ms

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
        result: dict[str, Any] = {
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
            # Directive 16: structured telemetry
            "latency": self._latency_summary(),
            "total_tokens_used": self.total_tokens_used,
            "provider_errors": dict(self.provider_errors),
            "retries": self.retries,
            "context_reuses": self.context_reuses,
            "context_compactions": self.context_compactions,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "verification_duration_ms": round(self.verification_duration_ms, 1),
        }
        return result

    def _latency_summary(self) -> dict[str, Any]:
        """Compute latency distribution summary."""
        def _stats(values: list[float]) -> dict[str, Any]:
            if not values:
                return {"count": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "max_ms": 0}
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            return {
                "count": n,
                "mean_ms": round(sum(sorted_vals) / n, 1),
                "p50_ms": round(sorted_vals[n // 2], 1),
                "p95_ms": round(sorted_vals[int(n * 0.95)], 1) if n > 1 else round(sorted_vals[0], 1),
                "max_ms": round(sorted_vals[-1], 1),
            }

        return {
            "model": _stats(self.model_latencies_ms),
            "provider": _stats(self.provider_latencies_ms),
            "tools": {
                name: _stats(lats)
                for name, lats in self.tool_latencies_ms.items()
            },
        }
