"""Runtime governance for Phase 10.5.

Two small, pure (Rich-free, EventBus-free) state machines the runtime uses
to stay honest about progress:

``ConvergenceGovernor``
    Tracks repeated commands / tool calls / failure signatures across an
    execution and detects stagnation.  When the same failing command repeats
    without any intervening change, it classifies the situation (``stalled``)
    and recommends a strategy switch (e.g. TESTER → DEBUGGER).  It never
    decides anything by itself — it reports evidence so the runtime (the
    only authority) can act.

``BudgetGovernor``
    Tracks per-stage token/tool consumption (planning / exploration /
    implementation / debugging / verification).  A simple task may no longer
    burn the entire global token budget in one phase: each stage gets a
    bounded allowance, and the governor reports which stage is over budget.

Both are deterministic and unit-testable without a running runtime.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

# ── Convergence governor ──────────────────────────────────────────────────


@dataclass
class StagnationEvidence:
    """One piece of evidence that the execution is not converging."""

    kind: str  # repeated_command | repeated_failure | repeated_patch | no_progress
    detail: str
    count: int
    at: float = field(default_factory=time.time)


@dataclass
class StagnationReport:
    """Deterministic report of detected stagnation (evidence only)."""

    stalled: bool
    reason: str = ""
    evidence: list[StagnationEvidence] = field(default_factory=list)
    repeated_command: str = ""
    failure_signature: str = ""
    recommended_switch: str = ""  # e.g. TESTER -> DEBUGGER
    escalate_model: bool = False
    escalate_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stalled": self.stalled,
            "reason": self.reason,
            "evidence": [
                {"kind": e.kind, "detail": e.detail, "count": e.count} for e in self.evidence
            ],
            "repeated_command": self.repeated_command,
            "failure_signature": self.failure_signature,
            "recommended_switch": self.recommended_switch,
            "escalate_model": self.escalate_model,
            "escalate_reason": self.escalate_reason,
        }


def _signature(command: str, exit_code: int | None, stderr: str = "") -> str:
    """Deterministic failure signature: command + exit code + first stderr line."""
    first_err = next((ln.strip() for ln in (stderr or "").splitlines() if ln.strip()), "")
    key = f"{command}|{exit_code}|{first_err[:200]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


class ConvergenceGovernor:
    """Detects repeated operations and stagnation from real tool evidence.

    Thresholds are deliberately small so blind retry loops are caught early:
    a command that fails twice with the *same* signature is a signal to stop
    and diagnose, not to try again with the same context.
    """

    REPEAT_WARN_AT = 2
    REPEAT_STALL_AT = 3

    def __init__(
        self,
        *,
        repeat_warn_at: int = REPEAT_WARN_AT,
        repeat_stall_at: int = REPEAT_STALL_AT,
        max_no_progress_events: int = 4,
    ) -> None:
        self.repeat_warn_at = repeat_warn_at
        self.repeat_stall_at = repeat_stall_at
        self.max_no_progress_events = max_no_progress_events

        # command signature -> consecutive failures
        self._failures: dict[str, int] = {}
        # command (normalized) -> consecutive executions
        self._commands: dict[str, int] = {}
        # path -> consecutive identical patches (write/edit with same hash)
        self._patches: dict[str, tuple[str, int]] = {}  # path -> (content_hash, count)
        # action key -> consecutive repeats (any outcome)
        self._actions: dict[str, int] = {}
        self._last_action: str = ""
        self._no_progress_events: int = 0
        self._last_patch_at: float = 0.0
        self._evidence: list[StagnationEvidence] = []
        self._last_command: str = ""
        self._last_failure_sig: str = ""
        self._last_fail_command: str = ""

    # ── observation ─────────────────────────────────────────────────────

    def record_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        succeeded: bool,
        exit_code: int | None = None,
        stderr: str = "",
        action_key: str = "",
    ) -> None:
        """Record one real tool execution (call + outcome)."""
        key = action_key or f"{tool_name}:{_signature(str(args), None)}"

        # Repeated identical action (regardless of outcome).
        if key == self._last_action:
            self._actions[key] = self._actions.get(key, 0) + 1
        else:
            self._actions[key] = 1
            self._last_action = key

        if tool_name == "run_command":
            command = str(args.get("command", ""))
            self._last_command = command
            if not succeeded:
                sig = _signature(command, exit_code, stderr)
                # A different failing command starts a fresh failure context.
                if command != self._last_fail_command:
                    self._failures.clear()
                    self._commands.clear()
                    self._last_fail_command = command
                self._failures[sig] = self._failures.get(sig, 0) + 1
                self._last_failure_sig = sig
                # Only consecutive identical *failing* commands accumulate a
                # repeat count — a passing run resets it below, so legitimate
                # retest-after-edit cycles never false-positive.
                self._commands[command] = self._commands.get(command, 0) + 1
            else:
                # Success is progress for this command context: reset the
                # failure streak so a later failure starts counting anew.
                self._failures.clear()
                self._last_fail_command = ""
                self._commands.pop(command, None)
        elif tool_name in ("write_file", "edit_file"):
            path = str(args.get("path", args.get("file_path", "")))
            content = str(args.get("content", ""))
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
            prev = self._patches.get(path)
            if prev is not None and prev[0] == content_hash:
                self._patches[path] = (content_hash, prev[1] + 1)
            else:
                self._patches[path] = (content_hash, 1)
            self._last_patch_at = time.time()
            if succeeded:
                # A code change between failures is real progress: the old
                # failure evidence may be stale. The identical-patch counter
                # still catches writing the same content over and over.
                self._failures.clear()
                self._commands.clear()
                self._last_fail_command = ""

    def record_no_progress(self) -> None:
        """Record an iteration that produced no meaningful progress."""
        self._no_progress_events += 1

    # ── classification ──────────────────────────────────────────────────

    def report(self, *, max_repeats: int | None = None) -> StagnationReport:
        """Deterministic stagnation report from the recorded evidence."""
        stall_at = max_repeats or self.repeat_stall_at
        warn_at = self.repeat_warn_at
        evidence: list[StagnationEvidence] = []
        repeated_command = ""
        failure_sig = ""
        worst_count = 0

        # Repeated identical failing command (strongest signal).
        for sig, count in self._failures.items():
            if count >= warn_at:
                evidence.append(
                    StagnationEvidence("repeated_failure", f"same failure {count}x", count)
                )
            if count > worst_count:
                worst_count = count
                failure_sig = sig
        # Repeated identical command (even when mixed outcome).
        for cmd, count in self._commands.items():
            if count >= stall_at:
                evidence.append(
                    StagnationEvidence("repeated_command", f"{cmd} {count}x", count)
                )
                if not repeated_command:
                    repeated_command = cmd
                worst_count = max(worst_count, count)
        # Repeated identical patch with no progress between.
        for path, (_, count) in self._patches.items():
            if count >= stall_at:
                evidence.append(
                    StagnationEvidence("repeated_patch", f"identical patch to {path} {count}x", count)
                )
                worst_count = max(worst_count, count)
        # Repeated identical action.
        for key, count in self._actions.items():
            if count >= stall_at + 1:
                evidence.append(
                    StagnationEvidence("repeated_action", f"{key[:80]} {count}x", count)
                )
        # No-progress streak.
        if self._no_progress_events >= self.max_no_progress_events:
            evidence.append(
                StagnationEvidence(
                    "no_progress",
                    f"{self._no_progress_events} iterations without progress",
                    self._no_progress_events,
                )
            )

        if not evidence:
            return StagnationReport(stalled=False)

        stalled = (
            any(e.kind == "repeated_failure" and e.count >= stall_at for e in evidence)
            or any(e.kind == "repeated_command" and e.count >= stall_at for e in evidence)
            or any(e.kind == "repeated_patch" and e.count >= stall_at for e in evidence)
            or any(e.kind == "no_progress" for e in evidence)
        )
        if not stalled:
            # Warn-level evidence is still useful — the runtime decides whether
            # to diagnose now or keep going. Never drop observed evidence.
            return StagnationReport(stalled=False, evidence=evidence)

        reason_parts = [f"{e.detail}" for e in evidence[:2]]
        reason = "; ".join(reason_parts)

        report = StagnationReport(
            stalled=True,
            reason=reason,
            evidence=evidence,
            repeated_command=repeated_command,
            failure_signature=failure_sig,
        )

        # Strategy switch recommendation (runtime decides, governor suggests).
        if repeated_command and failure_sig:
            report.recommended_switch = "TESTER -> DEBUGGER"
        elif repeated_command:
            report.recommended_switch = "retry with diagnosis"
        if repeated_command and failure_sig and worst_count >= stall_at:
            report.escalate_model = True
            report.escalate_reason = "repeated identical failure without progress"

        return report

    def reset(self) -> None:
        """Reset all tracking (per task/run boundary)."""
        self._failures = {}
        self._commands = {}
        self._patches = {}
        self._actions = {}
        self._last_action = ""
        self._no_progress_events = 0
        self._evidence = []
        self._last_command = ""
        self._last_failure_sig = ""
        self._last_fail_command = ""


# ── Budget governor ───────────────────────────────────────────────────────


@dataclass
class StageBudget:
    """Bounded allowance for one execution stage."""

    name: str
    max_tool_calls: int = 40
    max_tokens: int = 60_000  # estimated prompt tokens
    tool_calls: int = 0
    tokens: int = 0


class BudgetGovernor:
    """Per-stage token/tool budgets (Part 14).

    Prevents a single phase (e.g. exploration or debugging) from consuming
    the entire global budget: each stage has its own allowance, and the
    governor reports which stage is at risk so the runtime can intervene.
    """

    STAGES = ("planning", "exploration", "implementation", "debugging", "verification")

    def __init__(self, *, default_tool_cap: int = 40, default_token_cap: int = 60_000) -> None:
        self._stages: dict[str, StageBudget] = {
            s: StageBudget(name=s, max_tool_calls=default_tool_cap, max_tokens=default_token_cap)
            for s in self.STAGES
        }
        self._current_stage: str = "planning"
        self._total_tools = 0
        self._total_tokens = 0

    def set_stage(self, stage: str) -> None:
        if stage in self._stages:
            self._current_stage = stage

    @property
    def current_stage(self) -> str:
        return self._current_stage

    def record_tool(self, *, stage: str | None = None) -> None:
        s = self._stages.get(stage or self._current_stage, self._stages["planning"])
        s.tool_calls += 1
        self._total_tools += 1

    def record_tokens(self, tokens: int, *, stage: str | None = None) -> None:
        if tokens <= 0:
            return
        s = self._stages.get(stage or self._current_stage, self._stages["planning"])
        s.tokens += tokens
        self._total_tokens += tokens

    def stage_over_budget(self, stage: str) -> bool:
        s = self._stages.get(stage)
        if s is None:
            return False
        return s.tool_calls > s.max_tool_calls or s.tokens > s.max_tokens

    def over_budget_stages(self) -> list[str]:
        return [s for s in self.STAGES if self.stage_over_budget(s)]

    def usage(self) -> dict[str, Any]:
        return {
            "total_tools": self._total_tools,
            "total_tokens": self._total_tokens,
            "current_stage": self._current_stage,
            "stages": {
                s: {
                    "tool_calls": st.tool_calls,
                    "tokens": st.tokens,
                    "max_tool_calls": st.max_tool_calls,
                    "max_tokens": st.max_tokens,
                    "over_budget": self.stage_over_budget(s),
                }
                for s, st in self._stages.items()
            },
        }

    def reset(self) -> None:
        self._stages = {
            s: StageBudget(name=s, max_tool_calls=st.max_tool_calls, max_tokens=st.max_tokens)
            for s, st in self._stages.items()
        }
        self._current_stage = "planning"
        self._total_tools = 0
        self._total_tokens = 0
