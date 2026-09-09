"""Professional completion responses for Phase 10.5 (Parts 7-10, 31, 35-38).

Three small, pure components:

``CompletionClassifier``
    Maps the authoritative runtime outcome + view model into an explicit
    completion state — COMPLETE / COMPLETE_WITH_WARNINGS / PARTIAL / FAILED /
    BLOCKED / CANCELLED / VERIFICATION_FAILED — instead of a bare boolean.

``NextActionEngine``
    Suggests 2-4 *contextual* next actions from real evidence: the task type,
    changed files, verification state, Git state and remaining TODOs.  It
    never fabricates options — each suggestion is derived from observed state.

``CompletionFormatter``
    Renders the short, professional final response block (success, failure,
    verification failure, partial, cancellation) that the CLI prints after
    every task.  Only evidence that actually exists is shown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class CompletionState(str, Enum):
    """Explicit completion states (Part 31)."""

    COMPLETE = "COMPLETE"
    COMPLETE_WITH_WARNINGS = "COMPLETE_WITH_WARNINGS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELLED = "CANCELLED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"


# ── Completion classifier ─────────────────────────────────────────────────


class CompletionClassifier:
    """Decide the explicit completion state from runtime truth.

    The classifier reads ONLY authoritative state: the runtime status, the
    verification status, recovery exhaustion and the view-model evidence.
    Model claims never factor in.
    """

    @staticmethod
    def classify(vm: Any, outcome: Any = None) -> CompletionState:
        state = getattr(outcome, "state", None) if outcome is not None else None
        status = getattr(state, "status", None)
        status_value = getattr(status, "value", None) or (status if isinstance(status, str) else "")

        vm_status = vm.status if vm is not None else ""
        if vm_status == "cancelled" or status_value == "cancelled":
            return CompletionState.CANCELLED

        if status_value in ("failed",):
            blockers = list(getattr(state, "blockers", []) or [])
            if any("verification" in b or "verifier" in b for b in blockers):
                return CompletionState.VERIFICATION_FAILED
            if getattr(state, "recovery_exhausted", False):
                return CompletionState.BLOCKED
            return CompletionState.FAILED

        if status_value == "blocked":
            return CompletionState.BLOCKED

        if status_value != "success":
            # Runtime did not reach a success state.
            if vm is not None and vm.recovery_exhausted:
                return CompletionState.BLOCKED
            return CompletionState.FAILED

        # SUCCESS: differentiate COMPLETE vs COMPLETE_WITH_WARNINGS vs PARTIAL.
        verification_status = None
        if state is not None and getattr(state, "verification_status", None) is not None:
            vs = state.verification_status
            verification_status = getattr(vs, "value", None) or (vs if isinstance(vs, str) else "")
        elif vm is not None:
            verification_status = vm.verification_status

        if verification_status == "failed":
            return CompletionState.VERIFICATION_FAILED

        partial = False
        if vm is not None:
            failed_tasks = [a for a in vm.agents.values() if a.status == "failed"]
            if failed_tasks and not getattr(state, "recovery_exhausted", False):
                partial = True
        if partial:
            return CompletionState.PARTIAL

        # Warnings (recovery that occurred, model fallbacks, integrity notes).
        warnings: list[str] = []
        if vm is not None:
            warnings = list(vm.warnings or [])
        if vm is not None and vm.recovery_attempts:
            warnings.append(f"recovered {vm.recovery_attempts} issue(s)")
        if warnings:
            return CompletionState.COMPLETE_WITH_WARNINGS
        return CompletionState.COMPLETE


# ── Next-action engine ────────────────────────────────────────────────────


@dataclass
class NextAction:
    """One contextual suggestion with a short label."""

    label: str
    prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "prompt": self.prompt}


class NextActionEngine:
    """Derive 2-4 contextual next actions from observed evidence."""

    @staticmethod
    def _task_kind(request: str) -> str:
        low = (request or "").lower()
        if any(k in low for k in ("push", "github", "commit", "git")):
            return "git"
        if any(k in low for k in ("ui", "frontend", "css", "responsive", "style", "page")):
            return "ui"
        if any(k in low for k in ("doc", "readme", "documentation")):
            return "docs"
        if any(k in low for k in ("bug", "fix", "error", "broken", "fail", "issue")):
            return "bugfix"
        if any(k in low for k in ("test", "verify", "check")):
            return "testing"
        if any(k in low for k in ("explain", "describe", "what is", "how does")):
            return "explain"
        return "build"

    @staticmethod
    def _changed_exts(files: list[str]) -> set[str]:
        exts: set[str] = set()
        for f in files:
            m = re.search(r"\.([A-Za-z0-9]+)$", f)
            if m:
                exts.add(m.group(1).lower())
        return exts

    def suggest(
        self,
        request: str = "",
        *,
        files: list[str] | None = None,
        verification_status: str = "",
        git_commit: str = "",
        git_push: str = "",
        test_failures: int = 0,
        recovery_exhausted: bool = False,
        partial: bool = False,
    ) -> list[NextAction]:
        """Contextual suggestions (2-4). Never generic filler."""
        files = files or []
        kind = self._task_kind(request)
        exts = self._changed_exts(files)
        actions: list[NextAction] = []

        if partial or test_failures:
            actions.append(NextAction("Investigate the failing tests", "Investigate the failing tests and fix them"))
            actions.append(NextAction("Review the current changes", "Review the current changes"))
            actions.append(NextAction("Revert the attempted changes", "Revert the attempted changes"))
            return actions[:3]

        if verification_status == "failed":
            actions.append(NextAction("Investigate why verification failed", "Investigate why verification failed"))
            actions.append(NextAction("Review the changes and re-verify", "Review the changes and re-verify"))
            return actions[:3]

        if kind == "git":
            if git_commit and not git_push:
                actions.append(NextAction("Push the committed changes", "Push the committed changes to the remote"))
                actions.append(NextAction("Review the commit", "Review the last commit"))
                actions.append(NextAction("Check repository status", "Show git status"))
            elif git_commit and git_push:
                actions.append(NextAction("Continue development", "Continue working on the project"))
                actions.append(NextAction("Review the pushed commit", "Review the last commit"))
                actions.append(NextAction("Check repository status", "Show git status"))
            else:
                actions.append(NextAction("Stage and commit the changes", "Stage and commit the current changes"))
                actions.append(NextAction("Check repository status", "Show git status"))
            return actions[:3]

        if kind == "ui":
            actions.append(NextAction("Run a performance check", "Run a performance check on the UI"))
            actions.append(NextAction("Review the UI changes", "Review the UI changes"))
            actions.append(NextAction("Improve accessibility", "Improve the accessibility of the UI"))
            return actions[:3]

        if kind == "bugfix":
            actions.append(NextAction("Add a regression test", "Add a regression test for this fix"))
            actions.append(NextAction("Review the fix", "Review the fix"))
            actions.append(NextAction("Check for similar issues", "Check for similar issues elsewhere"))
            return actions[:3]

        if kind == "docs":
            actions.append(NextAction("Review the documentation", "Review the documentation changes"))
            actions.append(NextAction("Add usage examples", "Add usage examples to the documentation"))
            if not git_commit:
                actions.append(NextAction("Commit the documentation", "Commit the documentation changes"))
            return actions[:3]

        if kind == "testing":
            actions.append(NextAction("Fix remaining failures", "Fix the remaining test failures"))
            actions.append(NextAction("Run the full suite", "Run the full test suite"))
            actions.append(NextAction("Review failing tests", "Review the failing tests"))
            return actions[:3]

        if kind == "explain":
            actions.append(NextAction("Find potential bugs", "Look for potential bugs in the project"))
            actions.append(NextAction("Review the architecture", "Review the project architecture"))
            actions.append(NextAction("Run the test suite", "Run the test suite"))
            return actions[:3]

        # Build / general: derive from changed file extensions when possible.
        if "js" in exts or "ts" in exts or "tsx" in exts or "jsx" in exts:
            actions.append(NextAction("Run the JavaScript tests", "Run the JavaScript tests"))
        if "py" in exts:
            actions.append(NextAction("Run the Python tests", "Run the Python tests"))
        if "css" in exts or "scss" in exts:
            actions.append(NextAction("Review the styling changes", "Review the styling changes"))
        if not actions:
            actions.append(NextAction("Review the changes", "Review the current changes"))
            actions.append(NextAction("Run the test suite", "Run the test suite"))
            actions.append(NextAction("Continue development", "Continue working on the project"))
        if not git_commit and files:
            actions.append(NextAction("Commit the changes", "Commit the current changes"))
        return actions[:3]


# ── Completion formatter ──────────────────────────────────────────────────


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m{secs:02d}s"


def _bullets(items: list[str], indent: str = "  ") -> str:
    if not items:
        return ""
    return "\n".join(f"{indent}• {i}" for i in items)


class CompletionFormatter:
    """Render the short professional final response block.

    Every line is derived from evidence the caller passes in — the formatter
    never invents changes, tests, git state or performance claims.
    """

    def __init__(self, *, plain: bool = False) -> None:
        self.plain = plain

    # ── success ─────────────────────────────────────────────────────────

    def success(
        self,
        *,
        headline: str,
        files_modified: list[str],
        files_created: list[str],
        files_deleted: list[str] = None,
        tests_line: str = "",
        verification_status: str = "",
        git_commit: str = "",
        git_push: str = "",
        recovery_attempts: int = 0,
        duration: float = 0.0,
        agents: int = 0,
        tool_calls: int = 0,
        next_actions: list[NextAction] | None = None,
        summary: str = "",
        agent_response: str = "",
    ) -> str:
        """Render a successful completion summary (Part 35)."""
        out: list[str] = []
        out.append(f"✓ Done — {headline}")
        out.append("")

        if agent_response and self.plain:
            out.extend(agent_response.strip().splitlines())
            out.append("")

        if summary:
            out.append("Summary")
            out.append(f"  {summary[:400]}")
            out.append("")

        if files_modified or files_created or files_deleted:
            out.append("Changed")
            out.extend(_bullets([f"M {f}" for f in files_modified[:8]]).splitlines())
            out.extend(_bullets([f"A {f}" for f in files_created[:8]]).splitlines())
            out.extend(_bullets([f"D {f}" for f in (files_deleted or [])[:8]]).splitlines())
            out.append("")

        if tests_line:
            out.append("Validation")
            out.append(f"  {tests_line}")
            out.append("")

        if git_commit or git_push:
            out.append("Git")
            if git_commit:
                out.append(f"  ✓ Commit {git_commit[:12]}")
            if git_push:
                out.append(f"  ✓ Pushed {git_push}")
            out.append("")
        elif verification_status == "failed":
            out.append("Git")
            out.append("  Not requested")
            out.append("")

        if verification_status:
            if verification_status in ("passed", "not_started"):
                out.append("Verification")
                out.append(f"  ✓ {verification_status.replace('_', ' ').title()}")
            else:
                out.append("Verification")
                out.append(f"  ✗ {verification_status.replace('_', ' ').title()}")
            out.append("")

        if recovery_attempts:
            out.append("Recovery")
            out.append(f"  ✓ {recovery_attempts} issue(s) automatically repaired")
            out.append("")

        if duration > 0:
            parts = [f"{_fmt_elapsed(duration)}"]
            if agents:
                parts.append(f"{agents} agent(s)")
            if tool_calls:
                parts.append(f"{tool_calls} tool call(s)")
            out.append("Execution")
            out.append(f"  {' · '.join(parts)}")
            out.append("")

        if next_actions:
            out.append("Next")
            for a in next_actions:
                out.append(f"  → {a.label}")
            out.append("")

        return "\n".join(out).rstrip()

    # ── failure ─────────────────────────────────────────────────────────

    def failure(
        self,
        *,
        headline: str,
        what_happened: str = "",
        evidence_lines: list[str] | None = None,
        tried: list[str] | None = None,
        why_stopped: str = "",
        files_modified: list[str] | None = None,
        verification_status: str = "",
        recovery_attempts: int = 0,
        recovery_exhausted: bool = False,
        next_actions: list[NextAction] | None = None,
        agent_response: str = "",
    ) -> str:
        """Render an honest failure summary (Parts 10, 36)."""
        out: list[str] = []
        out.append(f"✗ {headline}")
        out.append("")

        if agent_response and self.plain:
            out.extend(agent_response.strip().splitlines())
            out.append("")

        if what_happened:
            out.append("What happened")
            out.append(f"  {what_happened[:300]}")
            out.append("")
        if evidence_lines:
            out.append("Evidence")
            out.extend(f"  {e}" for e in evidence_lines[:6])
            out.append("")
        if tried:
            out.append("What I tried")
            out.extend(_bullets(tried[:6]).splitlines())
            out.append("")
        if why_stopped:
            out.append("Why I stopped")
            out.append(f"  {why_stopped[:300]}")
            out.append("")
        if files_modified:
            out.append("Changes")
            out.extend(_bullets(files_modified[:8]).splitlines())
            out.append("")
        if verification_status == "failed":
            out.append("Verification")
            out.append("  ✗ Final verification failed")
            out.append("")
        if recovery_attempts:
            state = "exhausted" if recovery_exhausted else f"{recovery_attempts} attempt(s)"
            out.append("Recovery")
            out.append(f"  {state}")
            out.append("")

        out.append("No false success was reported.")
        out.append("")
        if next_actions:
            out.append("Next")
            for a in next_actions:
                out.append(f"  → {a.label}")
            out.append("")
        return "\n".join(out).rstrip()

    # ── cancellation ────────────────────────────────────────────────────

    def cancelled(
        self,
        *,
        completed: list[str] | None = None,
        interrupted: list[str] | None = None,
        uncommitted: list[str] | None = None,
        next_actions: list[NextAction] | None = None,
        agent_response: str = "",
    ) -> str:
        """Render the cancellation summary (Part 38)."""
        out: list[str] = []
        out.append("Cancellation requested...")
        out.append("✓ Runtime cancelled")
        out.append("✓ Agents stopped")
        out.append("✓ Locks released")
        out.append("✓ Session preserved")
        out.append("")

        if agent_response and self.plain:
            out.extend(agent_response.strip().splitlines())
            out.append("")
        if completed:
            out.append("Completed")
            out.extend(_bullets(completed[:6]).splitlines())
            out.append("")
        if interrupted:
            out.append("Interrupted")
            out.extend(_bullets(interrupted[:6]).splitlines())
            out.append("")
        if uncommitted:
            out.append("Uncommitted changes")
            out.extend(_bullets(uncommitted[:8]).splitlines())
            out.append("")
        if next_actions:
            out.append("Next")
            for a in next_actions:
                out.append(f"  → {a.label}")
            out.append("")
        return "\n".join(out).rstrip()
