"""Phase 10.5 — CompletionClassifier / CompletionFormatter / NextActionEngine tests.

The completion UX must be derived exclusively from authoritative runtime
evidence: explicit completion states (Part 31), contextual next actions
(Part 8), short professional final responses (Parts 7, 35-38). No invented
changes, tests or git state.
"""

from __future__ import annotations

from types import SimpleNamespace

from harness_core.cli.completion import (
    CompletionClassifier,
    CompletionFormatter,
    CompletionState,
    NextAction,
    NextActionEngine,
)
from harness_core.cli.runtime_dashboard import AgentView, RuntimeViewModel

# ── CompletionClassifier ───────────────────────────────────────────────────


def _outcome(status: str, **state_kwargs):
    state = SimpleNamespace(status=SimpleNamespace(value=status), **state_kwargs)
    return SimpleNamespace(state=state)


def _vm(**kwargs) -> RuntimeViewModel:
    vm = RuntimeViewModel()
    for k, v in kwargs.items():
        setattr(vm, k, v)
    return vm


def test_classify_cancelled():
    vm = _vm(status="cancelled")
    assert CompletionClassifier.classify(vm) == CompletionState.CANCELLED


def test_classify_failed_with_verification_blocker():
    outcome = _outcome(
        "failed",
        blockers=["verification failed"],
        recovery_exhausted=False,
    )
    assert CompletionClassifier.classify(RuntimeViewModel(), outcome) == CompletionState.VERIFICATION_FAILED


def test_classify_failed_with_recovery_exhausted_is_blocked():
    outcome = _outcome("failed", blockers=["test_failure"], recovery_exhausted=True)
    assert CompletionClassifier.classify(RuntimeViewModel(), outcome) == CompletionState.BLOCKED


def test_classify_plain_failed():
    outcome = _outcome("failed", blockers=["test_failure"], recovery_exhausted=False)
    assert CompletionClassifier.classify(RuntimeViewModel(), outcome) == CompletionState.FAILED


def test_classify_success_verification_failed():
    vm = _vm(verification_status="failed")
    outcome = _outcome("success", verification_status=SimpleNamespace(value="failed"))
    assert CompletionClassifier.classify(vm, outcome) == CompletionState.VERIFICATION_FAILED


def test_classify_success_partial_when_agent_failed():
    vm = _vm(verification_status="passed")
    agent = AgentView(task_id="t1", name="tester")
    agent.status = "failed"
    vm.agents["t1"] = agent
    outcome = _outcome("success", verification_status=SimpleNamespace(value="passed"))
    assert CompletionClassifier.classify(vm, outcome) == CompletionState.PARTIAL


def test_classify_success_with_warnings():
    vm = _vm(verification_status="passed")
    vm.warnings.append("model fallback used")
    outcome = _outcome("success", verification_status=SimpleNamespace(value="passed"))
    assert CompletionClassifier.classify(vm, outcome) == CompletionState.COMPLETE_WITH_WARNINGS


def test_classify_success_recovery_is_complete_with_warnings():
    vm = _vm(verification_status="passed", recovery_attempts=1)
    outcome = _outcome("success", verification_status=SimpleNamespace(value="passed"))
    assert CompletionClassifier.classify(vm, outcome) == CompletionState.COMPLETE_WITH_WARNINGS


def test_classify_clean_success():
    outcome = _outcome("success", verification_status=SimpleNamespace(value="passed"))
    assert CompletionClassifier.classify(RuntimeViewModel(), outcome) == CompletionState.COMPLETE


# ── NextActionEngine ───────────────────────────────────────────────────────


def test_ui_task_suggests_ui_actions():
    acts = NextActionEngine().suggest("make the UI more responsive and fast")
    labels = [a.label for a in acts]
    assert any("performance" in label.lower() for label in labels)
    assert len(acts) <= 4


def test_git_commit_without_push_suggests_push():
    acts = NextActionEngine().suggest(
        "push this to github", git_commit="06a6ae3", git_push=""
    )
    labels = [a.label for a in acts]
    assert any("push" in label.lower() for label in labels)


def test_git_fully_pushed_suggests_continue():
    acts = NextActionEngine().suggest(
        "push this to github", git_commit="06a6ae3", git_push="origin/main"
    )
    labels = [a.label for a in acts]
    assert any("continue" in label.lower() for label in labels)


def test_bugfix_suggests_regression_test():
    acts = NextActionEngine().suggest("fix the login bug")
    labels = [a.label for a in acts]
    assert any("regression" in label.lower() for label in labels)


def test_explain_task_suggests_exploration():
    acts = NextActionEngine().suggest("explain this project")
    labels = [a.label for a in acts]
    assert any("bugs" in label.lower() for label in labels)
    assert any("architecture" in label.lower() for label in labels)


def test_failure_suggestions_are_repair_focused():
    acts = NextActionEngine().suggest(
        "build feature", partial=True, test_failures=2
    )
    labels = [a.label for a in acts]
    assert any("investigate" in label.lower() for label in labels)
    assert any("revert" in label.lower() for label in labels)


def test_verification_failed_suggests_reverify():
    acts = NextActionEngine().suggest("build feature", verification_status="failed")
    labels = [a.label for a in acts]
    assert any("verification" in label.lower() for label in labels)


def test_suggestions_carry_actionable_prompts():
    for a in NextActionEngine().suggest("build a todo app"):
        assert isinstance(a, NextAction)
        assert a.label and a.prompt


# ── CompletionFormatter ────────────────────────────────────────────────────


def test_success_summary_only_evidence_present():
    out = CompletionFormatter(plain=True).success(
        headline="UI responsiveness improved",
        files_modified=["script.js", "styles.css"],
        files_created=["perf-report.md"],
        tests_line="✓ 32/32 tests passed",
        verification_status="passed",
        git_commit="",
        git_push="",
        duration=24.5,
        next_actions=[NextAction("Review the UI changes", "")],
    )
    assert "✓ Done" in out
    assert "UI responsiveness improved" in out
    assert "script.js" in out
    assert "perf-report.md" in out
    assert "24.5s" in out
    assert "Commit" not in out  # no git evidence passed → nothing claimed
    assert "Next" in out


def test_success_summary_git_section_from_evidence():
    out = CompletionFormatter(plain=True).success(
        headline="Pushed to GitHub",
        files_modified=[],
        files_created=[],
        tests_line="",
        verification_status="passed",
        git_commit="06a6ae3abcdef",
        git_push="origin/main",
        duration=10,
        next_actions=[],
    )
    assert "✓ Commit 06a6ae3ab" in out
    assert "✓ Pushed origin/main" in out


def test_failure_summary_is_honest():
    out = CompletionFormatter(plain=True).failure(
        headline="Could not safely complete the task",
        what_happened="node test.js failed repeatedly.",
        evidence_lines=["exit code: 1", "failure signature: ..."],
        why_stopped="The same failure repeated without meaningful progress.",
        verification_status="failed",
    )
    assert "✗ Could not safely complete the task" in out
    assert "exit code: 1" in out
    assert "✗ Checks failed" in out
    assert "✓ Done" not in out


def test_cancelled_summary_reports_lifecycle():
    out = CompletionFormatter(plain=True).cancelled(
        completed=["ARCHITECT"],
        interrupted=["FRONTEND"],
        uncommitted=["script.js"],
        next_actions=[NextAction("Resume", "")],
    )
    assert "✓ Runtime cancelled" in out
    assert "✓ Locks released" in out
    assert "ARCHITECT" in out
    assert "FRONTEND" in out
    assert "script.js" in out
