"""Deterministic intent classification (Phase 10.6).

The runtime must understand what the user actually asked for BEFORE
planning, so that:

    "explain this project"      → read-only EXPLAIN plan (no test/fix tasks)
    "make the UI faster"        → MODIFY plan (inspect + implement + validate)
    "push this to github"       → GIT plan
    "update README and push"    → composite DOCUMENT + GIT

The classifier is intentionally lightweight and deterministic: obvious
requests must not require an extra model call. Model-assisted
classification (if ever added) must remain behind this same interface
so the runtime decisions stay in one place.

Read-only intents never acquire REQUIRED modification tasks. If a goal
mixes an explanation verb with a modification verb ("explain why tests
fail and fix them"), the intent is treated as modification work so the
explanation alone can never be declared complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    """Primary intent of a user request."""

    EXPLAIN = "EXPLAIN"          # read-only: describe/analyze/summarize
    QUESTION = "QUESTION"        # read-only: direct question about the code
    INSPECT = "INSPECT"          # read-only: show/report current state
    REVIEW = "REVIEW"            # read-only unless fixes are requested
    MODIFY = "MODIFY"            # change existing behavior/code
    FIX = "FIX"                  # diagnosis + modification + verification
    TEST = "TEST"                # run/validate tests
    DOCUMENT = "DOCUMENT"        # documentation changes
    GIT = "GIT"                  # git operations (commit/push/status)
    OTHER = "OTHER"


@dataclass(frozen=True)
class IntentResult:
    """Outcome of deterministic intent classification."""

    intent: Intent
    read_only: bool  # True → plan must not contain required modification tasks
    reason: str = ""


# ── Deterministic signal sets ─────────────────────────────────────────

_READ_VERBS = (
    "explain", "describe", "summarize", "understand", "overview",
    "walk me through", "show me", "what is", "what does", "what are",
    "how does", "how is", "why does", "why is", "architecture",
    "how it works", "tell me about", "analyze this",
    "review", "audit", "inspect", "check", "look at", "read",
)
_QUESTION_STARTERS = ("what", "how", "why", "which", "where", "when", "who", "does", "is there")

_MODIFY_VERBS = (
    "make", "change", "improve", "optimize", "implement", "build",
    "create", "write", "edit", "update", "fix", "repair", "refactor",
    "enhance", "modify", "remove", "delete", "upgrade", "redesign",
    "add", "configure", "rewrite", "rework",
)
_TEST_PHRASES = (
    "run the test", "run tests", "run test", "run pytest",
    "run npm test", "test suite", "run the tests",
)
_GIT_WORDS = ("push", "github", "git ", "commit", "remote", "pull request")
_DOC_WORDS = ("readme", "documentation", "docstring", "docs", "document")

# Verbs that only produce evidence, never modify the repository.
_READ_ONLY = ("explain", "describe", "summarize", "overview", "understand",
              "walk me through", "analyze", "inspect", "read", "review",
              "list", "show", "check", "search", "discover", "map",
              "audit", "look")


def _contains(low: str, words: tuple[str, ...]) -> bool:
    """Match whole words (with optional suffixes) to avoid substring collisions.

    "read" should not match "readme"; "how" should not match "show".
    """
    import re
    for w in words:
        # Match the word as a whole word with common suffixes stripped for comparison
        # e.g. "read" in "readme" → False; "read" in "read the file" → True
        pattern = r"(?<![a-z])" + re.escape(w) + r"(?:s|es|ing|ed)?(?![a-z])"
        if re.search(pattern, low):
            return True
    return False


def classify_intent(goal: str) -> IntentResult:
    """Classify the primary intent of a goal deterministically.

    Precedence: GIT > TEST > FIX/MODIFY > EXPLAIN/QUESTION/INSPECT.
    A modification verb anywhere in the goal makes it non-read-only,
    even when explanation verbs are also present (composite requests
    must not be satisfied by a read-only pass).
    """
    g = (goal or "").strip().lower()
    if not g:
        return IntentResult(Intent.OTHER, read_only=False, reason="empty_goal")

    has_mod = _contains(g, _MODIFY_VERBS)
    has_read = _contains(g, _READ_VERBS)
    has_test = _contains(g, _TEST_PHRASES) or g.endswith("tests") or " tests" in g
    has_git = _contains(g, _GIT_WORDS)
    has_doc = _contains(g, _DOC_WORDS)

    # Git dominates — commit/push are concrete actions.
    if has_git:
        return IntentResult(Intent.GIT, read_only=False, reason="git_keyword")

    # Testing dominates when the user asks to run/validate tests.
    if has_test:
        if "fix" in g or "repair" in g:
            return IntentResult(Intent.FIX, read_only=False, reason="fix_test")
        if has_mod and not has_read:
            return IntentResult(Intent.MODIFY, read_only=False, reason="test_modify")
        return IntentResult(Intent.TEST, read_only=False, reason="test_keyword")

    # Documentation request — "update the README" / "document this" / "add docs"
    # is a content edit, not a code change. DOCUMENT ranks above generic MODIFY
    # so the planner builds read+write+commit rather than inspect+implement+test.
    if has_doc and not has_read:
        if has_mod:
            return IntentResult(Intent.DOCUMENT, read_only=False, reason="doc_modify")
        return IntentResult(Intent.DOCUMENT, read_only=False, reason="doc_keyword")

    # Modification verbs (with or without explanation verbs) mean work.
    if has_mod:
        if has_test or _contains(g, ("failing", "bug", "error", "broken", "crash")):
            return IntentResult(Intent.FIX, read_only=False, reason="bug_modify")
        return IntentResult(Intent.MODIFY, read_only=False, reason="modify_verb")

    # Pure read-only intents — check INSPECT signals BEFORE the question-word
    # check to avoid "show"/"how" substring collisions.
    if has_read:
        # INSPECT: showing/reporting concrete state takes priority.
        if _contains(g, ("show me", "show the", "list", "status", "print", "display", "architecture")):
            return IntentResult(Intent.INSPECT, read_only=True, reason="inspect_verb")
        # EXPLAIN: why/how/what questions about behavior.
        if _contains(g, _QUESTION_STARTERS[:3] + ("why does", "why is", "how does", "how is")):
            return IntentResult(Intent.EXPLAIN, read_only=True, reason="explain_question")
        return IntentResult(Intent.EXPLAIN, read_only=True, reason="read_verb")

    # Question-shaped goals are read-only by default.
    first = g.split()[0] if g.split() else ""
    if first in _QUESTION_STARTERS:
        return IntentResult(Intent.QUESTION, read_only=True, reason="question_starter")

    return IntentResult(Intent.OTHER, read_only=False, reason="no_signal")


def is_read_only_verb(verb: str) -> bool:
    """True when the leading verb of a TODO step only inspects/analyzes."""
    v = (verb or "").strip().lower()
    first = v.split()[0] if v.split() else ""
    return first in _READ_ONLY
