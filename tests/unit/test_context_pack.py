"""Tests for the shared token estimator in harness_core.context.pack.

Scope note (Phase 2B): this file used to also certify ContextPackBuilder,
ContextPiece and ContextPack. Those classes are not reachable from the
runtime — nothing under src/ imports them — so their tests reported a
context-management feature as working while the agent loop never called it.
They were removed rather than left to give false confidence. The module
itself is kept; if ContextPackBuilder is ever wired into message assembly,
restore its tests from git history alongside that change.

`estimate_tokens` stays covered here because the agent loop does use it:
harness_core.agent.loop imports it to size every prompt it builds.
"""

from __future__ import annotations

from harness_core.context.pack import estimate_tokens


# ── Token estimation ────────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_empty(self):
        assert estimate_tokens('') == 1

    def test_short(self):
        assert estimate_tokens('hello') == 1  # 5 chars / 4 = 1

    def test_longer(self):
        assert estimate_tokens('a' * 100) == 25

    def test_newlines(self):
        assert estimate_tokens('hello\nworld\n') == 3  # 12 / 4 = 3


class TestTokenEstimationIsUsedByTheLoop:
    """Guard: the loop's prompt sizing must go through this one estimator.

    Three near-identical char/4 estimators existed in the codebase. If the
    loop stops importing this one, prompt budgets and these tests drift apart
    silently, so pin the dependency.
    """

    def test_loop_imports_the_shared_estimator(self):
        from harness_core.agent import loop as loop_module

        assert loop_module.estimate_tokens is estimate_tokens
