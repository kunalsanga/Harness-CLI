"""Tests for deterministic intent classification (Phase 10.6).

The core promise: read-only requests ("explain this project") never
acquire REQUIRED modification/test/fix tasks, and composite requests
("explain and fix") are never satisfied by a read-only pass alone.
"""

from __future__ import annotations

import pytest

from harness_core.agent.intent import Intent, classify_intent, is_read_only_verb
from harness_core.agent.todos import todo_category
from harness_core.agent.workflows import classify_workflow


class TestClassifyIntent:
    def test_explain_this_project_is_read_only_explain(self):
        r = classify_intent("explain this project")
        assert r.intent == Intent.EXPLAIN
        assert r.read_only is True

    def test_question_about_code_is_read_only(self):
        r = classify_intent("what does this code do?")
        assert r.read_only is True
        assert r.intent in (Intent.EXPLAIN, Intent.QUESTION)

    def test_show_architecture_is_read_only(self):
        r = classify_intent("show me the architecture")
        assert r.read_only is True
        assert r.intent == Intent.INSPECT

    def test_review_this_code_is_read_only(self):
        r = classify_intent("review this code")
        assert r.read_only is True
        assert r.intent == Intent.EXPLAIN

    def test_ui_modification_is_modify(self):
        r = classify_intent("make the UI more responsive and fast")
        assert r.intent == Intent.MODIFY
        assert r.read_only is False

    def test_fix_failing_tests_is_fix(self):
        r = classify_intent("fix the failing tests")
        assert r.intent == Intent.FIX
        assert r.read_only is False

    def test_run_the_tests_is_test(self):
        r = classify_intent("run the tests")
        assert r.intent == Intent.TEST
        assert r.read_only is False

    def test_update_readme_is_document(self):
        r = classify_intent("update the README with project docs")
        assert r.intent == Intent.DOCUMENT
        assert r.read_only is False

    def test_push_to_github_is_git(self):
        r = classify_intent("push this to github")
        assert r.intent == Intent.GIT
        assert r.read_only is False

    def test_composite_update_readme_and_push_is_git(self):
        r = classify_intent("update the README and push it to github")
        assert r.intent == Intent.GIT
        assert r.read_only is False

    def test_composite_explain_and_fix_is_not_read_only(self):
        # Explanation verb + modification verb → modification work wins.
        r = classify_intent("explain why tests fail and fix them")
        assert r.read_only is False
        assert r.intent != Intent.EXPLAIN

    def test_blank_goal_is_other(self):
        r = classify_intent("")
        assert r.intent == Intent.OTHER
        assert r.read_only is False


class TestReadOnlyStepFilter:
    """The planner's read-only filter drops modify/test steps."""

    def test_read_only_plan_keeps_inspection_only(self):
        steps = [
            "Read README",
            "Run the test suite",
            "Inspect bugs",
            "Review CSS",
            "Fix identified issues",
            "Discover workspace structure",
        ]
        kept = [s for s in steps if todo_category(s) == "inspect" or is_read_only_verb(s)]
        # "Run the test suite" and "Fix identified issues" must be dropped.
        assert "Run the test suite" not in kept
        assert "Fix identified issues" not in kept
        assert kept[0] == "Read README"
        assert "Review CSS" in kept
        assert "Discover workspace structure" in kept

    def test_read_only_verbs_are_recognized(self):
        assert is_read_only_verb("Inspect project structure")
        assert is_read_only_verb("Read key files")
        assert is_read_only_verb("Review the CSS")
        assert is_read_only_verb("Analyze current implementation")
        assert not is_read_only_verb("Implement the changes")
        assert not is_read_only_verb("Run the tests")
        assert not is_read_only_verb("Fix failing tests")


class TestWorkflowRouterIntent:
    """The workflow fast-path must respect intent too."""

    def test_pure_explain_routes_to_explain_workflow(self):
        assert classify_workflow("explain this project") == "explain"

    def test_push_routes_to_git_workflow(self):
        assert classify_workflow("push this to github") == "git_push"

    def test_run_tests_routes_to_test_workflow(self):
        assert classify_workflow("run the tests") == "test"

    def test_composite_explain_and_fix_is_not_read_only_workflow(self):
        # The read-only explain workflow must never swallow a requested fix.
        assert classify_workflow("explain why tests fail and fix them") is None

    def test_composite_update_readme_and_push_stays_git(self):
        assert classify_workflow("update the README and push it to github") == "git_push"

    def test_ui_improvement_uses_generic_loop(self):
        assert classify_workflow("make the UI more responsive and fast") is None
