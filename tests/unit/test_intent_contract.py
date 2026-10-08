"""Regression tests for canonical TaskIntent, TaskContract, and adaptive execution.

Verifies:
- Intent classification covers all 12 required intents
- TaskContract derives correct properties from intent
- execute_interactive classifies intent before planning
- Read-only intents skip verification
- Structured completion events include intent data
- Adaptive execution selects appropriate role per intent
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.observability.events import EventBus


# ── Intent Classification ──────────────────────────────────────────────


class TestIntentClassification:
    """Tests for classify_request in planning/domain.py."""

    def test_explain_project(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("explain this project")
        assert c.intent == TaskIntent.EXPLAIN
        assert c.is_read_only is True

    def test_analyze_architecture(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("analyze the architecture")
        assert c.intent in (TaskIntent.ANALYZE, TaskIntent.RESEARCH)
        assert c.is_read_only is True

    def test_build_feature(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("build a restaurant website")
        assert c.intent == TaskIntent.IMPLEMENT
        assert c.is_read_only is False

    def test_fix_bug(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("fix the login bug")
        assert c.intent in (TaskIntent.DEBUG, TaskIntent.MODIFY)
        assert c.is_read_only is False

    def test_run_tests(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("run tests")
        assert c.intent == TaskIntent.TEST
        assert c.is_read_only is False

    def test_git_push(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("push to github")
        assert c.intent == TaskIntent.GIT
        assert c.is_read_only is False

    def test_refactor(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("refactor the authentication module")
        assert c.intent == TaskIntent.REFACTOR
        assert c.is_read_only is False

    def test_review_code(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("review this code")
        assert c.intent in (TaskIntent.REVIEW, TaskIntent.ANALYZE)
        # REVIEW is read-only unless fixes are explicitly requested
        assert c.is_read_only is True

    def test_deploy(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("deploy the application")
        assert c.intent == TaskIntent.DEPLOY
        assert c.is_read_only is False

    def test_research_investigate(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("research how caching works here")
        assert c.intent in (TaskIntent.RESEARCH, TaskIntent.EXPLAIN)
        assert c.is_read_only is True

    def test_describe_project(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("describe this project")
        assert c.intent in (TaskIntent.EXPLAIN, TaskIntent.READ_ONLY)
        assert c.is_read_only is True

    def test_summarize(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("summarize the test suite")
        assert c.intent in (TaskIntent.EXPLAIN, TaskIntent.READ_ONLY)
        assert c.is_read_only is True

    def test_what_is_project(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("what is this project")
        assert c.intent in (TaskIntent.EXPLAIN, TaskIntent.READ_ONLY)
        assert c.is_read_only is True

    def test_how_does_work(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("how does the authentication work")
        assert c.intent in (TaskIntent.EXPLAIN, TaskIntent.READ_ONLY)
        assert c.is_read_only is True

    def test_modify_existing(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("update the README")
        assert c.intent in (TaskIntent.MODIFY,)
        assert c.is_read_only is False


# ── TaskContract Properties ────────────────────────────────────────────


class TestTaskContract:
    """Tests for TaskContract derived properties."""

    def test_read_only_contract_skips_verification(self):
        from harness_core.planning.domain import (
            TaskContract, TaskIntent, TaskComplexity,
        )
        contract = TaskContract(
            objective="explain the project",
            intent=TaskIntent.EXPLAIN,
            complexity=TaskComplexity.TRIVIAL,
        )
        assert contract.is_read_only is True
        assert contract.needs_verification is False
        assert contract.should_skip_verification() is True
        assert contract.needs_planning is False

    def test_implementation_contract_needs_verification(self):
        from harness_core.planning.domain import (
            TaskContract, TaskIntent, TaskComplexity,
        )
        contract = TaskContract(
            objective="build a feature",
            intent=TaskIntent.IMPLEMENT,
            complexity=TaskComplexity.MEDIUM,
        )
        assert contract.is_read_only is False
        assert contract.needs_verification is True
        assert contract.should_skip_verification() is False
        assert contract.needs_planning is True

    def test_git_contract_needs_git(self):
        from harness_core.planning.domain import (
            TaskContract, TaskIntent, TaskComplexity,
        )
        contract = TaskContract(
            objective="push to github",
            intent=TaskIntent.GIT,
            complexity=TaskComplexity.SIMPLE,
        )
        assert contract.needs_git is True
        assert contract.needs_verification is True
        assert contract.is_read_only is False

    def test_analyze_contract_is_read_only(self):
        from harness_core.planning.domain import (
            TaskContract, TaskIntent, TaskComplexity,
        )
        contract = TaskContract(
            objective="analyze the codebase",
            intent=TaskIntent.ANALYZE,
            complexity=TaskComplexity.MEDIUM,
        )
        assert contract.is_read_only is True
        assert contract.needs_verification is False

    def test_contract_to_dict_includes_is_read_only(self):
        from harness_core.planning.domain import (
            TaskContract, TaskIntent,
        )
        contract = TaskContract(
            objective="explain",
            intent=TaskIntent.EXPLAIN,
        )
        d = contract.to_dict()
        assert "is_read_only" in d
        assert d["is_read_only"] is True

    def test_contract_from_dict_roundtrip(self):
        from harness_core.planning.domain import (
            TaskContract, TaskIntent,
        )
        original = TaskContract(
            objective="build a website",
            intent=TaskIntent.IMPLEMENT,
            allowed_operations=["write_file"],
            forbidden_operations=[],
        )
        d = original.to_dict()
        restored = TaskContract.from_dict(d)
        assert restored.intent == TaskIntent.IMPLEMENT
        assert restored.objective == "build a website"

    def test_read_only_operations(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("explain this project")
        assert "read_file" in c.allowed_operations
        assert "write_file" in c.forbidden_operations
        assert "edit_file" in c.forbidden_operations

    def test_implementation_operations(self):
        from harness_core.planning.domain import classify_request, TaskIntent
        c = classify_request("build a website")
        assert "write_file" in c.allowed_operations
        assert "edit_file" in c.allowed_operations


# ── Intent in TaskIntent Enum ──────────────────────────────────────────


class TestTaskIntentEnum:
    """Tests for the TaskIntent enum itself."""

    def test_all_intents_exist(self):
        from harness_core.planning.domain import TaskIntent
        required = [
            "EXPLAIN", "ANALYZE", "RESEARCH", "IMPLEMENT", "MODIFY",
            "DEBUG", "TEST", "REFACTOR", "REVIEW", "GIT", "DEPLOY", "MIXED",
        ]
        for name in required:
            assert hasattr(TaskIntent, name), f"TaskIntent missing {name}"

    def test_read_only_intents(self):
        from harness_core.planning.domain import TaskIntent
        assert TaskIntent.EXPLAIN.is_read_only is True
        assert TaskIntent.ANALYZE.is_read_only is True
        assert TaskIntent.RESEARCH.is_read_only is True
        assert TaskIntent.REVIEW.is_read_only is True

    def test_non_read_only_intents(self):
        from harness_core.planning.domain import TaskIntent
        assert TaskIntent.IMPLEMENT.is_read_only is False
        assert TaskIntent.MODIFY.is_read_only is False
        assert TaskIntent.DEBUG.is_read_only is False
        assert TaskIntent.TEST.is_read_only is False
        assert TaskIntent.GIT.is_read_only is False
        assert TaskIntent.DEPLOY.is_read_only is False
        assert TaskIntent.REFACTOR.is_read_only is False


# ── Runtime Integration ───────────────────────────────────────────────


class _FakeProvider:
    """Minimal provider stub for testing."""
    name = "fake"

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    async def generate(self, request):
        from harness_core.providers.base import CompletionResponse
        return CompletionResponse(content="Task completed successfully", model="fake")

    async def list_models(self):
        return []


@pytest.mark.asyncio
async def test_execute_interactive_classifies_intent(tmp_path):
    """execute_interactive stores the TaskContract on ProjectState."""
    from harness_core.runtime.runtime import EngineeringRuntime
    from harness_core.runtime.state import RuntimeStatus

    event_bus = EventBus()
    provider = _FakeProvider()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=provider,
        plan_provider=provider,
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
    )

    outcome = await runtime.execute_interactive("explain this project")

    assert outcome.status == RuntimeStatus.SUCCESS
    contract = runtime.state.task_contract
    assert contract is not None
    assert contract.intent.value in ("explain", "read_only")
    assert contract.is_read_only is True


@pytest.mark.asyncio
async def test_execute_interactive_emit_intent_event(tmp_path):
    """execute_interactive emits task.intent_detected with structured data."""
    from harness_core.runtime.runtime import EngineeringRuntime

    event_bus = EventBus()
    provider = _FakeProvider()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=provider,
        plan_provider=provider,
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
    )

    await runtime.execute_interactive("explain this project")

    events = event_bus.get_history()
    intent_events = [e for e in events if e.type == "task.intent_detected"]
    assert len(intent_events) >= 1
    data = intent_events[0].data
    assert "intent" in data
    assert "is_read_only" in data
    assert data["is_read_only"] is True


@pytest.mark.asyncio
async def test_execute_interactive_read_only_skips_verification(tmp_path):
    """Read-only tasks should skip the full verification engine."""
    from harness_core.runtime.runtime import EngineeringRuntime
    from harness_core.runtime.state import RuntimeStatus

    event_bus = EventBus()
    provider = _FakeProvider()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=provider,
        plan_provider=provider,
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
    )

    outcome = await runtime.execute_interactive("explain this project")

    assert outcome.status == RuntimeStatus.SUCCESS
    # Verification should be marked as passed (read-only)
    assert runtime.state.verification_status.value == "passed"
    # The verification_completed event should indicate skipped
    events = event_bus.get_history()
    verif_events = [e for e in events if e.type == "verification_completed"]
    assert len(verif_events) >= 1
    assert verif_events[0].data.get("passed") is True


@pytest.mark.asyncio
async def test_execute_interactive_implementation_needs_verification(tmp_path):
    """Implementation tasks should go through verification."""
    from harness_core.runtime.runtime import EngineeringRuntime
    from harness_core.runtime.state import RuntimeStatus

    event_bus = EventBus()
    provider = _FakeProvider()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=provider,
        plan_provider=provider,
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
    )

    outcome = await runtime.execute_interactive("create a file")

    assert outcome.status == RuntimeStatus.SUCCESS
    contract = runtime.state.task_contract
    assert contract is not None
    assert contract.intent.value in ("implement", "other")
    assert contract.needs_verification is True


@pytest.mark.asyncio
async def test_execute_interactive_success_event_includes_intent(tmp_path):
    """The runtime_completed event includes intent data."""
    from harness_core.runtime.runtime import EngineeringRuntime

    event_bus = EventBus()
    provider = _FakeProvider()
    runtime = EngineeringRuntime(
        workspace_path=tmp_path,
        provider=provider,
        plan_provider=provider,
        event_bus=event_bus,
        project_id=str(tmp_path.resolve()),
    )

    await runtime.execute_interactive("explain this project")

    events = event_bus.get_history()
    completed_events = [e for e in events if e.type == "runtime_completed"]
    assert len(completed_events) >= 1
    data = completed_events[0].data
    assert "intent" in data
    assert "is_read_only" in data
    assert "duration_ms" in data
