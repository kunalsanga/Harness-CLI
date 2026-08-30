"""Regression tests for the autonomous execution governor.

Covers the production-readiness requirements:
- Workspace intelligence (never "I don't have visibility")
- Structured planning / TODO validation
- Zero-tool-call guard
- Repeated failure detection → diagnosis mode
- Stagnation detection → honest stop
- Diagnosis budget
- Context compaction
- Truthful completion via verification
- Test integrity protection
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.agent.loop import (
    AgentLoop,
    HISTORY_CHAR_BUDGET,
    KEEP_RECENT_CALLS,
    MAX_DIAGNOSIS_ITERATIONS,
    MAX_NO_TOOL_NUDGES,
)
from harness_core.agent.types import (
    AgentConfig,
    Task,
    TaskStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider
from harness_core.tools.filesystem import ReadFileTool, WriteFileTool
from harness_core.tools.shell import RunCommandTool
from harness_core.verification.engine import VerificationCheck, VerificationReport, VerificationResult
from harness_core.verification.integrity import check_test_integrity, is_test_like_path


# ── Mock provider ────────────────────────────────────────────────────────


class ScriptedProvider(ModelProvider):
    """Returns scripted responses in order, then a default text answer."""

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self._responses = responses
        self._call_count = 0

    @property
    def name(self) -> str:
        return "scripted"

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = CompletionResponse(content="Done.", model="scripted", provider="scripted")
        self._call_count += 1
        return resp

    async def stream(self, request: CompletionRequest):
        yield CompletionResponse(content="", model="scripted", provider="scripted")

    async def list_models(self) -> list[Any]:
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def text(content: str) -> CompletionResponse:
    return CompletionResponse(content=content, model="scripted", provider="scripted")


def plan_response() -> CompletionResponse:
    return text("1. Inspect project\n2. Implement changes\n3. Run tests")


def tool_response(tool: str, args: dict[str, Any], call_id: str = "call-x") -> CompletionResponse:
    return CompletionResponse(
        content=None,
        model="scripted",
        provider="scripted",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool, "arguments": json.dumps(args)},
            }
        ],
    )


class EventCollector:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.events: list[Event] = []
        bus = self.bus

        async def _collect(event: Event) -> None:
            self.events.append(event)

        bus.on("*", _collect)

    def types(self) -> list[str]:
        return [e.type for e in self.events]

    def of_type(self, event_type: str) -> list[Event]:
        return [e for e in self.events if e.type == event_type]


def make_loop(
    provider: ModelProvider,
    workspace: Path,
    tools: list | None = None,
    collector: EventCollector | None = None,
    max_iterations: int = 15,
    verify: bool = True,
) -> AgentLoop:
    return AgentLoop(
        provider=provider,
        tools=tools if tools is not None else [ReadFileTool(), WriteFileTool(), RunCommandTool()],
        workspace_root=workspace,
        config=AgentConfig(max_iterations=max_iterations, verify_on_complete=verify),
        event_bus=collector.bus if collector else EventBus(),
    )


# ── Workspace intelligence ───────────────────────────────────────────────


class TestWorkspaceIntelligence:
    @pytest.mark.asyncio
    async def test_system_prompt_contains_workspace_snapshot(self, tmp_path: Path):
        (tmp_path / "script.js").write_text("console.log(1);", encoding="utf-8")
        (tmp_path / "index.html").write_text("<html></html>", encoding="utf-8")

        loop = make_loop(ScriptedProvider([]), tmp_path)
        loop._project_info = await loop.context_engine.discover_project()

        prompt = loop._system_prompt()
        assert "script.js" in prompt
        assert "index.html" in prompt
        assert str(tmp_path) in prompt

    @pytest.mark.asyncio
    async def test_system_prompt_forbids_no_visibility_claims(self, tmp_path: Path):
        loop = make_loop(ScriptedProvider([]), tmp_path)
        loop._project_info = await loop.context_engine.discover_project()
        prompt = loop._system_prompt()
        assert "NEVER claim you lack visibility" in prompt
        assert "NEVER ask the user for information" in prompt

    @pytest.mark.asyncio
    async def test_context_includes_project_metadata(self, tmp_path: Path):
        (tmp_path / "app.py").write_text("print('hi')", encoding="utf-8")
        loop = make_loop(ScriptedProvider([]), tmp_path)
        info = await loop.context_engine.discover_project()
        pieces = await loop.context_engine.assemble_context("explain this project", info)
        sources = [p.source for p in pieces]
        assert "project_metadata" in sources or "project_structure" in sources


# ── Structured planning ──────────────────────────────────────────────────


class TestPlanningValidation:
    def test_rejects_conversational_todos(self, tmp_path: Path):
        loop = make_loop(ScriptedProvider([]), tmp_path)
        garbage = [
            "I don't have visibility into your project",
            "Please provide more context",
            "Project type?",
            "What technology are you using?",
        ]
        assert loop._validate_plan_steps(garbage) == []

    def test_keeps_executable_todos(self, tmp_path: Path):
        loop = make_loop(ScriptedProvider([]), tmp_path)
        steps = [
            "Inspect existing calculator",
            "Improve calculator UI",
            "Run tests",
            "Verify changes",
        ]
        assert loop._validate_plan_steps(steps) == steps

    def test_default_plan_fallback(self, tmp_path: Path):
        loop = make_loop(ScriptedProvider([]), tmp_path)
        loop._project_info = {"has_tests": True}
        steps = loop._default_plan_steps()
        assert steps, "default plan must never be empty"
        assert all("?" not in s for s in steps)
        assert any("test" in s.lower() for s in steps)

    @pytest.mark.asyncio
    async def test_garbage_plan_replaced_with_default(self, tmp_path: Path):
        (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            text("I don't have visibility into your project. Please provide more context."),
            text("Done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, tools=[], collector=collector)

        task = await loop.run("Enhance this project")

        created = collector.of_type("plan.created")
        assert created, "plan.created must be emitted even for fallback plans"
        steps = created[0].data["steps"]
        assert steps, "plan must not be empty"
        for step in steps:
            low = step.lower()
            assert "visibility" not in low
            assert "please provide" not in low
            assert "?" not in step


# ── Zero-tool-call guard ─────────────────────────────────────────────────


class TestZeroToolCallGuard:
    @pytest.mark.asyncio
    async def test_nudges_model_that_skips_tools(self, tmp_path: Path):
        (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            text("I don't have visibility into your specific project files."),
            text("Please share your project details."),
            text("Fine, here is an answer anyway."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)

        task = await loop.run("Enhance this project")

        nudges = collector.of_type("execution.nudge")
        assert len(nudges) == MAX_NO_TOOL_NUDGES
        # Bounded: after the nudges the text answer is accepted, never an infinite loop
        assert task.status == TaskStatus.COMPLETED
        assert task.iterations == MAX_NO_TOOL_NUDGES + 1

    @pytest.mark.asyncio
    async def test_no_nudge_when_no_tools(self, tmp_path: Path):
        (tmp_path / "app.py").write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([plan_response(), text("An explanation.")])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, tools=[], collector=collector)

        task = await loop.run("Explain this project")

        assert collector.of_type("execution.nudge") == []
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_nudge_when_model_uses_tools(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            tool_response("read_file", {"path": str(target)}, "c1"),
            text("The project defines x = 1."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)

        task = await loop.run("Explain this project")

        assert collector.of_type("execution.nudge") == []
        assert len(task.tool_calls) == 1


# ── Repeated failure → diagnosis ─────────────────────────────────────────


FAIL_CMD = "python -c \"import sys; sys.exit(1)\""


class TestRepeatedFailureGovernor:
    @pytest.mark.asyncio
    async def test_diagnosis_triggered_after_two_failures(self, tmp_path: Path):
        provider = ScriptedProvider([
            plan_response(),
            tool_response("run_command", {"command": FAIL_CMD}, "c1"),
            tool_response("run_command", {"command": FAIL_CMD}, "c2"),
            text("Diagnosed and done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)

        await loop.run("Run the tests")

        triggered = collector.of_type("diagnosis.triggered")
        assert len(triggered) == 1
        assert triggered[0].data["attempts"] == 2
        assert triggered[0].data["command"] == FAIL_CMD
        # Diagnosis guidance reached the model
        assert any("diagnosis" in c.lower() or "diagnose" in c.lower()
                   for c in loop._corrections)

    @pytest.mark.asyncio
    async def test_third_identical_attempt_blocked(self, tmp_path: Path):
        provider = ScriptedProvider([
            plan_response(),
            tool_response("run_command", {"command": FAIL_CMD}, "c1"),
            tool_response("run_command", {"command": FAIL_CMD}, "c2"),
            tool_response("run_command", {"command": FAIL_CMD}, "c3"),
            text("Done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)

        task = await loop.run("Run the tests")

        # The third identical attempt must be blocked without executing
        results = [tc.result for tc in task.tool_calls if tc.result]
        blocked = [r for r in results if r.error and "BLOCKED" in r.error]
        assert len(blocked) == 1
        # Only two actual executions of the command (both failed)
        executed_failures = [
            tc for tc in task.tool_calls
            if tc.result and tc.result.exit_code == 1
        ]
        assert len(executed_failures) == 2

    @pytest.mark.asyncio
    async def test_edit_resets_failure_counts(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            tool_response("run_command", {"command": FAIL_CMD}, "c1"),
            tool_response("run_command", {"command": FAIL_CMD}, "c2"),
            tool_response("write_file", {"path": str(target), "content": "x = 2"}, "c3"),
            tool_response("run_command", {"command": FAIL_CMD}, "c4"),
            text("Done."),
        ])
        loop = make_loop(provider, tmp_path)

        task = await loop.run("Fix and rerun")

        # After the edit, the retry is allowed (not BLOCKED)
        last = task.tool_calls[-1]
        assert last.result is not None
        assert last.result.error is None or "BLOCKED" not in (last.result.error or "")


# ── Stagnation detection ─────────────────────────────────────────────────


class TestStagnationDetection:
    @pytest.mark.asyncio
    async def test_repeated_same_action_stops_task(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        same_read = tool_response("read_file", {"path": str(target)}, "c1")
        provider = ScriptedProvider([plan_response()] + [same_read] * 6)
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)

        task = await loop.run("Enhance this project")

        assert task.status == TaskStatus.FAILED
        assert "no meaningful progress" in task.error.lower()
        assert "progress.stalled" in collector.types()
        failed_events = collector.of_type("task.failed")
        assert any(e.data.get("reason") == "stagnation" for e in failed_events)

    @pytest.mark.asyncio
    async def test_new_actions_count_as_progress(self, tmp_path: Path):
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("a = 1", encoding="utf-8")
        b.write_text("b = 2", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            tool_response("read_file", {"path": str(a)}, "c1"),
            tool_response("read_file", {"path": str(b)}, "c2"),
            text("Explained."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)

        task = await loop.run("Explain this project")

        assert task.status == TaskStatus.COMPLETED
        assert collector.of_type("progress.stalled") == []


# ── Diagnosis budget ─────────────────────────────────────────────────────


class TestDiagnosisBudget:
    @pytest.mark.asyncio
    async def test_diagnosis_exhaustion_stops_task(self, tmp_path: Path):
        responses = [plan_response()]
        responses.append(tool_response("run_command", {"command": FAIL_CMD}, "c1"))
        responses.append(tool_response("run_command", {"command": FAIL_CMD}, "c2"))
        # Distinct failing reads keep progress "new" but never resolve diagnosis
        for i in range(MAX_DIAGNOSIS_ITERATIONS + 2):
            responses.append(
                tool_response("read_file", {"path": str(tmp_path / f"missing_{i}.py")}, f"r{i}")
            )
        provider = ScriptedProvider(responses)
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector,
                         max_iterations=MAX_DIAGNOSIS_ITERATIONS + 8)

        task = await loop.run("Fix the failing tests")

        assert task.status == TaskStatus.FAILED
        assert "unable to resolve" in task.error.lower()
        failed_events = collector.of_type("task.failed")
        assert any(e.data.get("reason") == "diagnosis_exhausted" for e in failed_events)


# ── Context compaction ───────────────────────────────────────────────────


class TestContextCompaction:
    def test_old_tool_history_summarized(self, tmp_path: Path):
        loop = make_loop(ScriptedProvider([]), tmp_path)
        task = Task(goal="Big task")
        big = "x" * (HISTORY_CHAR_BUDGET // 8)
        for i in range(KEEP_RECENT_CALLS + 6):
            call = ToolCall(id=f"t{i}", tool_name="read_file", arguments={"path": f"f{i}.py"})
            call.result = ToolResult(status=ToolResultStatus.SUCCESS, output=big)
            task.tool_calls.append(call)

        messages = loop._build_messages(task)

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert len(tool_messages) == KEEP_RECENT_CALLS
        summaries = [m for m in messages if m["role"] == "system" and "TASK STATE" in m["content"]]
        assert len(summaries) == 1
        assert "Big task" in summaries[0]["content"]

    def test_small_history_kept_verbatim(self, tmp_path: Path):
        loop = make_loop(ScriptedProvider([]), tmp_path)
        task = Task(goal="Small task")
        for i in range(4):
            call = ToolCall(id=f"t{i}", tool_name="read_file", arguments={"path": f"f{i}.py"})
            call.result = ToolResult(status=ToolResultStatus.SUCCESS, output="small")
            task.tool_calls.append(call)

        messages = loop._build_messages(task)
        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert len(tool_messages) == 4


# ── Truthful completion via verification ─────────────────────────────────


def _failed_report() -> VerificationReport:
    report = VerificationReport()
    report.add_result(VerificationResult(
        check_name="pytest", passed=False, output="", error="1 failed",
    ))
    return report


def _passed_report() -> VerificationReport:
    report = VerificationReport()
    report.add_result(VerificationResult(check_name="pytest", passed=True, output="ok"))
    return report


class TestVerificationOnComplete:
    async def _run_write_task(self, tmp_path: Path, report: VerificationReport | None,
                              checks: list[VerificationCheck] | None,
                              collector: EventCollector) -> Task:
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            tool_response("write_file", {"path": str(target), "content": "x = 2"}, "c1"),
            text("I fixed it."),
        ])
        loop = make_loop(provider, tmp_path, collector=collector)
        loop.verification_engine.detect_ecosystem = AsyncMock(
            return_value=checks if checks is not None else []
        )
        if checks:
            loop.verification_engine.run_checks = AsyncMock(return_value=report)
        return await loop.run("Change x to 2")

    @pytest.mark.asyncio
    async def test_failed_verification_blocks_completion(self, tmp_path: Path):
        collector = EventCollector()
        task = await self._run_write_task(
            tmp_path, _failed_report(),
            [VerificationCheck(name="pytest", command="python -m pytest -q")],
            collector,
        )
        assert task.status == TaskStatus.FAILED
        assert task.verification_passed is False
        assert "verification failed" in task.error.lower()
        failed_events = collector.of_type("task.failed")
        assert any(e.data.get("reason") == "verification_failed" for e in failed_events)

    @pytest.mark.asyncio
    async def test_passed_verification_allows_completion(self, tmp_path: Path):
        collector = EventCollector()
        task = await self._run_write_task(
            tmp_path, _passed_report(),
            [VerificationCheck(name="pytest", command="python -m pytest -q")],
            collector,
        )
        assert task.status == TaskStatus.COMPLETED
        assert task.verification_passed is True
        assert "verification.completed" in collector.types()

    @pytest.mark.asyncio
    async def test_no_checks_verifies_file_existence(self, tmp_path: Path):
        collector = EventCollector()
        task = await self._run_write_task(tmp_path, None, [], collector)
        assert task.status == TaskStatus.COMPLETED
        assert task.verification_passed is True

    @pytest.mark.asyncio
    async def test_read_only_task_skips_verification(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            tool_response("read_file", {"path": str(target)}, "c1"),
            text("Explained."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)

        task = await loop.run("Explain this project")

        assert task.status == TaskStatus.COMPLETED
        assert task.verification_passed is None  # never verified — nothing changed
        assert "verification.started" not in collector.types()


# ── Test integrity ───────────────────────────────────────────────────────


class TestTestIntegrity:
    def test_test_like_path_detection(self):
        assert is_test_like_path("tests/test_calculator.py")
        assert is_test_like_path("test.js")
        assert is_test_like_path("src/app.test.ts")
        assert not is_test_like_path("src/calculator.py")
        assert not is_test_like_path("styles.css")

    def test_weakened_test_flagged(self, tmp_path: Path):
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, capture_output=True)
        subprocess.run(["git", "config", "user.name", "T"], cwd=repo, capture_output=True)
        test_file = repo / "test_calc.py"
        test_file.write_text(
            "def test_a():\n    assert 1 == 1\n    assert 2 == 2\n", encoding="utf-8"
        )
        subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True)

        # Agent weakens the test: removes an assertion
        test_file.write_text("def test_a():\n    assert 1 == 1\n", encoding="utf-8")

        report = check_test_integrity(repo, ["test_calc.py"])
        assert report.suspicious
        assert "removed" in report.warning.lower()

    def test_unmodified_tests_not_flagged(self, tmp_path: Path):
        report = check_test_integrity(tmp_path, ["src/app.py", "styles.css"])
        assert not report.suspicious
        assert report.test_files_modified == []

    @pytest.mark.asyncio
    async def test_loop_emits_integrity_warning_for_test_edits(self, tmp_path: Path):
        test_file = tmp_path / "test_app.py"
        test_file.write_text("def test_a():\n    assert True\n", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            tool_response(
                "write_file",
                {"path": str(test_file), "content": "def test_a():\n    pass\n"},
                "c1",
            ),
            text("Tests updated."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)
        loop.verification_engine.detect_ecosystem = AsyncMock(return_value=[])

        await loop.run("Fix the tests")

        warnings = collector.of_type("test_integrity.warning")
        assert warnings, "modifying a test file must raise an integrity warning"


# ── Phase tracking ───────────────────────────────────────────────────────


class TestPhaseTracking:
    @pytest.mark.asyncio
    async def test_diagnosis_phase_emitted(self, tmp_path: Path):
        provider = ScriptedProvider([
            plan_response(),
            tool_response("run_command", {"command": FAIL_CMD}, "c1"),
            tool_response("run_command", {"command": FAIL_CMD}, "c2"),
            text("Done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)

        await loop.run("Run tests")

        phases = [e.data["phase"] for e in collector.of_type("task.phase")]
        assert "diagnosing" in phases

    @pytest.mark.asyncio
    async def test_verifying_phase_emitted(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            tool_response("write_file", {"path": str(target), "content": "x = 2"}, "c1"),
            text("Done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)
        loop.verification_engine.detect_ecosystem = AsyncMock(return_value=[])

        await loop.run("Change x")

        phases = [e.data["phase"] for e in collector.of_type("task.phase")]
        assert "verifying" in phases

    @pytest.mark.asyncio
    async def test_completed_event_carries_truthful_data(self, tmp_path: Path):
        target = tmp_path / "app.py"
        target.write_text("x = 1", encoding="utf-8")
        provider = ScriptedProvider([
            plan_response(),
            tool_response("write_file", {"path": str(target), "content": "x = 2"}, "c1"),
            text("Done."),
        ])
        collector = EventCollector()
        loop = make_loop(provider, tmp_path, collector=collector)
        loop.verification_engine.detect_ecosystem = AsyncMock(return_value=[])

        await loop.run("Change x")

        completed = collector.of_type("task.completed")
        assert completed
        data = completed[0].data
        assert data["verification_passed"] is True
        assert str(target) in data["files_changed"]


# ── Model rotation & free-safety-net routing ─────────────────────────────


from harness_core.providers.base import ModelInfo, ModelProvider
from harness_core.routing.health import ModelHealthStatus, ModelHealthTracker
from harness_core.routing.router import ModelRouter, RouterConfig


class _NoToolProvider(ModelProvider):
    def __init__(self) -> None:
        self._name = "openrouter"

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request):
        return CompletionResponse(content="ok", model="m", provider=self._name)

    async def stream(self, request):
        yield CompletionResponse(content="", model="m", provider=self._name)

    async def list_models(self):
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


class TestModelRotation:
    def test_no_tool_usage_marks_unhealthy_until_cooldown(self):
        tracker = ModelHealthTracker()
        tracker.record_success("m1", latency_ms=10)
        assert tracker.get_state("m1").is_healthy

        tracker.record_no_tool_usage("m1", cooldown_seconds=60)
        state = tracker.get_state("m1")
        assert state.health_status == ModelHealthStatus.NO_TOOL_USE
        assert not state.is_healthy  # excluded while cooling down

    def test_no_tool_usage_recovers_after_cooldown(self):
        tracker = ModelHealthTracker()
        tracker.record_no_tool_usage("m1", cooldown_seconds=-1)  # already expired
        assert tracker.get_state("m1").is_healthy

    def test_success_clears_no_tool_status(self):
        tracker = ModelHealthTracker()
        tracker.record_no_tool_usage("m1", cooldown_seconds=60)
        assert not tracker.get_state("m1").is_healthy
        tracker.record_success("m1", latency_ms=5)
        assert tracker.get_state("m1").is_healthy


def _model(mid: str, is_free: bool, ctx: int) -> ModelInfo:
    return ModelInfo(
        id=mid, name=mid, provider="openrouter",
        supports_tools=True, is_free=is_free, context_window=ctx,
    )


def _router_with_models(models: list[ModelInfo], mode: str) -> ModelRouter:
    import time as _time
    provider = _NoToolProvider()
    router = ModelRouter(providers=[provider], config=RouterConfig(routing_mode=mode))
    router.providers = {"openrouter": provider}
    router._model_cache = models
    router._last_refresh = _time.time() + 300  # cache valid
    return router


class TestFreeSafetyNet:
    @pytest.mark.asyncio
    async def test_auto_chain_includes_free_models(self):
        # 5 high-context paid models outrank the free ones; safety net must
        # still place free models in the chain (accounts without credits 402).
        models = (
            [_model(f"paid-{i}", is_free=False, ctx=1_000_000 - i) for i in range(5)]
            + [_model("free-a", is_free=True, ctx=32000),
               _model("free-b", is_free=True, ctx=32000)]
        )
        router = _router_with_models(models, "auto")
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Fix the bug"}],
            tools=[{"type": "function", "function": {"name": "t"}}],
        )
        chain = await router.select_models(req)
        ids = [mid for mid, _ in chain]
        assert any(mid.startswith("free-") for mid in ids), (
            f"auto chain must include a free model safety net, got {ids}"
        )

    @pytest.mark.asyncio
    async def test_free_mode_chain_only_free(self):
        models = [
            _model("paid-a", is_free=False, ctx=1_000_000),
            _model("free-a", is_free=True, ctx=32000),
        ]
        router = _router_with_models(models, "free")
        req = CompletionRequest(messages=[{"role": "user", "content": "Fix"}])
        chain = await router.select_models(req)
        ids = [mid for mid, _ in chain]
        assert ids == ["free-a"]
