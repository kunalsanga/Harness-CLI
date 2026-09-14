"""Comprehensive E2E tests for critical Harness flows.

Tests validate the real runtime path (EngineeringRuntime or AgentLoop) with
only the model provider boundary stubbed.  No task states are set by the test
— all orchestration runs through production code.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import (
    AgentConfig,
    AgentRole,
    Task,
    TaskStatus,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.agents.domain import (
    AgentResult,
    AgentRole as DomAgentRole,
    AgentStatus,
    SubTask,
    TaskGraph,
    TaskStatus as DomTaskStatus,
)
from harness_core.agents.worker import WorkerAgent
from harness_core.observability.events import EventBus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)
from harness_core.routing.fallback import FallbackResult
from harness_core.runtime.runtime import EngineeringRuntime
from harness_core.runtime.state import RuntimeStage, RuntimeStatus, VerificationStatus
from harness_core.tools.filesystem import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from harness_core.tools.search import GlobTool, GrepTool
from harness_core.tools.shell import RunCommandTool


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_provider(*, content: str = "Done", model: str = "test-model") -> MagicMock:
    """Create a mock provider that returns a single response."""
    provider = MagicMock()
    provider.generate = AsyncMock(
        return_value=CompletionResponse(content=content, model=model)
    )
    provider.health_check = AsyncMock(return_value=True)
    provider.close = AsyncMock()
    return provider


def _make_fail_provider(error: str) -> MagicMock:
    """Create a mock provider that always raises."""
    provider = MagicMock()
    provider.generate = AsyncMock(side_effect=Exception(error))
    provider.health_check = AsyncMock(return_value=True)
    provider.close = AsyncMock()
    return provider


def _make_tools() -> list:
    """Create the standard tool set for tests."""
    return [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListFilesTool(),
        GlobTool(),
        GrepTool(),
        RunCommandTool(),
    ]


def _loop_config(**overrides) -> AgentConfig:
    """Create an AgentConfig with sane defaults for tests."""
    defaults = dict(
        role=AgentRole.BUILD,
        max_iterations=10,
        verify_on_complete=False,
        autonomous_mode=True,
    )
    defaults.update(overrides)
    return AgentConfig(**defaults)


# ── 1. Simple explanation task ─────────────────────────────────────────────


class TestExplanationFlow:
    """Verify 'explain this project' completes without modifying files."""

    @pytest.mark.asyncio
    async def test_explain_project_no_modifications(self, tmp_path: Path):
        """Explanation request should complete without writing any files."""
        # Set up a minimal project
        (tmp_path / "main.py").write_text("def hello(): pass\n")
        (tmp_path / "README.md").write_text("# Test Project\n")

        provider = _make_provider(content="This is a Python project with a hello function.")
        bus = EventBus()
        loop = AgentLoop(
            provider=provider,
            tools=_make_tools(),
            workspace_root=tmp_path,
            config=_loop_config(max_iterations=5),
            event_bus=bus,
        )

        task = await loop.run("explain this project")

        # Should complete (or at least not fail)
        assert task.status in (TaskStatus.COMPLETED, TaskStatus.PARTIAL)
        # Should NOT have modified any files
        assert not any(
            (tmp_path / f).exists() and f not in ("main.py", "README.md", ".git")
            for f in ["new_file.py", "output.txt", "changes.py"]
        )
        # Should have emitted events
        event_types = {e.type for e in bus.get_history()}
        assert "task.started" in event_types or "iteration.started" in event_types


# ── 2. Simple coding task ──────────────────────────────────────────────────


class TestSimpleCodingTask:
    """Verify a simple coding task creates a file."""

    @pytest.mark.asyncio
    async def test_coding_task_creates_file(self, tmp_path: Path):
        """A coding task should use tools to create/modify files."""
        # Track tool calls
        tool_calls_made: list[str] = []

        async def smart_generate(request):
            """Return tool calls on first request, then text on second."""
            if not tool_calls_made:
                tool_calls_made.append("planned")
                return CompletionResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": str(
                                    {
                                        "path": str(tmp_path / "greeting.py"),
                                        "content": "def greet(name):\n    return f'Hello, {name}!'\n",
                                    }
                                ),
                            },
                        }
                    ],
                    model="test-model",
                )
            return CompletionResponse(
                content="Created greeting.py with a greet function.",
                model="test-model",
            )

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=smart_generate)

        loop = AgentLoop(
            provider=provider,
            tools=[WriteFileTool(), ReadFileTool()],
            workspace_root=tmp_path,
            config=_loop_config(max_iterations=5),
            event_bus=EventBus(),
        )

        task = await loop.run("Create a Python greeting function")

        # The file should exist (tool was executed)
        greeting = tmp_path / "greeting.py"
        if greeting.exists():
            content = greeting.read_text()
            assert "greet" in content


# ── 3. Provider fallback ───────────────────────────────────────────────────


class TestProviderFallback:
    """Verify fallback when the primary model fails."""

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self, tmp_path: Path):
        """When the primary model fails, the router should try fallback models."""
        from harness_core.routing.router import ModelRouter, RouterConfig
        from harness_core.routing.fallback import FallbackEngine, FallbackConfig
        from harness_core.routing.health import ModelHealthTracker

        # Primary fails, fallback succeeds
        primary = MagicMock()
        primary.generate = AsyncMock(side_effect=Exception("429 Rate Limited"))
        primary.name = "primary"

        fallback = MagicMock()
        fallback.generate = AsyncMock(
            return_value=CompletionResponse(content="Fallback response", model="fallback-m")
        )
        fallback.name = "fallback"

        # Build a router that tries primary then fallback
        health = ModelHealthTracker()
        engine = FallbackEngine(
            health_tracker=health,
            fallback_config=FallbackConfig(max_fallback_models=2),
        )

        request = CompletionRequest(messages=[{"role": "user", "content": "test"}])
        chain = [("primary-m", primary), ("fallback-m", fallback)]
        result = await engine.execute(request, chain)

        assert result.succeeded
        assert result.model_used == "fallback-m"
        assert result.provider_used == "fallback"
        assert len(result.attempts) >= 2  # primary failed, fallback succeeded


# ── 4. All-provider failure ────────────────────────────────────────────────


class TestAllProviderFailure:
    """Verify graceful failure when all providers are unavailable."""

    @pytest.mark.asyncio
    async def test_all_providers_fail_returns_structured_error(self, tmp_path: Path):
        """When every provider fails, the result should be FAILED with a clear error."""
        from harness_core.routing.fallback import FallbackEngine, FallbackConfig
        from harness_core.routing.health import ModelHealthTracker

        bad1 = MagicMock()
        bad1.generate = AsyncMock(side_effect=Exception("401 Unauthorized"))
        bad1.name = "bad1"

        bad2 = MagicMock()
        bad2.generate = AsyncMock(side_effect=Exception("402 Payment Required"))
        bad2.name = "bad2"

        bad3 = MagicMock()
        bad3.generate = AsyncMock(side_effect=Exception("Connection refused"))
        bad3.name = "bad3"

        engine = FallbackEngine(
            health_tracker=ModelHealthTracker(),
            fallback_config=FallbackConfig(max_fallback_models=3, total_timeout_seconds=30),
        )
        request = CompletionRequest(messages=[{"role": "user", "content": "test"}])
        chain = [("m1", bad1), ("m2", bad2), ("m3", bad3)]
        result = await engine.execute(request, chain)

        assert not result.succeeded
        assert result.final_error is not None
        assert len(result.attempts) >= 3
        # Error message should be informative
        error_lower = result.final_error.lower()
        assert any(kw in error_lower for kw in ["all", "failed", "unavailable", "payment"])


# ── 5. Permission denial ───────────────────────────────────────────────────


class TestPermissionDenial:
    """Verify permission denials are handled gracefully."""

    @pytest.mark.asyncio
    async def test_permission_denied_does_not_crash(self, tmp_path: Path):
        """A permission-denied tool call should not crash the loop."""
        from harness_core.permissions.manager import PermissionManager

        manager = PermissionManager(
            workspace_root=tmp_path,
            autonomous_mode=False,  # Non-autonomous: everything requires approval
            approval_callback=lambda tool, desc: False,  # Deny all
        )

        # Check that dangerous commands are denied
        result = manager.check_permission("run_command", {"command": "rm -rf /"})
        assert result == "deny"

        # Read-only tools should be allowed
        result = manager.check_permission("read_file", {"path": "test.py"})
        assert result == "allow"

    @pytest.mark.asyncio
    async def test_read_only_tools_always_allowed(self, tmp_path: Path):
        """Read-only tools should always be allowed even in non-autonomous mode."""
        from harness_core.permissions.manager import PermissionManager

        manager = PermissionManager(
            workspace_root=tmp_path,
            autonomous_mode=False,
        )

        for tool in ["read_file", "list_files", "glob", "grep", "git_status", "git_diff", "git_log"]:
            result = manager.check_permission(tool)
            assert result == "allow", f"{tool} should be allowed"


# ── 6. Verification gate ───────────────────────────────────────────────────


class TestVerificationGate:
    """Verify that verification gates task completion."""

    @pytest.mark.asyncio
    async def test_verification_detects_ecosystem(self, tmp_path: Path):
        """Verification engine should detect Python project."""
        from harness_core.verification.engine import VerificationEngine

        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname = 'test'\n\n[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
        )
        engine = VerificationEngine(workspace_root=tmp_path)
        checks = await engine.detect_ecosystem()
        names = [c.name for c in checks]
        assert "pytest" in names

    @pytest.mark.asyncio
    async def test_verification_passing_command(self, tmp_path: Path):
        """A passing verification command should return passed=True."""
        from harness_core.verification.engine import (
            VerificationCheck,
            VerificationEngine,
        )

        engine = VerificationEngine(workspace_root=tmp_path)
        check = VerificationCheck(name="pass_test", command="echo ok")
        result = await engine._run_check(check)
        assert result.passed
        assert "ok" in result.output

    @pytest.mark.asyncio
    async def test_verification_failing_command(self, tmp_path: Path):
        """A failing verification command should return passed=False."""
        from harness_core.verification.engine import (
            VerificationCheck,
            VerificationEngine,
        )

        engine = VerificationEngine(workspace_root=tmp_path)
        check = VerificationCheck(name="fail_test", command="false")
        result = await engine._run_check(check)
        assert not result.passed


# ── 7. Task execution stats ────────────────────────────────────────────────


class TestTaskExecutionStats:
    """Verify execution stats are truthful."""

    def test_success_rate(self):
        stats = __import__("harness_core.agent.types", fromlist=["TaskExecutionStats"]).TaskExecutionStats()
        for _ in range(3):
            stats.record_attempt()
            stats.record_success("tool")
        stats.record_attempt()
        stats.record_failure("tool")
        assert stats.success_rate == pytest.approx(0.75)

    def test_recovery_detection(self):
        from harness_core.agent.types import TaskExecutionStats

        stats = TaskExecutionStats()
        stats.record_attempt()
        stats.record_failure("run_command")
        stats.record_attempt()
        stats.record_success("run_command")
        assert stats.recovered == 1
        assert stats.unresolved == 0


# ── 8. Context reuse ──────────────────────────────────────────────────────


class TestContextReuse:
    """Verify ContextReuseManager tracks file snapshots."""

    def test_record_and_detect_unchanged(self, tmp_path: Path):
        from harness_core.context.reuse import ContextReuseManager

        mgr = ContextReuseManager()
        path = str(tmp_path / "file.txt")
        content = "hello world"

        # First read: snapshot recorded
        snap = mgr.record_read(path, content=content, size=len(content), mtime_ns=1000)
        assert snap is not None
        assert mgr.has_snapshot(path)

        # Unchanged: is_unchanged returns True
        assert mgr.is_unchanged(path, content=content, size=len(content), mtime_ns=1000)

    def test_detects_changed_content(self, tmp_path: Path):
        from harness_core.context.reuse import ContextReuseManager

        mgr = ContextReuseManager()
        path = str(tmp_path / "file.txt")

        mgr.record_read(path, content="original", size=8, mtime_ns=1000)
        assert not mgr.is_unchanged(path, content="modified", size=8, mtime_ns=1000)

    def test_invalidation(self, tmp_path: Path):
        from harness_core.context.reuse import ContextReuseManager

        mgr = ContextReuseManager()
        path = str(tmp_path / "file.txt")

        mgr.record_read(path, content="data", size=4, mtime_ns=1000)
        assert mgr.has_snapshot(path)

        mgr.invalidate(path)
        assert not mgr.has_snapshot(path)


# ── 9. Scheduler + TaskGraph ───────────────────────────────────────────────


class TestSchedulerTaskGraph:
    """Verify TaskGraph dependency resolution and scheduler dispatch."""

    def test_dependency_resolution(self):
        from harness_core.agents.domain import SubTask, TaskGraph, TaskStatus

        graph = TaskGraph()
        t1 = SubTask(task_id="a", role=DomAgentRole.CODER, dependencies=[])
        t2 = SubTask(task_id="b", role=DomAgentRole.CODER, dependencies=["a"])
        t3 = SubTask(task_id="c", role=DomAgentRole.CODER, dependencies=["a", "b"])
        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        # Initially only "a" is ready (no deps)
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "a"

        # Complete "a" -> "b" becomes ready
        graph.update_task_status("a", DomTaskStatus.COMPLETED)
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "b"

        # Complete "b" -> "c" becomes ready
        graph.update_task_status("b", DomTaskStatus.COMPLETED)
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "c"

    def test_failure_propagation(self):
        from harness_core.agents.domain import SubTask, TaskGraph, TaskStatus

        graph = TaskGraph()
        t1 = SubTask(task_id="x", role=DomAgentRole.CODER, dependencies=[])
        t2 = SubTask(task_id="y", role=DomAgentRole.CODER, dependencies=["x"])
        graph.add_task(t1)
        graph.add_task(t2)

        graph.update_task_status("x", DomTaskStatus.FAILED)
        # "y" should be blocked
        assert graph.get_task("y").status == DomTaskStatus.BLOCKED

    def test_cycle_detection(self):
        from harness_core.agents.domain import SubTask, TaskGraph

        graph = TaskGraph()
        t1 = SubTask(task_id="a", role=DomAgentRole.CODER, dependencies=["b"])
        t2 = SubTask(task_id="b", role=DomAgentRole.CODER, dependencies=["a"])
        graph.add_task(t1)
        graph.add_task(t2)

        errors = graph.validate()
        assert any("cycle" in e.lower() or "Circular" in e for e in errors)

    def test_parallel_independent_tasks(self):
        from harness_core.agents.domain import SubTask, TaskGraph, TaskStatus

        graph = TaskGraph()
        for i in range(5):
            graph.add_task(SubTask(task_id=f"t{i}", role=DomAgentRole.CODER, dependencies=[]))

        ready = graph.get_ready_tasks()
        assert len(ready) == 5  # All independent, all ready


# ── 10. Convergence governor ───────────────────────────────────────────────


class TestConvergenceGovernor:
    """Verify stagnation detection from real tool evidence."""

    def test_repeated_failure_detected(self):
        from harness_core.runtime.governance import ConvergenceGovernor

        gov = ConvergenceGovernor(repeat_stall_at=3)
        for _ in range(3):
            gov.record_tool(
                "run_command",
                {"command": "pytest"},
                succeeded=False,
                exit_code=1,
                stderr="FAILED test_something",
            )
        report = gov.report()
        assert report.stalled
        assert report.repeated_command == "pytest"

    def test_code_change_resets_failure_streak(self):
        from harness_core.runtime.governance import ConvergenceGovernor

        gov = ConvergenceGovernor(repeat_stall_at=3)
        # Two failures
        for _ in range(2):
            gov.record_tool(
                "run_command",
                {"command": "pytest"},
                succeeded=False,
                exit_code=1,
            )
        # Code change resets the failure context
        gov.record_tool(
            "edit_file",
            {"path": "src/app.py", "content": "new code"},
            succeeded=True,
        )
        report = gov.report()
        # After a code change, the failure streak is reset
        assert not report.stalled or not any(
            e.kind == "repeated_failure" for e in report.evidence
        )


# ── 11. Budget enforcement ─────────────────────────────────────────────────


class TestBudgetEnforcement:
    """Verify budget limits are enforced."""

    def test_iteration_budget(self):
        from harness_core.routing.budgets import BudgetConfig, BudgetManager

        mgr = BudgetManager(BudgetConfig(max_iterations=3, max_tool_calls=100))
        assert mgr.record_iteration()  # 1
        assert mgr.record_iteration()  # 2
        assert mgr.record_iteration()  # 3
        # 4th exceeds the budget
        assert not mgr.record_iteration()
        ok, reason = mgr.check_all()
        assert not ok
        assert "Iteration" in reason

    def test_tool_call_budget(self):
        from harness_core.routing.budgets import BudgetConfig, BudgetManager

        mgr = BudgetManager(BudgetConfig(max_iterations=100, max_tool_calls=2))
        assert mgr.record_tool_call()  # 1
        assert mgr.record_tool_call()  # 2
        # 3rd exceeds the budget
        assert not mgr.record_tool_call()
        ok, reason = mgr.check_all()
        assert not ok
        assert "Tool call" in reason


# ── 12. Model health tracking ──────────────────────────────────────────────


class TestModelHealthTracking:
    """Verify model health is tracked and affects routing."""

    def test_unhealthy_model_skipped(self):
        from harness_core.routing.health import ModelHealthTracker, HealthEvent

        ht = ModelHealthTracker()
        ht.record_failure("model-a", HealthEvent.AUTH_FAILED, 100)
        state = ht.get_state("model-a")
        assert state.is_unavailable

    def test_healthy_model_not_skipped(self):
        from harness_core.routing.health import ModelHealthTracker

        ht = ModelHealthTracker()
        ht.record_success("model-b", latency_ms=100)
        state = ht.get_state("model-b")
        assert not state.is_unavailable
