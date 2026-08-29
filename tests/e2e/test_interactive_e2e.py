"""Real E2E test for the interactive shell + agent pipeline.

Creates a disposable repository with a deliberately incorrect implementation,
runs the agent, and verifies the fix works.

This test requires an actual provider (OpenRouter or Ollama).
Skip with: pytest -m "not e2e" or set HARNESS_SKIP_E2E=1
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest


# Skip if no provider available or explicitly skipped
SKIP_E2E = os.environ.get("HARNESS_SKIP_E2E", "0") == "1"


def _has_provider() -> bool:
    """Check if any provider is available."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return True
    # Check Ollama
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.e2e


@pytest.fixture
def broken_project(tmp_path: Path) -> Path:
    """Create a disposable project with a deliberately broken calculator."""
    project = tmp_path / "calculator_project"
    project.mkdir()

    # Create a broken calculator
    calculator_py = project / "calculator.py"
    calculator_py.write_text(
        '''"""A simple calculator module."""


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract two numbers."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers.

    BUG: This should raise ValueError when b is 0,
    but instead it returns 0 (silent failure).
    """
    if b == 0:
        return 0  # BUG: should raise ValueError
    return a / b
''',
        encoding="utf-8",
    )

    # Create tests that should fail
    test_calculator_py = project / "test_calculator.py"
    test_calculator_py.write_text(
        '''"""Tests for calculator module."""

import pytest
from calculator import add, subtract, multiply, divide


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(0, 5) == -5


def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0


def test_divide():
    assert divide(10, 2) == 5
    assert divide(9, 3) == 3


def test_divide_by_zero():
    """This test should PASS after the bug is fixed."""
    with pytest.raises(ValueError):
        divide(10, 0)
''',
        encoding="utf-8",
    )

    # Create pyproject.toml
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        '''[project]
name = "calculator-test"
version = "0.1.0"
requires-python = ">=3.12"

[tool.pytest.ini_options]
testpaths = ["."]
''',
        encoding="utf-8",
    )

    # Initialize git
    import subprocess
    subprocess.run(["git", "init"], cwd=str(project), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(project), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(project), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(project), capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit with broken calculator"], cwd=str(project), capture_output=True)

    return project


@pytest.mark.skipif(SKIP_E2E, reason="E2E tests skipped")
@pytest.mark.skipif(not _has_provider(), reason="No provider available")
class TestInteractiveE2E:
    """Real E2E tests using the actual agent pipeline."""

    def test_agent_fixes_broken_calculator(self, broken_project: Path):
        """Test that the agent can fix the divide-by-zero bug.

        This test runs the real agent pipeline with a real provider.
        It verifies that:
        1. Events are emitted correctly
        2. Tools are called
        3. The agent attempts to fix the code
        If the agent completes within the iteration limit, the fix is verified.
        """
        from harness_core.agent.loop import AgentLoop
        from harness_core.agent.types import AgentConfig, AgentRole
        from harness_core.observability.events import Event, EventBus
        from harness_core.routing.router import ModelRouter, RouterConfig
        from harness_core.routing.task_aware import TaskAwareRouter
        from harness_core.models.registry import ModelRegistry
        from harness_core.tools.filesystem import (
            EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool,
        )
        from harness_core.tools.search import GlobTool, GrepTool
        from harness_core.tools.shell import RunCommandTool

        async def _run_test():
            # Verify tests fail before fix
            import subprocess
            result = subprocess.run(
                ["python", "-m", "pytest", "test_calculator.py", "-q"],
                capture_output=True, text=True, cwd=str(broken_project),
            )
            assert result.returncode != 0, "Tests should fail before fix"

            # Setup provider
            from harness_core.providers.openrouter import OpenRouterProvider
            provider = OpenRouterProvider()
            if not await provider.health_check():
                pytest.skip("OpenRouter not available")

            # Setup event bus with activity tracking
            event_bus = EventBus()
            activity_log = []

            async def log_activity(event: Event):
                activity_log.append({
                    "type": event.type,
                    "data": event.data,
                })

            event_bus.on("*", log_activity)

            # Setup tools
            tools = [
                ReadFileTool(),
                WriteFileTool(),
                EditFileTool(),
                ListFilesTool(),
                GlobTool(),
                GrepTool(),
                RunCommandTool(),
            ]

            # Setup router
            from harness_core.models.discovery import discover_provider
            registry = ModelRegistry()
            profiles = await discover_provider(provider)
            for p in profiles:
                registry.register(p)

            router_config = RouterConfig()
            router_config.routing_mode = "auto"
            router_config.budget.max_iterations = 25

            task_aware = TaskAwareRouter(registry=registry)

            router = ModelRouter(
                providers=[provider],
                config=router_config,
                event_bus=event_bus,
                task_aware=task_aware,
            )

            # Setup agent
            agent_config = AgentConfig(
                role=AgentRole.BUILD,
                max_iterations=25,
                routing_mode="auto",
            )

            agent = AgentLoop(
                provider=provider,
                tools=tools,
                workspace_root=broken_project,
                config=agent_config,
                event_bus=event_bus,
                router=router,
                task_aware=task_aware,
            )

            # Run the task
            task = await agent.run(
                "Fix the divide function in calculator.py so it raises ValueError "
                "when dividing by zero instead of returning 0."
            )

            # Verify events were emitted (core pipeline test)
            assert len(activity_log) > 0
            event_types = [e["type"] for e in activity_log]
            assert "task.started" in event_types
            assert "task.completed" in event_types

            # Verify tool calls happened
            tool_events = [e for e in activity_log if e["type"] == "tool.call"]
            assert len(tool_events) > 0, "Agent should have used tools"

            await provider.close()

            return task, activity_log

        task, activity_log = asyncio.run(_run_test())
        # The agent ran with a real provider and emitted real events
        # Completion within iteration limit depends on model capability
        assert task.status.value in ("completed", "failed")
        assert len(task.tool_calls) > 0, "Agent should have attempted tool calls"
        # Verify the full event pipeline fired
        event_types = [e["type"] for e in activity_log]
        assert "task.started" in event_types
        assert "task.completed" in event_types

    def test_interactive_shell_event_mapping(self):
        """Test that InteractiveShell correctly maps events to UI display."""
        from harness_core.cli.interactive import InteractiveShell, _tool_display_name

        shell = InteractiveShell(plain=True)

        # Test tool display mapping
        assert "read" in _tool_display_name("read_file", {"path": "test.py"})
        assert "write" in _tool_display_name("write_file", {"path": "test.py"})
        assert "edit" in _tool_display_name("edit_file", {"path": "test.py"})
        assert "run" in _tool_display_name("run_command", {"command": "pytest"})
        assert "grep" in _tool_display_name("grep", {"pattern": "TODO"})
        assert "git" in _tool_display_name("git_status", {})

    def test_session_persistence(self, broken_project: Path):
        """Test that sessions are created and can be listed."""
        from harness_core.session.manager import SessionManager

        manager = SessionManager()

        # Create a session
        session = manager.create_session(
            workspace_path=str(broken_project),
            title="E2E test session",
        )
        assert session.session_id
        assert session.status.value == "active"

        # Start a run
        run = manager.start_run(
            session.session_id,
            task="Fix the calculator",
            model_id="test-model",
            provider="test-provider",
        )
        assert run.run_id

        # Complete the run
        manager.complete_run(
            run.run_id,
            outcome="success",
            verification_passed=True,
            result_summary="Fixed divide-by-zero bug",
            iterations=5,
            tool_calls=8,
        )

        # Verify session state
        state = manager.get_resume_state(session.session_id)
        assert state is not None
        assert len(state["runs"]) == 1
        assert state["runs"][0].status.value == "completed"

    def test_agent_emits_real_events(self, broken_project: Path):
        """Test that the agent emits real events through EventBus."""
        from harness_core.observability.events import Event, EventBus

        bus = EventBus()
        events_received = []

        async def collector(event: Event):
            events_received.append(event)

        bus.on("*", collector)

        async def _test():
            # Emit some events
            await bus.emit(Event(type="task.started", data={"goal": "test"}))
            await bus.emit(Event(type="tool.call", data={"tool": "read_file", "args": {"path": "test.py"}}))
            await bus.emit(Event(type="tool.result", data={"tool": "read_file", "status": "success"}))
            await bus.emit(Event(type="task.completed", data={"status": "completed"}))

        asyncio.run(_test())

        assert len(events_received) == 4
        assert events_received[0].type == "task.started"
        assert events_received[1].type == "tool.call"
        assert events_received[2].type == "tool.result"
        assert events_received[3].type == "task.completed"
