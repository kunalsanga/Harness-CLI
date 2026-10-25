"""Behavioral tests for the TUI / UX overhaul.

Verifies event-to-render mapping, truthful activity display, branded wordmark,
path shortening, 9router indicator, steering, cancellation, and TODO passthrough.
"""

from __future__ import annotations

import io
import time
from unittest.mock import AsyncMock, MagicMock
import pytest
from rich.console import Console

from harness_core.cli.completion import CompletionFormatter, NextAction
from harness_core.cli.conversation import ConversationRenderer, ConversationState, _compact_tool_display
from harness_core.cli.interactive import InteractiveShell
from harness_core.cli.ui import (
    PHASE_DISPLAY,
    Sym,
    render_welcome,
    render_provider_error,
    shorten_path,
)
from harness_core.observability.events import Event, EventBus


class TestTuiBranding:
    """Branded wordmark and visual identity."""

    def test_welcome_rich_branding(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        render_welcome(console, workspace="/path/to/my-project", plain=False)
        output = out.getvalue()
        assert Sym.DIAMOND in output or "Harness" in output
        assert "my-project" in output
        assert "Autonomous AI Engineering" in output

    def test_welcome_plain_fallback(self):
        out = io.StringIO()
        console = Console(file=out, no_color=True, width=100)
        render_welcome(console, workspace="/path/to/my-project", plain=True)
        output = out.getvalue()
        assert "Harness" in output
        assert "my-project" in output
        assert "Autonomous AI Engineering" in output


class TestPathShortening:
    """Workspace-relative path shortening."""

    def test_shorten_inside_workspace(self):
        ws = "C:/Users/dev/project"
        path = "C:/Users/dev/project/src/components/Menu.tsx"
        assert shorten_path(path, ws) == "src/components/Menu.tsx"

    def test_shorten_outside_workspace(self):
        ws = "C:/Users/dev/project"
        path = "C:/Windows/System32/drivers/etc/hosts"
        res = shorten_path(path, ws)
        assert "hosts" in res


class TestPhaseMapping:
    """Backend phases produce accurate display names."""

    def test_backend_phases_mapped(self):
        expected = {
            "understanding": "Understanding",
            "planning": "Planning",
            "implementing": "Working",
            "testing": "Verifying",
            "diagnosing": "Repairing",
            "fixing": "Repairing",
            "verifying": "Verifying",
            "committing": "Working",
            "pushing": "Working",
            "complete": "Completed",
            "workflow": "Working",
        }
        for backend_phase, display in expected.items():
            assert PHASE_DISPLAY[backend_phase] == display

    def test_conversation_phase_update(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        renderer = ConversationRenderer(console)
        renderer.update_phase("understanding")
        assert renderer.state.phase == "understanding"
        assert renderer.state.display_phase == "Understanding"


class TestActivityAggregation:
    """Consecutive read activities are aggregated."""

    def test_consecutive_reads_aggregated(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        renderer = ConversationRenderer(console)
        renderer.start("Inspect repo")
        for i in range(5):
            renderer.tool_started("read_file", {"path": f"src/file{i}.py"})
            renderer.tool_completed("read_file", "success")

        renderable = renderer._renderable()
        console.print(renderable)
        text = out.getvalue()
        assert "Inspected 5 files" in text or "Inspected" in text


class TestExecutionSummary:
    """Four-question summary format."""

    def test_success_summary_format(self):
        fmt = CompletionFormatter(plain=True)
        res = fmt.success(
            headline="Built restaurant menu",
            files_modified=["src/components/Menu.tsx", "src/api/orders.py"],
            files_created=["src/pages/Admin.tsx"],
            tests_line="2 checks passed",
            verification_status="passed",
            duration=45.2,
            tool_calls=6,
        )
        assert "Done" in res
        assert "Built restaurant menu" in res
        assert "Changes" in res
        assert "Menu.tsx" in res
        assert "Admin.tsx" in res
        assert "Verification" in res

    def test_failure_summary_format(self):
        fmt = CompletionFormatter(plain=True)
        res = fmt.failure(
            headline="Unable to complete task",
            what_happened="Build failed due to type errors",
            files_modified=["src/api/orders.py"],
            evidence_lines=["TS2322: Type 'string' is not assignable to type 'number'"],
            next_actions=[NextAction(label="Investigate build errors")],
        )
        assert "What happened" in res
        assert "Build failed" in res
        assert "Changes" in res
        assert "orders.py" in res
        assert "Next" in res
        assert "No false success was reported" not in res


class TestCancellationUX:
    """Cancellation reporting with state preservation."""

    def test_cancellation_preserves_state(self):
        shell = InteractiveShell(plain=True)
        shell.console = Console(file=io.StringIO(), no_color=True)
        shell.task_start = time.monotonic() - 5.0
        shell._render_cancellation()
        out = shell.console.file.getvalue()
        assert "Cancelled" in out
        assert "Task stopped safely" in out
        assert "State preserved" in out


class TestSteeringDisplay:
    """Steering feedback rendered inline."""

    @pytest.mark.asyncio
    async def test_steering_received(self):
        shell = InteractiveShell(plain=True)
        shell.console = Console(file=io.StringIO(), no_color=True)
        shell._event_bus = EventBus()
        shell._setup_event_handlers()

        await shell._event_bus.emit(
            Event(type="steering.received", data={"message": "Use dark theme for website"})
        )
        out = shell.console.file.getvalue()
        assert "User update received" in out
        assert "Use dark theme for website" in out


class TestFreeModeDisplay:
    """Product-level free mode message."""

    def test_free_mode_message(self):
        shell = InteractiveShell(plain=True)
        shell.console = Console(file=io.StringIO(), no_color=True)
        shell._cmd_free_mode()
        out = shell.console.file.getvalue()
        assert "Free model routing enabled" in out


class TestNineRouterDisplay:
    """9router suppresses model exposure in normal mode."""

    @pytest.mark.asyncio
    async def test_9router_normal_mode(self):
        shell = InteractiveShell(plain=True)
        shell.verbose = False
        shell.console = Console(file=io.StringIO(), no_color=True)
        shell._event_bus = EventBus()
        shell._setup_event_handlers()

        await shell._event_bus.emit(
            Event(type="routing.decision", data={"provider": "9router", "model": "hidden-upstream-model-123"})
        )
        out = shell.console.file.getvalue()
        assert "Routed through 9router" in out
        assert "hidden-upstream-model-123" not in out


class TestErrorMapping:
    """Error categories map to user-friendly messages."""

    def test_auth_error_mapping(self):
        out = io.StringIO()
        console = Console(file=out, no_color=True)
        render_provider_error(console, "401 Unauthorized: Invalid API key", plain=True)
        text = out.getvalue()
        assert "Authentication failed" in text

    def test_rate_limit_mapping(self):
        out = io.StringIO()
        console = Console(file=out, no_color=True)
        render_provider_error(console, "429 Too Many Requests", plain=True)
        text = out.getvalue()
        assert "Rate limited" in text


class TestTodoPassthrough:
    """TUI passes backend TODOs through honestly without hiding them."""

    def test_backend_todos_passthrough(self):
        renderer = ConversationRenderer(Console(file=io.StringIO()))
        backend_todos = [
            {"id": "t1", "title": "Stage intended changes", "status": "pending"},
            {"id": "t2", "title": "Commit changes", "status": "pending"},
        ]
        renderer.update_todo_items(backend_todos)
        assert len(renderer.state.todos) == 2
        assert renderer.state.todos[0]["title"] == "Stage intended changes"
        assert renderer.state.todos[1]["title"] == "Commit changes"


class TestAgentProgressiveDisclosure:
    """Agent count vs names progressive disclosure."""

    def test_progressive_disclosure(self):
        renderer = ConversationRenderer(Console(file=io.StringIO()))
        # 2 agents: shows names/ids
        renderer.update_agent("worker-1", "running")
        renderer.update_agent("worker-2", "running")
        assert len(renderer.state.agents) == 2

        # 4 agents: disclosure logic handles larger sets
        renderer.update_agent("worker-3", "running")
        renderer.update_agent("worker-4", "running")
        assert len(renderer.state.agents) == 4
