"""Tests for Harness CLI interaction UX.

Covers the rebuilt conversational terminal experience:
- Prompt rendered exactly once (spec §3)
- No fake thinking/display (spec §4, §5)
- Compact tool activity with semantic icons (spec §12)
- Completion status variants (spec §8)
- No giant chrome at startup (spec §1)
- Minimal input composer (spec §2)
"""

from __future__ import annotations

import io
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from rich.console import Console

from harness_core.cli.conversation import (
    ConversationRenderer,
    ConversationState,
    derive_intent,
)
from harness_core.cli.interactive import InteractiveShell


# ── Prompt rendered exactly once ─────────────────────────────────────────


class TestPromptRenderedOnce:
    """Spec §3: user prompt must appear exactly once in the terminal."""

    def test_rich_mode_prompt_not_printed_by_renderer(self):
        """Rich mode: ConversationRenderer does NOT print the prompt (prompt_toolkit handles it)."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=False)
        conv.start("explain this project")
        conv.stop()

        out = console.file.getvalue()
        # The renderer should NOT print the prompt — prompt_toolkit owns it.
        # It only prints a blank line for spacing.
        assert "explain this project" not in out, (
            "Renderer must not print prompt in rich mode (prompt_toolkit handles it)"
        )

    def test_plain_mode_prompt_once(self):
        """Plain mode: prompt printed once."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=True)
        conv.start("explain this project")
        conv.stop()

        out = console.file.getvalue()
        assert out.count("explain this project") == 1

    def test_prompt_not_duplicated_by_on_task_started(self):
        """on_task_started must NOT call conv.start() — that's done by _execute_task."""
        from harness_core.observability.events import Event, EventBus

        shell = InteractiveShell()
        shell.console = MagicMock()
        shell.plain = False
        shell._event_bus = EventBus()
        shell._setup_event_handlers()

        with MagicMock() as mock_conv:
            shell._conv = mock_conv
            event = Event(
                type="task.started",
                source="agent_loop",
                data={"goal": "explain this project"},
            )
            for handler in shell._event_bus._handlers.get("task.started", []):
                import asyncio
                asyncio.run(handler(event))

            mock_conv.start.assert_not_called()


# ── No fake thinking ────────────────────────────────────────────────────


class TestNoFakeThinking:
    """Spec §4, §5: never display fake internal reasoning."""

    def test_renderable_no_thinking_heading(self):
        """The live renderable must not contain 'Thinking' heading."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=False)
        conv.start("explain this project")

        renderable = conv._renderable()
        console.print(renderable)
        out = console.file.getvalue()

        assert "Thinking" not in out, "Must not display 'Thinking' heading"
        assert "Analyzing" not in out, "Must not display 'Analyzing' text"
        assert "Reasoning" not in out, "Must not display 'Reasoning' text"

    def test_intent_is_user_safe(self):
        """Intent messages must be short user-safe execution descriptions."""
        # Various goals should produce safe, short intents
        intents = [
            derive_intent("explain this project"),
            derive_intent("fix the tests"),
            derive_intent("add dark mode"),
            derive_intent("push to github"),
        ]
        for intent in intents:
            assert len(intent) < 120, f"Intent too long: {intent}"
            assert not intent.lower().startswith("i am thinking"), (
                f"Must not be chain-of-thought: {intent}"
            )


# ── Tool activity icons ─────────────────────────────────────────────────


class TestToolActivityIcons:
    """Spec §12: semantic icons for different tool types."""

    def test_read_gets_checkmark(self):
        """Read/list/glob tools should get ✓ icon on success."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=False)
        conv.start("test")
        conv.tool_started("read_file", {"path": "app.py"})
        conv.tool_completed("read_file", "success")
        conv.stop()

        out = console.file.getvalue()
        assert "✓" in out, "Read tool should show ✓ icon"

    def test_edit_gets_edit_icon(self):
        """Write/edit tools should get ✎ icon on success."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=False)
        conv.start("test")
        conv.tool_started("edit_file", {"path": "app.py"})
        conv.tool_completed("edit_file", "success")
        conv.stop()

        out = console.file.getvalue()
        assert "✎" in out, "Edit tool should show ✎ icon"

    def test_run_command_gets_play_icon(self):
        """Run_command should get ▶ icon."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=False)
        conv.start("test")
        conv.tool_started("run_command", {"command": "pytest"})
        conv.tool_completed("run_command", "success")
        conv.stop()

        out = console.file.getvalue()
        assert "▶" in out, "Run command should show ▶ icon"

    def test_failure_gets_cross_icon(self):
        """Failed tools should get ✗ icon."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=False)
        conv.start("test")
        conv.tool_started("run_command", {"command": "pytest"})
        conv.tool_completed("run_command", "error")
        conv.stop()

        out = console.file.getvalue()
        assert "✗" in out, "Failed tool should show ✗ icon"


# ── Completion status ───────────────────────────────────────────────────


class TestCompletionStatus:
    """Spec §8: concise completion indicators."""

    def test_success_completion(self):
        """Successful completion shows ✓ Done."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=True)
        conv.start("test")
        conv.render_completion(12.4, success=True, status="completed")
        conv.stop()

        out = console.file.getvalue()
        assert "✓" in out
        assert "Done" in out
        assert "s" in out  # elapsed time present

    def test_failure_completion(self):
        """Failed completion shows ✗ Failed."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=True)
        conv.start("test")
        conv.render_completion(8.2, success=False, status="failed")
        conv.stop()

        out = console.file.getvalue()
        assert "✗" in out
        assert "Failed" in out

    def test_cancelled_completion(self):
        """Cancelled shows ⏹ Cancelled."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=True)
        conv.start("test")
        conv.render_completion(3.1, success=False, status="cancelled")
        conv.stop()

        out = console.file.getvalue()
        assert "Cancelled" in out

    def test_paused_completion(self):
        """Paused shows ⏸ Paused."""
        console = Console(file=io.StringIO(), no_color=True, width=100)
        conv = ConversationRenderer(console, plain=True)
        conv.start("test")
        conv.render_completion(5.0, success=False, status="paused")
        conv.stop()

        out = console.file.getvalue()
        assert "Paused" in out


# ── Startup chrome ──────────────────────────────────────────────────────


class TestCompactStartup:
    """Spec §1: remove giant CLI chrome."""

    def test_welcome_no_ready_message(self):
        """Welcome must NOT display 'Ready.' message."""
        shell = InteractiveShell()
        shell.console = MagicMock()
        shell.workspace = "/tmp/test-project"

        shell._print_welcome()

        calls = shell.console.print.call_args_list
        ready_calls = [c for c in calls if "Ready" in str(c)]
        assert len(ready_calls) == 0, "Must not display 'Ready.' in welcome"

    def test_welcome_no_provider_model_routing(self):
        """Welcome must NOT display verbose provider/model/routing info."""
        shell = InteractiveShell()
        shell.console = MagicMock()
        shell.workspace = "/tmp/test-project"

        shell._print_welcome()

        calls = shell.console.print.call_args_list
        verbose_calls = [
            c for c in calls
            if any(k in str(c) for k in ("Provider:", "Model:", "Routing:"))
        ]
        assert len(verbose_calls) == 0, "Must not display provider/model/routing"

    def test_welcome_shows_project_name(self):
        """Welcome should show project name from directory."""
        shell = InteractiveShell()
        shell.console = MagicMock()
        shell.workspace = "/tmp/my-cool-project"

        shell._print_welcome()

        calls = shell.console.print.call_args_list
        has_project = any("my-cool-project" in str(c) for c in calls)
        assert has_project, "Welcome should show project name"

    def test_welcome_no_verbose_chrome(self):
        """Welcome must not show verbose chrome like separators or provider info."""
        shell = InteractiveShell()
        shell.console = MagicMock()
        shell.workspace = "/tmp/test"

        shell._print_welcome()

        calls = shell.console.print.call_args_list
        # Must not show separator, Ready, Provider, Model, Routing
        chrome = [c for c in calls if any(k in str(c) for k in ("─", "Ready", "Provider:", "Model:", "Routing:"))]
        assert len(chrome) == 0, f"Welcome must not show chrome, got: {chrome}"


# ── Input composer ──────────────────────────────────────────────────────


class TestInputComposer:
    """Spec §2: minimal input composer."""

    def test_prompt_text_has_no_box_borders(self):
        """The prompt text must not use box-drawing border characters."""
        import inspect
        from harness_core.cli.interactive import InteractiveShell
        source = inspect.getsource(InteractiveShell._read_input)
        # Must NOT contain box-drawing characters used for borders
        assert "┌" not in source, "Must not use top-left box border"
        assert "└" not in source, "Must not use bottom-left box border"
        assert "│" not in source, "Must not use vertical box border"
        assert "Enter a coding task" not in source, "Must use shorter placeholder"


# ── Conversation state ──────────────────────────────────────────────────


class TestConversationState:
    """State tracking for live conversation updates."""

    def test_items_tracked(self):
        """Tool activity items are tracked in state."""
        state = ConversationState(goal="test", started_at=time.time())
        state.add_running("Read app.py")
        state.add_running("Edit app.py")
        assert len(state.items) == 2
        state.complete_last("success")
        assert state.items[-1][2] == "success"

    def test_max_items_window(self):
        """State keeps at most max_items recent items."""
        state = ConversationState(goal="test", started_at=time.time(), max_items=3)
        for i in range(10):
            state.add_running(f"Tool {i}")
        assert len(state.items) == 3
        # Most recent items kept
        assert state.items[-1][1] == "Tool 9"

    def test_snapshot(self):
        """State can be serialized to a snapshot dict."""
        state = ConversationState(
            goal="test",
            started_at=time.time(),
        )
        state.add_running("Read app.py")
        snapshot = state.to_snapshot()
        assert snapshot["goal"] == "test"
        assert len(snapshot["items"]) == 1
