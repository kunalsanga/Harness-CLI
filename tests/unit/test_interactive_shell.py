"""Tests for the interactive terminal shell."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.cli.interactive import (
    InteractiveShell,
    _safe_str,
    _tool_display_name,
    __version__,
)


# ─── Helper Tests ─────────────────────────────────────────────────────────

class TestSafeStr:
    def test_redacts_openrouter_key(self):
        result = _safe_str("sk-or-v1-abc123def456ghi789jkl012mno345pqr678")
        assert "[REDACTED]" in result
        assert "sk-or-" not in result

    def test_redacts_generic_sk_key(self):
        result = _safe_str("sk-abc123def456ghi789jkl012mno345pqr678stu901")
        assert "[REDACTED]" in result

    def test_redacts_github_token(self):
        result = _safe_str("ghp_123456789012345678901234567890123456")
        assert "[REDACTED]" in result

    def test_redacts_bearer_token(self):
        result = _safe_str("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9")
        assert "[REDACTED]" in result

    def test_preserves_normal_strings(self):
        result = _safe_str("hello world")
        assert result == "hello world"

    def test_none_returns_empty(self):
        assert _safe_str(None) == ""


class TestToolDisplayName:
    def test_read_file(self):
        assert _tool_display_name("read_file", {"path": "foo.py"}) == "read foo.py"

    def test_write_file(self):
        assert _tool_display_name("write_file", {"path": "bar.py"}) == "write bar.py"

    def test_edit_file(self):
        assert _tool_display_name("edit_file", {"path": "baz.py"}) == "edit baz.py"

    def test_list_files(self):
        assert _tool_display_name("list_files", {"path": "src"}) == "list src"

    def test_list_files_default(self):
        assert _tool_display_name("list_files", {}) == "list ."

    def test_run_command(self):
        assert _tool_display_name("run_command", {"command": "pytest"}) == "run pytest"

    def test_run_command_long(self):
        cmd = "python -m pytest tests/ -v --tb=short -x"
        result = _tool_display_name("run_command", {"command": cmd})
        assert len(result) < 70
        assert result.startswith("run ")

    def test_grep(self):
        assert _tool_display_name("grep", {"pattern": "TODO", "path": "src"}) == 'grep "TODO" in src'

    def test_glob(self):
        assert _tool_display_name("glob", {"pattern": "**/*.py"}) == "glob **/*.py"

    def test_git_status(self):
        assert _tool_display_name("git_status", {}) == "git status"

    def test_git_diff(self):
        assert _tool_display_name("git_diff", {}) == "git diff"

    def test_git_log(self):
        assert _tool_display_name("git_log", {}) == "git log"

    def test_unknown_tool(self):
        assert _tool_display_name("custom_tool", {}) == "custom_tool"


# ─── Shell Init Tests ────────────────────────────────────────────────────

class TestInteractiveShellInit:
    def test_default_init(self):
        shell = InteractiveShell()
        assert shell.model is None
        assert shell.mode == "auto"
        assert shell.free is False
        assert shell.local is False
        assert shell.plain is False
        assert shell.max_iterations == 30
        assert shell.running is False

    def test_custom_init(self):
        shell = InteractiveShell(
            model="test-model",
            mode="free",
            free=True,
            local=True,
            plain=True,
            max_iterations=10,
            max_cost=5.0,
            workspace="/tmp/test",
        )
        assert shell.model == "test-model"
        assert shell.mode == "free"
        assert shell.free is True
        assert shell.local is True
        assert shell.plain is True
        assert shell.max_iterations == 10
        assert shell.max_cost == 5.0
        assert shell.workspace == "/tmp/test"

    def test_version_defined(self):
        assert __version__ == "0.1.0"


# ─── Slash Command Tests ─────────────────────────────────────────────────

class TestSlashCommands:
    def _make_shell(self) -> InteractiveShell:
        shell = InteractiveShell(plain=True)
        shell.console = MagicMock()
        return shell

    def test_non_command_returns_false(self):
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("fix the tests"))
        assert result is False

    def test_exit_returns_false(self):
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/exit"))
        assert result is False
        result = asyncio.run(shell._handle_command("/quit"))
        assert result is False

    def test_help_returns_true(self):
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/help"))
        assert result is True

    def test_status_returns_true(self):
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/status"))
        assert result is True

    def test_model_returns_true(self):
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/model"))
        assert result is True

    def test_clear_returns_true(self):
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/clear"))
        assert result is True

    def test_config_returns_true(self):
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/config"))
        assert result is True

    def test_unknown_command_returns_true(self):
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/unknown"))
        assert result is True


# ─── Event Handler Tests ─────────────────────────────────────────────────

class TestEventHandlers:
    def _make_shell(self) -> InteractiveShell:
        shell = InteractiveShell(plain=True)
        shell.console = MagicMock()
        return shell

    def test_setup_event_handlers_registers_listeners(self):
        shell = self._make_shell()
        mock_bus = MagicMock()
        shell._event_bus = mock_bus

        shell._setup_event_handlers()

        # Should register handlers for key events
        expected_events = [
            "task.started",
            "task.classified",
            "routing.decision",
            "iteration.started",
            "tool.call",
            "tool.result",
            "model.error",
            "task.completed",
        ]
        registered = [call[0][0] for call in mock_bus.on.call_args_list]
        for event_type in expected_events:
            assert event_type in registered, f"Missing handler for {event_type}"

    def test_setup_handles_none_bus(self):
        shell = self._make_shell()
        shell._event_bus = None
        # Should not raise
        shell._setup_event_handlers()


# ─── Tool Display During Execution ───────────────────────────────────────

class TestToolDisplay:
    def test_read_file_shows_path(self):
        display = _tool_display_name("read_file", {"path": "src/main.py"})
        assert "src/main.py" in display

    def test_run_command_shows_command(self):
        display = _tool_display_name("run_command", {"command": "uv run pytest"})
        assert "uv run pytest" in display

    def test_grep_shows_pattern(self):
        display = _tool_display_name("grep", {"pattern": "TODO", "path": "src"})
        assert "TODO" in display


# ─── Session Tests ───────────────────────────────────────────────────────

class TestSessionManagement:
    def test_shell_stores_session_id(self):
        shell = InteractiveShell()
        assert shell.session_id is None

    def test_shell_initializes_session_manager_as_none(self):
        shell = InteractiveShell()
        assert shell.session_manager is None


# ─── Plain Mode Tests ────────────────────────────────────────────────────

class TestPlainMode:
    def test_plain_mode_sets_console_properties(self):
        shell = InteractiveShell(plain=True)
        assert shell.plain is True

    def test_non_plain_mode(self):
        shell = InteractiveShell(plain=False)
        assert shell.plain is False


# ─── Status Tracking Tests ───────────────────────────────────────────────

class TestStatusTracking:
    def test_initial_counts(self):
        shell = InteractiveShell()
        assert shell.total_tool_calls == 0
        assert shell.total_iterations == 0
        assert shell.session_start == 0.0
        assert shell.task_start == 0.0

    def test_model_tracking(self):
        shell = InteractiveShell()
        assert shell.current_model == ""
        assert shell.current_provider == ""


# ─── Memory Command Tests ────────────────────────────────────────────────

class TestMemoryCommand:
    def test_memory_no_session(self):
        shell = InteractiveShell(plain=True)
        shell.console = MagicMock()
        shell.session_id = None
        shell._cmd_memory("")
        shell.console.print.assert_called_with("  [dim]No active session[/]")

    def test_memory_add_no_session(self):
        shell = InteractiveShell(plain=True)
        shell.console = MagicMock()
        shell.session_id = None
        shell._cmd_memory("add test memory")
        shell.console.print.assert_called_with("  [dim]No active session[/]")


# ─── Status Line Tests ───────────────────────────────────────────────────

class TestStatusLine:
    def test_status_line_no_session_start(self):
        shell = InteractiveShell(plain=True)
        shell.console = MagicMock()
        shell.session_start = 0
        shell._print_status_line()
        # Should print without error
        shell.console.print.assert_called_once()


# ─── Regression Tests for Async Bug Fix ──────────────────────────────────

class TestAsyncBugFix:
    """Regression tests for the asyncio.run() nested event loop bug.

    Previously, /models called asyncio.run() inside the already-running
    event loop, causing RuntimeError. All command handlers must now be
    properly awaited from the async main loop.
    """

    def _make_shell(self) -> InteractiveShell:
        shell = InteractiveShell(plain=True)
        shell.console = MagicMock()
        return shell

    def test_handle_command_is_coroutine(self):
        """_handle_command must be an async method."""
        import inspect
        shell = self._make_shell()
        assert inspect.iscoroutinefunction(shell._handle_command), "_handle_command must be async"

    def test_cmd_models_is_coroutine(self):
        """_cmd_models must be an async method."""
        import inspect
        shell = self._make_shell()
        assert inspect.iscoroutinefunction(shell._cmd_models), "_cmd_models must be async"

    def test_models_command_does_not_crash(self):
        """/models must not raise RuntimeError from nested asyncio.run()."""
        shell = self._make_shell()
        # This would previously raise:
        # RuntimeError: asyncio.run() cannot be called from a running event loop
        result = asyncio.run(shell._handle_command("/models"))
        assert result is True

    def test_models_returns_to_prompt(self):
        """/models must return control (True = command handled)."""
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/models"))
        assert result is True, "/models should return True (command handled)"

    def test_multiple_models_calls_work(self):
        """Multiple /models calls must not crash."""
        shell = self._make_shell()
        for _ in range(3):
            result = asyncio.run(shell._handle_command("/models"))
            assert result is True

    def test_sync_commands_still_work(self):
        """Synchronous commands must still work correctly."""
        shell = self._make_shell()
        # These should all work via await (they're sync internally)
        for cmd in ["/help", "/status", "/model", "/config", "/doctor",
                    "/history", "/clear", "/unknown"]:
            result = asyncio.run(shell._handle_command(cmd))
            assert result is True, f"{cmd} should return True"

    def test_exit_still_works(self):
        """Slash exit must still signal termination."""
        shell = self._make_shell()
        result = asyncio.run(shell._handle_command("/exit"))
        assert result is False, "/exit should return False"

    def test_no_nested_asyncio_run_in_models(self):
        """Verify _cmd_models does not contain asyncio.run() calls."""
        import inspect
        shell = self._make_shell()
        source = inspect.getsource(shell._cmd_models)
        assert "asyncio.run" not in source, "_cmd_models must not call asyncio.run()"

    def test_no_nested_asyncio_run_in_handle_command(self):
        """Verify _handle_command does not contain asyncio.run() calls."""
        import inspect
        shell = self._make_shell()
        source = inspect.getsource(shell._handle_command)
        assert "asyncio.run" not in source, "_handle_command must not call asyncio.run()"
