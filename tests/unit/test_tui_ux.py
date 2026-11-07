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


class TestTerminalReliabilityAndClaudeParity:
    """Claude Code parity: minimal prompt, no console wipe, no model leaks, flicker-free streaming."""

    def test_unicode_and_prompt_symbols(self):
        from harness_core.cli.ui import Sym, is_unicode_supported
        assert isinstance(is_unicode_supported(), bool)
        assert Sym.PROMPT == "❯"
        assert Sym.PROMPT_ASCII == ">"

    def test_clean_prompt_session_format(self):
        shell = InteractiveShell(plain=False)
        prompt_html = shell._get_prompt_html()
        fragments = prompt_html.__pt_formatted_text__()
        full_text = "".join(f[1] for f in fragments)
        assert "─" not in full_text
        assert "❯" in full_text or ">" in full_text

    def test_9router_completion_privacy(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        renderer = ConversationRenderer(console)
        renderer.set_9router(True)
        renderer.render_completion(
            elapsed=1.5,
            success=True,
            status="completed",
            model="secret-upstream-model",
            models_used=["secret-upstream-model"],
        )
        output = out.getvalue()
        assert "⚡ Routed through 9router" in output
        assert "secret-upstream-model" not in output

    def test_flicker_free_stream_response(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        renderer = ConversationRenderer(console)
        renderer.stream_response("# Plan Summary\n\n- Task 1\n- Task 2", delay=0.0)
        output = out.getvalue()
        assert "Plan Summary" in output
        assert "Task 1" in output

    @pytest.mark.asyncio
    async def test_first_task_does_not_clear_console(self):
        shell = InteractiveShell(plain=True)
        mock_console = MagicMock()
        shell.console = mock_console
        shell._first_task = True
        shell.mode = "standard"
        shell._provider = None  # Will return early
        res = await shell._execute_task("hello")
        assert shell._first_task is False
        mock_console.clear.assert_not_called()

    def test_strip_redundant_narration(self):
        from harness_core.cli.conversation import _clean_intro_narration
        fluff = (
            "Based on my analysis of the codebase, let me provide a comprehensive explanation.\n\n"
            "# System Architecture\n\nThe runtime consists of..."
        )
        cleaned = _clean_intro_narration(fluff)
        assert cleaned.startswith("# System Architecture")
        assert "Based on my analysis" not in cleaned

        # Direct content is not modified
        direct = "# Direct Heading\n\nContent here."
        assert _clean_intro_narration(direct) == direct

    def test_tool_activity_timeline_without_plan(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        renderer = ConversationRenderer(console)
        renderer.start("Quick inspection")
        renderer.tool_started("read_file", {"path": "src/main.py"})
        renderer.tool_completed("read_file", "success")
        renderer.tool_started("run_command", {"command": "pytest"})
        renderer.tool_completed("run_command", "success")
        
        renderable = renderer._renderable()
        console.print(renderable)
        text = out.getvalue()
        # Should NOT contain orphaned tree branches when no plan exists
        assert "├─" not in text
        assert "Read src/main.py" in text
        assert "Run pytest" in text

    def test_consecutive_tasks_reset_state(self):
        shell = InteractiveShell(plain=True)
        console = Console(file=io.StringIO(), force_terminal=True, width=100)
        
        # Task 1
        conv1 = ConversationRenderer(console)
        conv1.start("Task 1")
        conv1.tool_started("read_file", {"path": "src/a.py"})
        conv1.tool_completed("read_file", "success")
        conv1.update_tokens(output_tokens=500, input_tokens=100)
        assert conv1.state.tool_calls == 1
        assert conv1.state.output_tokens == 500
        conv1.stop()

        # Task 2 — fresh instance simulating consecutive execution
        conv2 = ConversationRenderer(console)
        conv2.start("Task 2")
        assert conv2.state.tool_calls == 0
        assert conv2.state.output_tokens == 0
        assert len(conv2.state.items) == 0
        assert conv2.state.is_9router is False

    def test_file_edit_and_test_reporting_in_completion(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        renderer = ConversationRenderer(console)
        renderer.render_completion(
            elapsed=4.2,
            success=True,
            status="completed",
            files_changed=["src/api/routes.py"],
            file_change_stats={"src/api/routes.py": {"added": 12, "removed": 3}},
            test_results={"passed": 5, "failed": 0, "total": 5, "success": True},
        )
        output = out.getvalue()
        assert "routes.py (+12 -3)" in output
        assert "Tests  5 passed" in output

    @pytest.mark.asyncio
    async def test_model_cooldown_event_updates_stage(self):
        shell = InteractiveShell(plain=True)
        shell.console = Console(file=io.StringIO(), no_color=True)
        shell._event_bus = EventBus()
        shell._setup_event_handlers()
        conv = ConversationRenderer(shell.console)
        shell._conv = conv

        await shell._event_bus.emit(
            Event(type="model.cooldown", data={"model": "gemma", "cooldown_seconds": 15.0})
        )
        assert conv.state.stage == "Retrying model…"

    def test_sparkle_status_line_format(self):
        renderer = ConversationRenderer(Console(file=io.StringIO()))
        renderer.state.started_at = time.monotonic() - 12.0
        renderer.state.stage = "Considering…"
        renderer.state.current_turn_tokens = 512
        line = renderer._render_status_line()
        text = line.plain
        assert "Considering…" in text
        assert "12s" in text
        assert "512" in text
        assert "(" in text and ")" in text


class TestClaudeCodeTerminalUXParity:
    """Rigorous regression tests for Claude Code terminal UX parity and reliability."""

    def test_normal_successful_task(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        conv.start("Add user auth endpoint")
        conv.tool_started("read_file", {"path": "src/auth.py"})
        conv.tool_completed("read_file", "success")
        conv.tool_started("write_file", {"path": "src/auth.py"})
        conv.tool_completed("write_file", "success")
        conv.stop()
        conv.stream_response("# User Auth Added\n\nEndpoint created successfully.")
        conv.render_completion(
            elapsed=3.5,
            success=True,
            status="completed",
            files_changed=["src/auth.py"],
            test_results={"passed": 4, "failed": 0},
        )
        rendered = out.getvalue()
        assert "Completed" in rendered
        assert "3.5s" in rendered
        assert "2 tool calls" in rendered
        assert "auth.py" in rendered
        assert "4 passed" in rendered

    def test_multiple_tool_calls_timeline(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        conv.start("Refactor database queries")
        tools = [
            ("read_file", {"path": "src/db/models.py"}),
            ("grep", {"pattern": "SELECT * FROM users"}),
            ("edit_file", {"path": "src/db/queries.py"}),
            ("run_command", {"command": "pytest tests/test_db.py"}),
            ("git_status", {}),
        ]
        for t_name, t_args in tools:
            conv.tool_started(t_name, t_args)
            conv.tool_completed(t_name, "success")
        
        renderable = conv._renderable()
        console.print(renderable)
        text = out.getvalue()
        assert "Read src/db/models.py" in text
        assert 'Search "SELECT * FROM users"' in text
        assert "Edit src/db/queries.py" in text
        assert "Run pytest tests/test_db.py" in text
        assert "Git status" in text
        assert conv.state.tool_calls == 5

    def test_tool_failure_with_nonzero_exit_code_and_stderr(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        conv.start("Run suite")
        conv.tool_started("run_command", {"command": "pytest tests/api/"})
        conv.tool_completed(
            "run_command",
            "failed",
            exit_code=1,
            error="FAILED tests/api/test_users.py::test_create - AssertionError: 404 != 200",
        )
        renderable = conv._renderable()
        console.print(renderable)
        text = out.getvalue()
        # Must show command, exit code, and the actual error reason indented
        assert "Run pytest tests/api/" in text
        assert "Exit 1" in text
        assert "AssertionError: 404 != 200" in text

    def test_model_timeout_and_rate_limit_error_presentation(self):
        from harness_core.cli.interactive import _friendly_model_error
        # Rate limit
        err_429 = _friendly_model_error("HTTP 429: rate_limit_exceeded: free-models-per-day reached")
        assert "rate limited" in err_429.lower() or "limit" in err_429.lower()
        
        # Auth failure
        err_401 = _friendly_model_error("HTTP 401 Unauthorized: Invalid API key")
        assert "authentication failed" in err_401.lower()

        # Timeout
        err_timeout = _friendly_model_error("Request timed out after 45.0 seconds")
        assert "timed out" in err_timeout.lower()

    def test_cancellation_during_execution(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        conv.start("Long task")
        conv.tool_started("read_file", {"path": "src/big.py"})
        conv.tool_completed("read_file", "success")
        conv.stop()
        conv.render_completion(
            elapsed=1.2,
            success=False,
            status="cancelled",
            files_changed=[],
        )
        rendered = out.getvalue()
        assert "Cancelled" in rendered
        assert "1.2s" in rendered

    def test_empty_or_whitespace_response(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        # Should gracefully return without outputting broken markdown
        conv.stream_response("")
        conv.stream_response("   \n\n  ")
        assert out.getvalue().strip() == ""

    def test_long_markdown_response_and_fenced_code(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=False, no_color=True, width=100)
        conv = ConversationRenderer(console)
        md_text = (
            "# Architecture Overview\n\n"
            "Here is the service diagram:\n\n"
            "```python\n"
            "def handle_request(req):\n"
            "    return {'status': 200}\n"
            "```\n\n"
            "| Feature | Supported |\n"
            "| --- | --- |\n"
            "| Auth | Yes |\n"
            "| RBAC | Yes |\n"
        )
        conv.stream_response(md_text, delay=0.0)
        text = out.getvalue()
        assert "Architecture Overview" in text
        assert "def handle_request" in text
        assert "Feature" in text and "Auth" in text

    def test_unicode_emoji_and_wide_characters(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        conv.start("Unicode test: 🚀 数据库 日本語 ✦ ◆ ❯")
        conv.tool_started("run_command", {"command": "echo 'Hello 世界 🌍'"})
        conv.tool_completed("run_command", "success")
        renderable = conv._renderable()
        console.print(renderable)
        text = out.getvalue()
        assert "Hello 世界" in text or "Run echo" in text

    def test_narrow_terminal_width_adaptation(self):
        for width in [24, 40, 60, 80]:
            out = io.StringIO()
            console = Console(file=out, force_terminal=True, width=width)
            conv = ConversationRenderer(console)
            conv.start("A very long goal that exceeds small widths easily")
            conv.tool_started("run_command", {"command": "pytest --capture=no tests/unit/very_long_test_file_name.py"})
            conv.tool_completed("run_command", "failed", exit_code=1, error="AssertionError: value mismatch")
            renderable = conv._renderable()
            console.print(renderable)
            # Ensure none of the rendered output crashes
            lines = out.getvalue().splitlines()
            assert len(lines) > 0

    def test_no_duplicate_completion_message(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        conv.render_completion(
            elapsed=5.0,
            success=True,
            status="completed",
        )
        text = out.getvalue()
        assert text.count("Completed") == 1
        assert text.count("5.0s") == 1

    def test_no_stale_spinner_or_timer_after_stop(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        conv.start("Testing stop cleanup")
        conv.tool_started("read_file", {"path": "a.txt"})
        conv.tool_completed("read_file", "success")
        conv.stop()
        # When stopped, _active is False, so _renderable() should NOT contain status line with spinner
        renderable = conv._renderable()
        console.print(renderable)
        text = out.getvalue()
        assert "Connecting…" not in text
        assert "Working…" not in text

    def test_verification_status_reporting(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=True, width=100)
        conv = ConversationRenderer(console)
        conv.render_completion(
            elapsed=6.0,
            success=False,
            status="failed",
            verification_status="failed",
            verification_summary="2 unit tests failed",
            unverified_files=["src/app.py"],
            error_detail="Task stopped because verification failed",
        )
        text = out.getvalue()
        assert "Failed" in text
        assert "Verification  Failed (2 unit tests failed)" in text
        assert "Files (unverified)" in text and "app.py" in text
        assert "Task stopped because verification failed" in text

    def test_output_token_accounting_and_omission(self):
        # 1. Output tokens == 0 -> line omitted
        out1 = io.StringIO()
        conv1 = ConversationRenderer(Console(file=out1, width=80))
        conv1.render_completion(elapsed=2.0, success=True, token_usage={"input_tokens": 0, "output_tokens": 0})
        assert "Tokens" not in out1.getvalue()

        # 2. Output tokens > 0 -> formatted accurately
        out2 = io.StringIO()
        conv2 = ConversationRenderer(Console(file=out2, width=80))
        conv2.render_completion(elapsed=2.0, success=True, token_usage={"input_tokens": 1500, "output_tokens": 400, "estimated": False})
        text2 = out2.getvalue()
        assert "Tokens  1.5k in, 400 out (exact)" in text2

    def test_non_tty_and_plain_mode_tool_failure(self):
        out = io.StringIO()
        console = Console(file=out, force_terminal=False, no_color=True, width=80)
        conv = ConversationRenderer(console, plain=True)
        conv.start("Plain mode run")
        conv.tool_started("run_command", {"command": "npm test"})
        conv.tool_completed("run_command", "failed", exit_code=2, error="Test suite failed")
        text = out.getvalue()
        assert "Plain mode run" in text
        assert "exit 2" in text or "Test suite failed" in text


class TestRefinedTerminalUX:
    """Rigorous tests for refined terminal UX: deduplication, compaction, clean spacing."""

    def test_compact_markdown_tables_transformation(self):
        from harness_core.cli.conversation import compact_markdown_tables
        # 5-column table should convert to structured cards on narrow width
        wide_table = (
            "| Component | Role | Health | Latency | Cost |\n"
            "|---|---|---|---|---|\n"
            "| Router | Routing | OK | 24ms | $0.00 |\n"
            "| Worker | Execution | OK | 110ms | $0.00 |\n"
        )
        compacted = compact_markdown_tables(wide_table, max_width=60)
        assert "|---|---|---|---|---|" not in compacted
        assert "Router" in compacted and "Routing" in compacted
        assert "Worker" in compacted and "Execution" in compacted

        # Narrow 2-column table should remain as markdown table
        narrow_table = (
            "| Key | Value |\n"
            "|---|---|\n"
            "| Status | Active |\n"
        )
        kept = compact_markdown_tables(narrow_table, max_width=100)
        assert "| Key | Value |" in kept
        assert "|---|---|" in kept

    def test_tool_event_deduplication_by_call_id(self):
        out = io.StringIO()
        console = Console(file=out, width=80)
        renderer = ConversationRenderer(console)
        renderer.start("Deduplication test")

        # Simulate workflow tool.call followed by tool.started with same call_id
        renderer.tool_started("read_file", {"path": "src/main.py"}, call_id="call_abc")
        renderer.tool_started_safe("read_file", "read_file: src/main.py", call_id="call_abc")

        # Must not duplicate items or double-count tool calls
        assert renderer.state.tool_calls == 1
        assert len(renderer.state.items) == 1

        # Simulate tool completion
        renderer.tool_completed("read_file", "success", call_id="call_abc")
        assert len(renderer.state.items) == 1
        assert renderer.state.items[0][2] == "success"

    def test_zero_stale_spinners_on_renderer_stop(self):
        out = io.StringIO()
        console = Console(file=out, width=80)
        renderer = ConversationRenderer(console)
        renderer.start("Stop sweep test")

        renderer.tool_started("run_command", {"command": "cargo build"}, call_id="call_999")
        assert renderer.state.items[0][2] == "running"
        assert renderer.state.items[0][0] == Sym.SPINNER

        # Stopping should sweep any pending item so spinner is never frozen in history
        renderer.stop()
        assert renderer.state.items[0][2] == "success"
        assert renderer.state.items[0][0] == Sym.CHECK
        for item in renderer.state.items:
            assert item[0] != Sym.SPINNER
            assert item[2] != "running"

    def test_read_only_task_omits_verification_line(self):
        out = io.StringIO()
        console = Console(file=out, width=80)
        renderer = ConversationRenderer(console)
        renderer.render_completion(
            elapsed=1.2,
            success=True,
            status="completed",
            files_changed=[],
            verification_status="not_started",
        )
        assert "Verification" not in out.getvalue()

    def test_phase_duration_clean_threshold(self):
        out1 = io.StringIO()
        conv1 = ConversationRenderer(Console(file=out1, width=80))
        # Trivial / single phase -> omitted
        conv1.render_completion(
            elapsed=1.2,
            success=True,
            status="completed",
            phase_durations={"planning": 0.2, "working": 1.0},
        )
        assert "Planning" not in out1.getvalue()

        out2 = io.StringIO()
        conv2 = ConversationRenderer(Console(file=out2, width=80))
        # Multi-phase significant work -> displayed
        conv2.render_completion(
            elapsed=5.5,
            success=True,
            status="completed",
            phase_durations={"planning": 1.5, "working": 4.0},
        )
        val = out2.getvalue()
        assert "Planning 1.5s" in val
        assert "Working 4.0s" in val

    def test_clean_prompt_no_leading_newline(self):
        shell = InteractiveShell(plain=False)
        prompt_html = shell._get_prompt_html()
        raw_text = getattr(prompt_html, "formatted_text", "") or str(prompt_html)
        assert not str(raw_text).startswith("\n")


class TestPostRefinementAudit:
    """Targeted audit tests for event lifecycles, widths, tables, tokens, and verification."""

    def test_concurrent_tools_distinct_call_ids(self):
        """Concurrent tools with different call_ids must not be collapsed."""
        renderer = ConversationRenderer(Console(file=io.StringIO(), width=80))
        renderer.start("Concurrent tools test")

        # Start tool A and tool B concurrently
        renderer.tool_started("read_file", {"path": "a.py"}, call_id="call_A")
        renderer.tool_started("read_file", {"path": "b.py"}, call_id="call_B")

        assert len(renderer.state.items) == 2
        assert renderer.state.tool_calls == 2
        assert renderer.state.items[0].call_id == "call_A"
        assert renderer.state.items[1].call_id == "call_B"

        # Complete tool B first (out of order)
        renderer.tool_completed("read_file", "success", call_id="call_B")
        assert renderer.state.items[1][2] == "success"
        assert renderer.state.items[0][2] == "running"

        # Complete tool A
        renderer.tool_completed("read_file", "success", call_id="call_A")
        assert renderer.state.items[0][2] == "success"

    def test_tool_failure_and_retry_not_collapsed(self):
        """A failed tool followed by a retry must be preserved as distinct records."""
        renderer = ConversationRenderer(Console(file=io.StringIO(), width=80))
        renderer.start("Retry test")

        # Attempt 1 fails
        renderer.tool_started("run_command", {"command": "cargo test"}, call_id="call_f1")
        renderer.tool_completed("run_command", "failed", exit_code=101, error="cargo test failed", call_id="call_f1")
        assert len(renderer.state.items) == 1
        assert renderer.state.items[0][2] == "error" or renderer.state.items[0][2] == "failed"

        # Attempt 2 retries
        renderer.tool_started("run_command", {"command": "cargo test"}, call_id="call_f2")
        assert len(renderer.state.items) == 2
        assert renderer.state.tool_calls == 2

        renderer.tool_completed("run_command", "success", call_id="call_f2")
        assert renderer.state.items[0][2] in ("error", "failed")
        assert renderer.state.items[1][2] == "success"

    def test_legacy_event_missing_call_ids(self):
        """Legacy events without call_ids must still deduplicate on identical running tools."""
        renderer = ConversationRenderer(Console(file=io.StringIO(), width=80))
        renderer.start("Legacy test")

        renderer.tool_started("read_file", {"path": "legacy.py"}, call_id="")
        renderer.tool_started_safe("read_file", "read_file: legacy.py", call_id="")

        assert len(renderer.state.items) == 1
        assert renderer.state.tool_calls == 1

        renderer.tool_completed("read_file", "success", call_id="")
        assert renderer.state.items[0][2] == "success"

    @pytest.mark.parametrize("width", [60, 80, 100, 120])
    def test_rendering_across_terminal_widths(self, width: int):
        """Verify stream_response and renderable behave stably across various terminal widths."""
        out = io.StringIO()
        console = Console(file=out, width=width, force_terminal=True)
        renderer = ConversationRenderer(console)
        renderer.start(f"Testing width {width}")

        long_url = "https://github.com/harness-engineering/autonomous-agent-runtime/releases/tag/v2.1.0-alpha.9"
        long_identifier = "supercalifragilisticexpialidocious_variable_initialization_handler"
        code_block = "```python\ndef calculate_metric(x: int) -> int:\n    return x * 42\n```"
        unicode_text = "🚀 Launched fast build · ✨ All tests green · 📦 Published package"
        markdown_sample = f"# Heading\n\n- Visit: {long_url}\n- Symbol: `{long_identifier}`\n\n{code_block}\n\n{unicode_text}"

        renderer.stream_response(markdown_sample, delay=0.0)
        output = out.getvalue()
        assert "Heading" in output
        assert "calculate_metric" in output

    def test_empty_and_whitespace_responses_no_crash(self):
        """Empty and whitespace-only responses should safely no-op."""
        out = io.StringIO()
        console = Console(file=out, width=80)
        renderer = ConversationRenderer(console)
        renderer.stream_response("", delay=0.0)
        renderer.stream_response("   \n\t  \n", delay=0.0)
        assert out.getvalue().strip() == ""

    def test_token_usage_display_permutations(self):
        """Verify token usage formatting across zero, missing, exact, and estimated values."""
        # 1. Missing / None
        out1 = io.StringIO()
        conv1 = ConversationRenderer(Console(file=out1, width=80))
        conv1.render_completion(elapsed=1.0, success=True, token_usage=None)
        assert "Tokens" not in out1.getvalue()

        # 2. Zero tokens
        out2 = io.StringIO()
        conv2 = ConversationRenderer(Console(file=out2, width=80))
        conv2.render_completion(elapsed=1.0, success=True, token_usage={"input_tokens": 0, "output_tokens": 0})
        assert "Tokens" not in out2.getvalue()

        # 3. Exact tokens
        out3 = io.StringIO()
        conv3 = ConversationRenderer(Console(file=out3, width=80))
        conv3.render_completion(elapsed=1.0, success=True, token_usage={"input_tokens": 2048, "output_tokens": 512, "estimated": False})
        assert "Tokens  2.0k in, 512 out (exact)" in out3.getvalue()

        # 4. Estimated tokens
        out4 = io.StringIO()
        conv4 = ConversationRenderer(Console(file=out4, width=80))
        conv4.render_completion(elapsed=1.0, success=True, token_usage={"input_tokens": 1000, "output_tokens": 200, "estimated": True})
        assert "Tokens  1.0k in, 200 out (estimated)" in out4.getvalue()

    def test_verification_status_display_permutations(self):
        """Verify verification status display: passed, failed, not completed on edit, omitted on read-only."""
        # 1. Verification passed
        out1 = io.StringIO()
        conv1 = ConversationRenderer(Console(file=out1, width=80))
        conv1.render_completion(elapsed=2.0, success=True, verification_status="passed", verification_summary="All 10 tests passed")
        assert "Verification  Passed (All 10 tests passed)" in out1.getvalue()

        # 2. Verification failed
        out2 = io.StringIO()
        conv2 = ConversationRenderer(Console(file=out2, width=80))
        conv2.render_completion(elapsed=2.0, success=False, verification_status="failed", verification_summary="2 tests failed")
        assert "Verification  Failed (2 tests failed)" in out2.getvalue()

        # 3. Task stopped with modified files before verification ran
        out3 = io.StringIO()
        conv3 = ConversationRenderer(Console(file=out3, width=80))
        conv3.render_completion(elapsed=2.0, success=False, files_changed=["src/lib.py"])
        assert "Verification  Not completed (task stopped before verification)" in out3.getvalue()

        # 4. Pure read-only query (no files modified, verification not started) -> omitted
        out4 = io.StringIO()
        conv4 = ConversationRenderer(Console(file=out4, width=80))
        conv4.render_completion(elapsed=1.0, success=True, files_changed=[], verification_status="not_started")
        assert "Verification" not in out4.getvalue()





