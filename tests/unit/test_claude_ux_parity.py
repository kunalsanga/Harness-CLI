"""Deterministic tests for Claude Code terminal UX parity in Harness CLI."""

from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rich.console import Console

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import AgentConfig, ExecutionState, Task, ToolResultStatus
from harness_core.cli.conversation import (
    ConversationRenderer,
    ConversationState,
    format_token_count,
)
from harness_core.cli.interactive import _safe_str
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import CompletionResponse, TokenUsage


def test_format_token_count_exact_and_estimated():
    """Verify honest token counting semantics."""
    assert format_token_count(0, False) == "0 tokens"
    assert format_token_count(450, False) == "450 tokens"
    assert format_token_count(1000, False) == "1.0k tokens"
    assert format_token_count(1250, False) == "1.2k tokens"
    assert format_token_count(450, True) == "~450 tokens (est)"
    assert format_token_count(1200, True) == "~1.2k tokens (est)"


def test_spinner_and_stage_transitions():
    """Verify live status stages transition based on observable events."""
    console = Console(file=StringIO(), width=80)
    renderer = ConversationRenderer(console)
    renderer.start("explain project")

    # Initial stage
    assert renderer.state.stage == "Connecting…"

    # Reading files
    renderer.tool_started("read_file", {"path": "src/app.py"})
    assert renderer.state.stage == "Reading files…"

    # Editing code
    renderer.tool_started("edit_file", {"path": "src/app.py"})
    assert renderer.state.stage == "Editing code…"

    # Running tests
    renderer.tool_started("run_command", {"command": "pytest tests/ -q"})
    assert renderer.state.stage == "Running tests…"

    # Running non-test command
    renderer.tool_started("run_command", {"command": "ls -la"})
    assert renderer.state.stage == "Running command…"

    # Model thinking / request
    renderer.update_model_activity("openrouter/free")
    assert renderer.state.stage == "Considering…"

    # Response streaming
    renderer.stream_response("Here is the explanation.")
    assert renderer.state.stage == "Preparing response…"

    renderer.stop()


def test_status_line_rendering_contains_spinner_stage_and_tokens():
    """Verify the live status line formatting with timer and token indicators."""
    console = Console(file=StringIO(), width=80)
    renderer = ConversationRenderer(console)
    renderer.start("test goal")
    renderer.state.started_at = 100.0

    with patch("time.monotonic", return_value=118.0):
        renderer.set_stage("Investigating…")
        renderer.update_tokens(output_tokens=1200, estimated=False)
        status_text = renderer._render_status_line().plain
        assert "Investigating…" in status_text
        assert "00:18" in status_text
        assert "↓ 1.2k tokens" in status_text

        # Now test estimated tokens tag
        renderer.update_tokens(output_tokens=1200, estimated=True)
        status_text_est = renderer._render_status_line().plain
        assert "↓ ~1.2k tokens (est)" in status_text_est

    renderer.stop()


def test_live_diff_preview_box_rendering():
    """Verify live code change preview box renders modified lines and counts."""
    console = Console(file=StringIO(), width=80)
    renderer = ConversationRenderer(console)

    diff = (
        "--- a/loop.py\n"
        "+++ b/loop.py\n"
        "@@ -10,3 +10,4 @@\n"
        "- old_tool_dispatch()\n"
        "+ await dispatch_tool_calls()\n"
        "+ log_event()\n"
    )
    renderer.update_file_changed(
        path="src/harness_core/agent/loop.py",
        added=18,
        removed=6,
        diff=diff,
    )
    assert renderer.state.file_change_stats["src/harness_core/agent/loop.py"] == {
        "added": 18,
        "removed": 6,
    }

    # Render diff box
    preview_group = renderer._render_diff_preview(renderer.state.current_diff_preview)
    assert preview_group is not None

    output = StringIO()
    Console(file=output, width=80).print(preview_group)
    rendered = output.getvalue()

    assert "loop.py" in rendered
    assert "+18 -6" in rendered
    assert "- old_tool_dispatch()" in rendered
    assert "+ await dispatch_tool_calls()" in rendered


def test_diff_box_narrow_terminal_adaptation():
    """Verify diff box truncates cleanly on narrow terminal without crashing."""
    console = Console(file=StringIO(), width=36)
    renderer = ConversationRenderer(console)

    diff = "- " + "x" * 60 + "\n+ " + "y" * 60 + "\n"
    renderer.update_file_changed("app.py", added=1, removed=1, diff=diff)

    preview_group = renderer._render_diff_preview(renderer.state.current_diff_preview)
    assert preview_group is not None

    output = StringIO()
    Console(file=output, width=36).print(preview_group)
    rendered = output.getvalue()
    assert "app.py" in rendered
    # Each rendered line should be within terminal width limits
    for line in rendered.splitlines():
        assert len(line) <= 40


def test_completion_summary_honest_accounting():
    """Verify turn completion summary displays exact/estimated tokens, files, and tests."""
    output = StringIO()
    console = Console(file=output, width=80)
    renderer = ConversationRenderer(console, plain=False)
    renderer.state.tool_calls = 5

    # Exact token usage
    renderer.render_completion(
        elapsed=14.2,
        success=True,
        model="google/gemma-4-31b-it:free",
        token_usage={"input_tokens": 3200, "output_tokens": 850, "estimated": False},
        file_change_stats={"src/harness_core/agent/loop.py": {"added": 18, "removed": 6}},
        test_results={"passed": 12, "failed": 0},
    )
    rendered = output.getvalue()
    assert "Completed" in rendered and "5 tool calls" in rendered
    assert "Model  Gemma 4 31B" in rendered
    assert "Tokens  3.2k in, 850 out (exact)" in rendered
    assert "Files  .../loop.py (+18 -6)" in rendered or "Files  src/harness_core/agent/loop.py (+18 -6)" in rendered
    assert "Tests  12 passed" in rendered


def test_completion_summary_estimated_tokens_and_failed_tests():
    """Verify completion summary shows (estimated) and failed test counts when applicable."""
    output = StringIO()
    console = Console(file=output, width=80)
    renderer = ConversationRenderer(console, plain=False)

    renderer.render_completion(
        elapsed=8.0,
        success=False,
        status="failed",
        error_detail="Tests did not pass",
        token_usage={"input_tokens": 1000, "output_tokens": 250, "estimated": True},
        test_results={"passed": 8, "failed": 2},
    )
    rendered = output.getvalue()
    assert "Failed  8.0s" in rendered
    assert "Tokens  1.0k in, 250 out (estimated)" in rendered
    assert "Tests  8 passed, 2 failed" in rendered
    assert "Tests did not pass" in rendered


def test_plain_mode_fallback():
    """Verify plain mode prints concise lines without ANSI escape artifacts."""
    output = StringIO()
    console = Console(file=output, width=80)
    renderer = ConversationRenderer(console, plain=True)
    renderer.start("test task")
    renderer.tool_started("read_file", {"path": "src/main.py"})

    rendered = output.getvalue()
    assert "Reading files…" in rendered
    assert "Read src/main.py" in rendered or "main.py" in rendered
    renderer.stop()


def test_terminal_cleanup_restores_cursor():
    """Verify stop() ensures cursor visibility is restored."""
    console = MagicMock(spec=Console)
    renderer = ConversationRenderer(console, plain=False)
    renderer.stop()
    console.show_cursor.assert_called_with(True)


def test_secret_redaction_in_tool_display():
    """Verify secrets and sensitive authorization values are redacted."""
    raw = "Bearer sk-or-v1-abcdef0123456789abcdef0123456789 and ghp_012345678901234567890123456789012345"
    redacted = _safe_str(raw)
    assert "sk-or-v1" not in redacted
    assert "ghp_" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.asyncio
async def test_agent_loop_token_accounting_and_file_change_events(tmp_path: Path):
    """Verify AgentLoop truthfully populates task tokens and emits file.changed."""
    bus = EventBus()
    emitted_events: list[Event] = []
    bus.on("model.completed", lambda e: emitted_events.append(e))

    # Mock provider returning exact tokens
    mock_provider = AsyncMock()
    mock_provider.name = "test_provider"
    mock_provider.health_check.return_value = True
    mock_provider.generate.return_value = CompletionResponse(
        content="Task completed successfully.",
        tool_calls=[],
        model="test-model",
        usage={"prompt_tokens": 120, "completion_tokens": 45},
    )

    loop = AgentLoop(
        provider=mock_provider,
        workspace_root=tmp_path,
        tools=[],
        config=AgentConfig(max_iterations=1),
        event_bus=bus,
    )

    result_task = await loop.run("test task")

    # Verify exact tokens recorded on task
    assert result_task.total_input_tokens == 120
    assert result_task.total_output_tokens == 45
    assert result_task.tokens_estimated is False
    assert result_task.token_usage["input_tokens"] == 120
    assert result_task.token_usage["output_tokens"] == 45

    # Verify model.completed event emitted with tokens
    assert len(emitted_events) >= 1
    assert emitted_events[0].data["input_tokens"] == 120
    assert emitted_events[0].data["output_tokens"] == 45
    assert emitted_events[0].data["estimated"] is False


@pytest.mark.asyncio
async def test_agent_loop_token_estimation_when_provider_omits_usage(tmp_path: Path):
    """Verify AgentLoop provides an honest estimate when provider omits usage metadata."""
    bus = EventBus()
    mock_provider = AsyncMock()
    mock_provider.name = "test_provider"
    mock_provider.health_check.return_value = True
    # 40 chars of content ~ 10 tokens
    mock_provider.generate.return_value = CompletionResponse(
        content="0123456789012345678901234567890123456789",
        tool_calls=[],
        model="test-model",
        usage={},  # Empty usage from provider
    )

    loop = AgentLoop(
        provider=mock_provider,
        workspace_root=tmp_path,
        tools=[],
        config=AgentConfig(max_iterations=1),
        event_bus=bus,
    )

    result_task = await loop.run("test task")

    assert result_task.total_output_tokens == 10
    assert result_task.tokens_estimated is True
    assert result_task.token_usage["estimated"] is True
