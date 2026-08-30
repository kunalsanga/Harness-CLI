"""Tests for git identity protection, task execution accounting, and truthful completion."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.agent.types import (
    AgentConfig,
    AgentRole,
    Task,
    TaskExecutionStats,
    TaskStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)


# ─── TaskExecutionStats Tests ──────────────────────────────────────────────


class TestTaskExecutionStats:
    """Test task execution accounting."""

    def test_initial_state(self):
        stats = TaskExecutionStats()
        assert stats.attempted == 0
        assert stats.succeeded == 0
        assert stats.failed == 0
        assert stats.recovered == 0
        assert stats.unresolved == 0
        assert not stats.has_unresolved_failures
        assert stats.success_rate == 0.0

    def test_record_success(self):
        stats = TaskExecutionStats()
        stats.record_attempt()
        stats.record_success("tool_a")
        assert stats.attempted == 1
        assert stats.succeeded == 1
        assert stats.failed == 0
        assert stats.recovered == 0
        assert stats.unresolved == 0
        assert stats.success_rate == 1.0

    def test_record_failure(self):
        stats = TaskExecutionStats()
        stats.record_attempt()
        stats.record_failure("tool_a")
        assert stats.attempted == 1
        assert stats.succeeded == 0
        assert stats.failed == 1
        assert stats.unresolved == 1
        assert stats.has_unresolved_failures

    def test_recovery_tracking(self):
        stats = TaskExecutionStats()
        # First attempt fails
        stats.record_attempt()
        stats.record_failure("tool_a")
        assert stats.unresolved == 1
        assert stats.has_unresolved_failures

        # Second attempt succeeds (recovery)
        stats.record_attempt()
        stats.record_success("tool_a")
        assert stats.recovered == 1
        assert stats.unresolved == 0
        assert not stats.has_unresolved_failures

    def test_multiple_failures_same_tool(self):
        stats = TaskExecutionStats()
        # Two failures on same tool
        stats.record_attempt()
        stats.record_failure("tool_a")
        stats.record_attempt()
        stats.record_failure("tool_a")
        assert stats.unresolved == 2

        # One success recovers both
        stats.record_attempt()
        stats.record_success("tool_a")
        assert stats.recovered == 2
        assert stats.unresolved == 0

    def test_mixed_tools(self):
        stats = TaskExecutionStats()
        # tool_a fails
        stats.record_attempt()
        stats.record_failure("tool_a")
        # tool_b succeeds
        stats.record_attempt()
        stats.record_success("tool_b")
        # tool_a succeeds (recovers)
        stats.record_attempt()
        stats.record_success("tool_a")
        # tool_c fails (unresolved)
        stats.record_attempt()
        stats.record_failure("tool_c")

        assert stats.recovered == 1
        assert stats.unresolved == 1
        assert stats.has_unresolved_failures

    def test_success_rate(self):
        stats = TaskExecutionStats()
        for _ in range(3):
            stats.record_attempt()
            stats.record_success("tool_a")
        stats.record_attempt()
        stats.record_failure("tool_b")
        assert stats.success_rate == pytest.approx(0.75)

    def test_permission_denied_not_counted_as_failure(self):
        stats = TaskExecutionStats()
        stats.record_permission_denied("tool_a")
        # record_permission_denied does NOT count as failure or attempted
        assert stats.failed == 0
        assert stats.unresolved == 0
        assert stats.attempted == 0

    def test_summary(self):
        stats = TaskExecutionStats()
        stats.record_attempt()
        stats.record_success("tool_a")
        stats.record_attempt()
        stats.record_failure("tool_b")
        stats.record_attempt()
        stats.record_success("tool_b")  # recovery
        summary = stats.summary()
        assert "Attempted: 3" in summary
        assert "Succeeded: 2" in summary
        assert "Failed: 1" in summary
        assert "Recovered: 1" in summary


# ─── Git Identity Tool Tests ──────────────────────────────────────────────


class TestGitIdentityTool:
    """Test git identity check tool."""

    @pytest.fixture
    def git_identity_tool(self):
        from harness_core.tools.git import GitIdentityTool
        return GitIdentityTool()

    @pytest.mark.asyncio
    async def test_check_identity_when_configured(self, git_identity_tool, tmp_path):
        """When user.name and user.email are configured, report success."""
        # Initialize git repo and configure identity locally
        proc = await asyncio.create_subprocess_exec(
            "git", "init", str(tmp_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        for cmd in [
            ["git", "config", "user.name", "Test User"],
            ["git", "config", "user.email", "test@example.com"],
        ]:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

        # Verify it works (reads from whatever cwd — global or local)
        result = await git_identity_tool.execute({})
        assert result.status == ToolResultStatus.SUCCESS
        assert "configured" in result.output.lower() or "user.name" in result.output
        assert result.metadata.get("configured") is True

    @pytest.mark.asyncio
    async def test_report_missing_identity_via_mock(self, git_identity_tool):
        """When git config returns empty, report error (don't invent one)."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            # Simulate empty name and email
            name_proc = AsyncMock()
            name_proc.communicate.return_value = (b"", b"")
            name_proc.returncode = 0

            email_proc = AsyncMock()
            email_proc.communicate.return_value = (b"", b"")
            email_proc.returncode = 0

            mock_exec.side_effect = [name_proc, email_proc]

            result = await git_identity_tool.execute({})
            assert result.status == ToolResultStatus.ERROR
            assert "Git identity is not configured" in result.error
            assert "user.name" in result.error


# ─── Agent Loop Completion Tests ──────────────────────────────────────────


class TestAgentLoopCompletion:
    """Test that unresolved failures prevent task completion."""

    @pytest.mark.asyncio
    async def test_unresolved_failure_blocks_completion(self):
        """Task with unresolved tool failure must NOT be marked COMPLETED."""
        from harness_core.agent.loop import AgentLoop

        # Create a task with a failed tool call
        task = Task(goal="test")
        task.tool_calls.append(ToolCall(
            tool_name="run_command",
            arguments={"command": "exit 1"},
            result=ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error="exit code 1",
                exit_code=1,
            ),
        ))

        loop = AgentLoop.__new__(AgentLoop)
        loop._recent_denials = []
        loop._consecutive_denials = 0
        loop._recent_failures = []
        loop._consecutive_failures = 0
        loop._last_command_hash = None

        # _should_block_completion should return True
        assert loop._should_block_completion(task) is True

    @pytest.mark.asyncio
    async def test_recovered_failure_allows_completion(self):
        """Task where failure was recovered should allow completion."""
        from harness_core.agent.loop import AgentLoop

        task = Task(goal="test")
        # First call failed
        task.tool_calls.append(ToolCall(
            tool_name="run_command",
            arguments={"command": "exit 1"},
            result=ToolResult(
                status=ToolResultStatus.ERROR,
                output="",
                error="exit code 1",
                exit_code=1,
            ),
        ))
        # Second call succeeded (recovery)
        task.tool_calls.append(ToolCall(
            tool_name="run_command",
            arguments={"command": "exit 0"},
            result=ToolResult(
                status=ToolResultStatus.SUCCESS,
                output="ok",
            ),
        ))

        loop = AgentLoop.__new__(AgentLoop)
        loop._recent_denials = []
        loop._consecutive_denials = 0
        loop._recent_failures = []
        loop._consecutive_failures = 0
        loop._last_command_hash = None

        assert loop._should_block_completion(task) is False

    @pytest.mark.asyncio
    async def test_permission_denied_does_not_block(self):
        """Permission denied is not a failure that blocks completion."""
        from harness_core.agent.loop import AgentLoop

        task = Task(goal="test")
        task.tool_calls.append(ToolCall(
            tool_name="run_command",
            arguments={"command": "rm -rf /"},
            result=ToolResult(
                status=ToolResultStatus.PERMISSION_DENIED,
                output="",
                error="denied",
            ),
        ))

        loop = AgentLoop.__new__(AgentLoop)
        loop._recent_denials = []
        loop._consecutive_denials = 0
        loop._recent_failures = []
        loop._consecutive_failures = 0
        loop._last_command_hash = None

        assert loop._should_block_completion(task) is False

    @pytest.mark.asyncio
    async def test_execution_stats_tracking(self):
        """Execution stats should be recorded in task."""
        from harness_core.agent.loop import AgentLoop

        task = Task(goal="test")
        stats = task.execution_stats

        # Simulate some tool calls
        stats.record_attempt()
        stats.record_success("read_file")
        stats.record_attempt()
        stats.record_failure("run_command")
        stats.record_attempt()
        stats.record_success("run_command")  # recovery

        assert stats.attempted == 3
        assert stats.succeeded == 2
        assert stats.failed == 1
        assert stats.recovered == 1
        assert stats.unresolved == 0
        assert not stats.has_unresolved_failures


# ─── Git Commit Identity Prevention Tests ─────────────────────────────────


class TestGitCommitIdentityPrevention:
    """Test that GitCommitTool prevents inventing identity."""

    @pytest.mark.asyncio
    async def test_reject_fake_identity_in_commit_message(self):
        """GitCommitTool should reject attempts to set identity via commit."""
        from harness_core.tools.git import GitCommitTool

        tool = GitCommitTool()
        result = await tool.execute({
            "message": 'git config user.name "Calculator User"',
        })
        assert result.status == ToolResultStatus.ERROR
        assert "must never invent" in result.error.lower()

    @pytest.mark.asyncio
    async def test_reject_fake_email_in_commit_message(self):
        """GitCommitTool should reject attempts to set email via commit."""
        from harness_core.tools.git import GitCommitTool

        tool = GitCommitTool()
        result = await tool.execute({
            "message": 'git config user.email "user@example.com"',
        })
        assert result.status == ToolResultStatus.ERROR
        assert "must never invent" in result.error.lower()


# ─── ToolResult Execution Failed Tests ─────────────────────────────────────


class TestToolResultExecutionFailed:
    """Test ToolResult.execution_failed property."""

    def test_success_not_failed(self):
        result = ToolResult(status=ToolResultStatus.SUCCESS, output="ok")
        assert not result.execution_failed

    def test_permission_denied_not_failed(self):
        result = ToolResult(status=ToolResultStatus.PERMISSION_DENIED, output="")
        assert not result.execution_failed

    def test_error_with_nonzero_exit_failed(self):
        result = ToolResult(
            status=ToolResultStatus.ERROR, output="", exit_code=1
        )
        assert result.execution_failed

    def test_error_without_exit_code_failed(self):
        result = ToolResult(
            status=ToolResultStatus.ERROR, output="", error="tool broken"
        )
        assert result.execution_failed

    def test_timeout_failed(self):
        result = ToolResult(
            status=ToolResultStatus.TIMEOUT, output="", error="timed out"
        )
        assert result.execution_failed

    def test_failure_category(self):
        assert ToolResult(
            status=ToolResultStatus.SUCCESS, output=""
        ).failure_category == "success"
        assert ToolResult(
            status=ToolResultStatus.PERMISSION_DENIED, output=""
        ).failure_category == "permission_denied"
        assert ToolResult(
            status=ToolResultStatus.TIMEOUT, output=""
        ).failure_category == "timeout"
        assert ToolResult(
            status=ToolResultStatus.ERROR, output="", exit_code=1
        ).failure_category == "execution_error"
        assert ToolResult(
            status=ToolResultStatus.ERROR, output="", error="broken"
        ).failure_category == "tool_error"
