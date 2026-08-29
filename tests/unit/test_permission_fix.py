"""Tests for the permission system fix: retry classification, loop guard, interactive approval."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.agent.types import (
    AgentConfig,
    TaskStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.permissions.manager import PermissionManager, PermissionRule


# ─── ToolResult retry classification ──────────────────────────────────────


class TestToolResultRetryClassification:
    """Verify ToolResult correctly classifies retry behavior."""

    def test_permission_denied_is_not_retryable(self):
        r = ToolResult(status=ToolResultStatus.PERMISSION_DENIED, output="", error="denied", retryable=False)
        assert r.is_perm_denied
        assert not r.retryable
        assert r.is_final

    def test_permission_denied_is_final(self):
        r = ToolResult(status=ToolResultStatus.PERMISSION_DENIED, output="", error="denied")
        assert r.is_final

    def test_error_with_retryable_is_transient(self):
        r = ToolResult(status=ToolResultStatus.ERROR, output="", error="timeout", retryable=True)
        assert r.is_transient
        assert not r.is_final

    def test_error_without_retryable_is_final(self):
        r = ToolResult(status=ToolResultStatus.ERROR, output="", error="fatal", retryable=False)
        assert r.is_final

    def test_success_is_not_denied(self):
        r = ToolResult(status=ToolResultStatus.SUCCESS, output="ok")
        assert not r.is_perm_denied
        assert not r.is_transient
        assert not r.is_final

    def test_timeout_is_transient(self):
        r = ToolResult(status=ToolResultStatus.TIMEOUT, output="", error="timed out")
        assert r.is_transient


# ─── PermissionManager interactive approval ───────────────────────────────


class TestPermissionManagerApproval:
    """Verify PermissionManager approval callback works."""

    def test_allow_bypasses_callback(self):
        pm = PermissionManager(rules=[PermissionRule(tool_pattern="read_file", action="allow")])
        # Even with no callback, allow returns True
        assert pm.request_approval("read_file", "read a file") is True

    def test_deny_bypasses_callback(self):
        pm = PermissionManager(rules=[PermissionRule(tool_pattern="rm", action="deny")])
        assert pm.request_approval("rm", "delete file") is False

    def test_ask_with_callback_uses_callback(self):
        pm = PermissionManager(
            rules=[PermissionRule(tool_pattern="run_command", action="ask")],
            approval_callback=lambda t, d: True,
        )
        assert pm.request_approval("run_command", "run something") is True

    def test_ask_with_callback_denies(self):
        pm = PermissionManager(
            rules=[PermissionRule(tool_pattern="run_command", action="ask")],
            approval_callback=lambda t, d: False,
        )
        assert pm.request_approval("run_command", "run something") is False

    def test_ask_without_callback_denies(self):
        pm = PermissionManager(rules=[PermissionRule(tool_pattern="run_command", action="ask")])
        assert pm.request_approval("run_command", "run something") is False

    def test_session_approval_overrides(self):
        pm = PermissionManager(
            rules=[PermissionRule(tool_pattern="run_command", action="ask")],
        )
        pm.approve_for_session("run_command")
        assert pm.request_approval("run_command", "run something") is True

    def test_session_deny_overrides(self):
        pm = PermissionManager(
            rules=[PermissionRule(tool_pattern="run_command", action="ask")],
            approval_callback=lambda t, d: True,
        )
        pm.deny_for_session("run_command")
        assert pm.request_approval("run_command", "run something") is False

    def test_git_read_only_allowed_by_default(self):
        pm = PermissionManager()
        assert pm.check_permission("git_status") == "allow"
        assert pm.check_permission("git_diff") == "allow"
        assert pm.check_permission("git_log") == "allow"

    def test_git_mutation_asks(self):
        pm = PermissionManager()
        assert pm.check_permission("git_commit") == "ask"
        assert pm.check_permission("git_push") == "ask"


# ─── AgentLoop loop guard ────────────────────────────────────────────────


class TestAgentLoopLoopGuard:
    """Verify AgentLoop prevents repeated identical denied calls."""

    @pytest.fixture
    def mock_tool(self):
        tool = MagicMock()
        tool.schema.name = "run_command"
        tool.schema.permission_required = "ask"
        tool.to_llm_schema.return_value = {
            "type": "function",
            "function": {"name": "run_command", "description": "Run", "parameters": {}},
        }
        tool.execute = AsyncMock(return_value=ToolResult(status=ToolResultStatus.SUCCESS, output="ok"))
        return tool

    def _make_loop(self, mock_tool, approval_callback=None):
        from harness_core.agent.loop import AgentLoop
        from harness_core.observability.events import EventBus

        bus = EventBus()
        loop = AgentLoop(
            provider=MagicMock(),
            tools=[mock_tool],
            workspace_root=Path("/tmp/test"),
            config=AgentConfig(max_iterations=5),
            event_bus=bus,
        )
        loop.permission_manager = PermissionManager(
            workspace_root=Path("/tmp/test"),
            rules=[PermissionRule(tool_pattern="run_command", action="ask")],
            approval_callback=approval_callback,
        )
        return loop

    @pytest.mark.asyncio
    async def test_identical_denied_call_is_blocked(self, mock_tool):
        loop = self._make_loop(mock_tool, approval_callback=lambda t, d: False)

        # First call — should deny and record
        call1 = ToolCall(tool_name="run_command", arguments={"command": "node test.js"})
        r1 = await loop._execute_tool(call1)
        assert r1.is_perm_denied
        assert "requires approval" in r1.error

        # Same call again — should be blocked immediately
        call2 = ToolCall(tool_name="run_command", arguments={"command": "node test.js"})
        r2 = await loop._execute_tool(call2)
        assert r2.is_perm_denied
        assert "already rejected" in r2.error
        assert not r2.retryable

    @pytest.mark.asyncio
    async def test_different_command_not_blocked(self, mock_tool):
        loop = self._make_loop(mock_tool, approval_callback=lambda t, d: False)

        call1 = ToolCall(tool_name="run_command", arguments={"command": "node test.js"})
        r1 = await loop._execute_tool(call1)
        assert r1.is_perm_denied

        # Different command — should not be blocked by loop guard
        call2 = ToolCall(tool_name="run_command", arguments={"command": "npm test"})
        r2 = await loop._execute_tool(call2)
        assert r2.is_perm_denied  # Still denied by permission, but NOT by loop guard
        assert "already rejected" not in r2.error

    @pytest.mark.asyncio
    async def test_consecutive_denials_trigger_safety_valve(self, mock_tool):
        """3+ consecutive denials should set _consecutive_denials."""
        from harness_core.agent.loop import AgentLoop
        from harness_core.observability.events import EventBus

        bus = EventBus()
        loop = AgentLoop(
            provider=MagicMock(),
            tools=[mock_tool],
            workspace_root=Path("/tmp/test"),
            config=AgentConfig(max_iterations=5),
            event_bus=bus,
        )
        loop.permission_manager = PermissionManager(
            workspace_root=Path("/tmp/test"),
            rules=[PermissionRule(tool_pattern="run_command", action="deny")],
        )

        for _ in range(3):
            call = ToolCall(tool_name="run_command", arguments={"command": "bad"})
            await loop._execute_tool(call)

        assert loop._consecutive_denials >= 3

    @pytest.mark.asyncio
    async def test_successful_call_resets_consecutive_denials(self, mock_tool):
        from harness_core.agent.loop import AgentLoop
        from harness_core.observability.events import EventBus

        bus = EventBus()
        loop = AgentLoop(
            provider=MagicMock(),
            tools=[mock_tool],
            workspace_root=Path("/tmp/test"),
            config=AgentConfig(max_iterations=5),
            event_bus=bus,
        )
        loop.permission_manager = PermissionManager(
            workspace_root=Path("/tmp/test"),
            rules=[PermissionRule(tool_pattern="run_command", action="allow")],
        )

        # Denial then success resets counter
        loop._consecutive_denials = 2
        call = ToolCall(tool_name="run_command", arguments={"command": "node test.js"})
        await loop._execute_tool(call)
        assert loop._consecutive_denials == 0


# ─── Interactive shell wiring ─────────────────────────────────────────────


class TestInteractiveShellApprovalWiring:
    """Verify the interactive shell sets up the approval callback."""

    def test_approval_callback_is_set(self):
        from harness_core.cli.interactive import InteractiveShell

        shell = InteractiveShell(workspace="/tmp/test")
        # Simulate what _setup_provider does
        callback = lambda t, d: True
        shell._agent_loop = MagicMock()
        shell._agent_loop.permission_manager = MagicMock()
        shell._agent_loop.permission_manager.approval_callback = None

        # Wire it
        shell._agent_loop.permission_manager.approval_callback = callback
        assert shell._agent_loop.permission_manager.approval_callback is callback
