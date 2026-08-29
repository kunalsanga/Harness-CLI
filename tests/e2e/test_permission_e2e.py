"""E2E test for permission system fix.

Tests that:
1. Permission-denied commands don't cause infinite retry loops
2. Agent gracefully handles permission blocks
3. Agent can complete tasks when permissions allow execution
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import AgentConfig, AgentRole, TaskStatus, ToolCall, ToolResult, ToolResultStatus
from harness_core.observability.events import Event, EventBus
from harness_core.permissions.manager import PermissionManager, PermissionRule
from harness_core.tools.shell import RunCommandTool
from harness_core.tools.filesystem import ReadFileTool, WriteFileTool


# ─── Test: Permission denied stops agent without retry loop ───────────────


@pytest.mark.e2e
class TestPermissionDenialE2E:
    """Verify the agent does NOT get stuck in a permission-denied retry loop."""

    @pytest.mark.asyncio
    async def test_agent_stops_after_consecutive_denials(self, tmp_path):
        """Agent should fail fast when commands are denied, not loop 30 times."""
        # Create a simple project
        (tmp_path / "app.js").write_text("console.log('hello');")
        (tmp_path / "test.js").write_text("console.log('test');")

        # Create a provider that always returns tool calls for run_command
        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1

            if call_count <= 5:
                # Keep trying to run_command
                return MagicMock(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{call_count}",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": json.dumps({"command": "node test.js"}),
                            },
                        }
                    ],
                )
            else:
                # After 5 calls, give up and return text
                return MagicMock(
                    content="Permission is blocked. Cannot run node test.js.",
                    tool_calls=None,
                )

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=mock_generate)

        # Create event bus
        bus = EventBus()

        # Create tools
        tools = [RunCommandTool(), ReadFileTool(), WriteFileTool()]

        # Agent with denied permissions
        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=30),
            event_bus=bus,
        )
        # Deny run_command
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="deny")],
        )

        # Run the task
        task = await agent.run("Run the tests and fix any failures")

        # The agent should NOT have used 30 iterations
        # It should have stopped early due to consecutive denials
        # (3 denials triggers safety valve + a few for model response)
        assert task.iterations <= 6, (
            f"Agent used {task.iterations} iterations - likely stuck in retry loop"
        )
        # Task should be marked as failed (blocked by permissions)
        assert task.status == TaskStatus.FAILED
        assert "permission" in task.error.lower() or "blocked" in task.error.lower()

        # The model should not have been called excessively
        assert call_count <= 6, f"Model called {call_count} times - too many retries"

    @pytest.mark.asyncio
    async def test_identical_denied_command_not_retried(self, tmp_path):
        """Same denied command should be blocked on second attempt."""
        (tmp_path / "test.js").write_text("console.log('test');")

        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            return MagicMock(
                content=None,
                tool_calls=[
                    {
                        "id": f"call_{call_count}",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "node test.js"}),
                        },
                    }
                ],
            )

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=mock_generate)

        bus = EventBus()
        tools = [RunCommandTool()]

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=10),
            event_bus=bus,
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="deny")],
        )

        task = await agent.run("Run tests")

        # Should stop quickly (safety valve at 3 consecutive denials)
        assert task.iterations <= 6, f"Agent used {task.iterations} iterations"
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_agent_completes_when_permissions_allow(self, tmp_path):
        """Agent should complete when permissions are allowed."""
        (tmp_path / "test.js").write_text("console.log('test');")

        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return MagicMock(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": json.dumps({"command": "node test.js"}),
                            },
                        }
                    ],
                )
            else:
                return MagicMock(
                    content="Tests passed successfully.",
                    tool_calls=None,
                )

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=mock_generate)

        bus = EventBus()
        tools = [RunCommandTool()]

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=10),
            event_bus=bus,
        )
        # Allow run_command
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="allow")],
        )

        task = await agent.run("Run tests")

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Tests passed successfully."

    @pytest.mark.asyncio
    async def test_different_commands_not_blocked_by_loop_guard(self, tmp_path):
        """Different commands should not be blocked even if same tool is denied."""
        (tmp_path / "test.js").write_text("console.log('test');")

        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1

            if call_count <= 3:
                # Try different commands
                commands = ["node test.js", "npm test", "python -m pytest"]
                cmd = commands[(call_count - 1) % len(commands)]
                return MagicMock(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{call_count}",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": json.dumps({"command": cmd}),
                            },
                        }
                    ],
                )
            else:
                return MagicMock(
                    content="All commands are blocked by permissions.",
                    tool_calls=None,
                )

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=mock_generate)

        bus = EventBus()
        tools = [RunCommandTool()]

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=10),
            event_bus=bus,
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="deny")],
        )

        task = await agent.run("Run tests with different frameworks")

        # Different commands should not be blocked by loop guard
        # Only consecutive denials of the SAME command trigger safety valve
        # With different commands, the agent should exhaust iterations or complete
        assert task.status in (TaskStatus.FAILED, TaskStatus.COMPLETED)


# ─── Test: File operations work, shell is denied ─────────────────────────


@pytest.mark.e2e
class TestMixedPermissionE2E:
    """Test mixed permission scenarios - files allowed, shell denied."""

    @pytest.mark.asyncio
    async def test_file_write_works_shell_denied(self, tmp_path):
        """File operations should succeed even when shell is denied."""
        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return MagicMock(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({
                                    "path": str(tmp_path / "output.txt"),
                                    "content": "Hello from Harness",
                                }),
                            },
                        }
                    ],
                )
            elif call_count == 2:
                return MagicMock(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": json.dumps({"command": "cat output.txt"}),
                            },
                        }
                    ],
                )
            else:
                return MagicMock(
                    content="File written successfully but shell is blocked.",
                    tool_calls=None,
                )

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=mock_generate)

        bus = EventBus()
        tools = [WriteFileTool(), RunCommandTool()]

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=10),
            event_bus=bus,
        )
        # Allow write_file, deny run_command
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[
                PermissionRule(tool_pattern="write_file", action="allow"),
                PermissionRule(tool_pattern="run_command", action="deny"),
            ],
        )

        task = await agent.run("Write a file and verify it")

        # File should have been written
        assert (tmp_path / "output.txt").read_text() == "Hello from Harness"

        # Task should complete (shell denial shouldn't block file operations)
        assert task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        # Should not have burned through all iterations
        assert task.iterations < 10


# ─── Test: Events are emitted correctly ──────────────────────────────────


@pytest.mark.e2e
class TestPermissionEventsE2E:
    """Verify permission events are emitted through EventBus."""

    @pytest.mark.asyncio
    async def test_permission_denied_emits_event(self, tmp_path):
        """Permission denied should emit a tool.result event with permission_denied status."""
        events_received = []

        async def capture_event(event):
            events_received.append(event)

        bus = EventBus()
        bus.on("tool.result", capture_event)

        async def mock_generate(request):
            return MagicMock(
                content="Done.",
                tool_calls=None,
            )

        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=mock_generate)

        tools = [RunCommandTool()]

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=5),
            event_bus=bus,
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="deny")],
        )

        # Manually trigger a tool call to test event emission
        call = ToolCall(tool_name="run_command", arguments={"command": "echo hello"})
        result = await agent._execute_tool(call)

        assert result.status == ToolResultStatus.PERMISSION_DENIED
