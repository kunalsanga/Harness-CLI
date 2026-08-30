"""Regression tests: tool execution failures must never be reported as task success.

These tests cover the critical invariant:
    TOOL FAILURE ≠ TASK SUCCESS

When a shell command exits non-zero, the agent loop must NOT mark the task
as COMPLETED. The runtime execution results are the source of truth, not
the model's text response.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import (
    AgentConfig,
    TaskStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.observability.events import Event, EventBus
from harness_core.permissions.manager import PermissionManager, PermissionRule
from harness_core.tools.shell import RunCommandTool


# ─── ToolResult structured failure info ───────────────────────────────────


class TestToolResultStructuredFields:
    """Verify ToolResult preserves exit_code, stderr, and failure category."""

    def test_success_has_exit_code_zero(self):
        r = ToolResult(
            status=ToolResultStatus.SUCCESS,
            output="ok",
            exit_code=0,
            stderr=None,
        )
        assert r.exit_code == 0
        assert r.stderr is None
        assert not r.execution_failed
        assert r.failure_category == "success"

    def test_nonzero_exit_code_is_failure(self):
        r = ToolResult(
            status=ToolResultStatus.ERROR,
            output="error output",
            error="Command failed with exit code 1",
            exit_code=1,
            stderr="actual error message",
        )
        assert r.exit_code == 1
        assert r.stderr == "actual error message"
        assert r.execution_failed
        assert r.failure_category == "execution_error"

    def test_exit_code_preserved(self):
        for code in [0, 1, 2, 127, 137]:
            r = ToolResult(
                status=ToolResultStatus.SUCCESS if code == 0 else ToolResultStatus.ERROR,
                output="",
                exit_code=code,
            )
            assert r.exit_code == code

    def test_stderr_preserved(self):
        stderr = "Error: Cannot find module 'missing'"
        r = ToolResult(
            status=ToolResultStatus.ERROR,
            output="",
            error="Command failed",
            exit_code=1,
            stderr=stderr,
        )
        assert r.stderr == stderr

    def test_permission_denied_is_not_execution_failure(self):
        r = ToolResult(
            status=ToolResultStatus.PERMISSION_DENIED,
            output="",
            error="Permission denied",
        )
        assert not r.execution_failed
        assert r.failure_category == "permission_denied"

    def test_timeout_is_execution_failure(self):
        r = ToolResult(
            status=ToolResultStatus.TIMEOUT,
            output="",
            error="Timed out after 30s",
        )
        assert r.execution_failed
        assert r.failure_category == "timeout"

    def test_tool_error_without_exit_code(self):
        r = ToolResult(
            status=ToolResultStatus.ERROR,
            output="",
            error="Unknown tool: foo",
        )
        assert r.execution_failed
        assert r.failure_category == "tool_error"


# ─── ShellTool preserves structured info ──────────────────────────────────


class TestShellToolStructuredResult:
    """Verify RunCommandTool populates exit_code and stderr."""

    @pytest.mark.asyncio
    async def test_successful_command(self, tmp_path: Path):
        tool = RunCommandTool(working_directory=str(tmp_path))
        result = await tool.execute({"command": "echo hello"})
        assert result.status == ToolResultStatus.SUCCESS
        assert result.exit_code == 0
        assert "hello" in result.output

    @pytest.mark.asyncio
    async def test_nonzero_exit_code(self, tmp_path: Path):
        script = tmp_path / "fail.py"
        script.write_text("import sys; sys.exit(1)")
        tool = RunCommandTool(working_directory=str(tmp_path))
        result = await tool.execute({"command": f"python {script}"})
        assert result.status == ToolResultStatus.ERROR
        assert result.exit_code == 1
        assert result.error is not None
        assert "exit code 1" in result.error

    @pytest.mark.asyncio
    async def test_exit_code_preserved(self, tmp_path: Path):
        # Use relative path to avoid Windows cmd /c path length issues
        script = tmp_path / "exit42.py"
        script.write_text("import sys; sys.exit(42)")
        tool = RunCommandTool(working_directory=str(tmp_path))
        result = await tool.execute({"command": "python exit42.py"})
        assert result.exit_code == 42
        assert "42" in result.error

    @pytest.mark.asyncio
    async def test_stderr_captured(self, tmp_path: Path):
        script = tmp_path / "stderr_test.py"
        script.write_text("import sys; sys.stderr.write('err_msg'); sys.exit(1)")
        tool = RunCommandTool(working_directory=str(tmp_path))
        result = await tool.execute({"command": "python stderr_test.py"})
        assert result.stderr is not None
        assert "err_msg" in result.stderr

    @pytest.mark.asyncio
    async def test_command_not_found(self, tmp_path: Path):
        tool = RunCommandTool(working_directory=str(tmp_path))
        result = await tool.execute({"command": "nonexistent_command_xyz"})
        # Command not found should be an error
        assert result.status == ToolResultStatus.ERROR
        assert result.exit_code is not None
        assert result.exit_code != 0


# ─── AgentLoop blocks false completion ────────────────────────────────────


class TestAgentLoopFalseCompletionPrevention:
    """The agent loop must NOT mark task as COMPLETED when tool calls failed."""

    @pytest.fixture
    def make_agent(self, tmp_path: Path):
        """Factory to create an agent with mock provider."""

        def _make(provider_generate, max_iterations=10):
            tools = [RunCommandTool()]
            agent = AgentLoop(
                provider=MagicMock(),
                tools=tools,
                workspace_root=tmp_path,
                config=AgentConfig(max_iterations=max_iterations),
                event_bus=EventBus(),
            )
            agent.provider.generate = AsyncMock(side_effect=provider_generate)
            # Allow all commands
            agent.permission_manager = PermissionManager(
                workspace_root=tmp_path,
                rules=[PermissionRule(tool_pattern="run_command", action="allow")],
            )
            return agent

        return _make

    @pytest.mark.asyncio
    async def test_failed_command_blocks_completion(self, tmp_path: Path, make_agent):
        """If a command fails and model claims success, task should FAIL."""
        # Create a failing test
        (tmp_path / "test.js").write_text("process.exit(1);")

        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # call_count 1 = planning phase (return a plan)
                return MagicMock(content="1. Run tests\n2. Fix issues", tool_calls=None)
            elif call_count == 2:
                # Run the failing test
                return MagicMock(
                    content=None,
                    tool_calls=[{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "node test.js", "cwd": str(tmp_path)}),
                        },
                    }],
                )
            else:
                # Model claims success despite failure
                return MagicMock(
                    content="The project is complete and working.",
                    tool_calls=None,
                )

        agent = make_agent(mock_generate)
        task = await agent.run("Execute the test suite")

        # Task must NOT be completed — the command failed
        assert task.status == TaskStatus.FAILED
        assert task.error is not None
        assert "failed" in task.error.lower() or "cannot" in task.error.lower()

    @pytest.mark.asyncio
    async def test_successful_command_allows_completion(self, tmp_path: Path, make_agent):
        """If a command succeeds, task should be COMPLETED."""
        (tmp_path / "test.js").write_text("process.exit(0);")

        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # Planning phase
                return MagicMock(content="1. Run tests\n2. Verify", tool_calls=None)
            elif call_count == 2:
                return MagicMock(
                    content=None,
                    tool_calls=[{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "node test.js", "cwd": str(tmp_path)}),
                        },
                    }],
                )
            else:
                return MagicMock(
                    content="All tests pass.",
                    tool_calls=None,
                )

        agent = make_agent(mock_generate)
        task = await agent.run("Execute the test suite")

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "All tests pass."

    @pytest.mark.asyncio
    async def test_recovery_after_failure_allows_completion(self, tmp_path: Path, make_agent):
        """Agent can recover: fail → fix → succeed → completion allowed."""
        test_file = tmp_path / "test.js"

        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                # Planning phase
                return MagicMock(content="1. Run tests\n2. Fix if needed\n3. Verify", tool_calls=None)
            elif call_count == 2:
                # First run: failing test
                test_file.write_text("process.exit(1);")
                return MagicMock(
                    content=None,
                    tool_calls=[{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "node test.js", "cwd": str(tmp_path)}),
                        },
                    }],
                )
            elif call_count == 3:
                # Agent "fixes" the code
                test_file.write_text("process.exit(0);")
                return MagicMock(
                    content=None,
                    tool_calls=[{
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "node test.js", "cwd": str(tmp_path)}),
                        },
                    }],
                )
            else:
                # Model reports success
                return MagicMock(
                    content="Tests now pass.",
                    tool_calls=None,
                )

        agent = make_agent(mock_generate)
        task = await agent.run("Repair issues and run suite")

        # The second run succeeded, so completion is allowed
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_tool_calls_allows_completion(self, tmp_path: Path, make_agent):
        """Tasks with no tool calls (planning) can complete normally."""
        async def mock_generate(request):
            return MagicMock(
                content="Here is my analysis of the project.",
                tool_calls=None,
            )

        agent = make_agent(mock_generate)
        task = await agent.run("Survey the project layout")

        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Here is my analysis of the project."

    @pytest.mark.asyncio
    async def test_permission_denied_does_not_block_completion(self, tmp_path: Path):
        """Permission denied on verification doesn't prevent completion of other work."""
        (tmp_path / "app.py").write_text("x = 1")

        tools = [RunCommandTool()]
        agent = AgentLoop(
            provider=MagicMock(),
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=10),
            event_bus=EventBus(),
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="deny")],
        )

        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Try to run (will be denied)
                return MagicMock(
                    content=None,
                    tool_calls=[{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "pytest"}),
                        },
                    }],
                )
            else:
                # Model reports what was done (permission blocked verification)
                return MagicMock(
                    content="Code was written. Verification blocked by permissions.",
                    tool_calls=None,
                )

        agent.provider.generate = AsyncMock(side_effect=mock_generate)
        task = await agent.run("Write code and verify")

        # Permission denied is NOT an execution failure, so completion is allowed
        assert task.status == TaskStatus.COMPLETED


# ─── Repetition detection ─────────────────────────────────────────────────


class TestRepetitionDetection:
    """Repeated identical failures should be detected."""

    @pytest.mark.asyncio
    async def test_repeated_same_failure_detected(self, tmp_path: Path):
        """Same failing command 3x without code changes → blocked."""
        tools = [RunCommandTool()]
        agent = AgentLoop(
            provider=MagicMock(),
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=20),
            event_bus=EventBus(),
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="allow")],
        )

        # Create a script that always fails
        fail_script = tmp_path / "always_fail.py"
        fail_script.write_text("import sys; sys.exit(1)")

        # Manually execute the same failing command multiple times
        for _ in range(3):
            call = ToolCall(
                tool_name="run_command",
                arguments={"command": "python always_fail.py", "cwd": str(tmp_path)},
            )
            result = await agent._execute_tool(call)

        # After 3 failures, the next identical call should be blocked
        call = ToolCall(
            tool_name="run_command",
            arguments={"command": "python always_fail.py", "cwd": str(tmp_path)},
        )
        result = await agent._execute_tool(call)
        assert result.status == ToolResultStatus.ERROR
        assert "already failed" in result.error.lower()
        assert not result.retryable

    @pytest.mark.asyncio
    async def test_different_command_not_affected(self, tmp_path: Path):
        """Different command should not be blocked by repetition detection."""
        tools = [RunCommandTool()]
        agent = AgentLoop(
            provider=MagicMock(),
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=20),
            event_bus=EventBus(),
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="allow")],
        )

        fail_script = tmp_path / "always_fail.py"
        fail_script.write_text("import sys; sys.exit(1)")

        # Fail on one command
        for _ in range(3):
            call = ToolCall(
                tool_name="run_command",
                arguments={"command": "python always_fail.py", "cwd": str(tmp_path)},
            )
            await agent._execute_tool(call)

        # Different command should still work
        call = ToolCall(
            tool_name="run_command",
            arguments={"command": "echo different", "cwd": str(tmp_path)},
        )
        result = await agent._execute_tool(call)
        assert result.status == ToolResultStatus.SUCCESS


# ─── Full integration: fail → fix → succeed ──────────────────────────────


class TestFullFailFixSucceedPipeline:
    """End-to-end test: command fails → agent diagnoses → fixes → verifies → completes."""

    @pytest.mark.asyncio
    async def test_full_recovery_pipeline(self, tmp_path: Path):
        """Simulate the full recovery flow."""
        test_file = tmp_path / "test.js"

        # Track state
        test_state = {"exit_code": 1}
        call_count = 0

        async def mock_generate(request):
            nonlocal call_count
            call_count += 1

            # Look at conversation history to understand what happened
            messages = request.messages
            last_tool_result = ""
            for msg in reversed(messages):
                if msg.get("role") == "tool":
                    last_tool_result = msg.get("content", "")
                    break

            if call_count <= 1:
                # Planning phase
                return MagicMock(content="1. Run test\n2. Fix if needed\n3. Verify", tool_calls=None)
            elif call_count == 2:
                # First: run the test (will fail)
                test_file.write_text("process.exit(1);")
                return MagicMock(
                    content=None,
                    tool_calls=[{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "node test.js", "cwd": str(tmp_path)}),
                        },
                    }],
                )
            elif call_count == 3:
                # Agent sees failure, fixes the code
                test_file.write_text("process.exit(0);")
                return MagicMock(
                    content=None,
                    tool_calls=[{
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "node test.js", "cwd": str(tmp_path)}),
                        },
                    }],
                )
            else:
                # Agent sees success, reports completion
                return MagicMock(
                    content="The test now passes. Task complete.",
                    tool_calls=None,
                )

        tools = [RunCommandTool()]
        agent = AgentLoop(
            provider=MagicMock(),
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=10),
            event_bus=EventBus(),
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="allow")],
        )
        agent.provider.generate = AsyncMock(side_effect=mock_generate)

        task = await agent.run("Repair and retest")

        # Task should complete because the second run succeeded
        assert task.status == TaskStatus.COMPLETED
        assert len(task.tool_calls) == 2
        # First tool call failed, second succeeded
        assert task.tool_calls[0].result.status == ToolResultStatus.ERROR
        assert task.tool_calls[1].result.status == ToolResultStatus.SUCCESS


# ─── Existing permission tests compatibility ──────────────────────────────


class TestExistingPermissionTestsStillPass:
    """Ensure the permission system still works correctly."""

    @pytest.mark.asyncio
    async def test_permission_denied_not_confused_with_execution_error(self, tmp_path: Path):
        """Permission denied ≠ execution error."""
        tools = [RunCommandTool()]
        agent = AgentLoop(
            provider=MagicMock(),
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=5),
            event_bus=EventBus(),
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="deny")],
            autonomous_mode=False,
        )

        call = ToolCall(
            tool_name="run_command",
            arguments={"command": "echo hello"},
        )
        result = await agent._execute_tool(call)


        assert result.status == ToolResultStatus.PERMISSION_DENIED
        assert not result.execution_failed
        assert result.failure_category == "permission_denied"

    @pytest.mark.asyncio
    async def test_consecutive_denial_safety_valve(self, tmp_path: Path):
        """3+ consecutive denials trigger safety valve."""
        tools = [RunCommandTool()]
        agent = AgentLoop(
            provider=MagicMock(),
            tools=tools,
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=30),
            event_bus=EventBus(),
        )
        agent.permission_manager = PermissionManager(
            workspace_root=tmp_path,
            rules=[PermissionRule(tool_pattern="run_command", action="deny")],
            autonomous_mode=False,
        )

        for _ in range(3):
            call = ToolCall(
                tool_name="run_command",
                arguments={"command": "echo test"},
            )
            await agent._execute_tool(call)

        assert agent._consecutive_denials >= 3
