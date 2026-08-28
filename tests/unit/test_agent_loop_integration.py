"""Regression tests for CLI → AgentLoop integration.

These tests verify that AgentLoop can be constructed with a full tool set
and that the tools are properly accessible through the loop's tool registry.
This class of bugs was first caught in the real end-to-end CLI run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import (
    AgentConfig,
    AgentRole,
    Task,
    TaskStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider
from harness_core.tools.base import Tool, ToolSchema
from harness_core.tools.filesystem import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from harness_core.tools.git import GitDiffTool, GitLogTool, GitStatusTool
from harness_core.tools.search import GlobTool, GrepTool
from harness_core.tools.shell import RunCommandTool


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_full_tool_set() -> list[Tool]:
    """Build the same tool set the CLI wires into AgentLoop."""
    return [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        ListFilesTool(),
        GlobTool(),
        GrepTool(),
        RunCommandTool(),
        GitStatusTool(),
        GitDiffTool(),
        GitLogTool(),
    ]


class MockProvider(ModelProvider):
    """Minimal mock provider for unit tests."""

    def __init__(self) -> None:
        self._call_count = 0
        self._responses: list[CompletionResponse] = []

    @property
    def name(self) -> str:
        return "mock"

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
        else:
            resp = CompletionResponse(content="Done.", model="mock", provider="mock")
        self._call_count += 1
        return resp

    async def stream(self, request: CompletionRequest):
        """Mock stream — not used in tests."""
        yield CompletionResponse(content="", model="mock", provider="mock")

    async def list_models(self) -> list[Any]:
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


# ── Tests ────────────────────────────────────────────────────────────────────


class TestAgentLoopConstruction:
    """Regression: AgentLoop must accept a tools list."""

    def test_agent_loop_requires_tools(self):
        """AgentLoop.__init__ must have a 'tools' parameter."""
        import inspect
        sig = inspect.signature(AgentLoop.__init__)
        assert "tools" in sig.parameters, (
            "AgentLoop.__init__ is missing the required 'tools' parameter"
        )

    def test_agent_loop_construction_with_tools(self):
        """AgentLoop can be constructed with a full tool set."""
        provider = MockProvider()
        tools = _build_full_tool_set()

        loop = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
            config=AgentConfig(),
            event_bus=EventBus(),
        )

        # All tools should be registered by name
        assert len(loop.tools) == len(tools)
        for tool in tools:
            assert tool.schema.name in loop.tools

    def test_agent_loop_tool_schemas_include_all_tools(self):
        """_tool_schemas() returns LLM-compatible schemas for every tool."""
        provider = MockProvider()
        tools = _build_full_tool_set()

        loop = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
        )

        schemas = loop._tool_schemas()
        schema_names = {s["function"]["name"] for s in schemas}
        expected_names = {t.schema.name for t in tools}
        assert schema_names == expected_names

    def test_agent_loop_empty_tools_list(self):
        """AgentLoop works with an empty tool list (no tools)."""
        provider = MockProvider()
        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=Path.cwd(),
        )
        assert len(loop.tools) == 0
        assert loop._tool_schemas() == []


class TestAgentLoopToolExecution:
    """Verify tool execution through the agent loop works correctly."""

    @pytest.mark.asyncio
    async def test_execute_known_tool(self):
        """AgentLoop can dispatch to a registered tool."""
        provider = MockProvider()
        tools = [ReadFileTool()]

        loop = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
        )

        call = ToolCall(
            id="test-1",
            tool_name="read_file",
            arguments={"path": "pyproject.toml"},
        )

        result = await loop._execute_tool(call)
        assert result.status in (ToolResultStatus.SUCCESS, ToolResultStatus.ERROR)
        assert call.result is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self):
        """Unknown tool name returns an error result, not an exception."""
        provider = MockProvider()
        tools = [ReadFileTool()]

        loop = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
        )

        call = ToolCall(
            id="test-2",
            tool_name="nonexistent_tool",
            arguments={},
        )

        result = await loop._execute_tool(call)
        assert result.status == ToolResultStatus.ERROR
        assert "Unknown tool" in result.error


class TestAgentLoopEndToEnd:
    """Integration test: agent loop completes a simple goal with mocked provider."""

    @pytest.mark.asyncio
    async def test_agent_loop_completes_without_tools(self):
        """Agent loop completes when model returns content (no tool calls)."""
        provider = MockProvider()
        provider._responses = [
            CompletionResponse(
                content="The project is a Python CLI tool.",
                model="mock",
                provider="mock",
            )
        ]

        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=Path.cwd(),
        )

        task = await loop.run("Explain this project")
        assert task.status == TaskStatus.COMPLETED
        assert task.result is not None
        assert "Python" in task.result
        assert task.iterations == 1

    @pytest.mark.asyncio
    async def test_agent_loop_with_tool_calls(self):
        """Agent loop executes tool calls then completes."""
        read_tool = ReadFileTool()
        tools = [read_tool]

        # First response: call read_file; second response: final answer
        first_resp = CompletionResponse(
            content=None,
            model="mock",
            provider="mock",
            tool_calls=[
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "pyproject.toml"}),
                    },
                }
            ],
        )
        second_resp = CompletionResponse(
            content="The project uses Python with rich for CLI output.",
            model="mock",
            provider="mock",
        )

        provider = MockProvider()
        provider._responses = [first_resp, second_resp]

        loop = AgentLoop(
            provider=provider,
            tools=tools,
            workspace_root=Path.cwd(),
        )

        task = await loop.run("Read pyproject.toml and explain the project")
        assert task.status == TaskStatus.COMPLETED
        assert len(task.tool_calls) == 1
        assert task.tool_calls[0].tool_name == "read_file"
        assert task.tool_calls[0].result is not None

    @pytest.mark.asyncio
    async def test_agent_loop_event_emission(self):
        """Agent loop emits events through the EventBus."""
        provider = MockProvider()
        provider._responses = [
            CompletionResponse(content="Done.", model="mock", provider="mock")
        ]

        events_received: list[Event] = []
        bus = EventBus()
        bus.on("*", lambda e: events_received.append(e))

        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=Path.cwd(),
            event_bus=bus,
        )

        task = await loop.run("Do something")
        event_types = [e.type for e in events_received]
        assert "task.started" in event_types
        assert "task.completed" in event_types


class TestAgentLoopPermissions:
    """Verify the permission system is still wired through the agent loop."""

    def test_permission_manager_is_initialized(self):
        """AgentLoop creates a PermissionManager with workspace root."""
        provider = MockProvider()
        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=Path("/tmp/test-workspace"),
        )
        assert loop.permission_manager is not None
        assert loop.permission_manager.workspace_root == Path("/tmp/test-workspace")

    def test_verification_engine_is_initialized(self):
        """AgentLoop creates a VerificationEngine."""
        provider = MockProvider()
        loop = AgentLoop(
            provider=provider,
            tools=[],
            workspace_root=Path.cwd(),
        )
        assert loop.verification_engine is not None


class TestFullToolSetConstruction:
    """Verify the CLI's tool set can be built and passed to AgentLoop."""

    def test_all_cli_tools_instantiate(self):
        """Every tool the CLI uses can be instantiated."""
        tools = _build_full_tool_set()
        assert len(tools) == 10

        tool_names = [t.schema.name for t in tools]
        expected = [
            "read_file", "write_file", "edit_file", "list_files",
            "glob", "grep", "run_command",
            "git_status", "git_diff", "git_log",
        ]
        assert tool_names == expected

    def test_all_tools_have_schemas(self):
        """Every tool exposes a valid schema."""
        for tool in _build_full_tool_set():
            schema = tool.schema
            assert isinstance(schema, ToolSchema)
            assert schema.name
            assert schema.description
            assert isinstance(schema.parameters, dict)

    def test_all_tools_have_permission_required(self):
        """Every tool declares a permission_required level."""
        for tool in _build_full_tool_set():
            assert tool.schema.permission_required in ("allow", "ask", "deny")
