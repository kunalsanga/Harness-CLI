"""Unit and regression tests for Claude Code parity improvements in Harness AgentLoop."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from typing import Any

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import AgentConfig, AgentRole, Task, ToolCall
from harness_core.observability.events import Event, EventBus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)
from harness_core.tools.filesystem import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    WriteFileTool,
)
from harness_core.tools.search import GlobTool, GrepTool
from harness_core.tools.shell import RunCommandTool


class DeterministicMockProvider(ModelProvider):
    """Deterministic mock provider that simulates specific turns of an agent execution."""

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self.responses = list(responses)
        self.turn = 0
        self.recorded_requests: list[CompletionRequest] = []

    @property
    def name(self) -> str:
        return "mock"

    async def health_check(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="mock-model", name="Mock Model", provider=self.name, supports_tools=True)]

    async def stream(self, request: CompletionRequest):
        resp = await self.generate(request)
        yield resp

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.recorded_requests.append(request)
        # Recognize planning phase so turn-by-turn responses are reserved for the execution loop
        is_planning = any(
            "planning an engineering task" in (m.get("content") or "").lower()
            for m in request.messages
        )
        if is_planning:
            return CompletionResponse(
                content="1. Inspect repository structure\n2. Implement the requested changes\n3. Verify changed behavior"
            )

        if self.turn < len(self.responses):
            resp = self.responses[self.turn]
            self.turn += 1
            return resp
        return CompletionResponse(content="Task completed successfully.")


@pytest.mark.asyncio
async def test_tool_name_aliasing(tmp_path: Path):
    """Claude Code tool names (Bash, Read, Edit, Write) are resolved to Harness tools."""
    tools = [ReadFileTool(), WriteFileTool(), EditFileTool(), RunCommandTool(working_directory=str(tmp_path))]
    loop = AgentLoop(
        provider=DeterministicMockProvider([]),
        tools=tools,
        workspace_root=tmp_path,
    )

    # Test file
    test_file = tmp_path / "hello.txt"
    test_file.write_text("Hello World", encoding="utf-8")

    # 1. 'Read' alias for 'read_file'
    read_call = ToolCall(tool_name="Read", arguments={"path": str(test_file)})
    result = await loop._execute_tool(read_call)
    assert result.success is True
    assert "Hello World" in result.output
    assert read_call.tool_name == "read_file"

    # 2. 'Edit' alias for 'edit_file'
    edit_call = ToolCall(
        tool_name="Edit",
        arguments={"path": str(test_file), "old_string": "World", "new_string": "Harness"},
    )
    result = await loop._execute_tool(edit_call)
    assert result.success is True
    assert test_file.read_text(encoding="utf-8") == "Hello Harness"
    assert edit_call.tool_name == "edit_file"

    # 3. 'Write' alias for 'write_file'
    write_call = ToolCall(
        tool_name="Write",
        arguments={"path": str(tmp_path / "new.txt"), "content": "Created via Write"},
    )
    result = await loop._execute_tool(write_call)
    assert result.success is True
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "Created via Write"
    assert write_call.tool_name == "write_file"


@pytest.mark.asyncio
async def test_tool_argument_aliasing(tmp_path: Path):
    """Alternative parameter names (file_path, old_str, new_str, cmd) are mapped seamlessly."""
    tools = [ReadFileTool(), EditFileTool(), RunCommandTool(working_directory=str(tmp_path))]
    loop = AgentLoop(
        provider=DeterministicMockProvider([]),
        tools=tools,
        workspace_root=tmp_path,
    )

    sample = tmp_path / "sample.py"
    sample.write_text("x = 1\n", encoding="utf-8")

    # file_path -> path for read_file
    call = ToolCall(tool_name="read_file", arguments={"file_path": str(sample)})
    res = await loop._execute_tool(call)
    assert res.success is True
    assert "x = 1" in res.output

    # old_str/new_str -> old_string/new_string for edit_file
    edit_call = ToolCall(
        tool_name="edit_file",
        arguments={"file_path": str(sample), "old_str": "x = 1", "new_str": "x = 42"},
    )
    res = await loop._execute_tool(edit_call)
    assert res.success is True
    assert sample.read_text(encoding="utf-8") == "x = 42\n"

    # cmd -> command for run_command
    cmd_call = ToolCall(tool_name="run_command", arguments={"cmd": "echo ok"})
    res = await loop._execute_tool(cmd_call)
    assert res.success is True
    assert "ok" in res.output.lower()


@pytest.mark.asyncio
async def test_malformed_argument_handling_does_not_crash(tmp_path: Path):
    """When a model produces malformed tool arguments, AgentLoop handles it without crashing."""
    # First turn: model returns malformed JSON arguments
    # Second turn: model gets error message and provides valid arguments
    # Third turn: model completes task
    sample = tmp_path / "data.txt"
    sample.write_text("Initial", encoding="utf-8")

    responses = [
        CompletionResponse(
            content="I will read data.txt",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{path: 'broken json no quotes}",  # invalid JSON
                    },
                }
            ],
        ),
        CompletionResponse(
            content="Correcting my call with valid JSON",
            tool_calls=[
                {
                    "id": "c2",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": str(sample)}),
                    },
                }
            ],
        ),
        CompletionResponse(content="I have read data.txt and finished."),
    ]

    mock_provider = DeterministicMockProvider(responses)
    bus = EventBus()
    tools = [ReadFileTool()]
    loop = AgentLoop(
        provider=mock_provider,
        tools=tools,
        workspace_root=tmp_path,
        event_bus=bus,
        config=AgentConfig(max_iterations=5),
    )

    task = await loop.run("Read data.txt and verify its contents")
    assert len(task.tool_calls) == 2
    # First call recorded the malformed argument error
    assert task.tool_calls[0].result.success is False
    assert "Malformed arguments" in task.tool_calls[0].result.error

    # Second call succeeded
    assert task.tool_calls[1].result.success is True
    assert "Initial" in task.tool_calls[1].result.output


@pytest.mark.asyncio
async def test_content_extracted_tool_calls(tmp_path: Path):
    """Models emitting tool calls in text (XML tags or markdown codeblocks) are extracted and executed."""
    sample = tmp_path / "calc.py"
    sample.write_text("def add(a, b): return a - b\n", encoding="utf-8")

    # Model generates tool call inside XML <tool_call> tags in content
    xml_content = (
        "I need to fix the subtraction bug in calc.py.\n"
        "<tool_call>\n"
        f'{{"name": "edit_file", "arguments": {{"path": "{str(sample).replace("\\\\", "/")}", "old_string": "a - b", "new_string": "a + b"}}}}\n'
        "</tool_call>"
    )

    responses = [
        CompletionResponse(content=xml_content, tool_calls=None),
        CompletionResponse(content="The bug is fixed."),
    ]

    mock_provider = DeterministicMockProvider(responses)
    tools = [EditFileTool()]
    loop = AgentLoop(
        provider=mock_provider,
        tools=tools,
        workspace_root=tmp_path,
        config=AgentConfig(max_iterations=4),
    )

    task = await loop.run("Fix the addition function in calc.py")
    assert len(task.tool_calls) >= 1
    assert task.tool_calls[0].tool_name == "edit_file"
    assert task.tool_calls[0].result.success is True
    assert sample.read_text(encoding="utf-8") == "def add(a, b): return a + b\n"


def test_tool_turn_block_and_thought_preservation():
    """Turn blocks preserve assistant thoughts and group parallel tool calls."""
    tc1 = ToolCall(
        id="c1",
        tool_name="read_file",
        arguments={"path": "a.txt"},
        iteration=1,
        thought="I will read both files.",
    )
    from harness_core.agent.types import ToolResults
    tc1.result = ToolResults.success("content a")

    tc2 = ToolCall(
        id="c2",
        tool_name="read_file",
        arguments={"path": "b.txt"},
        iteration=1,
        thought="I will read both files.",
    )
    tc2.result = ToolResults.success("content b")

    asst_msg, tool_msgs = AgentLoop._tool_turn_block([tc1, tc2])
    assert asst_msg["role"] == "assistant"
    assert asst_msg["content"] == "I will read both files."
    assert len(asst_msg["tool_calls"]) == 2
    assert asst_msg["tool_calls"][0]["id"] == "c1"
    assert asst_msg["tool_calls"][1]["id"] == "c2"

    assert len(tool_msgs) == 2
    assert tool_msgs[0]["role"] == "tool"
    assert tool_msgs[0]["tool_call_id"] == "c1"
    assert "content a" in tool_msgs[0]["content"]

    assert tool_msgs[1]["role"] == "tool"
    assert tool_msgs[1]["tool_call_id"] == "c2"
    assert "content b" in tool_msgs[1]["content"]


@pytest.mark.asyncio
async def test_deterministic_calculator_fix_e2e(tmp_path: Path):
    """Deterministic end-to-end bug fix scenario verifying tool calls, pytest execution, and verification."""
    # 1. Create broken calculator project
    calc_py = tmp_path / "calculator.py"
    calc_py.write_text(
        "def divide(a, b):\n"
        "    if b == 0:\n"
        "        return 0\n"
        "    return a / b\n",
        encoding="utf-8",
    )
    test_calc = tmp_path / "test_calculator.py"
    test_calc.write_text(
        "import pytest\n"
        "from calculator import divide\n\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5\n"
        "    with pytest.raises(ValueError):\n"
        "        divide(10, 0)\n",
        encoding="utf-8",
    )

    tools = [
        ReadFileTool(),
        EditFileTool(),
        RunCommandTool(working_directory=str(tmp_path)),
    ]

    event_bus = EventBus()
    events = []

    async def on_event(e: Event):
        events.append(e)

    event_bus.on("*", on_event)

    # Simulated model turns:
    # Turn 1: read calculator.py
    # Turn 2: edit calculator.py
    # Turn 3: run pytest to verify
    # Turn 4: final answer
    responses = [
        CompletionResponse(
            content="I will inspect calculator.py first.",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": str(calc_py)}),
                    },
                }
            ],
        ),
        CompletionResponse(
            content="Now I will fix the divide function to raise ValueError on zero divisor.",
            tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({
                            "path": str(calc_py),
                            "old_string": "    if b == 0:\n        return 0",
                            "new_string": "    if b == 0:\n        raise ValueError('Cannot divide by zero')",
                        }),
                    },
                }
            ],
        ),
        CompletionResponse(
            content="Running pytest to verify the fix.",
            tool_calls=[
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({"command": "pytest test_calculator.py -q"}),
                    },
                }
            ],
        ),
        CompletionResponse(
            content="All tests pass! The divide function now raises ValueError when dividing by zero."
        ),
    ]

    mock_provider = DeterministicMockProvider(responses)
    loop = AgentLoop(
        provider=mock_provider,
        tools=tools,
        workspace_root=tmp_path,
        event_bus=event_bus,
        config=AgentConfig(max_iterations=10),
    )

    task = await loop.run(
        "Fix the divide function in calculator.py so it raises ValueError when dividing by zero instead of returning 0."
    )

    # Verification of task outcome
    assert len(task.tool_calls) == 3
    tool_names = [tc.tool_name for tc in task.tool_calls]
    assert tool_names == ["read_file", "edit_file", "run_command"]
    assert all(tc.result.success for tc in task.tool_calls)

    # Verify calculator code actually modified
    content = calc_py.read_text(encoding="utf-8")
    assert "raise ValueError" in content

    # Verify event emission pipeline
    event_types = [e.type for e in events]
    assert "task.started" in event_types
    assert "tool.call" in event_types
    assert "tool.result" in event_types
    assert "task.completed" in event_types

    assert len([e for e in events if e.type == "tool.call"]) == 3


@pytest.mark.asyncio
async def test_tool_alias_precedence_and_registration_guard(tmp_path: Path):
    """Direct tool registrations take precedence, and aliases only resolve if target tool is registered."""
    from harness_core.tools.base import Tool, ToolSchema
    from harness_core.agent.types import ToolResult, ToolResults

    class CustomSearchTool(Tool):
        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(name="search", description="Custom search tool", parameters={"type": "object", "properties": {"q": {"type": "string"}}})

        async def execute(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResults.success(f"Custom search output: {arguments.get('q')}")

    # 1. Custom tool named 'search' takes precedence over alias 'search' -> 'grep'
    custom_tool = CustomSearchTool()
    loop_with_custom = AgentLoop(
        provider=DeterministicMockProvider([]),
        tools=[custom_tool],
        workspace_root=tmp_path,
    )
    call_custom = ToolCall(tool_name="search", arguments={"q": "antigravity"})
    res = await loop_with_custom._execute_tool(call_custom)
    assert res.success is True
    assert "Custom search output: antigravity" in res.output
    assert call_custom.tool_name == "search"  # Preserved, not overwritten to grep

    # 2. Alias does NOT resolve if target is not in registered tools
    # E.g. tools only has ReadFileTool. Call for 'Bash' should NOT resolve to 'run_command'
    loop_readonly = AgentLoop(
        provider=DeterministicMockProvider([]),
        tools=[ReadFileTool()],
        workspace_root=tmp_path,
    )
    call_unregistered = ToolCall(tool_name="Bash", arguments={"command": "ls"})
    res_unregistered = await loop_readonly._execute_tool(call_unregistered)
    assert res_unregistered.success is False
    assert "unknown tool" in res_unregistered.error.lower()
    # The call tool_name stays "Bash" rather than silently mutating to unregistered target
    assert call_unregistered.tool_name == "Bash"


def test_quoted_examples_and_ordinary_json_not_extracted():
    """Ordinary text with examples or normal JSON responses are not extracted as tool calls."""
    tools = [ReadFileTool(), EditFileTool(), RunCommandTool()]
    loop = AgentLoop(
        provider=DeterministicMockProvider([]),
        tools=tools,
    )

    # 1. Quoted example in markdown
    content_example = (
        "Here is an example of calling read_file:\n"
        "For example:\n"
        "```json\n"
        '{"name": "read_file", "arguments": {"path": "example.txt"}}\n'
        "```\n"
        "Let me know if you would like me to proceed."
    )
    assert loop._extract_tool_calls_from_content(content_example) == []

    # 2. Quoted example with XML tag
    content_xml_example = (
        "To inspect the file, the syntax is e.g. "
        '<tool_call>{"name": "read_file", "arguments": {"path": "test.txt"}}</tool_call>'
    )
    assert loop._extract_tool_calls_from_content(content_xml_example) == []

    # 3. Ordinary JSON response with status/result/description
    content_json_resp = (
        "The operation status was:\n"
        "```json\n"
        '{"status": "ok", "result": "success", "items": [1, 2, 3]}\n'
        "```\n"
    )
    assert loop._extract_tool_calls_from_content(content_json_resp) == []

    # 4. Valid unquoted tool call IS extracted
    valid_content = (
        "I will now inspect the test file.\n"
        "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"real.txt\"}}</tool_call>"
    )
    extracted = loop._extract_tool_calls_from_content(valid_content)
    assert len(extracted) == 1
    assert extracted[0]["function"]["name"] == "read_file"


def test_windows_path_and_escapes_preserved_in_json():
    """Windows drive paths are parsed cleanly without corrupting legitimate \\n, \\t, or quotes."""
    tools = [WriteFileTool()]
    loop = AgentLoop(
        provider=DeterministicMockProvider([]),
        tools=tools,
    )

    # Raw string containing Windows path with single backslashes and legitimate newlines
    raw_json = (
        '{"name": "write_file", "arguments": {'
        '"path": "C:\\Users\\test\\project\\output.txt", '
        '"content": "Line 1\\nLine 2\\tTabbed\\n\\\"Quoted\\\""}'
        '}'
    )
    content = f"```tool_call\n{raw_json}\n```"
    extracted = loop._extract_tool_calls_from_content(content)
    assert len(extracted) == 1
    args = json.loads(extracted[0]["function"]["arguments"])
    assert "Users" in args["path"]
    assert "Line 1\nLine 2\tTabbed\n\"Quoted\"" == args["content"]


@pytest.mark.asyncio
async def test_openrouter_health_check_defensive_and_exhaustion():
    """OpenRouter health check distinguishes free exhaustion, null payloads, and paid accounts."""
    from harness_core.providers.openrouter import OpenRouterProvider

    class MockResponse:
        def __init__(self, status_code: int, json_data: Any) -> None:
            self.status_code = status_code
            self._data = json_data

        def json(self):
            return self._data

    class MockClient:
        def __init__(self, key_response: MockResponse, is_closed: bool = False):
            self._key_resp = key_response
            self.is_closed = is_closed

        async def get(self, endpoint: str):
            return self._key_resp

        async def aclose(self):
            self.is_closed = True

    # 1. Free tier with 0 remaining is marked UNHEALTHY
    provider_exhausted = OpenRouterProvider(api_key="sk-or-test")
    provider_exhausted._client = MockClient(MockResponse(200, {
        "data": {"is_free_tier": True, "free_model_daily_requests": {"used": 50, "limit": 50, "remaining": 0}}
    }))
    assert await provider_exhausted.health_check() is False

    # 2. Free tier with remaining requests is HEALTHY
    provider_ok = OpenRouterProvider(api_key="sk-or-test")
    provider_ok._client = MockClient(MockResponse(200, {
        "data": {"is_free_tier": True, "free_model_daily_requests": {"used": 10, "limit": 50, "remaining": 40}}
    }))
    assert await provider_ok.health_check() is True

    # 3. Paid account (is_free_tier: False) is HEALTHY even without daily request quota
    provider_paid = OpenRouterProvider(api_key="sk-or-test")
    provider_paid._client = MockClient(MockResponse(200, {
        "data": {"is_free_tier": False, "limit": 1000}
    }))
    assert await provider_paid.health_check() is True

    # 4. Null data payload does not crash and returns True (200 with no free restrictions)
    provider_null = OpenRouterProvider(api_key="sk-or-test")
    provider_null._client = MockClient(MockResponse(200, {"data": None}))
    assert await provider_null.health_check() is True

    # 5. HTTP 401 returns False
    provider_auth_fail = OpenRouterProvider(api_key="sk-or-bad")
    provider_auth_fail._client = MockClient(MockResponse(401, {"error": "Invalid key"}))
    assert await provider_auth_fail.health_check() is False


@pytest.mark.asyncio
async def test_search_hidden_directories_and_dotfiles(tmp_path: Path):
    """Search correctly excludes hidden directories while permitting explicit dot-file searches."""
    # Create structure:
    # tmp_path/
    #   .git/config
    #   .env
    #   .gitignore
    #   src/app.py
    #   src/.secret.py
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config content", encoding="utf-8")

    (tmp_path / ".env").write_text("SECRET_KEY=12345\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("*.pyc\n.env\n", encoding="utf-8")

    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("print('hello world')\n", encoding="utf-8")
    (src / ".secret.py").write_text("hidden python file\n", encoding="utf-8")

    glob_tool = GlobTool()
    grep_tool = GrepTool()

    # 1. Standard search for '*.py' finds app.py, excludes .secret.py and .git
    res = await glob_tool.execute({"path": str(tmp_path), "pattern": "*.py"})
    assert res.success is True
    assert "app.py" in res.output
    assert ".secret.py" not in res.output
    assert ".git" not in res.output

    # 2. Explicit search for '.env' finds .env
    res_env = await glob_tool.execute({"path": str(tmp_path), "pattern": ".env"})
    assert res_env.success is True
    assert ".env" in res_env.output

    # 3. Explicit search for '.*' finds .env and .gitignore, but NOT files inside .git
    res_dot = await glob_tool.execute({"path": str(tmp_path), "pattern": ".*"})
    assert res_dot.success is True
    assert ".env" in res_dot.output
    assert ".gitignore" in res_dot.output
    assert "config" not in res_dot.output

    # 4. Grep with explicit dot-file include finds pattern inside .env
    grep_env = await grep_tool.execute({"path": str(tmp_path), "pattern": "SECRET_KEY", "include": ".env*"})
    assert grep_env.success is True
    assert "SECRET_KEY" in grep_env.output

    # 5. Grep without dot-file include does NOT search hidden files or .git
    grep_all = await grep_tool.execute({"path": str(tmp_path), "pattern": "git config content"})
    assert grep_all.success is True
    assert "(no matches)" in grep_all.output

