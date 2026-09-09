"""Regression tests for the prompt token ceiling (Phase 2B).

The failure these exist to prevent: an agent session died at 201,871 tokens
against a 196,608-token cap by accumulating whole-file dumps and raw command
output. The old `_build_messages` had a *trigger* for summarizing old history
but never an enforced *ceiling*, and its trigger required BOTH a long history
AND a large one — so a handful of enormous results produced no mitigation at
all. Every test here asserts the ceiling holds, not merely that compaction
was attempted.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from harness_core.agent.loop import (
    AgentLoop,
    DEFAULT_CONTEXT_TOKEN_BUDGET,
    HISTORY_CHAR_BUDGET,
    KEEP_RECENT_CALLS,
    MAX_TOOL_RESULT_TOKENS,
    message_tokens,
    messages_tokens,
    truncate_to_tokens,
)
from harness_core.agent.types import (
    AgentConfig,
    Task,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.observability.events import EventBus
from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelProvider
from harness_core.tools.filesystem import ReadFileTool


class SilentProvider(ModelProvider):
    """Never called: these tests exercise message assembly, not generation."""

    @property
    def name(self) -> str:
        return "silent"

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        return CompletionResponse(content="", model="silent", provider="silent")

    async def stream(self, request: CompletionRequest):
        yield CompletionResponse(content="", model="silent", provider="silent")

    async def list_models(self) -> list[Any]:
        return []

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


def make_loop(workspace: Path, budget: int | None = None) -> AgentLoop:
    config = AgentConfig(verify_on_complete=False)
    if budget is not None:
        config.context_token_budget = budget
    return AgentLoop(
        provider=SilentProvider(),
        tools=[ReadFileTool()],
        workspace_root=workspace,
        config=config,
        event_bus=EventBus(),
    )


def task_with_results(
    goal: str,
    count: int,
    output: str = "",
    error: str | None = None,
    status: ToolResultStatus = ToolResultStatus.SUCCESS,
) -> Task:
    task = Task(goal=goal)
    for i in range(count):
        call = ToolCall(id=f"t{i}", tool_name="read_file", arguments={"path": f"f{i}.py"})
        call.result = ToolResult(status=status, output=output, error=error)
        task.tool_calls.append(call)
    return task


def tool_contents(messages: list[dict[str, Any]]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "tool"]


def task_state_blocks(messages: list[dict[str, Any]]) -> list[str]:
    return [
        m["content"] for m in messages
        if m.get("role") == "system" and "TASK STATE" in (m.get("content") or "")
    ]


# ── Per-result truncation ────────────────────────────────────────────────


class TestTruncateToTokens:
    def test_short_text_untouched(self):
        assert truncate_to_tokens("hello world", 100) == "hello world"

    def test_exactly_at_limit_untouched(self):
        text = "x" * 400
        assert truncate_to_tokens(text, 100) == text

    def test_oversized_text_respects_the_cap(self):
        result = truncate_to_tokens("x" * 100_000, 1_000)
        assert len(result) <= 1_000 * 4

    def test_keeps_head_and_tail(self):
        text = "HEAD" + ("m" * 100_000) + "TAIL"
        result = truncate_to_tokens(text, 1_000)
        assert result.startswith("HEAD")
        assert result.endswith("TAIL")

    def test_says_that_content_was_removed(self):
        result = truncate_to_tokens("x" * 100_000, 1_000)
        assert "omitted by the harness" in result

    def test_zero_budget_yields_nothing(self):
        assert truncate_to_tokens("x" * 1_000, 0) == ""


# ── The ceiling itself ───────────────────────────────────────────────────


class TestTokenCeiling:
    def test_single_oversized_result_is_capped(self, tmp_path: Path):
        """One 2MB read used to enter the prompt verbatim (~500k tokens)."""
        loop = make_loop(tmp_path)
        task = task_with_results("Read one big file", 1, output="z" * 2_000_000)

        messages = loop._build_messages(task)

        assert messages_tokens(messages) <= DEFAULT_CONTEXT_TOKEN_BUDGET
        assert len(tool_contents(messages)) == 1
        assert len(tool_contents(messages)[0]) <= MAX_TOOL_RESULT_TOKENS * 4

    def test_few_but_enormous_results_stay_within_budget(self, tmp_path: Path):
        """The shape the old AND-condition missed: 8 calls, ~800k tokens."""
        loop = make_loop(tmp_path)
        task = task_with_results("Blowout", KEEP_RECENT_CALLS - 2, output="y" * 400_000)

        messages = loop._build_messages(task)

        assert messages_tokens(messages) <= DEFAULT_CONTEXT_TOKEN_BUDGET

    def test_many_enormous_results_stay_within_budget(self, tmp_path: Path):
        loop = make_loop(tmp_path)
        task = task_with_results("Many huge reads", 40, output="w" * 200_000)

        messages = loop._build_messages(task)

        assert messages_tokens(messages) <= DEFAULT_CONTEXT_TOKEN_BUDGET

    def test_errors_are_capped_too(self, tmp_path: Path):
        """Raw pytest output arrives on `error`, not `output`."""
        loop = make_loop(tmp_path)
        task = task_with_results(
            "Failing tests", 20, output="",
            error="E" * 300_000, status=ToolResultStatus.ERROR,
        )

        messages = loop._build_messages(task)

        assert messages_tokens(messages) <= DEFAULT_CONTEXT_TOKEN_BUDGET

    def test_configured_budget_is_honoured(self, tmp_path: Path):
        loop = make_loop(tmp_path, budget=20_000)
        task = task_with_results("Small window model", 30, output="q" * 200_000)

        messages = loop._build_messages(task)

        assert messages_tokens(messages) <= 20_000

    def test_tiny_budget_still_keeps_one_observation(self, tmp_path: Path):
        """Never send a tool-using prompt with no tool history at all."""
        loop = make_loop(tmp_path, budget=3_000)
        task = task_with_results("Cramped", 20, output="q" * 500_000)

        messages = loop._build_messages(task)

        assert messages_tokens(messages) <= 3_000
        assert len(tool_contents(messages)) == 1

    def test_corrections_are_counted_against_the_budget(self, tmp_path: Path):
        loop = make_loop(tmp_path, budget=30_000)
        loop._corrections = ["c" * 40_000, "d" * 40_000, "e" * 40_000]
        task = task_with_results("With guidance", 12, output="f" * 100_000)

        messages = loop._build_messages(task)

        # The corrections alone exceed the budget, so the ceiling cannot be
        # met — but history must have been surrendered trying.
        assert len(tool_contents(messages)) <= 1
        user_messages = [m for m in messages if m.get("role") == "user"]
        assert len(user_messages) == 4  # goal + at most 3 corrections

    def test_missing_config_field_falls_back_to_default(self, tmp_path: Path):
        """Older configs, or hand-rolled stand-ins, must not crash the loop."""
        loop = make_loop(tmp_path)
        loop.config = SimpleNamespace(autonomous_mode=True, verbose=False)

        assert loop._context_token_budget() == DEFAULT_CONTEXT_TOKEN_BUDGET

    def test_nonsense_budget_falls_back_to_default(self, tmp_path: Path):
        loop = make_loop(tmp_path, budget=0)

        assert loop._context_token_budget() == DEFAULT_CONTEXT_TOKEN_BUDGET


# ── Structural invariants ────────────────────────────────────────────────


class TestMessageStructureSurvivesTrimming:
    """Trimming must never produce a request the provider will reject."""

    def _assert_pairs_intact(self, messages: list[dict[str, Any]]) -> None:
        for i, msg in enumerate(messages):
            if msg.get("tool_calls"):
                call_id = msg["tool_calls"][0]["id"]
                assert i + 1 < len(messages), "assistant tool_call is the last message"
                nxt = messages[i + 1]
                assert nxt.get("role") == "tool"
                assert nxt.get("tool_call_id") == call_id
            if msg.get("role") == "tool":
                assert i > 0, "tool result with no preceding assistant message"
                assert messages[i - 1].get("tool_calls")

    def test_pairs_intact_under_heavy_trimming(self, tmp_path: Path):
        loop = make_loop(tmp_path, budget=10_000)
        task = task_with_results("Heavy", 25, output="p" * 300_000)

        self._assert_pairs_intact(loop._build_messages(task))

    def test_pairs_intact_when_nothing_is_trimmed(self, tmp_path: Path):
        loop = make_loop(tmp_path)
        task = task_with_results("Light", 5, output="small")

        self._assert_pairs_intact(loop._build_messages(task))

    def test_no_tool_result_is_left_empty(self, tmp_path: Path):
        loop = make_loop(tmp_path, budget=5_000)
        task = task_with_results("Cramped", 15, output="r" * 400_000)

        contents = tool_contents(loop._build_messages(task))

        assert contents
        for content in contents:
            assert content, "an empty tool result teaches the model nothing"

    def test_system_prompt_and_goal_always_survive(self, tmp_path: Path):
        loop = make_loop(tmp_path, budget=2_000)
        task = task_with_results("Preserve me", 30, output="s" * 500_000)

        messages = loop._build_messages(task)

        assert messages[0]["role"] == "system"
        assert any(m.get("content") == "Preserve me" for m in messages)


# ── Summary accuracy ─────────────────────────────────────────────────────


class TestTaskStateSummary:
    def test_no_summary_when_nothing_was_dropped(self, tmp_path: Path):
        loop = make_loop(tmp_path)
        task = task_with_results("All verbatim", 4, output="small")

        messages = loop._build_messages(task)

        assert task_state_blocks(messages) == []
        assert len(tool_contents(messages)) == 4

    def test_summary_count_matches_what_was_dropped(self, tmp_path: Path):
        loop = make_loop(tmp_path)
        dropped_expected = 6
        task = task_with_results(
            "Counted", KEEP_RECENT_CALLS + dropped_expected,
            output="x" * (HISTORY_CHAR_BUDGET // 8),
        )

        messages = loop._build_messages(task)
        blocks = task_state_blocks(messages)

        assert len(blocks) == 1
        assert f"{dropped_expected} earlier tool call(s)" in blocks[0]
        assert len(tool_contents(messages)) == KEEP_RECENT_CALLS

    def test_summary_reports_the_tools_that_were_used(self, tmp_path: Path):
        """Delegated to ContextCompactor — assert the delegation works."""
        loop = make_loop(tmp_path)
        task = task_with_results(
            "Tooling", KEEP_RECENT_CALLS + 4, output="x" * (HISTORY_CHAR_BUDGET // 8),
        )
        task.tool_calls[0].tool_name = "run_command"

        blocks = task_state_blocks(loop._build_messages(task))

        assert blocks
        assert "Tool calls made: 4" in blocks[0]
        assert "run_command" in blocks[0]

    def test_summary_survives_a_cramped_budget(self, tmp_path: Path):
        """The summary is the cheapest way to retain history, so keep it."""
        loop = make_loop(tmp_path, budget=8_000)
        task = task_with_results("Cramped", 30, output="t" * 300_000)

        messages = loop._build_messages(task)

        assert len(task_state_blocks(messages)) == 1
        assert messages_tokens(messages) <= 8_000


# ── Token accounting ─────────────────────────────────────────────────────


class TestMessageTokenAccounting:
    def test_tool_call_payload_is_counted(self):
        """A message with content=None still costs tokens for its arguments."""
        bare = {"role": "assistant", "content": None}
        with_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "t0",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            }],
        }

        assert message_tokens(with_call) > message_tokens(bare)

    def test_none_content_does_not_raise(self):
        assert message_tokens({"role": "assistant", "content": None}) > 0

    def test_total_is_the_sum_of_parts(self):
        messages = [
            {"role": "system", "content": "a" * 400},
            {"role": "user", "content": "b" * 400},
        ]

        assert messages_tokens(messages) == sum(message_tokens(m) for m in messages)
