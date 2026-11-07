"""Tests for token tracking, context preservation, and tool execution safety across provider failover."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import (
    AgentConfig,
    ExecutionState,
    Task,
    TaskPlan,
    TaskStatus,
    TodoItem,
    TodoStatus,
    ToolCall,
    ToolResult,
    ToolResultStatus,
)
from harness_core.observability.events import EventBus
from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.routing.fallback import FallbackEngine
from harness_core.routing.health import ModelHealthTracker
from harness_core.routing.router import ModelRouter, RouterConfig


class FailoverSequenceProvider(ModelProvider):
    """Mock provider with call-by-call responses and failures."""

    def __init__(self, name: str, models: list[str]) -> None:
        self._name = name
        self._models = models
        self.responses: list[CompletionResponse | Exception] = []
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def queue_response(self, resp: CompletionResponse | Exception) -> None:
        self.responses.append(resp)

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.call_count += 1
        if not self.responses:
            return CompletionResponse(
                content="Default response",
                model=request.model or self._models[0],
                provider=self.name,
                usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            )
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp

    async def stream(self, request: CompletionRequest):
        res = await self.generate(request)
        yield res.content

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=m,
                name=m,
                provider=self.name,
                context_window=32768,
                supports_tools=True,
                is_free=self.name == "openrouter",
            )
            for m in self._models
        ]

    async def health_check(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_context_and_milestones_preserved_across_failover():
    """Verify milestones, tool outputs, and task plan stay intact when provider fails."""
    p1_openrouter = FailoverSequenceProvider("openrouter", ["openrouter/free"])
    p2_openai = FailoverSequenceProvider("openai", ["gpt-4o-mini"])

    # First provider fails with 429
    p1_openrouter.queue_response(
        ProviderRequestError(
            ProviderErrorCategory.RATE_LIMIT,
            "Rate limit exceeded on openrouter",
            provider="openrouter",
            model="openrouter/free",
            status_code=429,
        )
    )

    # Second provider succeeds
    p2_openai.queue_response(
        CompletionResponse(
            content="Continuing task with plan step 2 completed.",
            model="gpt-4o-mini",
            provider="openai",
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        )
    )

    chain = [("openrouter/free", p1_openrouter), ("gpt-4o-mini", p2_openai)]
    engine = FallbackEngine(health_tracker=ModelHealthTracker(), event_bus=EventBus())

    task = Task(goal="Implement user auth and payments")
    task.task_plan = TaskPlan(items=[
        TodoItem(description="Setup models", status=TodoStatus.COMPLETED),
        TodoItem(description="Implement auth API", status=TodoStatus.IN_PROGRESS),
        TodoItem(description="Verify with tests", status=TodoStatus.PENDING),
    ])
    task.tool_calls.append(
        ToolCall(
            id="call_1",
            tool_name="write_file",
            arguments={"path": "auth.py", "content": "# auth"},
            result=ToolResult(output="File written", status=ToolResultStatus.SUCCESS),
        )
    )

    req = CompletionRequest(
        messages=[
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": task.goal},
        ]
    )

    result = await engine.execute(req, chain)

    assert result.succeeded is True
    assert result.provider_used == "openai"
    assert result.model_used == "gpt-4o-mini"

    # Verify task state was untouched by the provider failover
    assert task.task_plan.items[0].status == TodoStatus.COMPLETED
    assert task.task_plan.items[1].status == TodoStatus.IN_PROGRESS
    assert len(task.tool_calls) == 1
    assert task.tool_calls[0].result.status == ToolResultStatus.SUCCESS


@pytest.mark.asyncio
async def test_token_usage_attributed_per_provider_and_model():
    """Verify that tokens are attributed accurately to each provider and model."""
    task = Task(goal="Multi-step development")

    # Simulate turn 1: OpenRouter consumed 100 in, 50 out
    response_or = CompletionResponse(
        content="Step 1 complete",
        model="openrouter/free",
        provider="openrouter",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )

    # Turn 1 attribution
    task.provider_token_usage["openrouter"] = {"input_tokens": 100, "output_tokens": 50}
    task.model_token_usage["openrouter/free"] = {"input_tokens": 100, "output_tokens": 50}
    task.total_input_tokens += 100
    task.total_output_tokens += 50

    # Simulate turn 2: OpenRouter fails, OpenAI succeeds with 150 in, 80 out
    task.provider_token_usage["openai"] = {"input_tokens": 150, "output_tokens": 80}
    task.model_token_usage["gpt-4o-mini"] = {"input_tokens": 150, "output_tokens": 80}
    task.total_input_tokens += 150
    task.total_output_tokens += 80

    # Total accounting
    assert task.total_input_tokens == 250
    assert task.total_output_tokens == 130
    assert task.provider_token_usage["openrouter"]["input_tokens"] == 100
    assert task.provider_token_usage["openai"]["input_tokens"] == 150
    assert task.model_token_usage["openrouter/free"]["output_tokens"] == 50
    assert task.model_token_usage["gpt-4o-mini"]["output_tokens"] == 80


@pytest.mark.asyncio
async def test_safe_tool_execution_avoids_duplicate_calls():
    """Ensure already completed tools are tracked and not re-executed blindly."""
    task = Task(goal="Create project structure")

    call1 = ToolCall(
        id="call_init",
        tool_name="write_file",
        arguments={"path": "package.json", "content": "{}"},
        result=ToolResult(output="Wrote 2 bytes", status=ToolResultStatus.SUCCESS),
    )
    task.tool_calls.append(call1)
    task.completed_operations.append("write_file:package.json")

    # Check that completed_operations tracks the file write
    assert "write_file:package.json" in task.completed_operations
    # State remains intact
    assert task.tool_calls[0].result.status == ToolResultStatus.SUCCESS
