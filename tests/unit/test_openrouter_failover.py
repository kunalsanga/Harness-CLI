"""Focused tests for provider-neutral free-model failover (OpenRouter adapter)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
    ProviderErrorCategory,
    ProviderRequestError,
)
from harness_core.providers.openrouter import OpenRouterProvider
from harness_core.agent.loop import AgentLoop
from harness_core.agent.types import AgentConfig, TaskStatus, ToolResult, ToolResultStatus
from harness_core.routing.fallback import (
    ErrorClassification,
    FallbackConfig,
    RetryConfig,
    classify_error,
)
from harness_core.tools.base import Tool, ToolSchema
from harness_core.routing.router import (
    ModelRouter,
    RouterConfig,
)


# Deterministic free catalog for tests — not a core routing constant.
FREE_POOL = (
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free",
)


class FakeOpenRouter(ModelProvider):
    def __init__(self, outcomes: list[CompletionResponse | Exception]) -> None:
        self.api_key = "unit-test-secret"
        self.outcomes = list(outcomes)
        self.requests: list[CompletionRequest] = []
        self.configured_models = [
            ModelInfo(
                id=model_id,
                name=model_id,
                provider="openrouter",
                supports_tools=True,
                is_free=True,
                context_window=32000 - index,
                tags=["dynamic-route"] if model_id == "openrouter/free" else [],
            )
            for index, model_id in enumerate(FREE_POOL)
        ]

    @property
    def name(self) -> str:
        return "openrouter"

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        yield "unused"

    async def list_models(self) -> list[ModelInfo]:
        return list(self.configured_models)

    async def routing_hints(self, mode: str) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="openrouter/free",
                name="OpenRouter free router",
                provider=self.name,
                supports_tools=True,
                is_free=True,
                tags=["dynamic-route"],
            )
        ]

    async def health_check(self) -> bool:
        return True


def make_router(provider: FakeOpenRouter) -> ModelRouter:
    return ModelRouter(
        providers=[provider],
        config=RouterConfig(
            routing_mode="free",
            fallback=FallbackConfig(retry=RetryConfig(max_retries=0, jitter=False)),
            max_fallback_chain=len(FREE_POOL),
        ),
    )


def test_free_catalog_is_provider_boundary_not_core_constant() -> None:
    router_mod = __import__("harness_core.routing.router", fromlist=["RouterConfig"])
    assert not hasattr(router_mod, "DEFAULT_OPENROUTER_FREE_MODELS")
    assert not hasattr(RouterConfig(), "openrouter_free_models")
    assert all(mid.endswith(":free") or mid == "openrouter/free" for mid in FREE_POOL)


@pytest.mark.asyncio
async def test_success_uses_primary_from_discovered_free_catalog() -> None:
    provider = FakeOpenRouter([
        CompletionResponse(content="ok", model=FREE_POOL[0])
    ])
    router = make_router(provider)
    result = await router.execute(CompletionRequest(messages=[{"role": "user", "content": "hi"}]))

    assert result.succeeded
    assert result.model_used == FREE_POOL[0]
    assert [request.model for request in provider.requests] == [FREE_POOL[0]]
    assert router.current_model == FREE_POOL[0]
    assert router.attempt_count == 1


@pytest.mark.asyncio
async def test_openrouter_free_is_valid_and_remains_openrouter_provider() -> None:
    provider = FakeOpenRouter([CompletionResponse(content="dynamic free result")])
    router = make_router(provider)
    result = await router.execute(CompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="openrouter/free",
    ))

    assert result.succeeded
    assert provider.requests[0].model == "openrouter/free"
    assert result.provider_used == "openrouter"


@pytest.mark.asyncio
async def test_429_fails_over_and_reuses_identical_conversation() -> None:
    provider = FakeOpenRouter([
        RuntimeError("OpenRouter 429 rate limit (model=gemma)"),
        CompletionResponse(content="continued", model=FREE_POOL[1]),
    ])
    router = make_router(provider)
    messages = [
        {"role": "user", "content": "Implement the feature"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "read_file"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "existing workspace contents"},
    ]
    result = await router.execute(CompletionRequest(messages=messages, tools=[{"type": "function"}]))

    assert result.succeeded
    assert [request.model for request in provider.requests] == list(FREE_POOL[:2])
    assert all(request.messages is messages for request in provider.requests)
    assert all(request.messages[-1]["content"] == "existing workspace contents" for request in provider.requests)
    assert router.failed_models_for_current_request == [FREE_POOL[0]]
    assert router.current_model == FREE_POOL[1]
    assert router.attempt_count == 2
    assert router.failover_reason == ErrorClassification.RATE_LIMITED.value


@pytest.mark.asyncio
async def test_two_failures_continue_to_third_model() -> None:
    provider = FakeOpenRouter([
        RuntimeError("429 Too Many Requests"),
        RuntimeError("503 upstream temporarily unavailable"),
        CompletionResponse(content="success", model=FREE_POOL[2]),
    ])
    result = await make_router(provider).execute(
        CompletionRequest(messages=[{"role": "user", "content": "same task"}])
    )
    assert result.succeeded
    assert [request.model for request in provider.requests] == list(FREE_POOL[:3])


@pytest.mark.asyncio
async def test_exhaustion_is_bounded_and_reports_attempted_models() -> None:
    provider = FakeOpenRouter([RuntimeError("429 rate limit") for _ in FREE_POOL])
    router = make_router(provider)
    result = await router.execute(CompletionRequest(messages=[{"role": "user", "content": "task"}]))

    assert not result.succeeded
    assert len(provider.requests) == len(FREE_POOL)
    assert len({request.model for request in provider.requests}) == len(FREE_POOL)
    assert all(model in (result.final_error or "") for model in FREE_POOL)
    assert "rate limited" in (result.final_error or "").lower()


@pytest.mark.asyncio
async def test_auth_failure_stops_without_rotating_models() -> None:
    provider = FakeOpenRouter([RuntimeError("OpenRouter 401 invalid API key")])
    result = await make_router(provider).execute(
        CompletionRequest(messages=[{"role": "user", "content": "task"}])
    )
    assert not result.succeeded
    assert len(provider.requests) == 1
    assert "OPENROUTER_API_KEY" in (result.final_error or "")
    assert "unit-test-secret" not in (result.final_error or "")


@pytest.mark.asyncio
async def test_provider_credential_is_never_logged(caplog: pytest.LogCaptureFixture) -> None:
    provider = FakeOpenRouter([RuntimeError("401 invalid credential unit-test-secret")])
    await make_router(provider).execute(
        CompletionRequest(messages=[{"role": "user", "content": "task"}])
    )
    assert "unit-test-secret" not in caplog.text


def test_only_model_request_failures_are_classified_for_failover() -> None:
    assert classify_error(RuntimeError("HTTP 429 rate limit")) == ErrorClassification.RATE_LIMITED
    assert classify_error(RuntimeError("model temporarily unavailable")) == ErrorClassification.MODEL_UNAVAILABLE
    assert classify_error(RuntimeError("request timed out")) == ErrorClassification.RETRYABLE
    assert classify_error(RuntimeError("HTTP 401 unauthorized")) == ErrorClassification.PERMANENT
    assert classify_error(RuntimeError("file does not exist")) == ErrorClassification.UNKNOWN
    assert classify_error(RuntimeError("test suite failed")) == ErrorClassification.UNKNOWN


def test_openrouter_wrapped_provider_unavailable_is_model_unavailable() -> None:
    """The exact live Nemotron failure: HTTP 200 wrap, code 502,
    error_type provider_unavailable — must classify as failover-eligible."""
    err = RuntimeError(
        "OpenRouter error (code 502, provider_unavailable): Upstream error from Nvidia: "
        "ResourceExhausted: Worker local total request limit reached (16/16) "
        "(model=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free)"
    )
    assert classify_error(err) == ErrorClassification.MODEL_UNAVAILABLE
    assert classify_error(ProviderRequestError(
        ProviderErrorCategory.MODEL_UNAVAILABLE,
        "Upstream error from Nvidia",
        provider="openrouter",
        status_code=502,
        provider_code="provider_unavailable",
    )) == ErrorClassification.MODEL_UNAVAILABLE


@pytest.mark.asyncio
async def test_wrapped_provider_unavailable_fails_over_to_openrouter_free() -> None:
    """Gemma 31B 429 → Gemma 26B 429 → Nemotron wrapped 502/provider_unavailable
    → openrouter/free succeeds. The task must NOT terminate on Nemotron."""
    provider = FakeOpenRouter([
        RuntimeError("OpenRouter 429 rate limit (model=google/gemma-4-31b-it:free)"),
        RuntimeError("OpenRouter 429 rate limit (model=google/gemma-4-26b-a4b-it:free)"),
        RuntimeError(
            "OpenRouter error (code 502, provider_unavailable): Upstream error from Nvidia: "
            "ResourceExhausted: Worker local total request limit reached (16/16) "
            "(model=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free)"
        ),
        CompletionResponse(content="final answer", model="openrouter/free"),
    ])
    result = await make_router(provider).execute(
        CompletionRequest(messages=[{"role": "user", "content": "continue the same task"}])
    )
    assert result.succeeded
    assert [request.model for request in provider.requests] == list(FREE_POOL)
    assert result.model_used == "openrouter/free"
    assert result.attempt_count == 4


@pytest.mark.asyncio
async def test_openrouter_wraps_200_error_and_preserves_code_and_type() -> None:
    """The provider must surface structured code + error_type from a
    200-wrapped OpenRouter error so the classifier can route it."""
    p = OpenRouterProvider(api_key="sk-or-test")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "error": {
            "code": 502,
            "message": "Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (16/16)",
            "metadata": {"error_type": "provider_unavailable"},
        }
    })
    client = AsyncMock()
    client.post = AsyncMock(return_value=mock_resp)
    client.timeout = MagicMock(read=120)
    p._get_client = AsyncMock(return_value=client)

    with pytest.raises(ProviderRequestError) as ei:
        await p.generate(CompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        ))
    err = ei.value
    assert err.category == ProviderErrorCategory.MODEL_UNAVAILABLE
    assert err.status_code == 502
    assert err.provider_code in {502, "502", "provider_unavailable"}
    assert "Upstream error from Nvidia" in str(err)
    assert "sk-or-test" not in str(err)


@pytest.mark.asyncio
async def test_failed_model_is_cooled_down_for_the_next_request_then_pool_recovers() -> None:
    provider = FakeOpenRouter([
        RuntimeError("429 rate limit"),
        CompletionResponse(content="fallback success"),
        CompletionResponse(content="primary recovered"),
    ])
    router = make_router(provider)
    messages = [{"role": "user", "content": "task"}]

    first = await router.execute(CompletionRequest(messages=messages))
    second = await router.execute(CompletionRequest(messages=messages))

    assert first.succeeded and second.succeeded
    assert [request.model for request in provider.requests] == [
        FREE_POOL[0],
        FREE_POOL[1],
        FREE_POOL[1],
    ]


@pytest.mark.asyncio
async def test_model_request_failure_does_not_replay_successful_tool_operation() -> None:
    """Fallback operates before AgentLoop processes a tool-call response."""
    tool_executions = 0

    def tool_operation() -> None:
        nonlocal tool_executions
        tool_executions += 1

    provider = FakeOpenRouter([
        CompletionResponse(tool_calls=[{"id": "once", "type": "function", "function": {
            "name": "write_file", "arguments": "{}"
        }}]),
        RuntimeError("429 rate limit"),
        CompletionResponse(content="continued"),
    ])
    router = make_router(provider)

    first = await router.execute(CompletionRequest(messages=[{"role": "user", "content": "task"}]))
    assert first.succeeded and first.response is not None
    tool_operation()  # AgentLoop executes only after a successful model response.
    second = await router.execute(CompletionRequest(messages=[
        {"role": "user", "content": "task"},
        {"role": "assistant", "tool_calls": first.response.tool_calls},
        {"role": "tool", "tool_call_id": "once", "content": "tool result"},
    ]))

    assert second.succeeded
    assert tool_executions == 1
    assert [request.model for request in provider.requests] == [
        FREE_POOL[0],
        FREE_POOL[0],
        FREE_POOL[1],
    ]


@pytest.mark.asyncio
async def test_agent_task_continues_after_mid_task_failover() -> None:
    """A completed tool action and its result survive a model switch."""
    class WriteOnceTool(Tool):
        executions = 0

        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(
                name="write_file",
                description="Write a workspace file",
                parameters={"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"},
                }, "required": ["path", "content"]},
            )

        async def execute(self, arguments: dict) -> ToolResult:
            self.executions += 1
            (Path.cwd() / "__harness_failover_test__.txt").write_text(arguments["content"], encoding="utf-8")
            return ToolResult(status=ToolResultStatus.SUCCESS, output="file written")

    artifact = Path.cwd() / "__harness_failover_test__.txt"
    assert not artifact.exists(), "refusing to overwrite a pre-existing test artifact"
    tool = WriteOnceTool()
    provider = FakeOpenRouter([
        CompletionResponse(content="1. Create __harness_failover_test__.txt", model=FREE_POOL[0]),
        CompletionResponse(tool_calls=[{
                "id": "write-once",
                "type": "function",
                "function": {"name": "write_file", "arguments": '{"path":"__harness_failover_test__.txt","content":"ok"}'},
        }], model=FREE_POOL[0]),
        RuntimeError("OpenRouter 429 rate limit"),
        CompletionResponse(content="File created successfully.", model=FREE_POOL[1]),
    ])
    router = make_router(provider)
    loop = AgentLoop(
        provider=provider,
        router=router,
        tools=[tool],
        workspace_root=Path.cwd(),
        config=AgentConfig(max_iterations=3, verify_on_complete=False),
    )

    try:
        task = await loop.run("Create a file named __harness_failover_test__.txt containing ok")

        assert task.status == TaskStatus.COMPLETED, (task.error, task.failure_reason, [r.model for r in provider.requests])
        assert artifact.read_text(encoding="utf-8") == "ok"
        models_used = [request.model for request in provider.requests]
        assert len(models_used) >= 3
        # Mid-task failover must change model while keeping the same conversation.
        assert models_used[-1] != models_used[-2]
        assert models_used[-1] in FREE_POOL
        assert tool.executions == 1
        assert len(task.tool_calls) == 1
        assert provider.requests[-2].messages is provider.requests[-1].messages
        assert provider.requests[-1].messages[-1]["role"] == "tool"
        assert provider.requests[-1].messages[-1]["tool_call_id"] == "write-once"
        assert "file written" in provider.requests[-1].messages[-1]["content"]
    finally:
        artifact.unlink(missing_ok=True)
