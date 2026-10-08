"""Model health + fail-fast orchestration tests (fake providers, deterministic).

Covers the P0 routing audit:
A  healthy model succeeds
B  429 → cooldown → next model
C  wrapped 502 provider_unavailable → cooldown → next model
D  timeout → cooldown → next model
E  all unavailable → immediate structured failure (no long waits)
F  401 → stop rotation
G  403 → stop rotation
H  Retry-After honored in cooldown
I  new request after failure still skips unhealthy model
J  routing-mode switch does not erase health state
K  openrouter/free unavailable → health state → not immediately retried
L  successful failover preserves task state
M  successful tool call is not repeated after failover
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)
from harness_core.routing.fallback import (
    ERROR_CATEGORY_RATE_LIMIT,
    ERROR_CATEGORY_TIMEOUT,
    FallbackConfig,
    FallbackEngine,
    RetryConfig,
    classify_error,
    extract_retry_after,
    normalized_error_category,
)
from harness_core.routing.health import (
    HealthEvent,
    ModelHealthStatus,
    ModelHealthTracker,
)
from harness_core.routing.router import (
    ModelRouter,
    RouterConfig,
)
from harness_core.agent.types import ToolResult, ToolResultStatus


# Local free catalog for health/failover tests — not imported from core routing.
FREE_POOL = (
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "openrouter/free",
)


def make_request(**kwargs: Any) -> CompletionRequest:
    return CompletionRequest(messages=[{"role": "user", "content": "task"}], **kwargs)


class FakeProvider(ModelProvider):
    """Scripted provider: pops one outcome per generate() call."""

    def __init__(self, outcomes: list[CompletionResponse | Exception]) -> None:
        self.api_key = "unit-test-secret"
        self.outcomes = list(outcomes)
        self.requests: list[CompletionRequest] = []
        # Simulated in-request blocking for timeout tests.
        self.block_seconds: float = 0.0
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
        if self.block_seconds:
            await asyncio.sleep(self.block_seconds)
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


def make_engine(provider: FakeProvider, **cfg: Any) -> FallbackEngine:
    return FallbackEngine(
        fallback_config=FallbackConfig(
            retry=RetryConfig(max_retries=0, jitter=False),
            model_attempt_timeout_seconds=cfg.pop("attempt_timeout", 5.0),
            total_timeout_seconds=cfg.pop("total_timeout", 30.0),
            **cfg,
        )
    )


def make_router(provider: FakeProvider, **cfg: Any) -> ModelRouter:
    return ModelRouter(
        providers=[provider],
        config=RouterConfig(
            routing_mode="free",
            max_fallback_chain=len(FREE_POOL),
            fallback=FallbackConfig(
                retry=RetryConfig(max_retries=0, jitter=False),
                model_attempt_timeout_seconds=cfg.pop("attempt_timeout", 5.0),
                total_timeout_seconds=cfg.pop("total_timeout", 30.0),
            ),
        ),
    )


POOL = list(FREE_POOL)


# ── A. healthy model succeeds ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_healthy_model_succeeds() -> None:
    provider = FakeProvider([CompletionResponse(content="ok", model=POOL[0])])
    result = await make_engine(provider).execute(make_request(), [(POOL[0], provider)])
    assert result.succeeded
    assert result.model_used == POOL[0]
    assert result.error_category == ""


# ── B. 429 → cooldown → next model ──────────────────────────────────────

@pytest.mark.asyncio
async def test_b_429_sets_cooldown_and_fails_over() -> None:
    provider = FakeProvider([
        Exception("OpenRouter 429 Too Many Requests"),
        CompletionResponse(content="second"),
    ])
    engine = make_engine(provider)
    result = await engine.execute(make_request(), [(POOL[0], provider), (POOL[1], provider)])

    assert result.succeeded
    assert [r.model for r in provider.requests] == [POOL[0], POOL[1]]
    state = engine.health.get_state(POOL[0])
    assert state.health_status == ModelHealthStatus.RATE_LIMITED
    assert state.cooldown_remaining() > 0
    assert state.last_error_category == "rate_limit"
    # Immediate re-selection must be blocked.
    assert not engine.health.get_state(POOL[0]).is_healthy


# ── C. wrapped 502 provider_unavailable → cooldown → next model ─────────

@pytest.mark.asyncio
async def test_c_wrapped_provider_unavailable_sets_cooldown_and_fails_over() -> None:
    err = RuntimeError(
        "OpenRouter error (code 502, provider_unavailable): Upstream error from Nvidia: "
        "ResourceExhausted (model=nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free)"
    )
    provider = FakeProvider([err, CompletionResponse(content="next")])
    engine = make_engine(provider)
    result = await engine.execute(make_request(), [(POOL[2], provider), (POOL[3], provider)])

    assert result.succeeded
    state = engine.health.get_state(POOL[2])
    assert state.health_status == ModelHealthStatus.TEMPORARILY_UNAVAILABLE
    assert state.cooldown_remaining() > 0
    assert state.last_error_category == "model_unavailable"
    assert result.attempts[0]["category"] == "model_unavailable"


# ── D. timeout → cooldown → next model ──────────────────────────────────

@pytest.mark.asyncio
async def test_d_timeout_sets_cooldown_and_fails_over() -> None:
    provider = FakeProvider([CompletionResponse(content="never"), CompletionResponse(content="next")])
    # block_seconds must only affect the FIRST call — clear it inside the wrapper
    provider.block_seconds = 10.0  # exceeds the 1s attempt timeout
    engine = make_engine(provider, attempt_timeout=1.0)
    calls = {"n": 0}
    original_generate = provider.generate

    async def _block_once(request: CompletionRequest) -> CompletionResponse:
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.sleep(10.0)  # hard-block first attempt past the timeout
            return await original_generate(request)  # pragma: no cover - cancelled first
        provider.block_seconds = 0.0  # subsequent calls are instant
        return await original_generate(request)

    provider.generate = _block_once  # type: ignore[method-assign]
    result = await engine.execute(make_request(), [(POOL[0], provider), (POOL[1], provider)])

    assert result.succeeded, result.final_error
    # The timed-out attempt was cancelled before the provider recorded it,
    # so only the successful second attempt appears in requests.
    assert [r.model for r in provider.requests] == [POOL[1]]
    state = engine.health.get_state(POOL[0])
    assert state.cooldown_remaining() > 0
    assert state.last_error_category == "timeout"
    assert result.attempts[0]["category"] == "timeout"


# ── E. all unavailable → immediate structured failure ───────────────────

@pytest.mark.asyncio
async def test_e_all_unavailable_fails_fast_without_long_waits() -> None:
    # All four models are already cooling down from earlier evidence.
    engine = make_engine(FakeProvider([]))
    for i, mid in enumerate(POOL):
        engine.health.record_failure(mid, HealthEvent.RATE_LIMIT_429)
        engine.health.record_failure(mid, HealthEvent.RATE_LIMIT_429)

    start = time.time()
    provider = FakeProvider([])  # nothing should be called
    result = await engine.execute(make_request(), [(m, provider) for m in POOL])
    elapsed = time.time() - start

    assert not result.succeeded
    assert elapsed < 1.0, f"fail-fast violated: took {elapsed:.1f}s"
    assert len(provider.requests) == 0
    assert "No configured free model is currently available" in (result.final_error or "")
    for mid in POOL:
        assert mid in (result.final_error or "")
    assert "No paid model was used" in (result.final_error or "")


# ── F/G. 401/403 stop rotation ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_f_401_stops_rotation() -> None:
    provider = FakeProvider([Exception("OpenRouter 401 invalid API key")])
    result = await make_engine(provider).execute(
        make_request(), [(POOL[0], provider), (POOL[1], provider)]
    )
    assert not result.succeeded
    assert [r.model for r in provider.requests] == [POOL[0]]
    assert "OPENROUTER_API_KEY" in (result.final_error or "")


@pytest.mark.asyncio
async def test_g_403_stops_rotation() -> None:
    provider = FakeProvider([Exception("Client error '403 Forbidden'")])
    result = await make_engine(provider).execute(
        make_request(), [(POOL[0], provider), (POOL[1], provider)]
    )
    assert not result.succeeded
    assert [r.model for r in provider.requests] == [POOL[0]]


# ── H. Retry-After honored ──────────────────────────────────────────────

def test_h_retry_after_extraction() -> None:
    assert extract_retry_after(Exception("429 rate limited, retry-after: 45")) == 45.0
    assert extract_retry_after(Exception("Retry-After: 12")) == 12.0
    assert extract_retry_after(Exception("retry_after 30")) == 30.0
    assert extract_retry_after(Exception("no hint here")) is None
    assert extract_retry_after(Exception("retry-after: 99999")) is None  # unbounded hint rejected


@pytest.mark.asyncio
async def test_h_retry_after_respected_in_cooldown() -> None:
    provider = FakeProvider([
        Exception("OpenRouter 429 rate limit. retry-after: 120"),
        CompletionResponse(content="next"),
    ])
    engine = make_engine(provider)
    result = await engine.execute(make_request(), [(POOL[0], provider), (POOL[1], provider)])

    assert result.succeeded
    state = engine.health.get_state(POOL[0])
    # Cooldown must honor the provider's 120s hint, not the 60s default.
    assert 110.0 < state.cooldown_remaining() <= 120.0


# ── I. health survives a new request chain (new AgentLoop equivalent) ───

@pytest.mark.asyncio
async def test_i_new_request_skips_previously_failed_model() -> None:
    provider = FakeProvider([
        Exception("OpenRouter 429 rate limit"),
        CompletionResponse(content="first ok"),
        CompletionResponse(content="second ok"),
    ])
    router = make_router(provider)
    first = await router.execute(make_request())
    assert first.succeeded

    # A brand-new FallbackEngine with the SAME tracker (new AgentLoop in the
    # same session) must still skip the cooling-down model.
    engine2 = FallbackEngine(
        health_tracker=router.health,
        fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0, jitter=False)),
    )
    second = await engine2.execute(make_request(), [(m, provider) for m in POOL])
    assert second.succeeded
    # Pool[0] was skipped (cooling down) — first attempted model is pool[1].
    assert second.attempts[0] == {"model": POOL[0], "status": "skipped", "reason": "model unavailable (auth failed)"} or second.attempts[0]["status"] == "skipped"
    assert provider.requests[-1].model == POOL[1]


# ── J. routing-mode switch must not erase health ────────────────────────

def test_j_mode_switch_preserves_health() -> None:
    tracker = ModelHealthTracker()
    tracker.record_failure(POOL[0], HealthEvent.RATE_LIMIT_429)
    before = tracker.get_state(POOL[0]).cooldown_remaining()
    assert before > 0

    # /free only mutates RouterConfig flags; the tracker is untouched.
    config = RouterConfig(routing_mode="auto")
    config.routing_mode = "free"
    config.prefer_free = True

    assert tracker.get_state(POOL[0]).cooldown_remaining() > 0
    assert not tracker.get_state(POOL[0]).is_healthy


# ── K. openrouter/free is a normal pool member for health purposes ──────

@pytest.mark.asyncio
async def test_k_openrouter_free_failure_gets_cooldown_not_immediately_retried() -> None:
    err = RuntimeError("OpenRouter error (code 502, provider_unavailable): upstream busy")
    provider = FakeProvider([err, err])
    engine = make_engine(provider)
    # openrouter/free fails on both requests
    first = await engine.execute(make_request(), [(POOL[3], provider)])
    assert not first.succeeded
    assert engine.health.get_state(POOL[3]).cooldown_remaining() > 0

    second = await engine.execute(make_request(), [(POOL[3], provider)])
    assert not second.succeeded
    assert len(provider.requests) == 1, "cooling-down model must not be re-attempted"


# ── L/M. task-state preservation across failover (loop-level) ───────────

@pytest.mark.asyncio
async def test_l_m_failover_preserves_task_and_does_not_repeat_tools(tmp_path: Path) -> None:
    from harness_core.agent.loop import AgentLoop
    from harness_core.agent.types import AgentConfig, TaskStatus
    from harness_core.tools.base import Tool, ToolSchema

    class OnceTool(Tool):
        executions = 0

        @property
        def schema(self) -> ToolSchema:
            return ToolSchema(
                name="read_file",
                description="Read a workspace file",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            )

        async def execute(self, arguments: dict) -> ToolResult:
            self.executions += 1
            return ToolResult(status=ToolResultStatus.SUCCESS, output="file contents")

    tool = OnceTool()
    provider = FakeProvider([
        # Read-only work skips the extra planning request.
        # Model A: asks for the tool
        CompletionResponse(tool_calls=[{
            "id": "read-1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
        }], model=POOL[0]),
        # Model A (after tool result): rate limited
        Exception("OpenRouter 429 rate limit"),
        # Model B: finishes the task
        CompletionResponse(content="Done.", model=POOL[1]),
    ])
    router = make_router(provider)
    loop = AgentLoop(
        provider=provider,
        router=router,
        tools=[tool],
        workspace_root=tmp_path,
        config=AgentConfig(max_iterations=3, verify_on_complete=False),
    )

    task = await loop.run("Read the file")

    assert task.status == TaskStatus.COMPLETED, (task.error, task.failure_reason)
    assert tool.executions == 1, "tool must execute exactly once"
    assert len(task.tool_calls) == 1
    assert task.model_fallbacks >= 1
    # The failover request carried the SAME conversation including the tool result.
    failover_request = provider.requests[-1]
    assert failover_request.messages[-1]["role"] == "tool"
    assert failover_request.messages[-1]["tool_call_id"] == "read-1"
    assert "file contents" in failover_request.messages[-1]["content"]


# ── Normalized categories ────────────────────────────────────────────────

def test_normalized_categories() -> None:
    from harness_core.routing.fallback import ErrorClassification
    assert normalized_error_category(ErrorClassification.RATE_LIMITED, "429") == ERROR_CATEGORY_RATE_LIMIT
    assert normalized_error_category(ErrorClassification.RETRYABLE, "request timed out") == ERROR_CATEGORY_TIMEOUT
    assert normalized_error_category(ErrorClassification.PERMANENT, "401 unauthorized") == "auth"
    assert normalized_error_category(ErrorClassification.PERMANENT, "402 payment") == "payment"
    assert normalized_error_category(ErrorClassification.MODEL_UNAVAILABLE, "provider_unavailable") == "model_unavailable"
    assert normalized_error_category(ErrorClassification.UNKNOWN, "weird") == "unknown"
    assert classify_error(Exception("provider_unavailable")) == __import__(
        "harness_core.routing.fallback", fromlist=["ErrorClassification"]
    ).ErrorClassification.MODEL_UNAVAILABLE


# ── Config surface ───────────────────────────────────────────────────────

def test_attempt_timeout_is_configurable() -> None:
    from harness_core.routing.router import RouterConfig as RC
    cfg = RC.from_dict({"fallback": {"model_attempt_timeout_seconds": 45}})
    assert cfg.fallback.model_attempt_timeout_seconds == 45.0
    assert RC().fallback.model_attempt_timeout_seconds == 90.0
