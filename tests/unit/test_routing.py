"""Comprehensive tests for the routing subsystem (Milestone 2).

Covers: scoring, routing, fallback, retry, 429 handling, timeout handling,
budget enforcement, free-model selection, tool compatibility.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness_core.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelInfo,
    ModelProvider,
)
from harness_core.routing.budgets import BudgetConfig, BudgetManager
from harness_core.routing.fallback import (
    ErrorClassification,
    FallbackConfig,
    FallbackEngine,
    FallbackResult,
    RetryConfig,
    classify_error,
)
from harness_core.routing.health import HealthEvent, ModelHealthState, ModelHealthTracker
from harness_core.routing.router import RouterConfig, ModelRouter, RoutingDecision
from harness_core.routing.scoring import (
    ScoringContext,
    ScoringWeights,
    compute_model_score,
    rank_models,
    score_capability,
    score_context_fit,
    score_cost,
    score_free_bonus,
    score_latency,
    score_reliability,
    score_task_fit,
    score_tool_support,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_model(
    model_id: str,
    provider: str = "openrouter",
    context_window: int = 128000,
    supports_tools: bool = True,
    is_free: bool = False,
    cost_in: float = 0.0,
    cost_out: float = 0.0,
    latency_ms: float = 0.0,
    reliability: float = 1.0,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name=model_id,
        provider=provider,
        context_window=context_window,
        supports_tools=supports_tools,
        cost_per_1k_input=cost_in,
        cost_per_1k_output=cost_out,
        is_free=is_free,
        latency_ms=latency_ms,
        reliability=reliability,
    )


class MockProvider(ModelProvider):
    """Mock provider for testing."""

    def __init__(
        self,
        name: str = "mock",
        response: CompletionResponse | None = None,
        models: list[ModelInfo] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self._name = name
        self._response = response or CompletionResponse(content="OK", model="mock", provider="mock")
        self._models = models or []
        self._raise_error = raise_error
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def generate(self, request: CompletionRequest) -> CompletionResponse:
        self._call_count += 1
        if self._raise_error:
            raise self._raise_error
        return self._response

    async def stream(self, request: CompletionRequest):
        yield self._response

    async def list_models(self) -> list[ModelInfo]:
        return list(self._models)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass


# ── Scoring Tests ────────────────────────────────────────────────────────────


class TestScoringWeights:
    def test_default_weights_sum_to_one(self):
        w = ScoringWeights().normalized()
        total = (
            w.capability + w.task_fit + w.tool_support + w.context_fit
            + w.cost + w.reliability + w.latency + w.free_bonus
        )
        assert abs(total - 1.0) < 1e-6

    def test_custom_weights_normalize(self):
        # When only capability and cost are non-zero, they dominate
        w = ScoringWeights(
            capability=5, task_fit=0, tool_support=0, context_fit=0,
            cost=5, reliability=0, latency=0, free_bonus=0,
        ).normalized()
        assert abs(w.capability - 0.5) < 1e-6
        assert abs(w.cost - 0.5) < 1e-6


class TestScoreCapability:
    def test_large_context_gets_high_score(self):
        m = _make_model("big", context_window=1_000_000)
        score = score_capability(m, ScoringContext())
        assert score > 0.8

    def test_small_context_gets_low_score(self):
        m = _make_model("small", context_window=4096)
        score = score_capability(m, ScoringContext())
        assert score < 0.3

    def test_zero_context_gets_zero(self):
        m = _make_model("zero", context_window=0)
        score = score_capability(m, ScoringContext())
        assert score == 0.0


class TestScoreTaskFit:
    def test_coding_model_for_coding_task(self):
        m = _make_model("deepseek-coder-v2")
        ctx = ScoringContext(task_description="Fix the failing tests")
        score = score_task_fit(m, ctx)
        assert score > 0.5

    def test_generic_model_for_coding_task(self):
        m = _make_model("llama-3.1-8b")
        ctx = ScoringContext(task_description="Fix the failing tests")
        score = score_task_fit(m, ctx)
        assert score >= 0.3  # baseline minus penalty

    def test_baseline_for_unknown_task(self):
        m = _make_model("some-model")
        ctx = ScoringContext(task_description="Do something")
        score = score_task_fit(m, ctx)
        assert 0.3 <= score <= 0.8


class TestScoreToolSupport:
    def test_tool_support_required_and_present(self):
        m = _make_model("x", supports_tools=True)
        score = score_tool_support(m, ScoringContext(requires_tools=True))
        assert score == 1.0

    def test_tool_support_required_and_absent(self):
        m = _make_model("x", supports_tools=False)
        score = score_tool_support(m, ScoringContext(requires_tools=True))
        assert score == 0.0

    def test_tool_support_not_required(self):
        m = _make_model("x", supports_tools=False)
        score = score_tool_support(m, ScoringContext(requires_tools=False))
        assert score == 1.0


class TestScoreContextFit:
    def test_plenty_of_room(self):
        m = _make_model("big", context_window=1_000_000)
        score = score_context_fit(m, ScoringContext(estimated_context_tokens=10000))
        assert score == 1.0

    def test_barely_fits(self):
        m = _make_model("tight", context_window=10000)
        score = score_context_fit(m, ScoringContext(estimated_context_tokens=9500))
        assert score <= 0.5

    def test_doesnt_fit(self):
        m = _make_model("tiny", context_window=1000)
        score = score_context_fit(m, ScoringContext(estimated_context_tokens=5000))
        assert score == 0.0


class TestScoreCost:
    def test_free_model_gets_max(self):
        m = _make_model("free", is_free=True)
        score = score_cost(m, ScoringContext())
        assert score == 1.0

    def test_expensive_model_gets_low_score(self):
        # $3.0/1k tokens is very expensive
        m = _make_model("expensive", cost_in=1.0, cost_out=2.0)
        score = score_cost(m, ScoringContext())
        assert score < 0.3

    def test_cheap_model_gets_high_score(self):
        m = _make_model("cheap", cost_in=0.001, cost_out=0.002)
        score = score_cost(m, ScoringContext())
        assert score > 0.6


class TestScoreReliability:
    def test_perfect_reliability(self):
        m = _make_model("perfect", reliability=1.0)
        score = score_reliability(m, ScoringContext())
        assert score == 1.0

    def test_zero_reliability(self):
        m = _make_model("broken", reliability=0.0)
        score = score_reliability(m, ScoringContext())
        assert score == 0.0


class TestScoreLatency:
    def test_fast_model(self):
        m = _make_model("fast", latency_ms=100)
        score = score_latency(m, ScoringContext())
        assert score > 0.9

    def test_slow_model(self):
        m = _make_model("slow", latency_ms=10000)
        score = score_latency(m, ScoringContext())
        assert score < 0.2

    def test_unknown_latency_gets_neutral(self):
        m = _make_model("unknown", latency_ms=0)
        score = score_latency(m, ScoringContext())
        assert score == 0.5


class TestScoreFreeBonus:
    def test_prefer_free_with_free_model(self):
        m = _make_model("free", is_free=True)
        score = score_free_bonus(m, ScoringContext(prefer_free=True))
        assert score == 1.0

    def test_prefer_free_with_paid_model(self):
        m = _make_model("paid", is_free=False)
        score = score_free_bonus(m, ScoringContext(prefer_free=True))
        assert score == 0.0

    def test_not_prefer_free(self):
        m = _make_model("any")
        score = score_free_bonus(m, ScoringContext(prefer_free=False))
        assert score == 0.5


class TestComputeModelScore:
    def test_free_tool_model_scores_high_for_free_task(self):
        m = _make_model("free-coder", is_free=True, supports_tools=True, context_window=262144)
        ctx = ScoringContext(
            task_description="Fix the bug",
            requires_tools=True,
            prefer_free=True,
        )
        score = compute_model_score(m, ctx)
        assert score > 0.6

    def test_no_tool_model_scores_lower_for_tool_task(self):
        # Tool support contributes to score; without it, score is lower
        m_with = _make_model("with-tools", supports_tools=True)
        m_without = _make_model("no-tools", supports_tools=False)
        ctx = ScoringContext(requires_tools=True)
        score_with = compute_model_score(m_with, ctx)
        score_without = compute_model_score(m_without, ctx)
        assert score_with > score_without

    def test_score_bounded(self):
        m = _make_model("test", is_free=True, supports_tools=True, context_window=1_000_000)
        ctx = ScoringContext(prefer_free=True)
        score = compute_model_score(m, ctx)
        assert 0.0 <= score <= 1.0


class TestRankModels:
    def test_free_model_ranks_higher_when_prefer_free(self):
        models = [
            _make_model("paid", is_free=False, context_window=128000),
            _make_model("free", is_free=True, context_window=128000),
        ]
        ctx = ScoringContext(prefer_free=True, requires_tools=False)
        ranked = rank_models(models, ctx)
        assert ranked[0][0].id == "free"

    def test_models_without_tools_excluded_when_required(self):
        models = [
            _make_model("no-tools", supports_tools=False),
            _make_model("has-tools", supports_tools=True),
        ]
        ctx = ScoringContext(requires_tools=True)
        ranked = rank_models(models, ctx)
        assert ranked[0][0].id == "has-tools"


# ── Health Tracker Tests ─────────────────────────────────────────────────────


class TestHealthTracker:
    def test_initial_state_neutral(self):
        tracker = ModelHealthTracker()
        assert tracker.get_reliability("unknown") == 0.5

    def test_success_increases_reliability(self):
        tracker = ModelHealthTracker()
        for _ in range(10):
            tracker.record_success("m1", latency_ms=100)
        assert tracker.get_reliability("m1") == 1.0

    def test_failures_decrease_reliability(self):
        tracker = ModelHealthTracker()
        tracker.record_success("m1")
        tracker.record_failure("m1", HealthEvent.SERVER_ERROR_5XX)
        tracker.record_failure("m1", HealthEvent.SERVER_ERROR_5XX)
        assert tracker.get_reliability("m1") < 0.5

    def test_rate_limit_marks_unhealthy(self):
        tracker = ModelHealthTracker(default_cooldown_seconds=1)
        tracker.record_failure("m1", HealthEvent.RATE_LIMIT_429)
        state = tracker.get_state("m1")
        assert state.is_rate_limited

    def test_consecutive_failures_mark_unhealthy(self):
        tracker = ModelHealthTracker()
        for _ in range(5):
            tracker.record_failure("m1", HealthEvent.SERVER_ERROR_5XX)
        assert not tracker.get_state("m1").is_healthy

    def test_success_resets_consecutive_failures(self):
        tracker = ModelHealthTracker()
        for _ in range(4):
            tracker.record_failure("m1", HealthEvent.SERVER_ERROR_5XX)
        tracker.record_success("m1")
        assert tracker.get_state("m1").consecutive_failures == 0

    def test_latency_tracking(self):
        tracker = ModelHealthTracker()
        tracker.record_success("m1", latency_ms=100)
        tracker.record_success("m1", latency_ms=200)
        assert tracker.get_state("m1").avg_latency_ms == 150.0

    def test_cost_tracking(self):
        tracker = ModelHealthTracker()
        tracker.record_success("m1", cost=0.01)
        tracker.record_success("m1", cost=0.02)
        assert abs(tracker.get_state("m1").estimated_cost - 0.03) < 1e-6

    def test_to_dict(self):
        tracker = ModelHealthTracker()
        tracker.record_success("m1")
        d = tracker.to_dict()
        assert "m1" in d
        assert d["m1"]["successes"] == 1

    def test_reset(self):
        tracker = ModelHealthTracker()
        tracker.record_success("m1")
        tracker.reset("m1")
        assert tracker.get_reliability("m1") == 0.5  # back to neutral


# ── Error Classification Tests ──────────────────────────────────────────────


class TestErrorClassification:
    def test_429_is_rate_limited(self):
        assert classify_error(Exception("429 Too Many Requests")) == ErrorClassification.RATE_LIMITED

    def test_rate_limit_in_message(self):
        assert classify_error(Exception("rate limit exceeded")) == ErrorClassification.RATE_LIMITED

    def test_401_is_permanent(self):
        assert classify_error(Exception("401 Unauthorized")) == ErrorClassification.PERMANENT

    def test_404_is_permanent(self):
        assert classify_error(Exception("404 Not Found")) == ErrorClassification.PERMANENT

    def test_timeout_is_retryable(self):
        assert classify_error(Exception("Connection timed out")) == ErrorClassification.RETRYABLE

    def test_500_is_retryable(self):
        assert classify_error(Exception("500 Internal Server Error")) == ErrorClassification.RETRYABLE

    def test_context_overflow_is_rate_limited(self):
        assert classify_error(Exception("maximum context length exceeded")) == ErrorClassification.RATE_LIMITED


# ── Retry Config Tests ──────────────────────────────────────────────────────


class TestRetryConfig:
    def test_exponential_backoff(self):
        cfg = RetryConfig(base_delay_seconds=1.0, backoff_factor=2.0, jitter=False)
        assert cfg.delay_for_attempt(0) == 1.0
        assert cfg.delay_for_attempt(1) == 2.0
        assert cfg.delay_for_attempt(2) == 4.0

    def test_max_delay_cap(self):
        cfg = RetryConfig(base_delay_seconds=1.0, max_delay_seconds=5.0, jitter=False)
        assert cfg.delay_for_attempt(10) == 5.0

    def test_jitter_reduces_delay(self):
        cfg = RetryConfig(base_delay_seconds=10.0, jitter=True)
        # With jitter, delay should be between 5.0 and 10.0
        delays = [cfg.delay_for_attempt(0) for _ in range(20)]
        assert all(5.0 <= d <= 10.0 for d in delays)


# ── Fallback Engine Tests ───────────────────────────────────────────────────


class TestFallbackEngine:
    @pytest.mark.asyncio
    async def test_primary_success(self):
        provider = MockProvider(response=CompletionResponse(content="OK", model="m1"))
        engine = FallbackEngine()
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])

        result = await engine.execute(req, [("m1", provider)])
        assert result.succeeded
        assert result.model_used == "m1"
        assert result.attempt_count == 1

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_succeeds(self):
        p1 = MockProvider(name="p1", raise_error=Exception("429 Too Many Requests"))
        p2 = MockProvider(name="p2", response=CompletionResponse(content="OK", model="m2"))

        engine = FallbackEngine()
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])

        result = await engine.execute(req, [("m1", p1), ("m2", p2)])
        assert result.succeeded
        assert result.model_used == "m2"
        assert result.attempt_count == 2

    @pytest.mark.asyncio
    async def test_all_models_fail(self):
        p1 = MockProvider(name="p1", raise_error=Exception("500 Server Error"))
        p2 = MockProvider(name="p2", raise_error=Exception("500 Server Error"))

        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0))
        )
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])

        result = await engine.execute(req, [("m1", p1), ("m2", p2)])
        assert not result.succeeded
        assert result.final_error is not None

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        call_count = 0

        class FlakyProvider(MockProvider):
            async def generate(self, request):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise Exception("503 Service Unavailable")
                return CompletionResponse(content="OK after retry", model="m1")

        provider = FlakyProvider(name="retry-provider")

        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=3, jitter=False, base_delay_seconds=0.01))
        )
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])

        result = await engine.execute(req, [("m1", provider)])
        assert result.succeeded
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_permanent_error(self):
        call_count = 0

        class PermFailProvider(MockProvider):
            async def generate(self, request):
                nonlocal call_count
                call_count += 1
                raise Exception("401 Unauthorized")

        provider = PermFailProvider(name="perm-provider")

        engine = FallbackEngine(
            fallback_config=FallbackConfig(retry=RetryConfig(max_retries=3))
        )
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])

        result = await engine.execute(req, [("m1", provider)])
        assert not result.succeeded
        assert call_count == 1  # no retry


# ── Budget Tests ─────────────────────────────────────────────────────────────


class TestBudgetManager:
    def test_iterations_budget(self):
        mgr = BudgetManager(BudgetConfig(max_iterations=3))
        assert mgr.record_iteration() is True
        assert mgr.record_iteration() is True
        assert mgr.record_iteration() is True
        assert mgr.record_iteration() is False  # exceeded

    def test_tool_call_budget(self):
        mgr = BudgetManager(BudgetConfig(max_tool_calls=2))
        assert mgr.record_tool_call() is True
        assert mgr.record_tool_call() is True
        assert mgr.record_tool_call() is False

    def test_cost_budget(self):
        mgr = BudgetManager(BudgetConfig(max_cost=1.0))
        assert mgr.record_cost(0.5) is True  # total = 0.5
        assert mgr.record_cost(0.4) is True  # total = 0.9
        assert mgr.record_cost(0.2) is False  # total = 1.1 > 1.0
        ok, reason = mgr.check_all()
        assert not ok
        assert "Cost" in reason

    def test_check_all_ok(self):
        mgr = BudgetManager(BudgetConfig(max_iterations=10))
        mgr.record_iteration()
        ok, reason = mgr.check_all()
        assert ok
        assert reason is None

    def test_per_model_limits(self):
        mgr = BudgetManager(BudgetConfig(max_calls_per_model=2))
        mgr.record_tokens(model_id="m1")
        mgr.record_tokens(model_id="m1")
        ok, reason = mgr.check_model_limit("m1")
        assert not ok
        assert "call limit" in reason

    def test_remaining_budget(self):
        mgr = BudgetManager(BudgetConfig(max_iterations=10, max_tool_calls=20))
        mgr.record_iteration()
        mgr.record_tool_call()
        remaining = mgr.remaining_budget()
        assert remaining["iterations_remaining"] == 9
        assert remaining["tool_calls_remaining"] == 19

    def test_reset(self):
        mgr = BudgetManager(BudgetConfig(max_iterations=5))
        mgr.record_iteration()
        mgr.record_iteration()
        mgr.reset()
        assert mgr.state.iterations == 0

    def test_timeout(self):
        cfg = BudgetConfig(timeout_seconds=0.01)
        mgr = BudgetManager(cfg)
        time.sleep(0.02)
        ok, reason = mgr.check_all()
        assert not ok
        assert "Timeout" in reason

    def test_to_dict(self):
        mgr = BudgetManager(BudgetConfig(max_iterations=5))
        d = mgr.to_dict()
        assert "config" in d
        assert "state" in d
        assert "remaining" in d


# ── Model Router Tests ──────────────────────────────────────────────────────


class TestModelRouter:
    @pytest.mark.asyncio
    async def test_refresh_models(self):
        provider = MockProvider(models=[
            _make_model("m1", provider="mock", is_free=True),
            _make_model("m2", provider="mock", is_free=False),
        ])
        router = ModelRouter(providers=[provider])
        models = await router.refresh_models()
        assert len(models) == 2

    @pytest.mark.asyncio
    async def test_select_models_filters_no_tools(self):
        provider = MockProvider(models=[
            _make_model("no-tools", provider="mock", supports_tools=False),
            _make_model("has-tools", provider="mock", supports_tools=True),
        ])
        router = ModelRouter(providers=[provider])
        req = CompletionRequest(
            messages=[{"role": "user", "content": "Fix the bug"}],
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        chain = await router.select_models(req)
        model_ids = [mid for mid, _ in chain]
        assert "has-tools" in model_ids
        assert "no-tools" not in model_ids

    @pytest.mark.asyncio
    async def test_routing_decision_recorded(self):
        provider = MockProvider(models=[
            _make_model("m1", provider="mock", is_free=True),
        ])
        router = ModelRouter(providers=[provider])
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])
        await router.select_models(req)
        decisions = router.get_routing_decisions()
        assert len(decisions) >= 1
        assert decisions[0].selected_model == "m1"

    @pytest.mark.asyncio
    async def test_execute_uses_router(self):
        provider = MockProvider(
            response=CompletionResponse(content="routed!", model="m1"),
            models=[_make_model("m1", provider="mock", supports_tools=True)],
        )
        router = ModelRouter(providers=[provider])
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])
        result = await router.execute(req)
        assert result.succeeded
        assert result.model_used == "m1"

    @pytest.mark.asyncio
    async def test_execute_fallback_on_429(self):
        p1 = MockProvider(
            name="p1",
            raise_error=Exception("429 Too Many Requests"),
            models=[_make_model("m1", provider="p1")],
        )
        p2 = MockProvider(
            name="p2",
            response=CompletionResponse(content="fallback!", model="m2"),
            models=[_make_model("m2", provider="p2")],
        )

        router = ModelRouter(
            providers=[p1, p2],
            config=RouterConfig(fallback=FallbackConfig(retry=RetryConfig(max_retries=0))),
        )
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])
        result = await router.execute(req)
        assert result.succeeded
        assert result.model_used == "m2"

    @pytest.mark.asyncio
    async def test_budget_exceeded_returns_error(self):
        router = ModelRouter(config=RouterConfig(
            budget=BudgetConfig(max_iterations=0),
        ))
        # Pre-exceed the budget
        router.budget.state.iterations = 1
        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])
        result = await router.execute(req)
        assert not result.succeeded
        assert "Budget" in result.final_error

    @pytest.mark.asyncio
    async def test_health_report(self):
        provider = MockProvider(models=[_make_model("m1", provider="mock")])
        router = ModelRouter(providers=[provider])
        await router.refresh_models()
        report = router.get_health_report()
        assert isinstance(report, dict)

    def test_budget_status(self):
        router = ModelRouter()
        status = router.get_budget_status()
        assert "config" in status
        assert "remaining" in status


# ── Router Config Tests ─────────────────────────────────────────────────────


class TestRouterConfig:
    def test_from_dict(self):
        data = {
            "routing_mode": "free",
            "prefer_free": True,
            "scoring_weights": {"capability": 0.3, "cost": 0.3},
            "budget": {"max_iterations": 50, "max_cost": 2.0},
        }
        cfg = RouterConfig.from_dict(data)
        assert cfg.routing_mode == "free"
        assert cfg.prefer_free is True
        assert cfg.scoring_weights.capability == 0.3
        assert cfg.budget.max_iterations == 50
        assert cfg.budget.max_cost == 2.0

    def test_default_config(self):
        cfg = RouterConfig()
        assert cfg.routing_mode == "auto"
        assert cfg.prefer_free is False
        assert cfg.budget.max_iterations == 30


# ── Integration Test: 429 → Fallback ────────────────────────────────────────


class TestIntegration429Fallback:
    """Integration test simulating: Model A → 429 → Model B → success."""

    @pytest.mark.asyncio
    async def test_429_fallback_to_second_model(self):
        # Model A: always returns 429
        p_a = MockProvider(name="provider-a")
        p_a = MockProvider(
            name="provider-a",
            raise_error=Exception("429 Too Many Requests"),
            models=[_make_model("model-a", provider="provider-a")],
        )

        # Model B: succeeds with tool call
        p_b = MockProvider(
            name="provider-b",
            response=CompletionResponse(
                content="I'll read the file for you.",
                model="model-b",
                provider="provider-b",
                tool_calls=[{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "README.md"}),
                    },
                }],
            ),
            models=[_make_model("model-b", provider="provider-b", supports_tools=True)],
        )

        router = ModelRouter(
            providers=[p_a, p_b],
            config=RouterConfig(
                fallback=FallbackConfig(retry=RetryConfig(max_retries=0)),
            ),
        )

        request = CompletionRequest(
            messages=[{"role": "user", "content": "Read the README"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }],
        )

        result = await router.execute(request)

        # Assertions
        assert result.succeeded, f"Expected success but got: {result.final_error}"
        assert result.model_used == "model-b"
        assert result.provider_used == "provider-b"
        assert result.attempt_count == 2  # model-a failed, model-b succeeded
        assert result.response is not None
        assert len(result.response.tool_calls) == 1
        assert result.response.tool_calls[0]["function"]["name"] == "read_file"

        # Verify health tracking
        health_a = router.health.get_state("model-a")
        assert health_a.rate_limit_hits == 1
        assert health_a.is_rate_limited

        health_b = router.health.get_state("model-b")
        assert health_b.successes == 1

    @pytest.mark.asyncio
    async def test_timeout_fallback_to_second_model(self):
        p_a = MockProvider(name="slow")
        p_a = MockProvider(
            name="slow",
            raise_error=asyncio.TimeoutError("Request timed out"),
            models=[_make_model("slow-model", provider="slow")],
        )
        p_b = MockProvider(
            name="fast",
            response=CompletionResponse(content="Done", model="fast-model"),
            models=[_make_model("fast-model", provider="fast", supports_tools=True)],
        )

        router = ModelRouter(
            providers=[p_a, p_b],
            config=RouterConfig(
                fallback=FallbackConfig(retry=RetryConfig(max_retries=0)),
            ),
        )

        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])
        result = await router.execute(req)
        assert result.succeeded
        assert result.model_used == "fast-model"

    @pytest.mark.asyncio
    async def test_cascading_429s(self):
        """All models return 429 — should fail gracefully."""
        providers = []
        for i in range(3):
            p = MockProvider(
                name=f"p{i}",
                raise_error=Exception("429 Too Many Requests"),
                models=[_make_model(f"m{i}", provider=f"p{i}")],
            )
            providers.append(p)

        router = ModelRouter(
            providers=providers,
            config=RouterConfig(
                fallback=FallbackConfig(retry=RetryConfig(max_retries=0)),
            ),
        )

        req = CompletionRequest(messages=[{"role": "user", "content": "test"}])
        result = await router.execute(req)
        assert not result.succeeded
        assert "All" in result.final_error or "failed" in result.final_error
