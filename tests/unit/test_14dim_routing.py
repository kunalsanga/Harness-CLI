"""Tests for the 14-dimension scoring system (M3.6 Phase 2)."""

import pytest
from harness_core.providers.base import ModelInfo
from harness_core.routing.scoring import (
    ScoringContext,
    ScoringWeights,
    compute_model_score,
    score_task_type_fit,
    score_capability_fit,
    score_history_success,
    score_history_latency,
    score_tool_efficiency,
    score_user_preference,
)


def _make_model(
    model_id: str = "test-model",
    context_window: int = 32000,
    supports_tools: bool = True,
    is_free: bool = False,
    cost_per_1k_input: float = 0.0,
    latency_ms: float = 0.0,
    reliability: float = 1.0,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        name=model_id,
        provider="test",
        context_window=context_window,
        supports_tools=supports_tools,
        is_free=is_free,
        cost_per_1k_input=cost_per_1k_input,
        latency_ms=latency_ms,
        reliability=reliability,
    )


class TestScoringWeights:
    def test_default_weights_sum_to_one(self):
        w = ScoringWeights()
        n = w.normalized()
        total = (
            n.capability + n.task_fit + n.tool_support + n.context_fit
            + n.cost + n.reliability + n.latency + n.free_bonus
            + n.task_type_fit + n.capability_fit
            + n.history_success + n.history_latency
            + n.tool_efficiency + n.user_preference
        )
        assert abs(total - 1.0) < 0.001

    def test_all_14_dimensions_present(self):
        w = ScoringWeights()
        n = w.normalized()
        assert n.task_type_fit > 0
        assert n.capability_fit > 0
        assert n.history_success > 0
        assert n.history_latency > 0
        assert n.tool_efficiency > 0
        assert n.user_preference > 0


class TestTaskTypeFitScoring:
    def test_bug_fix_model_for_bug_fix_task(self):
        ctx = ScoringContext(task_type="bug_fix")
        model = _make_model(model_id="deepseek-coder-33b")
        score = score_task_type_fit(model, ctx)
        assert score > 0.5  # should score well

    def test_unknown_task_type(self):
        ctx = ScoringContext(task_type="")
        model = _make_model()
        score = score_task_type_fit(model, ctx)
        assert score == 0.5  # neutral

    def test_coding_model_for_implementation_task(self):
        ctx = ScoringContext(task_type="implementation")
        model = _make_model(model_id="deepseek-coder-33b")
        score = score_task_type_fit(model, ctx)
        assert score >= 0.8  # strong alignment

    def test_research_task_with_research_model(self):
        ctx = ScoringContext(task_type="research")
        model = _make_model(model_id="claude-3.5-sonnet")
        score = score_task_type_fit(model, ctx)
        assert score >= 0.7


class TestCapabilityFitScoring:
    def test_with_capability_data(self):
        ctx = ScoringContext(
            model_capability_scores={"coding": 0.9, "tool_use": 0.8}
        )
        model = _make_model()
        score = score_capability_fit(model, ctx)
        assert score == pytest.approx(0.85)

    def test_without_capability_data(self):
        ctx = ScoringContext()
        model = _make_model()
        score = score_capability_fit(model, ctx)
        assert score == 0.5  # neutral


class TestHistoryScoring:
    def test_high_success_rate(self):
        ctx = ScoringContext(historical_success_rate=0.95)
        score = score_history_success(_make_model(), ctx)
        assert score == pytest.approx(0.95)

    def test_low_success_rate(self):
        ctx = ScoringContext(historical_success_rate=0.3)
        score = score_history_success(_make_model(), ctx)
        assert score == pytest.approx(0.3)

    def test_no_history(self):
        ctx = ScoringContext()
        score = score_history_success(_make_model(), ctx)
        assert score == 0.5  # neutral


class TestHistoryLatencyScoring:
    def test_fast_latency(self):
        ctx = ScoringContext(historical_avg_latency_ms=500)
        score = score_history_latency(_make_model(), ctx)
        assert score > 0.8

    def test_slow_latency(self):
        ctx = ScoringContext(historical_avg_latency_ms=8000)
        score = score_history_latency(_make_model(), ctx)
        assert score < 0.3

    def test_no_history(self):
        ctx = ScoringContext()
        score = score_history_latency(_make_model(), ctx)
        assert score == 0.5  # neutral


class TestToolEfficiencyScoring:
    def test_efficient(self):
        ctx = ScoringContext(historical_tool_efficiency=0.3)  # low ratio = efficient
        score = score_tool_efficiency(_make_model(), ctx)
        assert score > 0.5

    def test_inefficient(self):
        ctx = ScoringContext(historical_tool_efficiency=3.0)  # high ratio = inefficient
        score = score_tool_efficiency(_make_model(), ctx)
        assert score < 0.5

    def test_no_data(self):
        ctx = ScoringContext()
        score = score_tool_efficiency(_make_model(), ctx)
        assert score == 0.5


class TestUserPreferenceScoring:
    def test_selected_model(self):
        ctx = ScoringContext(user_selected_model="my-model")
        model = _make_model(model_id="my-model")
        score = score_user_preference(model, ctx)
        assert score == 1.0

    def test_non_selected_model(self):
        ctx = ScoringContext(user_selected_model="my-model")
        model = _make_model(model_id="other-model")
        score = score_user_preference(model, ctx)
        assert score == 0.5

    def test_no_preference(self):
        ctx = ScoringContext()
        model = _make_model()
        score = score_user_preference(model, ctx)
        assert score == 0.5


class TestComputeModelScore:
    def test_score_increases_with_task_type_fit(self):
        ctx_no_type = ScoringContext()
        ctx_with_type = ScoringContext(task_type="bug_fix")
        model = _make_model(model_id="deepseek-coder-33b")

        score_no = compute_model_score(model, ctx_no_type)
        score_with = compute_model_score(model, ctx_with_type)

        # With task type should be at least as good
        assert score_with >= score_no

    def test_score_increases_with_capability_data(self):
        ctx_no_caps = ScoringContext()
        ctx_with_caps = ScoringContext(
            model_capability_scores={"coding": 0.95, "tool_use": 0.90}
        )
        model = _make_model()

        score_no = compute_model_score(model, ctx_no_caps)
        score_with = compute_model_score(model, ctx_with_caps)

        assert score_with > score_no

    def test_score_increases_with_history(self):
        ctx_no_hist = ScoringContext()
        ctx_with_hist = ScoringContext(historical_success_rate=0.95)
        model = _make_model()

        score_no = compute_model_score(model, ctx_no_hist)
        score_with = compute_model_score(model, ctx_with_hist)

        assert score_with > score_no

    def test_user_selected_model_gets_bonus(self):
        ctx = ScoringContext(user_selected_model="selected")
        selected = _make_model(model_id="selected")
        other = _make_model(model_id="other")

        score_selected = compute_model_score(selected, ctx)
        score_other = compute_model_score(other, ctx)

        assert score_selected > score_other

    def test_free_model_preferred_when_configured(self):
        ctx = ScoringContext(prefer_free=True)
        free = _make_model(model_id="free-model", is_free=True)
        paid = _make_model(model_id="paid-model", is_free=False)

        score_free = compute_model_score(free, ctx)
        score_paid = compute_model_score(paid, ctx)

        assert score_free > score_paid

    def test_tool_support_required(self):
        ctx = ScoringContext(requires_tools=True)
        with_tools = _make_model(supports_tools=True)
        without_tools = _make_model(supports_tools=False)

        score_with = compute_model_score(with_tools, ctx)
        score_without = compute_model_score(without_tools, ctx)

        assert score_with > score_without

    def test_score_range(self):
        ctx = ScoringContext()
        model = _make_model()
        score = compute_model_score(model, ctx)
        assert 0.0 <= score <= 1.0

    def test_ranking_order(self):
        """Test that a better model scores higher than a worse one."""
        ctx = ScoringContext(task_type="bug_fix", requires_tools=True)
        good = _make_model(
            model_id="deepseek-coder-33b",
            context_window=128000,
            supports_tools=True,
            latency_ms=500,
        )
        bad = _make_model(
            model_id="small-model",
            context_window=4096,
            supports_tools=False,
            latency_ms=5000,
        )

        score_good = compute_model_score(good, ctx)
        score_bad = compute_model_score(bad, ctx)

        assert score_good > score_bad
