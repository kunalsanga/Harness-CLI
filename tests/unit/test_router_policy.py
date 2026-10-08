"""Phase 10 — per-agent model_policy routing tests (Part 10).

Proves that role policies ("reasoning_high", "coding", "fast", "cheap")
passed through ``routing_mode_override`` are translated into the scoring
context the 14-dimension ranking consumes, so an architect actually gets a
reasoning-biased chain and a tester gets a fast/cheap chain.
"""

from __future__ import annotations

import asyncio

from harness_core.observability.events import EventBus
from harness_core.providers.base import CompletionRequest
from harness_core.routing.router import ModelRouter, RouterConfig


def _make_request(messages: list[dict] | None = None) -> CompletionRequest:
    return CompletionRequest(
        model="",
        messages=messages or [{"role": "user", "content": "Implement the API"}],
        tools=[{"type": "function"}],
    )


def _router() -> ModelRouter:
    return ModelRouter(providers={}, config=RouterConfig(), event_bus=EventBus())


def test_reasoning_high_policy_adds_reasoning_tags():
    router = _router()
    ctx = router._build_scoring_context(_make_request(), "reasoning_high")
    assert "reasoning" in ctx.task_tags
    assert "plan" in ctx.task_tags


def test_coding_policy_adds_coding_tag():
    router = _router()
    ctx = router._build_scoring_context(_make_request(), "coding")
    assert "coding" in ctx.task_tags


def test_fast_policy_prefers_free_when_no_user_model():
    router = _router()
    ctx = router._build_scoring_context(_make_request(), "fast")
    assert ctx.prefer_free is True
    assert "fast" in ctx.task_tags


def test_fast_policy_respects_user_selected_model():
    router = _router()
    request = _make_request()
    request.model = "some-user-pinned-model"
    ctx = router._build_scoring_context(request, "fast")
    # The user pinned a model: free-preference must not be forced on.
    assert ctx.prefer_free is False


def test_auto_policy_unchanged():
    router = _router()
    ctx = router._build_scoring_context(_make_request(), "auto")
    assert ctx.prefer_free is False
    assert "reasoning" not in ctx.task_tags


def test_role_policy_is_recorded_in_routing_decision():
    """The routing decision records the active policy/mode end-to-end."""
    from harness_core.providers.base import CompletionResponse, ModelInfo, ModelProvider

    class _P(ModelProvider):
        name = "fake"

        async def health_check(self) -> bool:
            return True

        async def close(self) -> None:
            return None

        async def list_models(self):
            return [
                ModelInfo(
                    id="fast-mini", name="fast-mini", provider="fake",
                    context_window=32000, supports_tools=True, is_free=True,
                    latency_ms=50, reliability=0.9,
                ),
                ModelInfo(
                    id="reasoner-pro", name="reasoner-pro", provider="fake",
                    context_window=128000, supports_tools=True, is_free=False,
                    latency_ms=500, reliability=0.95,
                ),
            ]

        async def generate(self, request):
            return CompletionResponse(content="ok", model="fast-mini")

        async def stream(self, request):
            yield CompletionResponse(content="ok", model="fast-mini")

    # allow_paid_models=True isolates the variable under test: role-policy
    # ordering. The paid-opt-in routing gate has its own dedicated tests.
    router = ModelRouter(
        providers=[_P()],
        config=RouterConfig(allow_paid_models=True),
        event_bus=EventBus(),
    )

    async def _select(policy: str) -> list:
        return await router.select_models(_make_request(), routing_mode_override=policy)

    # reasoning_high ranks the reasoning model first...
    chain = asyncio.run(_select("reasoning_high"))
    assert chain, "expected a non-empty chain"
    assert chain[0][0] == "reasoner-pro"

    # ...and fast prefers the fast/free model.
    chain_fast = asyncio.run(_select("fast"))
    assert chain_fast, "expected a non-empty chain"
    assert chain_fast[0][0] == "fast-mini"
