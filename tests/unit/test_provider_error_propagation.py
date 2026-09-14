"""Regression: provider error propagation, fallback, and secret leakage.

Covers §10 of the provider-bug task:
- OpenRouter minimal chat (mocked), tool-enabled, streaming
- HTTP error codes 401/403/400/404/429/5xx/timeout/network
- Malformed provider response
- Free fallback per-model reasons preserved, never "unknown"
- No secret leakage (API key / Authorization)
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock
import httpx

from harness_core.providers.base import CompletionRequest, CompletionResponse, ModelInfo, ModelProvider
from harness_core.providers.openrouter import OpenRouterProvider
from harness_core.routing.fallback import FallbackEngine, FallbackConfig, RetryConfig, classify_error, ErrorClassification
from harness_core.routing.health import HealthEvent, ModelHealthTracker
from harness_core.routing.router import ModelRouter, RouterConfig


class FakeProvider(ModelProvider):
    def __init__(self, name="openrouter"):
        self._name = name
        self._fn = AsyncMock()
    @property
    def name(self): return self._name
    async def generate(self, r): return await self._fn(r)
    async def stream(self, r):
        yield ""
    async def list_models(self): return []
    async def health_check(self): return True


def _req(with_tools=False):
    tools = None
    if with_tools:
        tools = [{"type":"function","function":{"name":"read_file","description":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]
    return CompletionRequest(messages=[{"role":"user","content":"Reply with exactly: HARNESS_PROVIDER_OK"}], tools=tools)


def _make_request(status_code: int, json_body: dict, request_url="https://openrouter.ai/api/v1/chat/completions"):
    req = httpx.Request("POST", request_url)
    return httpx.Response(status_code=status_code, json=json_body, request=req)

# ── OpenRouter minimal / error extraction ────────────────────────────────

@pytest.mark.asyncio
async def test_minimal_no_tools_succeeds():
    p = OpenRouterProvider(api_key="sk-or-test")
    # mock httpx client
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices":[{"message":{"content":"HARNESS_PROVIDER_OK","tool_calls":[]},"finish_reason":"stop"}],"model":"m","usage":{}}
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    client = AsyncMock()
    client.post = AsyncMock(return_value=mock_resp)
    client.timeout = MagicMock(read=120)
    p._client = client
    # also need _get_client to return same
    p._get_client = AsyncMock(return_value=client)
    r = await p.generate(_req(with_tools=False))
    assert r.content == "HARNESS_PROVIDER_OK"
    assert r.tool_calls == []

@pytest.mark.asyncio
async def test_tool_enabled_request():
    p = OpenRouterProvider(api_key="sk-or-test")
    mock_resp = MagicMock()
    mock_resp.json.side_effect = [
        # raise_for_status needs no json; .json() for success
        {"choices":[{"message":{"content":"","tool_calls":[{"id":"1","type":"function","function":{"name":"read_file","arguments":'{"path":"README.md"}'}}]},"finish_reason":"tool_calls"}],"model":"m","usage":{}}
    ]
    # Make _extract not called; raise_for_status is no-op
    mock_resp.raise_for_status = MagicMock()
    mock_resp.status_code = 200
    # client.post must return mock_resp
    client = AsyncMock()
    # ensure mock_resp.json returns dict on first call
    mock_resp.json = MagicMock(return_value={"choices":[{"message":{"content":"","tool_calls":[{"id":"1","type":"function","function":{"name":"read_file","arguments":'{"path":"README.md"}'}}]},"finish_reason":"tool_calls"}],"model":"m","usage":{}})
    client.post = AsyncMock(return_value=mock_resp)
    client.timeout = MagicMock(read=120)
    p._get_client = AsyncMock(return_value=client)
    r = await p.generate(_req(with_tools=True))
    assert len(r.tool_calls) == 1
    assert r.tool_calls[0]["function"]["name"] == "read_file"

@pytest.mark.asyncio
async def test_401_preserved_with_model_id_and_no_secret():
    p = OpenRouterProvider(api_key="sk-or-v1-secret1234567890")
    err_resp = _make_request(401, {"error":{"message":"Unauthorized","code":401}})
    exc = httpx.HTTPStatusError("401", request=err_resp.request, response=err_resp)
    client = AsyncMock()
    client.post = AsyncMock(side_effect=exc)
    client.timeout = MagicMock(read=120)
    p._get_client = AsyncMock(return_value=client)
    with pytest.raises(RuntimeError) as ei:
        await p.generate(CompletionRequest(messages=[{"role":"user","content":"hi"}], model="openrouter/auto"))
    msg = str(ei.value)
    assert "401" in msg
    assert "openrouter/auto" in msg
    assert "sk-or-" not in msg
    assert "secret" not in msg.lower()

@pytest.mark.asyncio
async def test_429_preserved():
    p = OpenRouterProvider(api_key="k")
    err_resp = _make_request(429, {"error":{"message":"Rate limited"}})
    exc = httpx.HTTPStatusError("429", request=err_resp.request, response=err_resp)
    client = AsyncMock(); client.post = AsyncMock(side_effect=exc); client.timeout = MagicMock(read=120)
    p._get_client = AsyncMock(return_value=client)
    with pytest.raises(RuntimeError) as ei:
        await p.generate(_req())
    assert "429" in str(ei.value)

@pytest.mark.asyncio
async def test_400_invalid_request_preserved():
    p = OpenRouterProvider(api_key="k")
    err_resp = _make_request(400, {"error":{"message":"Invalid tool schema"}})
    exc = httpx.HTTPStatusError("400", request=err_resp.request, response=err_resp)
    client = AsyncMock(); client.post = AsyncMock(side_effect=exc); client.timeout = MagicMock(read=120)
    p._get_client = AsyncMock(return_value=client)
    with pytest.raises(RuntimeError) as ei:
        await p.generate(_req(with_tools=True))
    assert "400" in str(ei.value)
    assert "Invalid tool" in str(ei.value)

@pytest.mark.asyncio
async def test_404_model_unavailable():
    p = OpenRouterProvider(api_key="k")
    err_resp = _make_request(404, {"error":{"message":"Model not found"}})
    exc = httpx.HTTPStatusError("404", request=err_resp.request, response=err_resp)
    client = AsyncMock(); client.post = AsyncMock(side_effect=exc); client.timeout = MagicMock(read=120)
    p._get_client = AsyncMock(return_value=client)
    with pytest.raises(RuntimeError) as ei:
        await p.generate(CompletionRequest(messages=[{"role":"user","content":"hi"}], model="not-a-model"))
    assert "404" in str(ei.value)

@pytest.mark.asyncio
async def test_5xx_and_timeout_network():
    for code in [500, 502, 503]:
        assert classify_error(Exception(f"{code} Server Error")) == ErrorClassification.RETRYABLE
    assert classify_error(Exception("Connection timed out")) == ErrorClassification.RETRYABLE
    assert classify_error(Exception("Network error")) == ErrorClassification.RETRYABLE

@pytest.mark.asyncio
async def test_malformed_provider_response_raises_typed():
    p = OpenRouterProvider(api_key="k")
    bad = MagicMock()
    bad.json.side_effect = ValueError("not json")
    bad.status_code = 200
    bad.raise_for_status = MagicMock()
    client = AsyncMock(); client.post = AsyncMock(return_value=bad); client.timeout = MagicMock(read=120)
    p._get_client = AsyncMock(return_value=client)
    with pytest.raises(RuntimeError) as ei:
        await p.generate(_req())
    assert "invalid response" in str(ei.value).lower()

@pytest.mark.asyncio
async def test_streaming_yields_content():
    p = OpenRouterProvider(api_key="k")
    # Simulate streamed chunks
    async def _aiter():
        yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
        yield 'data: [DONE]'
    mock_stream_resp = MagicMock()
    mock_stream_resp.raise_for_status = MagicMock()
    mock_stream_resp.aiter_lines = lambda: _aiter()
    class FakeCtx:
        async def __aenter__(self): return mock_stream_resp
        async def __aexit__(self, *a): return False
    client = MagicMock()
    client.stream = MagicMock(return_value=FakeCtx())
    client.timeout = MagicMock(read=120)
    p._get_client = AsyncMock(return_value=client)
    out = []
    async for ch in p.stream(_req()):
        out.append(ch)
    assert "".join(out) == "hello"

# ── Fallback: per-model reasons preserved, never unknown ────────────────

@pytest.mark.asyncio
async def test_fallback_never_unknown_when_all_skipped():
    engine = FallbackEngine(fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0)))
    # Mark first as unavailable so it's skipped
    engine.health.record_failure("m1", HealthEvent.AUTH_FAILED)
    engine.health.record_failure("m2", HealthEvent.AUTH_FAILED)
    p1 = FakeProvider("openrouter"); p2 = FakeProvider("openrouter")
    # Neither should be called; but ensure attempts reflect skipped
    chain = [("m1", p1), ("m2", p2)]
    res = await engine.execute(_req(), chain)
    assert not res.succeeded
    assert "unknown" not in res.final_error.lower()
    assert "skipped" in res.final_error.lower()

@pytest.mark.asyncio
async def test_fallback_per_model_429_400_404_preserved():
    engine = FallbackEngine(fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0)))
    p = FakeProvider("openrouter")
    async def _gen(r):
        if r.model == "m-a": raise Exception("429 Too Many Requests")
        if r.model == "m-b": raise Exception("400 Invalid tool schema")
        if r.model == "m-c": raise Exception("404 Model not found")
        return CompletionResponse(content="ok")
    p._fn = _gen
    chain = [("m-a", p), ("m-b", p), ("m-c", p), ("m-d", p)]
    # m-d also fails with 500 to have a last error
    orig = p._fn
    async def _gen2(r):
        if r.model == "m-d": raise Exception("500 Internal Server Error")
        return await orig(r)
    p._fn = _gen2
    res = await engine.execute(_req(), chain)
    assert not res.succeeded
    assert "unknown" not in res.final_error.lower()
    # Should contain per-model breakdown or status codes
    low = res.final_error.lower()
    assert "429" in res.final_error or "rate limit" in low
    assert any("400" in a.get("error","") or "500" in a.get("error","") for a in res.attempts if a.get("status")=="error")

@pytest.mark.asyncio
async def test_fallback_empty_message_not_unknown():
    engine = FallbackEngine(fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0)))
    p = FakeProvider("openrouter")
    p._fn = AsyncMock(side_effect=Exception(""))  # empty message
    res = await engine.execute(_req(), [("m1", p)])
    assert not res.succeeded
    assert "unknown" not in res.final_error.lower()
    assert "empty message" in res.final_error.lower()

@pytest.mark.asyncio
async def test_no_secret_in_final_error():
    engine = FallbackEngine(fallback_config=FallbackConfig(retry=RetryConfig(max_retries=0)))
    p = FakeProvider("openrouter")
    p._fn = AsyncMock(side_effect=Exception("OpenRouter 401 Unauthorized (model=openrouter/auto)"))
    res = await engine.execute(_req(), [("m1", p)])
    assert "sk-or" not in res.final_error
    assert "Authorization" not in res.final_error

# ── Classifier & orchestrator: provider failures not UNKNOWN ──────────

def _make_agent_result(errors, status_failed=True):
    from harness_core.agents.domain import AgentResult, AgentStatus, SubTask
    from harness_core.agents.domain import AgentRole as DomainRole
    # minimal AgentResult
    r = AgentResult(status=AgentStatus.FAILED if status_failed else AgentStatus.COMPLETED, errors=errors, summary="", files_changed=[], findings=[], tests_total=0, tests_passed=0)
    return r

def test_classifier_maps_401_not_unknown():
    from harness_core.recovery.classifier import FailureClassifier, FailureCategory
    from harness_core.agents.domain import SubTask, AgentRole as DomainRole
    task = SubTask(task_id="t1", description="do", role=DomainRole.CODER)
    res = _make_agent_result(["OpenRouter 401 Unauthorized (model=openrouter/auto)"])
    c = FailureClassifier.classify(task, res)
    assert c.category == FailureCategory.MODEL_FAILURE
    assert "401" in c.summary or "auth" in c.summary.lower()

def test_classifier_maps_429():
    from harness_core.recovery.classifier import FailureClassifier, FailureCategory
    from harness_core.agents.domain import SubTask, AgentRole as DomainRole
    task = SubTask(task_id="t1", description="do", role=DomainRole.CODER)
    res = _make_agent_result(["429 Too Many Requests for https://openrouter.ai/api/v1/chat/completions"])
    c = FailureClassifier.classify(task, res)
    assert c.category == FailureCategory.MODEL_FAILURE

@pytest.mark.asyncio
async def test_orchestrator_skips_provider_failure():
    from harness_core.recovery.orchestrator import RecoveryOrchestrator
    from harness_core.recovery.classifier import FailureClassification, FailureCategory
    from harness_core.agents.domain import SubTask, TaskGraph, AgentRole as DomainRole
    from harness_core.observability.events import EventBus
    from harness_core.agents.registry import AgentRegistry
    bus = EventBus(); reg = AgentRegistry()
    provider = FakeProvider("openrouter")
    orch = RecoveryOrchestrator(event_bus=bus, registry=reg, provider=provider, max_attempts=3)
    g = TaskGraph()
    t = SubTask(task_id="task1", description="do", role=DomainRole.CODER)
    g.add_task(t)
    cls = FailureClassification(category=FailureCategory.MODEL_FAILURE, summary="Provider rate limited (429): 429", evidence=["429"])
    ok = await orch.handle_failure(g, t, cls)
    assert ok is False
    assert any(e.type == "recovery_skipped" for e in bus.get_history())
