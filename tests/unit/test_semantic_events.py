"""Regression tests for semantic events, sanitization, stream normalization (Parts 8-10, 17)."""

from __future__ import annotations

import pytest

from harness_core.observability.semantic import (
    CONTEXT_COMPACTION_COMPLETED,
    RUN_STARTED,
    STEERING_APPLIED,
    TOOL_COMPLETED,
    make_event,
    model_event_data,
    normalize_model_stream_chunk,
    redact_sensitive,
    tool_lifecycle_payload,
)
from harness_core.observability.events import EventBus
from harness_core.providers.stream_events import (
    normalize_stream_error,
    normalize_stream_event,
)
from harness_core.providers.base import CompletionResponse


# ── Sanitization (no secret leakage) ─────────────────────────────────────


class TestSanitization:
    def test_api_key_redacted(self):
        out = redact_sensitive({"api_key": "sk-super-secret", "model": "x"})
        assert out["api_key"] == "[REDACTED]"
        assert out["model"] == "x"

    def test_authorization_header_redacted(self):
        out = redact_sensitive({"headers": {"Authorization": "Bearer abc", "Accept": "json"}})
        assert out["headers"]["Authorization"] == "[REDACTED]"
        assert out["headers"]["Accept"] == "json"

    def test_nested_token_redacted(self):
        out = redact_sensitive({"a": {"b": {"credential": "x"}}})
        assert out["a"]["b"]["credential"] == "[REDACTED]"

    def test_text_bounded(self):
        out = redact_sensitive({"error": "x" * 99999})
        assert len(out["error"]) <= 2000

    def test_list_bounded(self):
        out = redact_sensitive({"files": [f"f{i}" for i in range(500)]})
        assert len(out["files"]) <= 50

    def test_make_event_stamps_run_id(self):
        ev = make_event(RUN_STARTED, "test", {"goal": "g"}, run_id="r1")
        assert ev.data["run_id"] == "r1"
        assert ev.type == RUN_STARTED

    def test_make_event_sanitizes(self):
        ev = make_event(TOOL_COMPLETED, "test", {"api_key": "secret"})
        assert ev.data["api_key"] == "[REDACTED]"

    def test_no_secret_leak_through_eventbus(self):
        """End-to-end: emit an event carrying a credential-shaped key and
        verify no handler ever sees the raw value."""
        bus = EventBus()
        seen = {}

        async def handler(ev):
            seen.update(ev.data)

        bus.on("*", handler)

        async def flow():
            ev = make_event(STEERING_APPLIED, "test", {
                "text": "do x", "password": "hunter2",
            }, run_id="r")
            await bus.emit(ev)

        import asyncio
        asyncio.run(flow())
        assert seen["password"] == "[REDACTED]"
        assert seen["text"] == "do x"

    def test_tool_lifecycle_payload_shape(self):
        p = tool_lifecycle_payload(
            tool_name="edit_file",
            call_id="c1",
            status="completed",
            summary="edit: a.py",
            duration_ms=12.3,
            error_class="",
            affected_files=["a.py"],
        )
        assert p["tool"] == "edit_file"
        assert p["status"] == "completed"
        assert p["duration_ms"] == 12.3
        assert p["affected_files"] == ["a.py"]

    def test_model_event_data(self):
        d = model_event_data("m1", "openrouter", "started", iteration=1)
        assert d["model"] == "m1" and d["kind"] == "started"


# ── Provider stream normalization (provider neutrality) ──────────────────


class TestStreamNormalization:
    def test_string_delta(self):
        ev = normalize_stream_event("hello ")
        assert ev == {"kind": "delta", "text": "hello "}

    def test_empty_string_returns_none(self):
        assert normalize_stream_event("") is None
        assert normalize_stream_event(None) is None

    def test_response_delta(self):
        resp = CompletionResponse(content="text", model="m", provider="p")
        assert normalize_stream_event(resp) == {"kind": "delta", "text": "text"}

    def test_response_tool_call(self):
        resp = CompletionResponse(
            content="",
            tool_calls=[{"id": "1", "function": {"name": "read_file", "arguments": "{}"}}],
        )
        ev = normalize_stream_event(resp)
        assert ev["kind"] == "tool_call_started"
        assert ev["tool_calls"][0]["name"] == "read_file"

    def test_error_chunk(self):
        ev = normalize_stream_event({"error": "rate limited"})
        assert ev["kind"] == "error"

    def test_finish_reason(self):
        resp = CompletionResponse(content="", finish_reason="stop")
        assert normalize_stream_event(resp)["kind"] == "completed"

    def test_unknown_shape_degrades_safely(self):
        ev = normalize_stream_event(object())
        assert ev is not None and ev["kind"] in ("completed",)

    def test_exception_normalization_bounded(self):
        err = normalize_stream_error(RuntimeError("x" * 5000))
        assert err["kind"] == "error"
        assert len(err["error"]) <= 500

    def test_normalize_model_stream_chunk_parity(self):
        assert normalize_model_stream_chunk("hi") == {"kind": "delta", "text": "hi"}
        resp = CompletionResponse(content="", finish_reason="stop")
        assert normalize_model_stream_chunk(resp)["kind"] == "completed"


# ── Agent loop integration: semantic events + steering at the boundary ───


class _FakeProvider:
    """Provider that reads a file, then finishes after seeing steering."""

    def __init__(self):
        self.model = "fake"
        self.calls = 0

    @property
    def name(self):
        return "fake"

    async def generate(self, request):
        self.calls += 1
        # Planning requests carry no tool schemas; only answer them with text.
        if not getattr(request, "tools", None):
            return CompletionResponse(
                content="1. Read the file\n2. Report findings",
                model="fake",
                provider="fake",
            )
        if self.calls <= 2:
            return CompletionResponse(
                content="",
                tool_calls=[{
                    "id": "t1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "hello.txt"}',
                    },
                }],
                model="fake",
                provider="fake",
            )
        # Finish with text
        return CompletionResponse(
            content="done",
            model="fake",
            provider="fake",
            finish_reason="stop",
        )

    async def stream(self, request):
        yield CompletionResponse(content="done", model="fake", provider="fake")

    async def list_models(self):
        return []

    async def health_check(self):
        return True


class TestLoopSemanticEvents:
    @pytest.mark.asyncio
    async def test_run_lifecycle_events_and_steering(self, tmp_path):
        """End-to-end-ish: request → context discovery → model → tool →
        steering mid-run → continue → completion, all with semantic events."""
        (tmp_path / "hello.txt").write_text("hi\n")
        from harness_core.agent.loop import AgentLoop
        from harness_core.agent.types import AgentConfig, TaskStatus
        from harness_core.tools.filesystem import ReadFileTool
        from harness_core.observability import semantic

        provider = _FakeProvider()
        loop = AgentLoop(
            provider=provider,
            tools=[ReadFileTool()],
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=5, verify_on_complete=False),
        )
        # Queue steering BEFORE the run: it must be applied at the first
        # step boundary (iteration 1).
        await loop.submit_steering_message("keep the file small")

        task = await loop.run("read hello.txt")

        assert task.status in (TaskStatus.COMPLETED, TaskStatus.PARTIAL)
        # Tool actually executed.
        assert any(tc.tool_name == "read_file" for tc in task.tool_calls)

        history = loop.event_bus.get_history()
        types = {e.type for e in history}

        # Canonical lifecycle events all present.
        assert semantic.RUN_STARTED in types
        assert semantic.CONTEXT_DISCOVERY_STARTED in types
        assert semantic.CONTEXT_DISCOVERY_COMPLETED in types
        assert semantic.TOOL_STARTED in types
        assert semantic.TOOL_COMPLETED in types
        assert semantic.STEERING_RECEIVED in types
        assert semantic.STEERING_APPLIED in types
        assert semantic.RUN_COMPLETED in types

        # run_id stamped on semantic events for whole-run reconstruction.
        started = next(e for e in history if e.type == semantic.RUN_STARTED)
        assert started.data.get("run_id") == loop.run_id

        # Steering was rendered into corrections (injected at boundary).
        assert any("keep the file small" in c for c in loop._corrections)

        # No secrets / raw output dumped in tool events.
        for e in history:
            assert e.data.get("api_key") is None

    @pytest.mark.asyncio
    async def test_quiet_marking_for_readonly_tools(self, tmp_path):
        (tmp_path / "a.txt").write_text("x\n")
        from harness_core.agent.loop import AgentLoop
        from harness_core.agent.types import AgentConfig
        from harness_core.tools.filesystem import ReadFileTool
        from harness_core.observability import semantic

        loop = AgentLoop(
            provider=_FakeProvider(),
            tools=[ReadFileTool()],
            workspace_root=tmp_path,
            config=AgentConfig(max_iterations=5, verify_on_complete=False),
        )
        await loop.run("read a.txt")
        started = [e for e in loop.event_bus.get_history()
                   if e.type == semantic.TOOL_STARTED]
        assert started and all(e.data.get("quiet") is True for e in started)
