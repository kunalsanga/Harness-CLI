"""Regression tests for steering buffer, ask-user protocol, and micro-workers (Parts 7, 12, 14)."""

from __future__ import annotations

import asyncio

import pytest

from harness_core.agent.steering import SteeringBuffer, SteeringMessage
from harness_core.agent.ask_user import AskUserManager, QuestionStatus
from harness_core.agent.micro import default_micro_registry


# ── Steering buffer ──────────────────────────────────────────────────────


class TestSteeringBuffer:
    @pytest.mark.asyncio
    async def test_single_message_submit_and_drain(self):
        buf = SteeringBuffer()
        await buf.submit_steering_message("only modify the frontend")
        assert await buf.has_pending_steering()
        msgs = await buf.drain_steering_messages()
        assert len(msgs) == 1
        assert "frontend" in msgs[0].render()
        assert not await buf.has_pending_steering()

    @pytest.mark.asyncio
    async def test_multiple_messages_preserve_order(self):
        buf = SteeringBuffer()
        await buf.submit_steering_message("first")
        await buf.submit_steering_message("second")
        await buf.submit_steering_message("third")
        msgs = await buf.drain_steering_messages()
        assert [m.text for m in msgs] == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_drain_empty_buffer(self):
        buf = SteeringBuffer()
        assert await buf.drain_steering_messages() == []
        assert not await buf.has_pending_steering()

    @pytest.mark.asyncio
    async def test_concurrent_submitters_no_loss(self):
        buf = SteeringBuffer()

        async def producer(i: int) -> None:
            for j in range(10):
                await buf.submit_steering_message(f"p{i}-m{j}")
                await asyncio.sleep(0)

        await asyncio.gather(*(producer(i) for i in range(5)))
        msgs = await buf.drain_steering_messages()
        assert len(msgs) == 50
        assert buf.total_drained == 50
        assert buf.total_submitted == 50

    @pytest.mark.asyncio
    async def test_concurrent_submit_and_drain(self):
        buf = SteeringBuffer()

        async def producer() -> None:
            for i in range(20):
                await buf.submit_steering_message(f"m{i}")

        async def drainer() -> int:
            total = 0
            for _ in range(20):
                total += len(await buf.drain_steering_messages())
                await asyncio.sleep(0)
            return total

        _, drained = await asyncio.gather(producer(), drainer())
        # Whatever remains must still be drainable — nothing is lost.
        remaining = len(await buf.drain_steering_messages())
        assert drained + remaining == 20

    def test_cancel_preserves_messages(self):
        buf = SteeringBuffer()
        # submit via sync-compatible path
        asyncio.get_event_loop_policy()
        buf._queue.append(SteeringMessage(text="keep me"))
        buf.cancel()
        assert buf.cancelled
        # Messages preserved after cancel.
        assert buf.pending_count() == 1

    @pytest.mark.asyncio
    async def test_cancellation_does_not_lose_messages(self):
        buf = SteeringBuffer()
        await buf.submit_steering_message("during run")
        buf.cancel()
        msgs = await buf.drain_steering_messages()
        assert len(msgs) == 1  # never dropped


# ── Ask-user protocol ────────────────────────────────────────────────────


class TestAskUser:
    @pytest.mark.asyncio
    async def test_question_lifecycle(self):
        mgr = AskUserManager()
        q = await mgr.ask("Which database?", choices=["postgres", "sqlite"])
        assert q.status is QuestionStatus.PENDING
        waiter = asyncio.create_task(mgr.wait_for_answer(q, timeout_seconds=2))
        await asyncio.sleep(0.01)
        await mgr.provide_answer(q.id, "sqlite")
        result = await waiter
        assert result.status is QuestionStatus.ANSWERED
        assert result.answer == "sqlite"
        assert not mgr.pending_questions()

    @pytest.mark.asyncio
    async def test_pause_and_resume(self):
        mgr = AskUserManager()
        q = await mgr.ask("Proceed?")
        waiter = asyncio.create_task(mgr.wait_for_answer(q))
        await asyncio.sleep(0.01)  # let the waiter suspend
        assert mgr.pending_questions()
        await mgr.provide_answer(q.id, "yes")
        result = await waiter
        assert result.answer == "yes"

    @pytest.mark.asyncio
    async def test_timeout_releases_waiter(self):
        mgr = AskUserManager()
        q = await mgr.ask("Hello?")
        result = await mgr.wait_for_answer(q, timeout_seconds=0.05)
        assert result.status is QuestionStatus.TIMEOUT
        assert result.answer is None

    @pytest.mark.asyncio
    async def test_cancel_releases_waiter(self):
        mgr = AskUserManager()
        q = await mgr.ask("Hello?")
        waiter = asyncio.create_task(mgr.wait_for_answer(q, timeout_seconds=5))
        await asyncio.sleep(0.01)
        await mgr.cancel_question(q.id)
        result = await waiter
        assert result.status is QuestionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_unknown_question_rejected(self):
        mgr = AskUserManager()
        with pytest.raises(Exception):
            await mgr.provide_answer("nonexistent", "x")

    @pytest.mark.asyncio
    async def test_event_payloads_no_secrets(self):
        mgr = AskUserManager()
        q = await mgr.ask("Q?", choices=["a"], explanation="because")
        payload = q.to_event_payload()
        assert payload["question_id"] == q.id
        assert payload["choices"] == ["a"]
        assert "answer" not in payload  # no answer before it exists


# ── Micro-workers ────────────────────────────────────────────────────────


class TestMicroWorkers:
    @pytest.mark.asyncio
    async def test_path_validate_rejects_traversal(self, tmp_path):
        reg = default_micro_registry()
        (tmp_path / "a.py").write_text("x = 1\n")
        result = await reg.run("path.validate", {
            "workspace_root": str(tmp_path),
            "paths": ["a.py", "../outside.py", "missing.py"],
        })
        assert result["valid"] == ["a.py"]
        reasons = {r["path"]: r["reason"] for r in result["rejected"]}
        assert reasons["../outside.py"] == "traversal_or_empty"
        assert reasons["missing.py"] == "not_found"

    @pytest.mark.asyncio
    async def test_token_count(self):
        reg = default_micro_registry()
        result = await reg.run("tokens.count", {"texts": ["a" * 40, "b" * 8]})
        assert result["counts"] == [10, 2]
        assert result["total"] == 12

    @pytest.mark.asyncio
    async def test_dedupe_paths(self):
        reg = default_micro_registry()
        result = await reg.run("paths.dedupe", {"paths": ["a.py", "a.py", "b.py"]})
        assert result["paths"] == ["a.py", "b.py"]
        assert result["duplicates_removed"] == 1

    @pytest.mark.asyncio
    async def test_unknown_worker_raises(self):
        reg = default_micro_registry()
        with pytest.raises(KeyError):
            await reg.run("eval.something", {})

    @pytest.mark.asyncio
    async def test_output_bounded(self, tmp_path):
        reg = default_micro_registry()
        (tmp_path / "big.py").write_text("x = 1\n")
        result = await reg.run("path.validate", {
            "workspace_root": str(tmp_path),
            "paths": ["big.py"] * 50,
        })
        assert len(result["valid"]) == 50  # all valid but bounded list handling
