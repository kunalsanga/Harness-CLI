"""Ask-user backend protocol (milestone Part 14).

Backend contracts for an agent to request user input mid-run and pause
safely until the response arrives. The TUI implementation is owned by the
UI layer; the runtime only needs the contracts to exist and be safe.

Flow:

    agent.question_requested  (question + optional choices + run/task id)
       ↓  runtime awaits — the loop is suspended, not killed
    user answers (via UI calling provide_answer)
       ↓
    agent.question_answered   (answer flows back to the agent)

Cancellation safety: if the run is cancelled while paused, waiters are
released with a cancellation answer rather than hanging forever.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestionStatus(str, Enum):
    PENDING = "pending"
    ANSWERED = "answered"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AgentQuestion:
    """A question the agent asks the user."""

    question: str
    choices: list[str] = field(default_factory=list)
    allow_free_text: bool = True
    explanation: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str = ""
    task_id: str = ""
    created_at: float = field(default_factory=time.time)
    status: QuestionStatus = QuestionStatus.PENDING
    answer: str | None = None
    answered_at: float | None = None

    def to_event_payload(self) -> dict[str, Any]:
        """Semantic payload for agent.question_requested."""
        return {
            "question_id": self.id,
            "question": self.question[:500],
            "choices": list(self.choices),
            "allow_free_text": self.allow_free_text,
            "explanation": self.explanation[:300],
            "run_id": self.run_id,
            "task_id": self.task_id,
        }


class AskUserError(Exception):
    """Raised on protocol misuse (e.g. answering an unknown question)."""


class AskUserManager:
    """Pending-question registry bridging the agent loop and the UI.

    The loop creates a question and awaits ``wait_for_answer``; the UI
    layer resolves it with ``provide_answer``. Async-safe: the registry is
    keyed by question id and each waiter gets its own event.
    """

    def __init__(self, default_timeout_seconds: float | None = None) -> None:
        self._pending: dict[str, AgentQuestion] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        self._default_timeout = default_timeout_seconds

    async def ask(
        self,
        question: str,
        choices: list[str] | None = None,
        explanation: str = "",
        allow_free_text: bool = True,
        timeout_seconds: float | None = None,
    ) -> AgentQuestion:
        """Register a question. Does NOT wait — see wait_for_answer.

        Splitting registration from waiting lets the caller emit the
        semantic event between the two, so the UI can start rendering the
        question before the loop suspends.
        """
        q = AgentQuestion(
            question=question,
            choices=list(choices or []),
            allow_free_text=allow_free_text,
            explanation=explanation,
        )
        async with self._lock:
            self._pending[q.id] = q
            self._events[q.id] = asyncio.Event()
        return q

    async def wait_for_answer(
        self, question: AgentQuestion, timeout_seconds: float | None = None
    ) -> AgentQuestion:
        """Suspend until the user answers, cancels, or times out.

        Returns the same question with status/answer filled in. Never
        raises on timeout/cancel — the caller inspects status and continues
        gracefully (e.g. proceeds with the free-text default).
        """
        ev = self._events.get(question.id)
        if ev is None:
            raise AskUserError(f"unknown question: {question.id}")
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout
        try:
            if timeout is not None:
                await asyncio.wait_for(ev.wait(), timeout=timeout)
            else:
                await ev.wait()
        except (asyncio.TimeoutError, TimeoutError):
            question.status = QuestionStatus.TIMEOUT
            question.answer = None
            return question
        # Answer (or cancellation) already applied by provide_answer/cancel.
        return question

    async def provide_answer(self, question_id: str, answer: str) -> AgentQuestion:
        """Resolve a pending question with the user's answer."""
        async with self._lock:
            q = self._pending.get(question_id)
            ev = self._events.get(question_id)
            if q is None or ev is None:
                raise AskUserError(f"unknown question: {question_id}")
            q.answer = answer
            q.status = QuestionStatus.ANSWERED
            q.answered_at = time.time()
        ev.set()
        return q

    async def cancel_question(self, question_id: str) -> AgentQuestion:
        """Cancel a pending question (run cancelled / UI closed).

        Waiters are released so the loop never hangs.
        """
        async with self._lock:
            q = self._pending.get(question_id)
            ev = self._events.get(question_id)
            if q is None or ev is None:
                raise AskUserError(f"unknown question: {question_id}")
            q.status = QuestionStatus.CANCELLED
            q.answer = None
        ev.set()
        return q

    def pending_questions(self) -> list[AgentQuestion]:
        return [q for q in self._pending.values() if q.status is QuestionStatus.PENDING]

    def cleanup(self, question_id: str) -> None:
        """Drop registry entries once the caller has consumed the result."""
        self._pending.pop(question_id, None)
        self._events.pop(question_id, None)


def question_answered_payload(q: AgentQuestion) -> dict[str, Any]:
    """Semantic payload for agent.question_answered."""
    return {
        "question_id": q.id,
        "status": q.status.value,
        "answer": (q.answer[:500] if q.answer is not None else None),
        "answered_at": q.answered_at,
        "run_id": q.run_id,
        "task_id": q.task_id,
    }
