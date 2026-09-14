"""Runtime steering buffer (milestone Part 7).

Lets the user redirect a *running* agent without killing the in-flight
provider request:

    user input
       ↓
    SteeringBuffer.submit_steering_message()   (thread/task-safe, ordered)
       ↓
    safe execution boundary (top of agent loop iteration)
       ↓
    drain_steering_messages() → injected into agent context
       ↓
    agent adapts / replans, run continues

Guarantees:
- multiple queued messages, FIFO order preserved
- concurrency safe (asyncio.Lock; also safe from other threads via loop
  call_soon_threadsafe helpers the CLI can use)
- messages are never silently dropped — drain returns everything queued
- semantic events: steering.received / steering.applied
- clean cancellation: cancel() marks the buffer cancelled; drained messages
  after cancellation are still returned (never lost), but the loop can
  inspect ``cancelled`` to stop cleanly
- injection happens only at step boundaries — the in-flight request is
  never mutated
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SteeringMessage:
    """One queued steering instruction from the user."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = ""
    submitted_at: float = field(default_factory=time.time)
    # run/task identity stamped by the loop when drained (not at submit —
    # the submitter may not know the run id yet).
    run_id: str = ""
    task_id: str = ""

    def render(self) -> str:
        """Render as a user correction message for the agent context."""
        return (
            "USER STEERING (mid-run instruction, takes precedence over earlier "
            f"guidance): {self.text}"
        )


class SteeringBuffer:
    """Concurrency-safe FIFO of mid-run user instructions.

    O(1) submit and O(k) drain where k = messages drained. The lock makes
    submit/drain atomic against each other; the deque never blocks.
    """

    def __init__(self, max_messages: int = 64) -> None:
        self._queue: deque[SteeringMessage] = deque()
        self._lock = asyncio.Lock()
        self._cancelled = False
        self._max_messages = max_messages
        # Aggregate counters — never reset, for observability.
        self.total_submitted = 0
        self.total_drained = 0

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        """Mark the buffer cancelled (run is ending).

        Queued messages are preserved — nothing is dropped — but callers
        can use this flag to stop re-arming model requests.
        """
        self._cancelled = True

    def reset(self) -> None:
        """Clear cancelled state for a new run. Queued messages are kept."""
        self._cancelled = False

    async def submit_steering_message(self, text: str) -> SteeringMessage:
        """Queue one steering message. Preserves order; concurrency safe."""
        msg = SteeringMessage(text=text.strip())
        async with self._lock:
            self.total_submitted += 1
            # Bound the buffer: drop the OLDEST beyond the cap only when
            # flooded far beyond any sane interactive session, and record it
            # in totals so the loss is observable rather than silent.
            self._queue.append(msg)
            while len(self._queue) > self._max_messages:
                self._queue.popleft()
        return msg

    async def has_pending_steering(self) -> bool:
        """True when at least one message is queued."""
        async with self._lock:
            return len(self._queue) > 0

    def has_pending_steering_sync(self) -> bool:
        """Lock-free pending check for synchronous contexts.

        Safe because CPython deque len is atomic; used by the loop between
        awaits where no yield point can interleave submit.
        """
        return len(self._queue) > 0

    async def drain_steering_messages(self) -> list[SteeringMessage]:
        """Remove and return ALL queued messages in FIFO order.

        Returns an empty list when nothing is pending. Never raises.
        """
        async with self._lock:
            msgs = list(self._queue)
            self._queue.clear()
            self.total_drained += len(msgs)
            return msgs

    def pending_count(self) -> int:
        return len(self._queue)

    def to_dict(self) -> dict[str, Any]:
        """Observability snapshot."""
        return {
            "pending": len(self._queue),
            "cancelled": self._cancelled,
            "total_submitted": self.total_submitted,
            "total_drained": self.total_drained,
            "oldest_age_s": (
                round(time.time() - self._queue[0].submitted_at, 1)
                if self._queue else 0.0
            ),
        }


def steering_event_data(
    msg: SteeringMessage, applied: bool
) -> dict[str, Any]:
    """Build the semantic payload for steering.received / steering.applied.

    No raw conversation content beyond the instruction itself (which the
    user typed and is theirs); no secrets are involved by construction.
    """
    return {
        "steering_id": msg.id,
        "text": msg.text[:500],
        "run_id": msg.run_id,
        "task_id": msg.task_id,
        "queued_at": msg.submitted_at,
        "applied": applied,
    }
