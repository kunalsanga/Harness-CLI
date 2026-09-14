"""Canonical semantic runtime events (milestone Parts 8/9).

The EventBus is the single runtime→UI contract. Provider-specific events
never cross this boundary: providers are normalized upstream (see
``providers/stream_events``) and only these stable event families reach
the TUI:

    run.*        run.started / run.completed / run.failed / run.cancelled
    intent.*     intent.detected
    context.*    discovery_started / discovery_completed / file_selected /
                 compaction_started / compaction_completed
    agent.*      started / status_changed / completed / failed /
                 question_requested / question_answered
    tool.*       started / completed / failed
    worker.*     spawned / started / completed / failed
    verification.started / completed / failed
    recovery.*   started / completed / exhausted
    steering.*   received / applied
    model.*      started / delta / completed / error / switched
    todo.updated (pre-existing, preserved)

Compatibility: existing event names (task.started, tool.call, tool.result,
todo.updated, verification.*, model.switched, ...) remain authoritative for
current consumers; new canonical names are additive. ``run_id`` is stamped
on every event so a whole run can be reconstructed from the stream.
"""

from __future__ import annotations

import time
from typing import Any

# ── Canonical event names (additive; existing names preserved) ──────────

RUN_STARTED = "run.started"
RUN_COMPLETED = "run.completed"
RUN_FAILED = "run.failed"
RUN_CANCELLED = "run.cancelled"

INTENT_DETECTED = "intent.detected"

CONTEXT_DISCOVERY_STARTED = "context.discovery_started"
CONTEXT_DISCOVERY_COMPLETED = "context.discovery_completed"
CONTEXT_FILE_SELECTED = "context.file_selected"
CONTEXT_COMPACTION_STARTED = "context.compaction_started"
CONTEXT_COMPACTION_COMPLETED = "context.compaction_completed"

AGENT_STARTED = "agent.started"
AGENT_STATUS_CHANGED = "agent.status_changed"
AGENT_COMPLETED = "agent.completed"
AGENT_FAILED = "agent.failed"
AGENT_QUESTION_REQUESTED = "agent.question_requested"
AGENT_QUESTION_ANSWERED = "agent.question_answered"

TOOL_STARTED = "tool.started"
TOOL_COMPLETED = "tool.completed"
TOOL_FAILED = "tool.failed"

WORKER_SPAWNED = "worker.spawned"
WORKER_STARTED = "worker.started"
WORKER_COMPLETED = "worker.completed"
WORKER_FAILED = "worker.failed"

VERIFICATION_STARTED = "verification.started"
VERIFICATION_COMPLETED = "verification.completed"
VERIFICATION_FAILED = "verification.failed"

RECOVERY_STARTED = "recovery.started"
RECOVERY_COMPLETED = "recovery.completed"
RECOVERY_EXHAUSTED = "recovery.exhausted"

STEERING_RECEIVED = "steering.received"
STEERING_APPLIED = "steering.applied"

MODEL_STARTED = "model.started"
MODEL_DELTA = "model.delta"
MODEL_COMPLETED = "model.completed"
MODEL_ERROR = "model.error"          # pre-existing name — reused
MODEL_SWITCHED = "model.switched"    # pre-existing name — reused

TODO_UPDATED = "todo.updated"        # pre-existing name — reused

# Events that are safe to mark "quiet" for ghost/background exploration UI
# (Part 13). Deterministic read-only operations only.
QUIET_EVENT_TYPES = frozenset({
    "tool.call", "tool.result",
    TOOL_STARTED, TOOL_COMPLETED,
    CONTEXT_FILE_SELECTED,
})

# Credential-shaped keys that must never appear in event payloads.
_SENSITIVE_KEYS = frozenset({
    "api_key", "authorization", "auth_header", "password", "secret",
    "token", "credential", "credentials", "private_key", "cookie",
})

_MAX_TEXT = 2000
_MAX_LIST = 50


def redact_sensitive(data: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``data`` safe for the event stream.

    Drops credential-shaped keys recursively and truncates oversized text
    so no giant payload (or key material) can leak into the UI or logs.
    Model output is untrusted input — this is the choke point.
    """
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        k = str(key)
        if k.lower() in _SENSITIVE_KEYS:
            cleaned[k] = "[REDACTED]"
            continue
        if isinstance(value, dict):
            cleaned[k] = redact_sensitive(value)
        elif isinstance(value, list):
            cleaned[k] = [
                redact_sensitive(v) if isinstance(v, dict)
                else (v[:_MAX_TEXT] if isinstance(v, str) else v)
                for v in value[:_MAX_LIST]
            ]
        elif isinstance(value, str):
            cleaned[k] = value[:_MAX_TEXT]
        else:
            cleaned[k] = value
    return cleaned


def make_event(
    event_type: str,
    source: str,
    data: dict[str, Any],
    *,
    run_id: str = "",
    task_id: str = "",
    agent_id: str = "",
    quiet: bool = False,
) -> Any:
    """Build a semantic Event with identity + sanitization applied.

    Returns the existing ``Event`` dataclass so all current consumers keep
    working; adds ``run_id``/``quiet`` via the metadata convention (data
    keys) because Event's shape is frozen for compatibility.
    """
    from harness_core.observability.events import Event

    payload = redact_sensitive(data)
    if run_id:
        payload.setdefault("run_id", run_id)
    if task_id:
        payload.setdefault("task_id", task_id)
    if agent_id:
        payload.setdefault("agent_id", agent_id)
    if quiet:
        payload["quiet"] = True
    return Event(
        type=event_type,
        source=source,
        data=payload,
        timestamp=time.time(),
    )


# ── Tool lifecycle helper (Part 10) ──────────────────────────────────────

def tool_lifecycle_payload(
    *,
    tool_name: str,
    call_id: str,
    status: str,
    summary: str = "",
    duration_ms: float = 0.0,
    error_class: str = "",
    affected_files: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical metadata for tool.started/completed/failed events.

    Captures what the UI needs (name, safe summary, timing, status, error
    classification, affected files) without dumping raw tool output.
    """
    payload: dict[str, Any] = {
        "tool": tool_name,
        "call_id": call_id,
        "status": status,
        "summary": (summary or "")[:300],
        "duration_ms": round(duration_ms, 1),
        "error_class": error_class,
        "affected_files": (affected_files or [])[:20],
    }
    if extra:
        payload.update(redact_sensitive(extra))
    return payload


# ── Model stream normalization (Part 9) ─────────────────────────────────

def normalize_model_stream_chunk(chunk: Any) -> dict[str, Any] | None:
    """Normalize one provider stream chunk into a semantic model event dict.

    Accepts whatever a provider's ``stream()`` yields (str deltas or
    CompletionResponse-like objects) and maps it onto the stable model
    event vocabulary:

        {kind: "delta"}       — text delta
        {kind: "completed"}   — model completed
        {kind: "error"}       — model error
        {kind: "tool_call"}   — tool call announced by the model
        None                  — nothing user-facing in this chunk

    Reasoning/thinking chunks are reduced to safe status metadata only —
    hidden chain-of-thought never crosses the boundary.
    """
    if chunk is None:
        return None
    if isinstance(chunk, str):
        return {"kind": "delta", "text": chunk} if chunk else None
    # CompletionResponse-like
    if isinstance(chunk, dict):
        if chunk.get("error"):
            return {
                "kind": "error",
                "error": str(chunk["error"])[:500],
            }
        text = chunk.get("content") or ""
        if text:
            return {"kind": "delta", "text": text}
        return None
    content = getattr(chunk, "content", "") or ""
    if getattr(chunk, "tool_calls", None):
        return {"kind": "tool_call", "tool_calls": chunk.tool_calls}
    if content:
        return {"kind": "delta", "text": content}
    finish = getattr(chunk, "finish_reason", "")
    if finish:
        return {"kind": "completed", "finish_reason": finish}
    return None


def model_event_data(
    model: str,
    provider: str,
    kind: str,
    **extra: Any,
) -> dict[str, Any]:
    """Uniform payload for model.* events (no secrets, bounded text)."""
    data: dict[str, Any] = {
        "model": model,
        "provider": provider,
        "kind": kind,
    }
    data.update(extra)
    return redact_sensitive(data)
