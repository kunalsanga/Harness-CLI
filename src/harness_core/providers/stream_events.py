"""Provider stream event normalization (milestone Part 9).

Maps provider-specific streaming onto the stable Harness model event
vocabulary so the TUI never sees provider formats:

    Provider.stream()
       ↓ normalize_stream_event()
    {kind: delta|tool_call_started|tool_call_arguments|tool_result|
           error|completed|reasoning_status, ...}

All text is bounded; reasoning content is reduced to safe status metadata
(never leaked chain-of-thought); errors are truncated and secret-free by
construction (providers raise RuntimeError with bounded messages).
"""

from __future__ import annotations

import json
from typing import Any

_MAX_TEXT = 4_000


def _bound(text: str) -> str:
    return text[:_MAX_TEXT]


def normalize_stream_event(raw: Any) -> dict[str, Any] | None:
    """Normalize one raw provider stream item to a semantic event dict.

    Handles the two shapes providers actually produce today:

    - ``str`` — plain text delta (OpenRouter/Ollama/LiteLLM/NVIDIA all
      yield content deltas as strings)
    - ``CompletionResponse``-like — non-streaming fallback shape

    Returns None for chunks with no user-facing content (keepalives,
    empty deltas). Unknown shapes return a generic completed marker so a
    new provider version can never wedge the TUI.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return {"kind": "delta", "text": _bound(raw)} if raw else None
    if isinstance(raw, dict):
        err = raw.get("error")
        if err:
            return {"kind": "error", "error": _bound(str(err))}
        content = raw.get("content")
        if content:
            return {"kind": "delta", "text": _bound(str(content))}
        return None
    # CompletionResponse-like object
    if getattr(raw, "tool_calls", None):
        calls = []
        for tc in raw.tool_calls[:10]:
            func = tc.get("function", {}) if isinstance(tc, dict) else {}
            calls.append({
                "id": (tc.get("id", "") if isinstance(tc, dict) else ""),
                "name": str(func.get("name", "")),
                "arguments": _bound(str(func.get("arguments", ""))),
            })
        return {"kind": "tool_call_started", "tool_calls": calls}
    content = getattr(raw, "content", "") or ""
    if content:
        return {"kind": "delta", "text": _bound(content)}
    finish = getattr(raw, "finish_reason", "") or ""
    if finish:
        return {"kind": "completed", "finish_reason": str(finish)}
    # Unknown object shape — degrade safely.
    return {"kind": "completed", "finish_reason": "unknown_chunk"}


def normalize_stream_error(exc: BaseException) -> dict[str, Any]:
    """Normalize a stream exception into a model.error payload.

    Providers raise RuntimeError with messages that are already bounded and
    contain no Authorization headers (they are stripped before request
    send); we still truncate defensively here.
    """
    msg = str(exc)[:500]
    return {"kind": "error", "error": msg}


def tool_arguments_delta_state(arguments_so_far: str) -> dict[str, Any]:
    """Incremental tool-call argument stream state (safe partial view).

    Providers that stream tool arguments give partial JSON; we expose only
    the length and a bounded tail so the UI can show progress without
    parsing untrusted partial JSON.
    """
    return {
        "arguments_bytes": len(arguments_so_far),
        "tail": _bound(arguments_so_far[-200:]) if arguments_so_far else "",
    }
