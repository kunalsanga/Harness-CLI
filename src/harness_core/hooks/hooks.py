"""Hook system — lifecycle callbacks for extensions.

Hooks fire at well-defined lifecycle points. They may:
- Observe events (read-only)
- Modify allowed metadata
- Reject operations (where policy permits)

Hooks must NOT bypass security/permissions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class HookEvent(str, Enum):
    """Lifecycle events that hooks can subscribe to."""

    BEFORE_RUN = "before_run"
    AFTER_RUN = "after_run"
    BEFORE_AGENT = "before_agent"
    AFTER_AGENT = "after_agent"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    ON_ERROR = "on_error"
    ON_SESSION_RESUME = "on_session_resume"
    ON_CHECKPOINT = "on_checkpoint"
    ON_VERIFICATION = "on_verification"


@dataclass
class HookResult:
    """Result of a hook execution."""

    hook_id: str
    event: HookEvent
    success: bool
    modified_context: dict[str, Any] | None = None
    rejected: bool = False
    rejection_reason: str = ""
    duration_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "hook_id": self.hook_id,
            "event": self.event.value,
            "success": self.success,
            "rejected": self.rejected,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.rejection_reason:
            d["rejection_reason"] = self.rejection_reason
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class RegisteredHook:
    """A registered hook with its handler."""

    hook_id: str
    event: HookEvent
    handler: Callable[..., Any]
    name: str = ""
    priority: int = 100  # Lower = runs first
    enabled: bool = True
    source: str = ""  # plugin name, etc.
    description: str = ""


class HookRegistry:
    """Registry for lifecycle hooks.

    Supports:
    - Register/unregister hooks by event
    - Execute hooks in priority order
    - Collect modified context
    - Detect rejections
    - Error isolation per hook
    """

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[RegisteredHook]] = {}
        self._next_id: int = 1

    def register(
        self,
        event: HookEvent,
        handler: Callable[..., Any],
        name: str = "",
        priority: int = 100,
        source: str = "",
        description: str = "",
    ) -> str:
        """Register a hook. Returns hook_id."""
        hook_id = f"hook_{self._next_id}"
        self._next_id += 1

        hook = RegisteredHook(
            hook_id=hook_id,
            event=event,
            handler=handler,
            name=name or hook_id,
            priority=priority,
            source=source,
            description=description,
        )

        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(hook)
        self._hooks[event].sort(key=lambda h: h.priority)
        return hook_id

    def unregister(self, hook_id: str) -> bool:
        """Unregister a hook by ID."""
        for event, hooks in self._hooks.items():
            for i, h in enumerate(hooks):
                if h.hook_id == hook_id:
                    hooks.pop(i)
                    return True
        return False

    def enable(self, hook_id: str) -> bool:
        for hooks in self._hooks.values():
            for h in hooks:
                if h.hook_id == hook_id:
                    h.enabled = True
                    return True
        return False

    def disable(self, hook_id: str) -> bool:
        for hooks in self._hooks.values():
            for h in hooks:
                if h.hook_id == hook_id:
                    h.enabled = False
                    return True
        return False

    def list_hooks(self, event: HookEvent | None = None) -> list[RegisteredHook]:
        """List hooks, optionally filtered by event."""
        if event:
            return [h for h in self._hooks.get(event, []) if h.enabled]
        all_hooks = []
        for hooks in self._hooks.values():
            all_hooks.extend(h for h in hooks if h.enabled)
        return sorted(all_hooks, key=lambda h: h.priority)

    def execute(
        self,
        event: HookEvent,
        context: dict[str, Any] | None = None,
    ) -> list[HookResult]:
        """Execute all hooks for an event.

        Returns a list of HookResults. If any hook rejects, the
        context carries the rejection forward.
        """
        hooks = self._hooks.get(event, [])
        if not hooks:
            return []

        ctx = dict(context or {})
        results: list[HookResult] = []
        rejected = False

        for hook in hooks:
            if not hook.enabled:
                continue
            if rejected:
                # Skip remaining hooks after rejection
                continue

            start = time.monotonic()
            try:
                result = hook.handler(ctx)
                duration = (time.monotonic() - start) * 1000

                # Hook may return a dict to modify context
                modified = None
                is_rejected = False
                rejection_reason = ""

                if isinstance(result, dict):
                    if result.get("reject"):
                        is_rejected = True
                        rejection_reason = result.get("reason", "Hook rejected")
                        rejected = True
                    else:
                        modified = result
                        ctx.update(result)

                hook_result = HookResult(
                    hook_id=hook.hook_id,
                    event=event,
                    success=True,
                    modified_context=modified,
                    rejected=is_rejected,
                    rejection_reason=rejection_reason,
                    duration_ms=duration,
                )

            except Exception as e:
                duration = (time.monotonic() - start) * 1000
                hook_result = HookResult(
                    hook_id=hook.hook_id,
                    event=event,
                    success=False,
                    duration_ms=duration,
                    error=f"{type(e).__name__}: {str(e)[:200]}",
                )

            results.append(hook_result)

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get hook statistics."""
        total = sum(len(h) for h in self._hooks.values())
        by_event = {e.value: len(h) for e, h in self._hooks.items() if h}
        sources = set()
        for hooks in self._hooks.values():
            for h in hooks:
                if h.source:
                    sources.add(h.source)
        return {
            "total_hooks": total,
            "events": by_event,
            "sources": sorted(sources),
        }
