"""Event bus for observability and inter-component communication."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class Event:
    """An observable event."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    type: str = ""
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """Decoupled event system for observability."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._history: list[Event] = []

    def on(self, event_type: str, handler: EventHandler) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Remove a handler for an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    async def emit(self, event: Event) -> None:
        """Emit an event to all registered handlers."""
        self._history.append(event)
        handlers = self._handlers.get(event.type, []) + self._handlers.get("*", [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception:
                pass  # Don't let handler errors break event emission

    def get_history(self, event_type: str | None = None) -> list[Event]:
        """Get event history, optionally filtered by type."""
        if event_type:
            return [e for e in self._history if e.type == event_type]
        return list(self._history)
