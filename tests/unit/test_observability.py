"""Unit tests for the event bus."""

import pytest

from harness_core.observability.events import Event, EventBus


class TestEventBus:
    """Tests for EventBus."""

    @pytest.mark.asyncio
    async def test_emit_and_receive(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.on("test", handler)
        await bus.emit(Event(type="test", data={"key": "value"}))

        assert len(received) == 1
        assert received[0].data["key"] == "value"

    @pytest.mark.asyncio
    async def test_wildcard_handler(self):
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event.type)

        bus.on("*", handler)
        await bus.emit(Event(type="foo"))
        await bus.emit(Event(type="bar"))

        assert received == ["foo", "bar"]

    @pytest.mark.asyncio
    async def test_handler_error_does_not_break(self):
        bus = EventBus()
        received = []

        async def bad_handler(event: Event):
            raise ValueError("oops")

        async def good_handler(event: Event):
            received.append(event)

        bus.on("test", bad_handler)
        bus.on("test", good_handler)
        await bus.emit(Event(type="test"))

        assert len(received) == 1

    def test_event_history(self):
        bus = EventBus()
        bus._history.append(Event(type="a"))
        bus._history.append(Event(type="b"))
        bus._history.append(Event(type="a"))

        assert len(bus.get_history()) == 3
        assert len(bus.get_history("a")) == 2
