import pytest
from unittest.mock import MagicMock
from harness_core.observability.events import EventBus
from harness_core.agents.domain import AgentMessage, MessageType, TaskGraph, SubTask, AgentRole
from harness_core.agents.registry import AgentRegistry
from harness_core.agents.message_bus import AgentMessageBus, MessageValidationError

@pytest.fixture
def message_bus():
    event_bus = EventBus()
    registry = AgentRegistry()
    task_graph = TaskGraph()
    task_graph.add_task(SubTask(task_id="sender_task", role=AgentRole.ARCHITECT))
    task_graph.add_task(SubTask(task_id="recipient_task", role=AgentRole.BACKEND))
    
    return AgentMessageBus(event_bus, task_graph, registry)


@pytest.mark.asyncio
async def test_send_valid_message(message_bus):
    msg = AgentMessage(
        sender_task_id="sender_task",
        recipient_task_id="recipient_task",
        message_type=MessageType.HANDOFF,
        payload={"contract": "api-contract.md"}
    )
    result = await message_bus.send(msg)
    assert result is True
    
    inbox = message_bus.get_messages_for("recipient_task")
    assert len(inbox) == 1
    assert inbox[0].message_id == msg.message_id


@pytest.mark.asyncio
async def test_idempotent_duplicate_message(message_bus):
    msg = AgentMessage(
        sender_task_id="sender_task",
        recipient_task_id="recipient_task",
        message_type=MessageType.HANDOFF,
        payload={"data": "test"}
    )
    # Send twice
    await message_bus.send(msg)
    await message_bus.send(msg)
    
    inbox = message_bus.get_messages_for("recipient_task")
    # Should only appear once
    assert len(inbox) == 1


@pytest.mark.asyncio
async def test_invalid_sender_task(message_bus):
    msg = AgentMessage(
        sender_task_id="nonexistent_sender",
        recipient_task_id="recipient_task",
        message_type=MessageType.HANDOFF,
    )
    result = await message_bus.send(msg)
    assert result is False


@pytest.mark.asyncio
async def test_oversized_payload(message_bus):
    large_payload = {"data": "x" * 60000}
    msg = AgentMessage(
        sender_task_id="sender_task",
        recipient_task_id="recipient_task",
        message_type=MessageType.INFORMATION,
        payload=large_payload
    )
    result = await message_bus.send(msg)
    assert result is False


@pytest.mark.asyncio
async def test_message_ordering(message_bus):
    m1 = AgentMessage(sender_task_id="sender_task", recipient_task_id="recipient_task", payload={"seq": 1})
    m2 = AgentMessage(sender_task_id="sender_task", recipient_task_id="recipient_task", payload={"seq": 2})
    
    await message_bus.send(m1)
    await message_bus.send(m2)
    
    inbox = message_bus.get_messages_for("recipient_task")
    assert len(inbox) == 2
    assert inbox[0].payload["seq"] == 1
    assert inbox[1].payload["seq"] == 2
