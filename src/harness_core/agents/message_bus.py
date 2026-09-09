"""
Message bus for agent-to-agent communication in Phase 6.
Distinct from EventBus which is for system observability.
"""

from __future__ import annotations

import logging
import asyncio
from typing import Callable, Coroutine, Any

from harness_core.agents.domain import AgentMessage, TaskGraph
from harness_core.agents.registry import AgentRegistry
from harness_core.observability.events import EventBus, Event


logger = logging.getLogger(__name__)
MessageHandler = Callable[[AgentMessage], Coroutine[Any, Any, None]]

class MessageValidationError(Exception):
    """Raised when a message fails validation."""
    pass


class AgentMessageBus:
    """
    Dedicated message bus for inter-agent communication.
    Provides typed handoffs, validation, ordering, and idempotency.
    """

    def __init__(self, event_bus: EventBus, task_graph: TaskGraph, registry: AgentRegistry) -> None:
        self.event_bus = event_bus
        self.task_graph = task_graph
        self.registry = registry
        
        # message_id -> AgentMessage
        self._processed_ids: set[str] = set()
        
        # recipient_agent_id -> list of messages
        self._inboxes: dict[str, list[AgentMessage]] = {}
        
        # recipient_agent_id -> subscribers (e.g. WorkerAgent inbox handler)
        self._subscribers: dict[str, list[MessageHandler]] = {}
        
        # Mutex for concurrency safety
        self._mutex = asyncio.Lock()

    async def send(self, message: AgentMessage) -> bool:
        """
        Validate and route a message.
        Returns True if delivered/queued, False if rejected.
        """
        async with self._mutex:
            if message.message_id in self._processed_ids:
                logger.debug(f"Ignoring duplicate message {message.message_id}")
                return True
                
            try:
                self._validate_message(message)
            except MessageValidationError as e:
                await self._emit_observability("agent_message_failed", message, error=str(e))
                logger.warning(f"Message {message.message_id} validation failed: {e}")
                return False

            self._processed_ids.add(message.message_id)

            # Route to recipient
            # If recipient_agent_id is empty but recipient_task_id is set, the task might not be running yet,
            # so we queue it in a task-based inbox instead, or let the worker pull it by task_id later.
            # To simplify, we index by either agent_id or task_id.
            recipient = message.recipient_agent_id or message.recipient_task_id
            
            if recipient not in self._inboxes:
                self._inboxes[recipient] = []
            
            self._inboxes[recipient].append(message)
            
            # Emit observability
            if message.message_type.name == "HANDOFF":
                await self._emit_observability("agent_handoff_created", message)
            else:
                await self._emit_observability("agent_message_sent", message)
                
            # Notify subscribers
            if recipient in self._subscribers:
                for handler in self._subscribers[recipient]:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler for {recipient}: {e}")

            return True

    def _validate_message(self, message: AgentMessage) -> None:
        """
        Validate message constraints.
        - Sender exists and matches context
        - Recipient exists
        - Payload size limits
        """
        # 1. Payload size limit
        payload_size = len(str(message.payload))
        if payload_size > 50000:
            raise MessageValidationError(f"Payload size {payload_size} exceeds 50000 bytes limit.")
            
        # 2. Required fields
        if not message.sender_agent_id and not message.sender_task_id:
            raise MessageValidationError("Message must have a sender.")
        if not message.recipient_agent_id and not message.recipient_task_id:
            raise MessageValidationError("Message must have a recipient.")
            
        # 3. Task validation
        if message.sender_task_id:
            if not self.task_graph.get_task(message.sender_task_id):
                raise MessageValidationError(f"Sender task {message.sender_task_id} does not exist in TaskGraph.")
        
        if message.recipient_task_id:
            if not self.task_graph.get_task(message.recipient_task_id):
                raise MessageValidationError(f"Recipient task {message.recipient_task_id} does not exist in TaskGraph.")

    def subscribe(self, recipient_id: str, handler: MessageHandler) -> None:
        """Subscribe to messages for a specific recipient agent or task ID."""
        if recipient_id not in self._subscribers:
            self._subscribers[recipient_id] = []
        self._subscribers[recipient_id].append(handler)
        
    def get_messages_for(self, recipient_id: str) -> list[AgentMessage]:
        """Get all pending/historical messages for a recipient in order."""
        return self._inboxes.get(recipient_id, []).copy()

    async def _emit_observability(self, event_type: str, message: AgentMessage, error: str = "") -> None:
        """Emit metadata-only event to EventBus."""
        data = {
            "message_id": message.message_id,
            "sender_agent_id": message.sender_agent_id,
            "sender_task_id": message.sender_task_id,
            "recipient_agent_id": message.recipient_agent_id,
            "recipient_task_id": message.recipient_task_id,
            "message_type": message.message_type.value,
            "correlation_id": message.correlation_id,
        }
        if error:
            data["error"] = error
            
        await self.event_bus.emit(Event(
            type=event_type,
            source="AgentMessageBus",
            data=data
        ))
