"""Steering buffer for asynchronous user guidance during execution."""

from collections import deque
from dataclasses import dataclass
import time
from typing import Optional

@dataclass
class SteeringMessage:
    content: str
    timestamp: float

class SteeringBuffer:
    """Holds steering messages injected by the user during execution.
    
    The AgentLoop consumes from this buffer at safe execution boundaries
    (e.g., between model iterations).
    """
    
    def __init__(self) -> None:
        self._messages: deque[SteeringMessage] = deque()
        
    def add(self, content: str) -> None:
        self._messages.append(SteeringMessage(content=content, timestamp=time.time()))
        
    def has_messages(self) -> bool:
        return len(self._messages) > 0
        
    def pop_all(self) -> list[SteeringMessage]:
        messages = list(self._messages)
        self._messages.clear()
        return messages
