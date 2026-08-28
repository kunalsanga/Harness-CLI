"""Base tool abstraction."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from harness_core.agent.types import ToolResult


@dataclass
class ToolSchema:
    """Schema for a tool."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    permission_required: str = "allow"
    timeout_seconds: float = 30.0
    tags: list[str] = field(default_factory=list)


class Tool(abc.ABC):
    """Abstract base class for tools."""

    @property
    @abc.abstractmethod
    def schema(self) -> ToolSchema:
        """Tool schema."""

    @abc.abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments."""

    async def validate(self, arguments: dict[str, Any]) -> bool:
        """Validate arguments before execution."""
        return True

    def to_llm_schema(self) -> dict[str, Any]:
        """Convert to LLM tool schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.schema.name,
                "description": self.schema.description,
                "parameters": self.schema.parameters,
            },
        }
