"""Extension context — safe API surface for extensions to interact with Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .manifest import ExtensionManifest, Permission


class ExtensionContext:
    """Context provided to extensions for interacting with Harness.

    Extensions receive this context instead of direct access to internals.
    Provides controlled access to:
    - Workspace path
    - Configuration
    - Tool registration
    - Event emission
    - Permission checking
    """

    def __init__(
        self,
        manifest: ExtensionManifest,
        workspace_path: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.manifest = manifest
        self.workspace_path = workspace_path
        self.config = config or {}
        self._registered_tools: list[dict[str, Any]] = []
        self._registered_providers: list[dict[str, Any]] = []
        self._registered_agents: list[dict[str, Any]] = []
        self._hooks: dict[str, list] = {}
        self._events: list[dict[str, Any]] = []

    def has_permission(self, permission: Permission) -> bool:
        """Check if this extension has a specific permission."""
        return permission in self.manifest.permissions

    def require_permission(self, permission: Permission) -> None:
        """Raise if permission not granted."""
        if not self.has_permission(permission):
            raise PermissionError(
                f"Extension '{self.manifest.name}' requires permission "
                f"'{permission.value}' which is not granted."
            )

    def register_tool(self, tool_def: dict[str, Any]) -> None:
        """Register a tool provided by this extension."""
        self.require_permission(Permission.TOOLS)
        tool_def["_extension"] = self.manifest.name
        self._registered_tools.append(tool_def)

    def register_provider(self, provider_def: dict[str, Any]) -> None:
        """Register a model provider provided by this extension."""
        self.require_permission(Permission.MODELS)
        provider_def["_extension"] = self.manifest.name
        self._registered_providers.append(provider_def)

    def register_agent(self, agent_def: dict[str, Any]) -> None:
        """Register an agent provided by this extension."""
        agent_def["_extension"] = self.manifest.name
        self._registered_agents.append(agent_def)

    def register_hook(self, event_type: str, callback) -> None:
        """Register a lifecycle hook."""
        self._hooks.setdefault(event_type, []).append(callback)

    def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event (for observability, not bypassing security)."""
        self._events.append({
            "type": event_type,
            "data": data,
            "extension": self.manifest.name,
        })

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get extension-specific configuration."""
        ext_config = self.config.get(self.manifest.name, {})
        return ext_config.get(key, default)

    def get_registered_tools(self) -> list[dict[str, Any]]:
        """Get tools registered by this extension."""
        return list(self._registered_tools)

    def get_registered_providers(self) -> list[dict[str, Any]]:
        """Get providers registered by this extension."""
        return list(self._registered_providers)

    def get_registered_agents(self) -> list[dict[str, Any]]:
        """Get agents registered by this extension."""
        return list(self._registered_agents)

    def get_hooks(self) -> dict[str, list]:
        """Get all registered hooks."""
        return dict(self._hooks)

    def get_events(self) -> list[dict[str, Any]]:
        """Get all emitted events."""
        return list(self._events)
