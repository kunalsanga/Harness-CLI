"""Extension manifest — metadata, types, and state for extensions."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ExtensionType(enum.Enum):
    """Types of extensions."""
    TOOL = "tool"
    PROVIDER = "provider"
    AGENT = "agent"
    HOOK = "hook"
    MCP = "mcp"


class ExtensionState(enum.Enum):
    """Extension lifecycle states."""
    DISCOVERED = "discovered"
    INSTALLED = "installed"
    ENABLED = "enabled"
    DISABLED = "disabled"
    FAILED = "failed"
    REMOVED = "removed"


class Permission(enum.Enum):
    """Extension permissions."""
    FILESYSTEM_READ = "filesystem.read"
    FILESYSTEM_WRITE = "filesystem.write"
    SHELL_EXECUTE = "shell.execute"
    NETWORK = "network"
    GIT = "git"
    MODELS = "models"
    SESSIONS = "sessions"
    TOOLS = "tools"


@dataclass
class ExtensionManifest:
    """Manifest describing an extension.

    Parsed from harness-plugin.yaml or equivalent.
    """

    name: str = ""
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    extension_type: ExtensionType = ExtensionType.TOOL
    entrypoint: str = ""
    capabilities: list[str] = field(default_factory=list)
    permissions: list[Permission] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    requires_harness: str = ">=0.1.0,<2.0.0"
    config_schema: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    # Runtime state
    state: ExtensionState = ExtensionState.DISCOVERED
    path: str = ""
    load_error: str = ""
    loaded_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "type": self.extension_type.value,
            "entrypoint": self.entrypoint,
            "capabilities": self.capabilities,
            "permissions": [p.value for p in self.permissions],
            "dependencies": self.dependencies,
            "requires_harness": self.requires_harness,
            "enabled": self.enabled,
            "state": self.state.value,
            "path": self.path,
            "load_error": self.load_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtensionManifest:
        """Create manifest from a dictionary (parsed YAML/JSON)."""
        perms = []
        for p in data.get("permissions", []):
            try:
                perms.append(Permission(p))
            except ValueError:
                pass

        ext_type = ExtensionType.TOOL
        try:
            ext_type = ExtensionType(data.get("type", "tool"))
        except ValueError:
            pass

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            extension_type=ext_type,
            entrypoint=data.get("entrypoint", ""),
            capabilities=data.get("capabilities", []),
            permissions=perms,
            dependencies=data.get("dependencies", []),
            requires_harness=data.get("requires_harness", ">=0.1.0,<2.0.0"),
            config_schema=data.get("config_schema", {}),
            enabled=data.get("enabled", True),
        )

    @classmethod
    def from_yaml(cls, yaml_text: str) -> ExtensionManifest:
        """Parse manifest from YAML text."""
        try:
            import yaml
            data = yaml.safe_load(yaml_text) or {}
        except ImportError:
            # Fallback: try JSON
            import json
            data = json.loads(yaml_text)
        return cls.from_dict(data)

    def validate(self) -> list[str]:
        """Validate the manifest. Returns list of errors."""
        errors = []
        if not self.name:
            errors.append("Missing 'name'")
        if not self.version:
            errors.append("Missing 'version'")
        if not self.entrypoint:
            errors.append("Missing 'entrypoint'")
        if self.extension_type == ExtensionType.TOOL and not self.entrypoint:
            errors.append("Tool extension requires 'entrypoint'")
        return errors
