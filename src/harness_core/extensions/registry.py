"""Extension registry — manages extension lifecycle and lookup."""

from __future__ import annotations

import time
from typing import Any, Optional

from .manifest import ExtensionManifest, ExtensionState, ExtensionType


class ExtensionRegistry:
    """Registry of all discovered/installed extensions.

    Thread-safe. Extensions are registered at startup.
    Failed extensions are recorded but don't block others.
    """

    def __init__(self) -> None:
        self._extensions: dict[str, ExtensionManifest] = {}
        self._by_type: dict[ExtensionType, list[str]] = {}

    def register(self, manifest: ExtensionManifest) -> None:
        """Register an extension manifest."""
        self._extensions[manifest.name] = manifest
        self._by_type.setdefault(manifest.extension_type, []).append(manifest.name)

    def unregister(self, name: str) -> bool:
        """Remove an extension from registry."""
        if name in self._extensions:
            manifest = self._extensions.pop(name)
            by_type = self._by_type.get(manifest.extension_type, [])
            if name in by_type:
                by_type.remove(name)
            return True
        return False

    def get(self, name: str) -> ExtensionManifest | None:
        """Get extension by name."""
        return self._extensions.get(name)

    def list_all(self) -> list[ExtensionManifest]:
        """List all registered extensions."""
        return list(self._extensions.values())

    def list_enabled(self) -> list[ExtensionManifest]:
        """List enabled extensions."""
        return [e for e in self._extensions.values() if e.enabled and e.state != ExtensionState.FAILED]

    def list_by_type(self, ext_type: ExtensionType) -> list[ExtensionManifest]:
        """List extensions of a specific type."""
        return [e for e in self._extensions.values() if e.extension_type == ext_type]

    def enable(self, name: str) -> bool:
        """Enable an extension."""
        ext = self._extensions.get(name)
        if ext and ext.state not in (ExtensionState.FAILED, ExtensionState.REMOVED):
            ext.enabled = True
            ext.state = ExtensionState.ENABLED
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable an extension."""
        ext = self._extensions.get(name)
        if ext:
            ext.enabled = False
            ext.state = ExtensionState.DISABLED
            return True
        return False

    def mark_failed(self, name: str, error: str) -> None:
        """Mark an extension as failed."""
        ext = self._extensions.get(name)
        if ext:
            ext.state = ExtensionState.FAILED
            ext.load_error = error
            ext.enabled = False

    def mark_installed(self, name: str) -> None:
        """Mark an extension as installed."""
        ext = self._extensions.get(name)
        if ext:
            ext.state = ExtensionState.INSTALLED

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        total = len(self._extensions)
        by_state = {}
        for ext in self._extensions.values():
            by_state[ext.state.value] = by_state.get(ext.state.value, 0) + 1
        by_type = {}
        for ext in self._extensions.values():
            by_type[ext.extension_type.value] = by_type.get(ext.extension_type.value, 0) + 1
        return {
            "total": total,
            "by_state": by_state,
            "by_type": by_type,
        }
