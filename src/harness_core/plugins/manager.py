"""Plugin manager — high-level plugin lifecycle management."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from ..extensions.context import ExtensionContext
from ..extensions.loader import ExtensionLoader
from ..extensions.manifest import ExtensionManifest, ExtensionState, ExtensionType
from ..extensions.registry import ExtensionRegistry


class PluginManager:
    """Manages local filesystem plugins.

    Lifecycle:
    discover → install → enable → disable → remove

    Plugins live in ~/.harness/plugins/ or project .harness/plugins/.
    """

    def __init__(
        self,
        registry: ExtensionRegistry | None = None,
        global_plugin_dir: str | None = None,
        project_plugin_dir: str | None = None,
    ) -> None:
        self.registry = registry or ExtensionRegistry()
        self.loader = ExtensionLoader(registry=self.registry)
        self.global_plugin_dir = Path(global_plugin_dir or self._default_global_dir())
        self.project_plugin_dir = Path(project_plugin_dir) if project_plugin_dir else None
        self._contexts: dict[str, ExtensionContext] = {}

    @staticmethod
    def _default_global_dir() -> str:
        """Default global plugin directory."""
        import os
        if os.name == "nt":
            return str(Path(os.environ.get("APPDATA", "~")) / "harness" / "plugins")
        return str(Path.home() / ".harness" / "plugins")

    def discover(self) -> list[ExtensionManifest]:
        """Discover all plugins in configured directories."""
        dirs = [str(self.global_plugin_dir)]
        if self.project_plugin_dir and self.project_plugin_dir.exists():
            dirs.append(str(self.project_plugin_dir))

        self.global_plugin_dir.mkdir(parents=True, exist_ok=True)
        return self.loader.discover(dirs)

    def install(self, source_path: str) -> ExtensionManifest | None:
        """Install a plugin from a local directory.

        Copies the plugin to the global plugin directory.
        Returns the manifest if successful.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Plugin source not found: {source_path}")

        # Load manifest
        manifest = self.loader._load_manifest_from_dir(source)
        if manifest is None:
            raise ValueError(f"No valid manifest found in {source_path}")

        # Validate
        errors = manifest.validate()
        if errors:
            raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

        # Check for duplicate
        if self.registry.get(manifest.name):
            raise ValueError(f"Plugin '{manifest.name}' is already installed")

        # Copy to plugin directory
        dest = self.global_plugin_dir / manifest.name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

        # Register
        manifest.path = str(dest)
        manifest.state = ExtensionState.INSTALLED
        self.registry.register(manifest)

        return manifest

    def enable(self, name: str) -> bool:
        """Enable a plugin."""
        manifest = self.registry.get(name)
        if manifest is None:
            return False
        if manifest.state == ExtensionState.FAILED:
            return False
        manifest.enabled = True
        manifest.state = ExtensionState.ENABLED
        return True

    def disable(self, name: str) -> bool:
        """Disable a plugin."""
        manifest = self.registry.get(name)
        if manifest is None:
            return False
        manifest.enabled = False
        manifest.state = ExtensionState.DISABLED
        return True

    def remove(self, name: str) -> bool:
        """Remove a plugin."""
        manifest = self.registry.get(name)
        if manifest is None:
            return False

        # Unload if loaded
        self.loader.unload(name)

        # Remove directory
        if manifest.path:
            plugin_path = Path(manifest.path)
            if plugin_path.exists():
                shutil.rmtree(plugin_path, ignore_errors=True)

        manifest.state = ExtensionState.REMOVED
        self.registry.unregister(name)
        self._contexts.pop(name, None)
        return True

    def load(self, name: str) -> bool:
        """Load a specific plugin."""
        manifest = self.registry.get(name)
        if manifest is None or not manifest.enabled:
            return False
        return self.loader.load_extension(manifest)

    def load_all(self) -> dict[str, bool]:
        """Load all enabled plugins."""
        results = {}
        for manifest in self.registry.list_enabled():
            results[manifest.name] = self.loader.load_extension(manifest)
        return results

    def get_context(self, name: str) -> ExtensionContext | None:
        """Get the extension context for a loaded plugin."""
        if name in self._contexts:
            return self._contexts[name]

        manifest = self.registry.get(name)
        if manifest is None:
            return None

        ctx = ExtensionContext(manifest=manifest)
        self._contexts[name] = ctx
        return ctx

    def inspect(self, name: str) -> dict[str, Any] | None:
        """Get detailed info about a plugin."""
        manifest = self.registry.get(name)
        if manifest is None:
            return None
        return manifest.to_dict()

    def list_all(self) -> list[dict[str, Any]]:
        """List all plugins with their status."""
        return [m.to_dict() for m in self.registry.list_all()]

    def list_enabled(self) -> list[dict[str, Any]]:
        """List enabled plugins."""
        return [m.to_dict() for m in self.registry.list_enabled()]

    def get_stats(self) -> dict[str, Any]:
        """Get plugin statistics."""
        return self.registry.get_stats()
