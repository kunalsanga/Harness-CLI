"""Extension loader — discovers and loads extensions from filesystem."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any

from .manifest import ExtensionManifest, ExtensionState, ExtensionType
from .registry import ExtensionRegistry


class ExtensionLoader:
    """Discovers and loads extensions from the filesystem.

    Supports:
    - Local plugin directories
    - Python entrypoint loading
    - Manifest parsing (YAML/JSON)
    - Safe error isolation
    """

    def __init__(
        self,
        registry: ExtensionRegistry | None = None,
        plugin_dirs: list[str] | None = None,
    ) -> None:
        self.registry = registry or ExtensionRegistry()
        self.plugin_dirs = plugin_dirs or []
        self._loaded_modules: dict[str, Any] = {}

    def discover(self, search_dirs: list[str] | None = None) -> list[ExtensionManifest]:
        """Discover extensions in search directories.

        Looks for:
        - harness-plugin.yaml files
        - __init__.py with manifest metadata
        """
        dirs = search_dirs or self.plugin_dirs
        discovered = []

        for search_dir in dirs:
            dir_path = Path(search_dir)
            if not dir_path.exists():
                continue

            # Look for plugin directories
            for entry in dir_path.iterdir():
                if not entry.is_dir():
                    continue
                if entry.name.startswith(".") or entry.name.startswith("_"):
                    continue

                manifest = self._load_manifest_from_dir(entry)
                if manifest:
                    manifest.path = str(entry)
                    manifest.state = ExtensionState.DISCOVERED
                    self.registry.register(manifest)
                    discovered.append(manifest)

        return discovered

    def _load_manifest_from_dir(self, dir_path: Path) -> ExtensionManifest | None:
        """Load manifest from a plugin directory."""
        # Try YAML manifest
        yaml_path = dir_path / "harness-plugin.yaml"
        if yaml_path.exists():
            try:
                return ExtensionManifest.from_yaml(yaml_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Try JSON manifest
        json_path = dir_path / "harness-plugin.json"
        if json_path.exists():
            try:
                import json
                data = json.loads(json_path.read_text(encoding="utf-8"))
                return ExtensionManifest.from_dict(data)
            except Exception:
                pass

        # Try __init__.py with MANIFEST dict
        init_path = dir_path / "__init__.py"
        if init_path.exists():
            try:
                content = init_path.read_text(encoding="utf-8")
                if "MANIFEST" in content:
                    # Try to extract MANIFEST dict from source
                    spec = importlib.util.spec_from_file_location(
                        f"_harness_ext_{dir_path.name}", str(init_path)
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        if hasattr(mod, "MANIFEST"):
                            return ExtensionManifest.from_dict(mod.MANIFEST)
            except Exception:
                pass

        return None

    def load_extension(self, manifest: ExtensionManifest) -> bool:
        """Load an extension by its manifest.

        Returns True if loaded successfully.
        """
        try:
            # Validate manifest
            errors = manifest.validate()
            if errors:
                manifest.state = ExtensionState.FAILED
                manifest.load_error = "; ".join(errors)
                self.registry.mark_failed(manifest.name, manifest.load_error)
                return False

            # Load the module
            ext_path = Path(manifest.path)
            if manifest.entrypoint:
                module_path = ext_path / manifest.entrypoint
            else:
                module_path = ext_path / "__init__.py"

            if not module_path.exists():
                manifest.state = ExtensionState.FAILED
                manifest.load_error = f"Entrypoint not found: {module_path}"
                self.registry.mark_failed(manifest.name, manifest.load_error)
                return False

            # Import the module
            module_name = f"harness_ext_{manifest.name}"
            spec = importlib.util.spec_from_file_location(module_name, str(module_path))
            if spec is None or spec.loader is None:
                manifest.state = ExtensionState.FAILED
                manifest.load_error = f"Cannot load module spec: {module_path}"
                self.registry.mark_failed(manifest.name, manifest.load_error)
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self._loaded_modules[manifest.name] = module
            manifest.state = ExtensionState.ENABLED
            manifest.loaded_at = time.time()
            self.registry.mark_installed(manifest.name)
            return True

        except Exception as e:
            manifest.state = ExtensionState.FAILED
            manifest.load_error = f"{type(e).__name__}: {str(e)[:200]}"
            self.registry.mark_failed(manifest.name, manifest.load_error)
            return False

    def load_all(self) -> dict[str, bool]:
        """Load all discovered extensions. Returns {name: success}."""
        results = {}
        for manifest in self.registry.list_all():
            if manifest.enabled:
                results[manifest.name] = self.load_extension(manifest)
            else:
                results[manifest.name] = False
        return results

    def get_module(self, name: str) -> Any | None:
        """Get the loaded module for an extension."""
        return self._loaded_modules.get(name)

    def unload(self, name: str) -> bool:
        """Unload an extension module."""
        if name in self._loaded_modules:
            module = self._loaded_modules.pop(name)
            module_name = f"harness_ext_{name}"
            sys.modules.pop(module_name, None)
            return True
        return False
