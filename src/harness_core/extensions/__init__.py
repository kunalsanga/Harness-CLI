"""Extension system for Harness — generic abstraction for plugins, MCP, hooks."""

from .manifest import ExtensionManifest, ExtensionType, ExtensionState
from .registry import ExtensionRegistry
from .loader import ExtensionLoader
from .context import ExtensionContext

__all__ = [
    "ExtensionManifest",
    "ExtensionType",
    "ExtensionState",
    "ExtensionRegistry",
    "ExtensionLoader",
    "ExtensionContext",
]
