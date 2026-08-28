"""User-facing error system for Harness Engineering CLI.

Every error provides:
- what happened
- why it happened
- what the user can do next
"""

from .errors import (
    HarnessError,
    ConfigurationError,
    ProviderError,
    ModelError,
    PermissionError,
    ToolError,
    WorkspaceError,
    VerificationError,
    ExtensionError,
    SessionError,
)

__all__ = [
    "HarnessError",
    "ConfigurationError",
    "ProviderError",
    "ModelError",
    "PermissionError",
    "ToolError",
    "WorkspaceError",
    "VerificationError",
    "ExtensionError",
    "SessionError",
]
