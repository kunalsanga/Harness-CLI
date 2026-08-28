"""Structured error classes with user-facing messages.

Each error provides:
- what happened
- why it happened
- what the user can do next
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base error for all Harness errors."""

    def __init__(
        self,
        message: str,
        *,
        what: str = "",
        why: str = "",
        fix: str = "",
        hint: str = "",
    ) -> None:
        self.what = what or message
        self.why = why
        self.fix = fix
        self.hint = hint
        super().__init__(message)

    def user_message(self) -> str:
        """Format a user-friendly error message."""
        parts = [f"Error: {self.what}"]
        if self.why:
            parts.append(f"  Why: {self.why}")
        if self.fix:
            parts.append(f"  Fix: {self.fix}")
        if self.hint:
            parts.append(f"  Hint: {self.hint}")
        return "\n".join(parts)


class ConfigurationError(HarnessError):
    """Configuration-related errors (missing config, invalid values)."""


class ProviderError(HarnessError):
    """Provider-related errors (connection, authentication, API)."""


class ModelError(HarnessError):
    """Model-related errors (unavailable, rate-limited, invalid response)."""


class PermissionError(HarnessError):
    """Permission-related errors (operation blocked by policy)."""


class ToolError(HarnessError):
    """Tool execution errors (command failure, file error)."""


class WorkspaceError(HarnessError):
    """Workspace-related errors (not a repo, missing files)."""


class VerificationError(HarnessError):
    """Verification errors (tests fail, type check fails)."""


class ExtensionError(HarnessError):
    """Extension/plugin errors (load failure, compatibility)."""


class SessionError(HarnessError):
    """Session-related errors (not found, corrupt, conflict)."""
