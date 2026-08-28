"""Permission manager for controlling agent access."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PermissionRule:
    """A permission rule."""

    tool_pattern: str
    action: str = "ask"  # allow, ask, deny
    conditions: dict[str, Any] = field(default_factory=dict)


class PermissionManager:
    """Manages permissions for agent actions."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        rules: list[PermissionRule] | None = None,
    ) -> None:
        self.workspace_root = workspace_root or Path.cwd()
        self.rules = rules or self._default_rules()
        self._pending_approvals: dict[str, bool] = {}

    def _default_rules(self) -> list[PermissionRule]:
        """Default permission rules."""
        return [
            PermissionRule(tool_pattern="read_file", action="allow"),
            PermissionRule(tool_pattern="list_files", action="allow"),
            PermissionRule(tool_pattern="glob", action="allow"),
            PermissionRule(tool_pattern="grep", action="allow"),
            PermissionRule(tool_pattern="search_code", action="allow"),
            PermissionRule(tool_pattern="edit_file", action="allow"),
            PermissionRule(tool_pattern="write_file", action="allow"),
            PermissionRule(tool_pattern="run_command", action="ask"),
            PermissionRule(tool_pattern="git_commit", action="ask"),
            PermissionRule(tool_pattern="git_push", action="ask"),
            PermissionRule(tool_pattern="network", action="ask"),
        ]

    def check_permission(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Check permission for a tool. Returns 'allow', 'ask', or 'deny'."""
        for rule in self.rules:
            if rule.tool_pattern in tool_name:
                return rule.action
        return "ask"  # Default to ask

    def is_within_workspace(self, path: str | Path) -> bool:
        """Check if a path is within the workspace."""
        try:
            resolved = Path(path).resolve()
            workspace = self.workspace_root.resolve()
            return resolved == workspace or workspace in resolved.parents
        except (ValueError, OSError):
            return False

    def is_protected_path(self, path: str | Path) -> bool:
        """Check if a path is protected (secrets, credentials)."""
        protected_patterns = [
            ".env",
            ".env.",
            "credentials",
            "private_key",
            "ssh_key",
            ".secret",
            "token",
        ]
        path_str = str(path).lower()
        return any(pattern in path_str for pattern in protected_patterns)

    def request_approval(self, tool_name: str, description: str) -> bool:
        """Request user approval for an action. Returns True if approved."""
        # In headless mode, auto-approve based on rules
        # In interactive mode, this would prompt the user
        permission = self.check_permission(tool_name)
        if permission == "allow":
            return True
        if permission == "deny":
            return False
        # For "ask" — in non-interactive mode, default to deny for safety
        return False
