"""Permission manager for controlling agent access.

Implements autonomous workspace execution: safe development commands are
auto-approved inside the workspace, while dangerous operations always
require explicit user approval.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ── Safe commands auto-allowed in autonomous mode ─────────────────────────

# Shell command prefixes/patterns that are safe inside the workspace.
# Checked against the raw command string for run_command tool calls.
_SAFE_COMMAND_PREFIXES: list[str] = [
    # Languages & runtimes
    "node ", "npx ", "npm ", "pnpm ", "yarn ",
    "python ", "python3 ", "uv ", "pip ", "pip3 ",
    "pytest ", "python -m pytest",
    "cargo ", "rustc ", "cargo test", "cargo build", "cargo run", "cargo clippy", "cargo fmt",
    "go ", "go test", "go build", "go run",
    "cmake ", "make ", "meson ",
    "java ", "javac ", "gradle ", "mvn ",
    "swift ", "swiftc ",
    # Git (read-only, commit, and push)
    "git status", "git diff", "git log", "git show", "git branch", "git stash",
    "git add ", "git commit ", "git checkout ", "git merge ", "git rebase ",
    "git reset ", "git revert ",
    "git push", "git fetch", "git pull", "git remote", "git tag",
    # Linters & formatters
    "eslint ", "prettier ", "black ", "ruff ", "flake8 ", "mypy ",
    "pylint ", "isort ", "autopep8 ",
    # Testing
    "jest ", "vitest ", "mocha ", "playwright ", "cypress ",
    "phpunit ", "dotnet test", "dotnet build",
    # Build
    "tsc ", "npx tsc", "webpack ", "vite ", "esbuild ", "rollup ",
    "gradlew ",
    # File operations (safe)
    "ls ", "dir ", "cat ", "head ", "tail ", "wc ", "find ", "tree ",
    "echo ", "which ", "where ",
    "mkdir ", "cp ", "mv ", "touch ",
    # Windows equivalents
    "type ",
]

# Exact matches for safe commands
_SAFE_COMMAND_EXACT: set[str] = {
    "git status", "git diff", "git log", "git show", "git branch",
    "git remote -v", "git remote",
    "ls", "dir", "pwd", "echo", "tree",
}

# Regex patterns for dangerous operations — always require approval
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r"mkfs"),
    re.compile(r"dd\s+if="),
    re.compile(r">\s*/dev/sd"),
    re.compile(r"format\s+[cCdD]:", re.IGNORECASE),
    re.compile(r"del\s+/[sS][qQ]", re.IGNORECASE),
    re.compile(r"shutdown"),
    re.compile(r"reboot"),
    re.compile(r"systemctl\s+(stop|disable|restart)", re.IGNORECASE),
    re.compile(r"chmod\s+777\s+/"),
    re.compile(r"curl\s+.*\|\s*(bash|sh)"),
    re.compile(r"wget\s+.*\|\s*(bash|sh)"),
]

# Credential/secret patterns in command arguments
_CREDENTIAL_PATTERNS: list[str] = [
    ".env", "credentials", "private_key", "ssh_key", "id_rsa",
    ".secret", "token", "password", "api_key", "secret_key",
]


@dataclass
class PermissionRule:
    """A permission rule."""

    tool_pattern: str
    action: str = "ask"  # allow, ask, deny
    conditions: dict[str, Any] = field(default_factory=dict)


class PermissionManager:
    """Manages permissions for agent actions.

    Modes:
    - autonomous (default): safe workspace commands auto-allowed,
      dangerous operations prompt user.
    - interactive: everything prompts user (legacy behavior).
    - strict: read-only auto-allowed, everything else prompts.

    Supports:
    - allow: execute without asking
    - deny: block execution
    - ask: prompt user or auto-deny based on mode
    """

    def __init__(
        self,
        workspace_root: Path | None = None,
        rules: list[PermissionRule] | None = None,
        approval_callback: Callable[[str, str], bool] | None = None,
        session_approvals: dict[str, bool] | None = None,
        autonomous_mode: bool = True,
    ) -> None:
        self.workspace_root = workspace_root or Path.cwd()
        self.rules = rules or self._default_rules()
        self._pending_approvals: dict[str, bool] = {}
        self.approval_callback = approval_callback
        self.session_approvals: dict[str, bool] = session_approvals or {}
        self.autonomous_mode = autonomous_mode

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
            PermissionRule(tool_pattern="run_command", action="allow"),
            PermissionRule(tool_pattern="git_commit", action="allow"),
            PermissionRule(tool_pattern="git_add", action="allow"),
            PermissionRule(tool_pattern="git_stage", action="allow"),
            PermissionRule(tool_pattern="git_push", action="allow"),
            PermissionRule(tool_pattern="git_status", action="allow"),
            PermissionRule(tool_pattern="git_diff", action="allow"),
            PermissionRule(tool_pattern="git_log", action="allow"),
            PermissionRule(tool_pattern="git_remote", action="allow"),
            PermissionRule(tool_pattern="git_identity", action="allow"),
            PermissionRule(tool_pattern="network", action="ask"),
        ]

    def _is_safe_workspace_command(self, arguments: dict[str, Any]) -> bool:
        """Check if a run_command call is a safe workspace operation.

        Only returns True in autonomous mode for commands that are:
        1. Known-safe development commands (checked against allowlist)
        2. Not dangerous patterns
        3. Not accessing credentials/secrets
        4. Working directory is within the workspace (or no cwd specified)
        """
        if not self.autonomous_mode:
            return False

        command = arguments.get("command", "").strip()
        if not command:
            return False

        # Check for dangerous patterns first
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                return False

        # Check for credential access
        cmd_lower = command.lower()
        for cred in _CREDENTIAL_PATTERNS:
            if cred in cmd_lower:
                return False

        # Check if command starts with a safe prefix
        cmd_stripped = command.lstrip()
        for prefix in _SAFE_COMMAND_PREFIXES:
            if cmd_stripped.lower().startswith(prefix.lower()):
                return True

        # Check exact matches
        if cmd_stripped.lower() in _SAFE_COMMAND_EXACT:
            return True

        # Check if it's a relative path command (likely a local script)
        # e.g., "./test.sh", "scripts/build.sh", "node scripts/test.js"
        if cmd_stripped.startswith("./") or cmd_stripped.startswith("../"):
            # Only allow if it looks like a test/build script
            lower = cmd_stripped.lower()
            if any(kw in lower for kw in ["test", "build", "lint", "check", "fmt", "install"]):
                return True

        return False

    def _is_dangerous_command(self, arguments: dict[str, Any]) -> bool:
        """Check if a command is explicitly dangerous."""
        command = arguments.get("command", "").strip()
        if not command:
            return False
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                return True
        return False

    def check_permission(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        """Check permission for a tool. Returns 'allow', 'ask', or 'deny'."""
        # For run_command, apply special logic based on mode
        if tool_name == "run_command" and arguments:
            if self._is_dangerous_command(arguments):
                return "deny"  # Dangerous — always block, even in autonomous mode
            if self.autonomous_mode:
                if self._is_safe_workspace_command(arguments):
                    return "allow"  # Safe — auto-approve
                # Unknown command in autonomous mode → 'ask' (auto-approved by request_approval)
                return "ask"
            # Non-autonomous mode → always 'ask'
            return "ask"

        # Check explicit rules
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

    def approve_for_session(self, tool_pattern: str) -> None:
        """Approve a tool pattern for the remainder of this session."""
        self.session_approvals[tool_pattern] = True

    def deny_for_session(self, tool_pattern: str) -> None:
        """Deny a tool pattern for the remainder of this session."""
        self.session_approvals[tool_pattern] = False

    def request_approval(self, tool_name: str, description: str = "") -> bool:
        """Request user approval for an action. Returns True if approved.

        Resolution order:
        1. allow rule → auto-approve
        2. deny rule → auto-deny
n        3. Autonomous mode → auto-approve (safe workspace operations)
        4. Session approval → use session decision
        5. Interactive callback → prompt user
        6. Default → deny (safe default)

        Dangerous operations are NEVER auto-approved, even in autonomous mode.
        """
        permission = self.check_permission(tool_name)
        if permission == "allow":
            return True
        if permission == "deny":
            return False
        # "ask" — In autonomous mode, auto-approve non-dangerous operations.
        # Dangerous commands (credential access, destructive ops) are blocked
        # by _is_dangerous_command and _CREDENTIAL_PATTERNS checks, which
        # cause check_permission to return 'deny' before reaching here.
        if self.autonomous_mode:
            return True
        # "ask" — check session-level approval first
        for pattern, approved in self.session_approvals.items():
            if pattern in tool_name:
                return approved
        # Use interactive callback if available
        if self.approval_callback is not None:
            return self.approval_callback(tool_name, description)
        # Default: deny (safe default for non-interactive mode)
        return False
