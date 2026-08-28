"""Hierarchical configuration system.

Precedence (highest wins):
  CLI flags
  > Session config
  > Project config (.harness/config.yaml)
  > User config (~/.harness/config.yaml)
  > Global defaults
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ConfigScope(str, Enum):
    DEFAULTS = "defaults"
    GLOBAL = "global"
    PROJECT = "project"
    SESSION = "session"
    CLI = "cli"


# Sensitive keys that should never be displayed in plain text
SENSITIVE_KEYS = {
    "api_key", "apikey", "api-key", "secret", "token",
    "password", "auth", "authorization", "credentials",
    "access_token", "refresh_token", "private_key",
}


@dataclass
class ConfigEntry:
    """A single configuration entry with its source scope."""

    key: str
    value: Any
    scope: ConfigScope
    source: str = ""  # file path or "cli"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self._safe_value(),
            "scope": self.scope.value,
            "source": self.source,
        }

    def _safe_value(self) -> Any:
        """Redact sensitive values."""
        key_lower = self.key.lower()
        parts = key_lower.split(".")
        # Check individual parts
        if any(p in SENSITIVE_KEYS for p in parts):
            return "[REDACTED]"
        # Check normalized key (replace . and - with _)
        normalized = key_lower.replace(".", "_").replace("-", "_")
        if any(s in normalized for s in SENSITIVE_KEYS):
            return "[REDACTED]"
        # Also check if any part contains a sensitive substring
        for part in parts:
            if any(s in part for s in ("secret", "token", "password", "credential", "private")):
                return "[REDACTED]"
        return self.value


class HarnessConfig:
    """Hierarchical configuration manager.

    Supports:
    - Multiple scopes with precedence
    - Environment variable override
    - Safe display (redacts secrets)
    - Validation
    """

    def __init__(self, project_root: str | None = None) -> None:
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._stores: dict[ConfigScope, dict[str, Any]] = {
            scope: {} for scope in ConfigScope
        }
        self._loaded_files: list[str] = []

        # Set defaults
        self._set_defaults()

    def _set_defaults(self) -> None:
        """Set built-in default values."""
        defaults = {
            "model.default": "auto",
            "model.free_default": "auto",
            "routing.mode": "auto",
            "routing.exploration_rate": 0.05,
            "session.retention_days": 90,
            "session.max_checkpoints": 50,
            "agent.max_agents": 8,
            "agent.max_parallel": 4,
            "agent.max_iterations_per_agent": 50,
            "agent.max_repair_cycles": 3,
            "agent.max_runtime_seconds": 600,
            "agent.max_cost_dollars": 10.0,
            "verification.enabled": True,
            "verification.auto_verify": True,
            "permissions.safe_mode": True,
            "extensions.enabled": True,
            "extensions.plugin_dir": "~/.harness/plugins",
            "mcp.enabled": False,
            "hooks.enabled": True,
            "output.json": False,
            "output.verbose": False,
        }
        self._stores[ConfigScope.DEFAULTS] = defaults

    def load_global(self) -> bool:
        """Load global user config from ~/.harness/config.yaml."""
        import yaml
        config_path = self._user_config_path()
        if not config_path.exists():
            return False
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if data and isinstance(data, dict):
                self._stores[ConfigScope.GLOBAL] = self._flatten(data)
                self._loaded_files.append(str(config_path))
                return True
        except Exception:
            pass
        return False

    def load_project(self) -> bool:
        """Load project config from .harness/config.yaml."""
        import yaml
        config_path = self.project_root / ".harness" / "config.yaml"
        if not config_path.exists():
            return False
        try:
            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if data and isinstance(data, dict):
                self._stores[ConfigScope.PROJECT] = self._flatten(data)
                self._loaded_files.append(str(config_path))
                return True
        except Exception:
            pass
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value, respecting precedence."""
        # Check scopes from highest to lowest precedence
        for scope in [ConfigScope.CLI, ConfigScope.SESSION, ConfigScope.PROJECT, ConfigScope.GLOBAL, ConfigScope.DEFAULTS]:
            store = self._stores.get(scope, {})
            if key in store:
                return store[key]

        # Check environment variable
        env_key = f"HARNESS_{key.upper().replace('.', '_')}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return self._coerce_env_value(env_val)

        return default

    def set(self, key: str, value: Any, scope: ConfigScope = ConfigScope.GLOBAL, source: str = "") -> None:
        """Set a config value at a specific scope."""
        self._stores[scope][key] = value

    def delete(self, key: str, scope: ConfigScope = ConfigScope.GLOBAL) -> bool:
        """Delete a config value from a scope."""
        if key in self._stores.get(scope, {}):
            del self._stores[scope][key]
            return True
        return False

    def show(self, redact_secrets: bool = True) -> dict[str, ConfigEntry]:
        """Show all effective config values."""
        result: dict[str, ConfigEntry] = {}
        # Collect all keys across all scopes
        all_keys: set[str] = set()
        for store in self._stores.values():
            all_keys.update(store.keys())

        for key in sorted(all_keys):
            for scope in [ConfigScope.CLI, ConfigScope.SESSION, ConfigScope.PROJECT, ConfigScope.GLOBAL, ConfigScope.DEFAULTS]:
                store = self._stores.get(scope, {})
                if key in store:
                    entry = ConfigEntry(
                        key=key,
                        value=store[key],
                        scope=scope,
                        source=self._scope_source(scope),
                    )
                    result[key] = entry
                    break

        return result

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of errors."""
        errors = []
        routing_mode = self.get("routing.mode", "auto")
        if routing_mode not in ("auto", "best", "coding", "debugging", "fast", "cheap", "free", "local", "reasoning"):
            errors.append(f"Invalid routing.mode: {routing_mode}")

        max_agents = self.get("agent.max_agents", 8)
        if not isinstance(max_agents, (int, float)) or max_agents < 1:
            errors.append(f"agent.max_agents must be >= 1, got {max_agents}")

        max_parallel = self.get("agent.max_parallel", 4)
        if not isinstance(max_parallel, (int, float)) or max_parallel < 1:
            errors.append(f"agent.max_parallel must be >= 1, got {max_parallel}")

        retention = self.get("session.retention_days", 90)
        if not isinstance(retention, (int, float)) or retention < 1:
            errors.append(f"session.retention_days must be >= 1, got {retention}")

        return errors

    def to_dict(self, redact: bool = True) -> dict[str, Any]:
        """Export all effective config as a dict."""
        entries = self.show(redact_secrets=redact)
        return {k: v.to_dict() for k, v in entries.items()}

    @staticmethod
    def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dict to dot-separated keys."""
        result = {}
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(HarnessConfig._flatten(v, full_key))
            else:
                result[full_key] = v
        return result

    @staticmethod
    def _user_config_path() -> Path:
        if os.name == "nt":
            return Path(os.environ.get("APPDATA", "~")) / "harness" / "config.yaml"
        return Path.home() / ".harness" / "config.yaml"

    @staticmethod
    def _scope_source(scope: ConfigScope) -> str:
        sources = {
            ConfigScope.DEFAULTS: "built-in defaults",
            ConfigScope.GLOBAL: str(HarnessConfig._user_config_path()),
            ConfigScope.PROJECT: ".harness/config.yaml",
            ConfigScope.SESSION: "session",
            ConfigScope.CLI: "command-line",
        }
        return sources.get(scope, "")

    @staticmethod
    def _coerce_env_value(val: str) -> Any:
        """Coerce environment variable string to appropriate type."""
        if val.lower() in ("true", "yes", "1"):
            return True
        if val.lower() in ("false", "no", "0"):
            return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val
