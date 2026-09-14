"""Centralized credential resolution for Harness providers.

Architecture:
    CredentialResolver
        ↓
    Provider Configuration
        ↓
    ModelGateway / Provider implementations

Credential precedence (first non-empty wins):
    1. Explicit environment variable (OPENROUTER_API_KEY, etc.)
    2. Project .env file (cwd or harness root)
    3. Persistent user credentials (~/.harness/credentials.json)
    4. Harness hosted authentication (future)

Security:
    - Never prints, logs, or exposes full secrets
    - Masked representation for display
    - Credentials never reach model prompts or EventBus events
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Provider-specific environment variable names
_PROVIDER_ENV_KEYS: dict[str, list[str]] = {
    "openrouter": ["OPENROUTER_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "nvidia": ["NVIDIA_API_KEY"],
    "ollama": [],  # No key needed, just host
}


def mask_secret(value: str, show_chars: int = 4) -> str:
    """Mask a secret for display. Shows first and last `show_chars` chars."""
    if not value:
        return ""
    if len(value) <= show_chars * 2 + 4:
        return "*" * len(value)
    return f"{value[:show_chars]}{'*' * (len(value) - show_chars * 2)}{value[-show_chars:]}"


@dataclass
class ProviderCredential:
    """Resolved credential for a provider."""

    provider: str
    api_key: str
    source: str  # "env", "project_env", "user_credential", "hosted"
    base_url: str | None = None
    default_model: str | None = None

    @property
    def is_valid(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    @property
    def masked_key(self) -> str:
        return mask_secret(self.api_key)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage (NEVER includes the actual key in logs)."""
        return {
            "provider": self.provider,
            "source": self.source,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "has_key": bool(self.api_key),
        }


class CredentialStore:
    """Persistent user-level credential storage.

    Stores credentials in ~/.harness/credentials.json with restricted
    file permissions. Credentials are stored as plain JSON on disk
    (the file itself should have restricted permissions on supported OSes).
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = config_dir or (Path.home() / ".harness")
        self._credentials_file = self.config_dir / "credentials.json"

    def _ensure_dir(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        # On Unix, restrict directory permissions
        try:
            import stat
            self.config_dir.chmod(stat.S_IRWXU)  # 700
        except (OSError, AttributeError):
            pass

    def save_credential(
        self,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> None:
        """Save a provider credential persistently."""
        self._ensure_dir()
        creds = self._load_all()
        creds[provider] = {
            "api_key": api_key,
            "base_url": base_url,
            "default_model": default_model,
        }
        self._credentials_file.write_text(
            json.dumps(creds, indent=2), encoding="utf-8"
        )
        # Restrict file permissions
        try:
            import stat
            self._credentials_file.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        except (OSError, AttributeError):
            pass

    def load_credential(self, provider: str) -> ProviderCredential | None:
        """Load a stored credential for a provider."""
        creds = self._load_all()
        data = creds.get(provider)
        if not data or not data.get("api_key"):
            return None
        return ProviderCredential(
            provider=provider,
            api_key=data["api_key"],
            source="user_credential",
            base_url=data.get("base_url"),
            default_model=data.get("default_model"),
        )

    def remove_credential(self, provider: str) -> bool:
        """Remove a stored credential. Returns True if it existed."""
        creds = self._load_all()
        if provider in creds:
            del creds[provider]
            self._credentials_file.write_text(
                json.dumps(creds, indent=2), encoding="utf-8"
            )
            return True
        return False

    def list_providers(self) -> list[str]:
        """List providers with stored credentials."""
        return list(self._load_all().keys())

    def _load_all(self) -> dict[str, dict]:
        if not self._credentials_file.exists():
            return {}
        try:
            return json.loads(self._credentials_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}


class CredentialResolver:
    """Centralized credential resolution.

    Resolves credentials for any provider using the configured precedence:
        1. Environment variable
        2. Project .env (already loaded into os.environ by dotenv loader)
        3. Persistent user credential (~/.harness/credentials.json)

    The ModelGateway and provider implementations should use this resolver
    instead of directly reading os.environ.
    """

    def __init__(self, credential_store: CredentialStore | None = None) -> None:
        self._store = credential_store or CredentialStore()

    def resolve(self, provider: str) -> ProviderCredential | None:
        """Resolve the best available credential for a provider.

        Returns None if no credential is available.
        """
        # 1. Environment variable
        env_key = self._resolve_env_key(provider)
        if env_key:
            return ProviderCredential(
                provider=provider,
                api_key=env_key,
                source="env",
            )

        # 2. Persistent user credential
        stored = self._store.load_credential(provider)
        if stored and stored.is_valid:
            return stored

        return None

    def resolve_for_provider(self, provider_name: str) -> ProviderCredential | None:
        """Alias for resolve() with clearer naming."""
        return self.resolve(provider_name)

    def has_credential(self, provider: str) -> bool:
        """Check if a credential is available for a provider."""
        cred = self.resolve(provider)
        return cred is not None and cred.is_valid

    def list_configured_providers(self) -> list[ProviderCredential]:
        """List all providers with resolved credentials."""
        results = []
        for provider_name in _PROVIDER_ENV_KEYS:
            cred = self.resolve(provider_name)
            if cred and cred.is_valid:
                results.append(cred)
        return results

    def _resolve_env_key(self, provider: str) -> str | None:
        """Check environment variables for a provider's API key."""
        env_keys = _PROVIDER_ENV_KEYS.get(provider, [])
        for key in env_keys:
            value = os.environ.get(key, "").strip()
            if value:
                return value
        return None

    @property
    def store(self) -> CredentialStore:
        return self._store
