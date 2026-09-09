"""Secret-stripping and credential redaction for memory entries.

All entries pass through :class:`MemorySanitizer` before persistence.
After sanitisation the content must contain no API keys, private keys,
JWTs, .env values, or credential patterns.

Security-reviewer entries receive an additional redaction pass that
strips internal hostnames, IP addresses, and absolute file paths so
that security reports cannot leak infrastructure details.
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from harness_core.memory.domain import MemoryEntry


_REDACTED = "[REDACTED]"


class MemorySanitizer:
    """Strips secrets, API keys, and credentials from memory content.

    The sanitizer is intentionally conservative — it prefers false
    positives (over-redaction) to false negatives (leaking secrets).
    """

    # ── Secret patterns ──────────────────────────────────────────────────
    # Order matters: more-specific patterns first so they win over the
    # generic token catch-all.
    _SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("openai_key", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
        ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
        ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("aws_secret", re.compile(r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key[\"'=\s:]+[A-Za-z0-9/+=]{40}")),
        ("github_pat", re.compile(r"ghp_[A-Za-z0-9]{36,}")),
        ("github_oauth", re.compile(r"gho_[A-Za-z0-9]{36,}")),
        ("slack_token", re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}")),
        ("pem_private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
        ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
        ("bearer_token", re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}")),
        ("google_api_key", re.compile(r"AIza[0-9A-Za-z\-_]{35}")),
        ("stripe_key", re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{20,}")),
        ("generic_password", re.compile(r"(?i)(?:password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"\\]{6,})['\"]?")),
        ("generic_api_key", re.compile(r"(?i)(?:api[_\-]?key|apikey|secret[_\-]?key)\s*[=:]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?")),
        ("connection_string", re.compile(r"(?i)(?:mongodb|postgres|mysql|redis)(\+srv)?://[^\s]+:[^\s]+@[^\s]+")),
        ("env_assignment", re.compile(r"(?im)^\s*(?:export\s+)?[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|API_KEY|CREDENTIAL)\s*=\s*['\"]?([^\s'\"\\#]{4,})['\"]?")),
    ]

    # Patterns used only for security-reviewer entries
    _INFRA_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        ("ipv4", re.compile(r"\b(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)){3}\b")),
        ("internal_host", re.compile(r"\b(?:[a-z0-9][a-z0-9\-]*\.)+(?:internal|local|corp|lan|home|localdomain)\b", re.IGNORECASE)),
        ("absolute_path", re.compile(r"(?:/home/[a-z0-9_\-]+|C:\\Users\\[A-Za-z0-9 _\-]+|/Users/[a-z0-9_\-]+|/var/log|/etc)(?:/[^\s\"'\\]*)?", re.IGNORECASE)),
    ]

    # ── Public API ───────────────────────────────────────────────────────

    def detect_secrets(self, content: str) -> list[str]:
        """Return a list of detected secret category names (no values).

        Useful for logging/telemetry without leaking the actual secret.
        """
        found: list[str] = []
        for name, pattern in self._SECRET_PATTERNS:
            if pattern.search(content):
                found.append(name)
        return found

    def sanitize(self, content: str, role: str = "") -> str:
        """Strip secrets (and infra details for security reviewers)."""
        out = content
        for _name, pattern in self._SECRET_PATTERNS:
            out = pattern.sub(_REDACTED, out)
        # Security-reviewer entries also have infra patterns stripped
        if role == "security_reviewer":
            for _name, pattern in self._INFRA_PATTERNS:
                out = pattern.sub(_REDACTED, out)
        return out

    def sanitize_entry(self, entry: "MemoryEntry") -> "MemoryEntry":
        """Return a sanitised copy of an entry (input not mutated)."""
        from dataclasses import replace

        new_content = self.sanitize(entry.content, role=entry.agent_role)
        # Also sanitise any string values in metadata
        new_metadata: dict[str, Any] = {}
        for k, v in entry.metadata.items():
            if isinstance(v, str):
                new_metadata[k] = self.sanitize(v, role=entry.agent_role)
            else:
                new_metadata[k] = v
        return replace(entry, content=new_content, metadata=new_metadata)
