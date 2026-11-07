"""CLI tests for multi-provider management, status inspection, testing, and secret masking."""

from __future__ import annotations

import json
import pytest
from typer.testing import CliRunner
from unittest.mock import AsyncMock, patch

from harness_core.cli.main import app
from harness_core.config.credentials import mask_secret

runner = CliRunner()


def test_secret_masking():
    """Verify secrets are masked and never exposed in full."""
    assert mask_secret("sk-or-v1-abcdef1234567890abcdef1234567890") != "sk-or-v1-abcdef1234567890abcdef1234567890"
    assert mask_secret("sk-or-v1-abcdef1234567890abcdef1234567890").startswith("sk-o")
    assert mask_secret("sk-or-v1-abcdef1234567890abcdef1234567890").endswith("7890")
    assert "*" in mask_secret("sk-or-v1-abcdef1234567890abcdef1234567890")
    assert mask_secret("") == ""
    assert mask_secret("short") == "*****"


def test_cli_providers_list():
    """Test `harness providers list` command."""
    res = runner.invoke(app, ["providers", "list"])
    assert res.exit_code == 0
    assert "openrouter" in res.stdout
    assert "openai" in res.stdout
    assert "nvidia" in res.stdout
    assert "freellmapi" in res.stdout


def test_cli_providers_list_json():
    """Test `harness providers list --json` command."""
    res = runner.invoke(app, ["providers", "list", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)
    names = [p["name"] for p in data]
    assert "openrouter" in names
    assert "openai" in names
    assert "nvidia" in names
    assert "freellmapi" in names


def test_cli_config_validate():
    """Test `harness config validate` command."""
    res = runner.invoke(app, ["config", "validate"])
    assert res.exit_code == 0
    assert "Target Providers Status:" in res.stdout
    assert "openrouter" in res.stdout
    assert "openai" in res.stdout
    assert "nvidia" in res.stdout
    assert "freellmapi" in res.stdout


def test_cli_providers_status_json():
    """Test `harness providers status --json`."""
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-fake-key-for-test-only-12345678"}):
        res = runner.invoke(app, ["providers", "status", "--json"])
        assert res.exit_code == 0
        data = json.loads(res.stdout)
        or_entry = next(d for d in data if d["provider"] == "openrouter")
        assert or_entry["status"] in ("Healthy", "Degraded", "Configured", "Temporarily unavailable")
        # Ensure secret is masked in the output
        assert "sk-or-fake-key-for-test-only-12345678" not in json.dumps(data)
        assert "*" in or_entry["credential"]

