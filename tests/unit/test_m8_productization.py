"""M8 tests — Error system, packaging, CLI UX."""

import os
from pathlib import Path

import pytest

from harness_core.errors.errors import (
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


class TestErrorSystem:
    def test_base_error(self):
        err = HarnessError("something broke", what="thing broke", why="bad input", fix="fix it")
        assert "thing broke" in err.user_message()
        assert "bad input" in err.user_message()
        assert "fix it" in err.user_message()

    def test_configuration_error(self):
        err = ConfigurationError(
            "missing config",
            what="No configuration found",
            why="harness init not run",
            fix="Run: harness init",
        )
        assert isinstance(err, HarnessError)
        msg = err.user_message()
        assert "No configuration found" in msg
        assert "Run: harness init" in msg

    def test_provider_error(self):
        err = ProviderError(
            "connection failed",
            what="Cannot connect to OpenRouter",
            why="API key invalid or missing",
            fix="Check OPENROUTER_API_KEY",
        )
        assert isinstance(err, HarnessError)

    def test_model_error(self):
        err = ModelError(
            "rate limited",
            what="Rate limit exceeded (429)",
            why="Too many requests",
            fix="Wait and retry, or use a different model",
        )
        assert isinstance(err, HarnessError)

    def test_permission_error(self):
        err = PermissionError(
            "blocked",
            what="Operation blocked by security policy",
            why="Dangerous shell command",
            fix="Review the command and approve if safe",
        )
        assert isinstance(err, HarnessError)

    def test_tool_error(self):
        err = ToolError(
            "command failed",
            what="Shell command returned non-zero exit code",
            why="pytest found failing tests",
            fix="Review the test output above",
        )
        assert isinstance(err, HarnessError)

    def test_workspace_error(self):
        err = WorkspaceError(
            "not a repo",
            what="Current directory is not a git repository",
            why="No .git directory found",
            fix="Run 'git init' or navigate to a git repository",
        )
        assert isinstance(err, HarnessError)

    def test_verification_error(self):
        err = VerificationError(
            "tests failed",
            what="3 of 42 tests failed",
            why="Agent changes introduced regressions",
            fix="Resume the session: harness session resume <id>",
        )
        assert isinstance(err, HarnessError)

    def test_extension_error(self):
        err = ExtensionError(
            "plugin failed",
            what="Plugin 'my-plugin' failed to load",
            why="SyntaxError in plugin.py",
            fix="Check the plugin code or disable it",
        )
        assert isinstance(err, HarnessError)

    def test_session_error(self):
        err = SessionError(
            "not found",
            what="Session 'abc123' not found",
            why="Session may have been deleted",
            fix="Run: harness session list",
        )
        assert isinstance(err, HarnessError)

    def test_error_minimal(self):
        err = HarnessError("simple error")
        msg = err.user_message()
        assert "Error: simple error" in msg
        # No why/fix/hint
        assert "Why:" not in msg
        assert "Fix:" not in msg

    def test_error_with_hint(self):
        err = HarnessError(
            "something",
            hint="Try running harness doctor",
        )
        msg = err.user_message()
        assert "Try running harness doctor" in msg

    def test_error_hierarchy(self):
        """All errors inherit from HarnessError."""
        assert issubclass(ConfigurationError, HarnessError)
        assert issubclass(ProviderError, HarnessError)
        assert issubclass(ModelError, HarnessError)
        assert issubclass(PermissionError, HarnessError)
        assert issubclass(ToolError, HarnessError)
        assert issubclass(WorkspaceError, HarnessError)
        assert issubclass(VerificationError, HarnessError)
        assert issubclass(ExtensionError, HarnessError)
        assert issubclass(SessionError, HarnessError)


class TestPackaging:
    def test_pyproject_exists(self):
        pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        assert pyproject.exists()

    def test_pyproject_has_entry_point(self):
        pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        assert 'harness = "harness_core.cli.main:app"' in content

    def test_pyproject_has_dependencies(self):
        pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        assert "typer" in content
        assert "rich" in content
        assert "httpx" in content

    def test_license_exists(self):
        license_file = Path(__file__).parent.parent.parent / "LICENSE"
        assert license_file.exists()
        content = license_file.read_text(encoding="utf-8")
        assert "MIT License" in content

    def test_changelog_exists(self):
        changelog = Path(__file__).parent.parent.parent / "CHANGELOG.md"
        assert changelog.exists()

    def test_readme_exists(self):
        readme = Path(__file__).parent.parent.parent / "README.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert "Harness" in content

    def test_src_package_exists(self):
        src = Path(__file__).parent.parent.parent / "src" / "harness_core"
        assert src.exists()
        assert (src / "__init__.py").exists()

    def test_cli_entry_point_importable(self):
        from harness_core.cli.main import app
        assert app is not None


class TestConfigRedaction:
    def test_api_key_redacted(self):
        from harness_core.config.config import ConfigEntry, ConfigScope
        entry = ConfigEntry(key="api.key", value="sk-secret123", scope=ConfigScope.CLI)
        assert entry._safe_value() == "[REDACTED]"

    def test_token_redacted(self):
        from harness_core.config.config import ConfigEntry, ConfigScope
        entry = ConfigEntry(key="auth.token", value="abc123", scope=ConfigScope.CLI)
        assert entry._safe_value() == "[REDACTED]"

    def test_password_redacted(self):
        from harness_core.config.config import ConfigEntry, ConfigScope
        entry = ConfigEntry(key="db.password", value="secret", scope=ConfigScope.CLI)
        assert entry._safe_value() == "[REDACTED]"

    def test_normal_value_not_redacted(self):
        from harness_core.config.config import ConfigEntry, ConfigScope
        entry = ConfigEntry(key="model.default", value="auto", scope=ConfigScope.CLI)
        assert entry._safe_value() == "auto"

    def test_secret_in_full_key(self):
        from harness_core.config.config import ConfigEntry, ConfigScope
        entry = ConfigEntry(key="my.secret_key", value="val", scope=ConfigScope.CLI)
        assert entry._safe_value() == "[REDACTED]"


class TestDoctorCommand:
    def test_doctor_runs(self):
        """Doctor command should run without crashing."""
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Doctor" in result.output or "Runtime" in result.output


class TestModelsListHelp:
    def test_models_help(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0

    def test_session_help(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["session", "--help"])
        assert result.exit_code == 0

    def test_agents_help(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["agents", "--help"])
        assert result.exit_code == 0

    def test_plugin_help(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["plugin", "--help"])
        assert result.exit_code == 0

    def test_providers_help(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["providers", "--help"])
        assert result.exit_code == 0

    def test_tools_help(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["tools", "--help"])
        assert result.exit_code == 0

    def test_mcp_help(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0

    def test_hooks_help(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["hooks", "--help"])
        assert result.exit_code == 0


class TestProvidersList:
    def test_providers_list(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["providers", "list"])
        assert result.exit_code == 0
        assert "OpenRouter" in result.output or "Provider" in result.output

    def test_providers_list_json(self):
        from typer.testing import CliRunner
        from harness_core.cli.main import app
        runner = CliRunner()
        result = runner.invoke(app, ["providers", "list", "--json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert len(data) >= 2
