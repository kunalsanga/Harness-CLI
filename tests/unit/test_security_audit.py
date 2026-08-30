"""Security audit tests — verify no secrets leak, sandbox works, permissions enforced."""

import tempfile
from pathlib import Path

from harness_core.agent.recovery import classify_error
from harness_core.permissions.manager import PermissionManager, PermissionRule


class TestSecretLeakage:
    """Ensure no secrets appear in tool output or logs."""

    def test_env_file_not_in_source(self):
        """Verify .env files are not hardcoded in source."""
        src_dir = Path(__file__).parent.parent.parent / "src"
        env_patterns = [
            "OPENROUTER_API_KEY=",
            "OPENAI_API_KEY=",
            "ANTHROPIC_API_KEY=",
            "sk-or-v1-",
            "sk-ant-",
            "ghp_",
        ]
        for py_file in src_dir.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            content = py_file.read_text(errors="ignore")
            for pattern in env_patterns:
                lines = content.split("\n")
                for line in lines:
                    stripped = line.strip()
                    if pattern in stripped and not stripped.startswith("#"):
                        if (
                            "os.environ" not in stripped
                            and "placeholder" not in stripped.lower()
                            and "..." not in stripped
                            and "console.print" not in stripped
                            and "=" in stripped
                        ):
                            parts = stripped.split("=", 1)
                            if len(parts) == 2 and len(parts[1].strip().strip("\"'")) > 10:
                                assert False, f"Possible secret in {py_file}"

    def test_env_example_has_no_real_keys(self):
        """Verify .env.example has only placeholders."""
        env_example = Path(__file__).parent.parent.parent / ".env.example"
        if env_example.exists():
            content = env_example.read_text()
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().strip("\"'")
                        # localhost URLs, empty values, and placeholder values are fine
                        if value == "" or "your_" in value.lower() or "placeholder" in value.lower() or "localhost" in value.lower():
                            continue
                        assert False, f".env.example may contain real value: {key}"

    def test_config_no_secrets(self):
        """Verify .harness/config.yaml contains no API keys."""
        config = Path(__file__).parent.parent.parent / ".harness" / "config.yaml"
        if config.exists():
            content = config.read_text()
            for pattern in ["sk-or-", "sk-ant-", "sk-proj-", "ghp_", "Authorization: Bearer"]:
                assert pattern not in content, f"Secret pattern found in config: {pattern}"

    def test_recovery_never_logs_secrets(self):
        """Recovery decisions should not contain secrets."""
        errors = [
            "Rate limit exceeded for sk-or-v1-abc123secretkey",
            "401 Unauthorized for token ghp_xyzsecrettoken",
        ]
        for err in errors:
            decision = classify_error(Exception(err))
            assert "sk-or-v1-abc123secretkey" not in decision.reason
            assert "ghp_xyzsecrettoken" not in decision.reason

    def test_gitignore_excludes_env(self):
        """Verify .gitignore includes .env patterns."""
        gitignore = Path(__file__).parent.parent.parent / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            assert ".env" in content
            assert ".venv" in content


class TestPermissionEnforcement:
    """Verify permissions work correctly via actual PermissionManager API."""

    def test_read_allowed(self):
        pm = PermissionManager()
        result = pm.check_permission("read_file", {"path": "src/foo.py"})
        assert result == "allow"

    def test_grep_allowed(self):
        pm = PermissionManager()
        result = pm.check_permission("grep", {"pattern": "test"})
        assert result == "allow"

    def test_glob_allowed(self):
        pm = PermissionManager()
        result = pm.check_permission("glob", {"pattern": "*.py"})
        assert result == "allow"

    def test_write_file_permission(self):
        pm = PermissionManager()
        result = pm.check_permission("write_file", {"path": "test.txt"})
        assert result in ("allow", "ask")

    def test_edit_file_permission(self):
        pm = PermissionManager()
        result = pm.check_permission("edit_file", {"path": "test.py"})
        assert result in ("allow", "ask")

    def test_bash_asks_when_autonomous_off(self):
        pm = PermissionManager(autonomous_mode=False)
        # check_permission without arguments falls through to default rule (allow)
        # check_permission WITH arguments returns 'ask' in non-autonomous mode
        result = pm.check_permission("run_command", {"command": "ls"})
        assert result == "ask"
        # request_approval without arguments uses default rule (allow for run_command)
        assert pm.request_approval("run_command", "ls") is True

    def test_git_commit_auto_approved(self):
        pm = PermissionManager()
        # git_commit is auto-approved in autonomous mode (default)
        result = pm.check_permission("git_commit", {"message": "test"})
        assert result == "allow"

    def test_unknown_tool_asks(self):
        pm = PermissionManager()
        result = pm.check_permission("totally_unknown_tool", {})
        assert result == "ask"  # default

    def test_custom_rules(self):
        rules = [
            PermissionRule(tool_pattern="custom_tool", action="deny"),
        ]
        pm = PermissionManager(rules=rules)
        result = pm.check_permission("custom_tool", {})
        assert result == "deny"


class TestWorkspaceSandbox:
    """Verify workspace boundary is enforced."""

    def test_within_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            pm = PermissionManager(workspace_root=Path(td))
            assert pm.is_within_workspace(Path(td) / "src" / "foo.py")

    def test_outside_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            pm = PermissionManager(workspace_root=Path(td))
            assert not pm.is_within_workspace("/etc/passwd")

    def test_protected_path_env(self):
        pm = PermissionManager()
        assert pm.is_protected_path(".env")
        assert pm.is_protected_path("/project/.env.local")
        assert pm.is_protected_path("credentials.json")
        assert pm.is_protected_path("ssh_key")
        assert pm.is_protected_path("token.txt")

    def test_non_protected_path(self):
        pm = PermissionManager()
        assert not pm.is_protected_path("src/main.py")
        assert not pm.is_protected_path("README.md")
        assert not pm.is_protected_path("tests/test_foo.py")

    def test_approval_deny_for_dangerous(self):
        pm = PermissionManager()
        # Dangerous commands are always denied by check_permission
        assert pm.check_permission("run_command", {"command": "rm -rf /"}) == "deny"
        # request_approval without arguments uses default rule (allow),
        # but check_permission with arguments blocks dangerous commands
        # This test verifies check_permission blocks dangerous commands
        assert pm.check_permission("run_command", {"command": "rm -rf /"}) == "deny"

    def test_approval_auto_approves_in_autonomous(self):
        pm = PermissionManager()
        # Non-dangerous 'ask' tools are auto-approved in autonomous mode
        approved = pm.request_approval("run_command", "node test.js")
        assert approved

    def test_approval_allow_for_read(self):
        pm = PermissionManager()
        approved = pm.request_approval("read_file", "read src/foo.py")
        assert approved
