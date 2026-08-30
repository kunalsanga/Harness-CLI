"""Unit tests for the permission system."""

import tempfile
from pathlib import Path

from harness_core.permissions.manager import PermissionManager, PermissionRule


class TestPermissionManager:
    """Tests for PermissionManager."""

    def test_default_permissions(self):
        pm = PermissionManager()
        assert pm.check_permission("read_file") == "allow"
        assert pm.check_permission("write_file") == "allow"
        # In autonomous mode (default), safe commands are auto-approved
        assert pm.check_permission("run_command", {"command": "node test.js"}) == "allow"

    def test_custom_rules(self):
        rules = [
            PermissionRule(tool_pattern="custom_tool", action="deny"),
        ]
        pm = PermissionManager(rules=rules)
        assert pm.check_permission("custom_tool") == "deny"

    def test_is_within_workspace(self):
        with tempfile.TemporaryDirectory() as d:
            pm = PermissionManager(workspace_root=Path(d))
            assert pm.is_within_workspace(Path(d) / "file.txt")
            assert not pm.is_within_workspace("/tmp/outside/file.txt")

    def test_is_protected_path(self):
        pm = PermissionManager()
        assert pm.is_protected_path(".env")
        assert pm.is_protected_path("credentials.json")
        assert pm.is_protected_path("private_key.pem")
        assert not pm.is_protected_path("README.md")

    def test_unknown_tool_defaults_to_ask(self):
        pm = PermissionManager()
        assert pm.check_permission("unknown_tool_xyz") == "ask"
        # In autonomous mode, 'ask' tools are auto-approved via request_approval
        assert pm.request_approval("unknown_tool_xyz") is True
        # Without autonomous mode, 'ask' tools are denied by default
        pm2 = PermissionManager(autonomous_mode=False)
        assert pm2.request_approval("unknown_tool_xyz") is False
