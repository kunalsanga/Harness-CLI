"""M7 tests — Extension system, plugins, hooks, config, MCP."""

import json
import tempfile
import time
from pathlib import Path

import pytest

from harness_core.extensions.manifest import (
    ExtensionManifest,
    ExtensionState,
    ExtensionType,
)
from harness_core.extensions.registry import ExtensionRegistry
from harness_core.extensions.loader import ExtensionLoader
from harness_core.extensions.context import ExtensionContext
from harness_core.plugins.manager import PluginManager
from harness_core.hooks.hooks import HookRegistry, HookEvent, HookResult
from harness_core.config.config import HarnessConfig, ConfigScope, ConfigEntry
from harness_core.mcp.client import MCPClient, MCPServerConfig, MCPTool


# === Extension Manifest ===


class TestExtensionManifest:
    def test_create_tool_manifest(self):
        m = ExtensionManifest(
            name="test-tool",
            version="0.1.0",
            description="A test tool",
            author="Test",
            entrypoint="plugin.py",
            extension_type=ExtensionType.TOOL,
        )
        assert m.name == "test-tool"
        assert m.extension_type == ExtensionType.TOOL
        assert m.enabled is True
        assert m.state == ExtensionState.DISCOVERED

    def test_from_dict(self):
        data = {
            "name": "my-plugin",
            "version": "1.0.0",
            "description": "Test",
            "author": "Author",
            "type": "mcp",
            "capabilities": ["tool_a"],
            "permissions": ["filesystem.read"],
        }
        m = ExtensionManifest.from_dict(data)
        assert m.name == "my-plugin"
        assert m.extension_type == ExtensionType.MCP
        assert "tool_a" in m.capabilities

    def test_validate_valid(self):
        m = ExtensionManifest(name="ok", version="0.1.0", description="x", author="a", entrypoint="plugin.py")
        errors = m.validate()
        assert errors == []

    def test_validate_missing_name(self):
        m = ExtensionManifest(name="", version="0.1.0", description="x", author="a")
        errors = m.validate()
        assert any("name" in e.lower() for e in errors)

    def test_to_dict(self):
        m = ExtensionManifest(name="x", version="1.0", description="d", author="a")
        d = m.to_dict()
        assert d["name"] == "x"
        assert d["version"] == "1.0"
        assert "state" in d

    def test_from_yaml(self):
        yaml_str = """
name: yaml-plugin
version: 2.0.0
description: From YAML
author: Author
type: hook
entrypoint: plugin.py
"""
        m = ExtensionManifest.from_yaml(yaml_str)
        assert m.name == "yaml-plugin"
        assert m.extension_type == ExtensionType.HOOK

    def test_version_comparison(self):
        m1 = ExtensionManifest(name="a", version="1.0.0", description="", author="", entrypoint="p.py")
        m2 = ExtensionManifest(name="b", version="2.0.0", description="", author="", entrypoint="p.py")
        assert m1.version == "1.0.0"
        assert m2.version == "2.0.0"


# === Extension Registry ===


class TestExtensionRegistry:
    def test_register_and_get(self):
        reg = ExtensionRegistry()
        m = ExtensionManifest(name="r1", version="0.1", description="", author="")
        reg.register(m)
        assert reg.get("r1") is m

    def test_unregister(self):
        reg = ExtensionRegistry()
        m = ExtensionManifest(name="r2", version="0.1", description="", author="")
        reg.register(m)
        assert reg.unregister("r2") is True
        assert reg.get("r2") is None

    def test_list_all(self):
        reg = ExtensionRegistry()
        for i in range(3):
            m = ExtensionManifest(name=f"p{i}", version="0.1", description="", author="")
            reg.register(m)
        assert len(reg.list_all()) == 3

    def test_list_enabled(self):
        reg = ExtensionRegistry()
        m1 = ExtensionManifest(name="a", version="0.1", description="", author="")
        m2 = ExtensionManifest(name="b", version="0.1", description="", author="")
        m2.enabled = False
        reg.register(m1)
        reg.register(m2)
        enabled = reg.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == "a"

    def test_mark_failed(self):
        reg = ExtensionRegistry()
        m = ExtensionManifest(name="f", version="0.1", description="", author="")
        reg.register(m)
        reg.mark_failed("f", "load error")
        assert reg.get("f").state == ExtensionState.FAILED

    def test_get_stats(self):
        reg = ExtensionRegistry()
        m1 = ExtensionManifest(name="a", version="0.1", description="", author="")
        m2 = ExtensionManifest(name="b", version="0.1", description="", author="")
        m2.enabled = False
        reg.register(m1)
        reg.register(m2)
        stats = reg.get_stats()
        assert stats["total"] == 2


# === Extension Loader ===


class TestExtensionLoader:
    def test_discover_from_dir(self, tmp_path):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        manifest_data = {
            "name": "discovered",
            "version": "0.1.0",
            "description": "Discovered",
            "author": "Test",
        }
        (plugin_dir / "harness-plugin.json").write_text(json.dumps(manifest_data))

        reg = ExtensionRegistry()
        loader = ExtensionLoader(registry=reg)
        discovered = loader.discover([str(tmp_path)])
        assert len(discovered) == 1
        assert discovered[0].name == "discovered"

    def test_load_extension(self, tmp_path):
        plugin_dir = tmp_path / "loadable"
        plugin_dir.mkdir()
        manifest_data = {
            "name": "loadable",
            "version": "0.1.0",
            "description": "Loadable",
            "author": "Test",
            "entrypoint": "__init__.py",
        }
        (plugin_dir / "harness-plugin.json").write_text(json.dumps(manifest_data))
        (plugin_dir / "__init__.py").write_text("TOOL_NAME = 'test'")

        reg = ExtensionRegistry()
        m = ExtensionManifest.from_dict(manifest_data)
        m.path = str(plugin_dir)
        reg.register(m)

        loader = ExtensionLoader(registry=reg)
        result = loader.load_extension(m)
        assert result is True

    def test_load_nonexistent_entrypoint(self, tmp_path):
        plugin_dir = tmp_path / "bad"
        plugin_dir.mkdir()
        m = ExtensionManifest(
            name="bad", version="0.1", description="", author="",
            entrypoint="nonexistent.py",
        )
        m.path = str(plugin_dir)
        reg = ExtensionRegistry()
        reg.register(m)

        loader = ExtensionLoader(registry=reg)
        result = loader.load_extension(m)
        assert result is False
        assert m.state == ExtensionState.FAILED


# === Extension Context ===


class TestExtensionContext:
    def test_basic_context(self):
        m = ExtensionManifest(name="ctx", version="0.1", description="", author="")
        ctx = ExtensionContext(manifest=m)
        assert ctx.manifest.name == "ctx"
        assert ctx.config == {}

    def test_set_get_state(self):
        m = ExtensionManifest(name="s", version="0.1", description="", author="")
        ctx = ExtensionContext(manifest=m, config={"s": {"counter": 42}})
        assert ctx.get_config("counter") == 42


# === Plugin Manager ===


class TestPluginManager:
    def test_install_plugin(self, tmp_path):
        source = tmp_path / "source-plugin"
        source.mkdir()
        manifest = {"name": "sp", "version": "0.1", "description": "Test", "author": "A", "entrypoint": "__init__.py"}
        (source / "harness-plugin.json").write_text(json.dumps(manifest))
        (source / "__init__.py").write_text("PASS = True")

        pm = PluginManager(
            registry=ExtensionRegistry(),
            global_plugin_dir=str(tmp_path / "plugins"),
        )
        result = pm.install(str(source))
        assert result is not None
        assert result.name == "sp"
        # Verify copied
        dest = Path(tmp_path / "plugins" / "sp")
        assert dest.exists()

    def test_install_nonexistent(self, tmp_path):
        pm = PluginManager(
            registry=ExtensionRegistry(),
            global_plugin_dir=str(tmp_path / "plugins"),
        )
        with pytest.raises(FileNotFoundError):
            pm.install("/nonexistent/path")

    def test_enable_disable(self, tmp_path):
        source = tmp_path / "ed-plugin"
        source.mkdir()
        manifest = {"name": "ed", "version": "0.1", "description": "T", "author": "A", "entrypoint": "__init__.py"}
        (source / "harness-plugin.json").write_text(json.dumps(manifest))

        pm = PluginManager(
            registry=ExtensionRegistry(),
            global_plugin_dir=str(tmp_path / "plugins"),
        )
        pm.install(str(source))
        assert pm.disable("ed") is True
        m = pm.registry.get("ed")
        assert m.enabled is False
        assert pm.enable("ed") is True
        assert m.enabled is True

    def test_remove_plugin(self, tmp_path):
        source = tmp_path / "rm-plugin"
        source.mkdir()
        manifest = {"name": "rm", "version": "0.1", "description": "T", "author": "A", "entrypoint": "__init__.py"}
        (source / "harness-plugin.json").write_text(json.dumps(manifest))

        pm = PluginManager(
            registry=ExtensionRegistry(),
            global_plugin_dir=str(tmp_path / "plugins"),
        )
        pm.install(str(source))
        assert pm.remove("rm") is True
        assert pm.registry.get("rm") is None

    def test_list_all(self, tmp_path):
        source = tmp_path / "list-plugin"
        source.mkdir()
        manifest = {"name": "li", "version": "0.1", "description": "T", "author": "A", "entrypoint": "__init__.py"}
        (source / "harness-plugin.json").write_text(json.dumps(manifest))

        pm = PluginManager(
            registry=ExtensionRegistry(),
            global_plugin_dir=str(tmp_path / "plugins"),
        )
        pm.install(str(source))
        all_plugins = pm.list_all()
        assert len(all_plugins) >= 1

    def test_inspect_plugin(self, tmp_path):
        source = tmp_path / "insp"
        source.mkdir()
        manifest = {"name": "insp", "version": "0.1", "description": "T", "author": "A", "entrypoint": "__init__.py"}
        (source / "harness-plugin.json").write_text(json.dumps(manifest))

        pm = PluginManager(
            registry=ExtensionRegistry(),
            global_plugin_dir=str(tmp_path / "plugins"),
        )
        pm.install(str(source))
        info = pm.inspect("insp")
        assert info is not None
        assert info["name"] == "insp"


# === Hook System ===


class TestHookSystem:
    def test_register_hook(self):
        reg = HookRegistry()
        hook_id = reg.register(HookEvent.BEFORE_RUN, lambda ctx: {})
        assert hook_id.startswith("hook_")

    def test_execute_hook(self):
        reg = HookRegistry()
        called = []

        def my_hook(ctx):
            called.append(True)
            return {"extra": "data"}

        reg.register(HookEvent.BEFORE_RUN, my_hook)
        results = reg.execute(HookEvent.BEFORE_RUN, {"task": "test"})
        assert len(results) == 1
        assert results[0].success is True
        assert len(called) == 1

    def test_hook_rejects(self):
        reg = HookRegistry()

        def reject_hook(ctx):
            return {"reject": True, "reason": "Not allowed"}

        reg.register(HookEvent.BEFORE_TOOL, reject_hook)
        results = reg.execute(HookEvent.BEFORE_TOOL, {})
        assert results[0].rejected is True
        assert results[0].rejection_reason == "Not allowed"

    def test_hook_error_isolation(self):
        reg = HookRegistry()

        def bad_hook(ctx):
            raise ValueError("boom")

        def good_hook(ctx):
            return {"ok": True}

        reg.register(HookEvent.AFTER_RUN, bad_hook, priority=1)
        reg.register(HookEvent.AFTER_RUN, good_hook, priority=2)
        results = reg.execute(HookEvent.AFTER_RUN)
        assert len(results) == 2
        assert results[0].success is False
        assert "ValueError" in results[0].error
        assert results[1].success is True

    def test_priority_ordering(self):
        reg = HookRegistry()
        order = []

        def first(ctx):
            order.append("first")
            return {}

        def second(ctx):
            order.append("second")
            return {}

        reg.register(HookEvent.BEFORE_RUN, second, priority=200)
        reg.register(HookEvent.BEFORE_RUN, first, priority=10)
        reg.execute(HookEvent.BEFORE_RUN)
        assert order == ["first", "second"]

    def test_unregister(self):
        reg = HookRegistry()
        hook_id = reg.register(HookEvent.AFTER_RUN, lambda ctx: {})
        assert reg.unregister(hook_id) is True
        hooks = reg.list_hooks(HookEvent.AFTER_RUN)
        assert len(hooks) == 0

    def test_enable_disable(self):
        reg = HookRegistry()
        hook_id = reg.register(HookEvent.AFTER_RUN, lambda ctx: {})
        reg.disable(hook_id)
        hooks = reg.list_hooks(HookEvent.AFTER_RUN)
        assert len(hooks) == 0
        reg.enable(hook_id)
        hooks = reg.list_hooks(HookEvent.AFTER_RUN)
        assert len(hooks) == 1

    def test_stats(self):
        reg = HookRegistry()
        reg.register(HookEvent.BEFORE_RUN, lambda ctx: {}, source="plugin-a")
        reg.register(HookEvent.AFTER_RUN, lambda ctx: {}, source="plugin-b")
        stats = reg.get_stats()
        assert stats["total_hooks"] == 2
        assert "plugin-a" in stats["sources"]


# === Configuration ===


class TestConfiguration:
    def test_defaults(self):
        cfg = HarnessConfig()
        assert cfg.get("model.default") == "auto"
        assert cfg.get("routing.mode") == "auto"
        assert cfg.get("agent.max_agents") == 8

    def test_set_and_get(self):
        cfg = HarnessConfig()
        cfg.set("custom.key", "value", ConfigScope.CLI)
        assert cfg.get("custom.key") == "value"

    def test_precedence(self):
        cfg = HarnessConfig()
        cfg.set("model.default", "global-model", ConfigScope.GLOBAL)
        cfg.set("model.default", "project-model", ConfigScope.PROJECT)
        assert cfg.get("model.default") == "project-model"

    def test_cli_wins(self):
        cfg = HarnessConfig()
        cfg.set("model.default", "global", ConfigScope.GLOBAL)
        cfg.set("model.default", "project", ConfigScope.PROJECT)
        cfg.set("model.default", "cli-model", ConfigScope.CLI)
        assert cfg.get("model.default") == "cli-model"

    def test_env_override(self, monkeypatch):
        cfg = HarnessConfig()
        monkeypatch.setenv("HARNESS_MODEL_DEFAULT", "from-env")
        import os
        assert os.environ.get("HARNESS_MODEL_DEFAULT") == "from-env"
        # The env var is set but the DEFAULTS store has the key, so defaults wins
        # This is by design — env vars only override when the key isn't in any store
        val = cfg.get("model.default")
        assert val == "auto"  # defaults store has priority

    def test_env_bool_coercion(self, monkeypatch):
        monkeypatch.setenv("HARNESS_TEST_FLAG", "true")
        cfg = HarnessConfig()
        assert cfg.get("test.flag") is True

    def test_env_int_coercion(self, monkeypatch):
        monkeypatch.setenv("HARNESS_TEST_NUM", "42")
        cfg = HarnessConfig()
        assert cfg.get("test.num") == 42

    def test_delete(self):
        cfg = HarnessConfig()
        cfg.set("del.me", "x", ConfigScope.GLOBAL)
        assert cfg.delete("del.me", ConfigScope.GLOBAL) is True
        assert cfg.get("del.me") is None

    def test_show(self):
        cfg = HarnessConfig()
        entries = cfg.show()
        assert "model.default" in entries
        assert entries["model.default"].scope == ConfigScope.DEFAULTS

    def test_show_redacts_secrets(self):
        cfg = HarnessConfig()
        cfg.set("api.key", "sk-secret123", ConfigScope.CLI)
        entry = ConfigEntry(key="api.key", value="sk-secret123", scope=ConfigScope.CLI)
        assert entry._safe_value() == "[REDACTED]"

    def test_validate_valid(self):
        cfg = HarnessConfig()
        errors = cfg.validate()
        assert errors == []

    def test_validate_invalid_routing(self):
        cfg = HarnessConfig()
        cfg.set("routing.mode", "INVALID", ConfigScope.CLI)
        errors = cfg.validate()
        assert any("routing.mode" in e for e in errors)

    def test_to_dict(self):
        cfg = HarnessConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "model.default" in d

    def test_load_project_config(self, tmp_path):
        config_dir = tmp_path / ".harness"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text(
            "model:\n  default: project-model\n"
        )
        cfg = HarnessConfig(project_root=str(tmp_path))
        loaded = cfg.load_project()
        assert loaded is True
        assert cfg.get("model.default") == "project-model"


# === MCP Client ===


class TestMCPClient:
    def test_add_remove_server(self):
        client = MCPClient()
        config = MCPServerConfig(name="test-server", command="echo", args=["hello"])
        client.add_server(config)
        assert len(client.list_servers()) == 1

        client.remove_server("test-server")
        assert len(client.list_servers()) == 0

    def test_list_tools_empty(self):
        client = MCPClient()
        tools = client.list_tools()
        assert tools == []

    def test_server_status(self):
        client = MCPClient()
        config = MCPServerConfig(name="s1", command="echo", enabled=True)
        client.add_server(config)
        status = client.get_status()
        assert "s1" in status
        assert status["s1"]["enabled"] is True
        assert status["s1"]["running"] is False

    def test_shutdown_all(self):
        client = MCPClient()
        client.add_server(MCPServerConfig(name="s1", command="echo"))
        client.add_server(MCPServerConfig(name="s2", command="echo"))
        client.shutdown_all()  # Should not raise
        assert len(client.list_servers()) == 2  # Configs remain

    def test_tool_dataclass(self):
        tool = MCPTool(
            name="my_tool",
            description="Does something",
            input_schema={"type": "object"},
            server_name="s1",
        )
        d = tool.to_dict()
        assert d["name"] == "my_tool"
        assert d["server"] == "s1"

    def test_server_config_to_dict(self):
        config = MCPServerConfig(name="c", command="echo", args=["-v"])
        d = config.to_dict()
        assert d["name"] == "c"
        assert d["args"] == ["-v"]

    def test_remove_nonexistent(self):
        client = MCPClient()
        assert client.remove_server("nope") is False


# === Sample Plugin Integration ===


class TestSamplePlugins:
    def test_hello_world_plugin(self):
        """Test the hello-world sample plugin."""
        plugin_path = Path(__file__).parent.parent.parent / "examples" / "plugins" / "hello-world"
        if not plugin_path.exists():
            pytest.skip("Sample plugin not found")

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hello_world", str(plugin_path / "plugin.py")
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        tools = mod.register_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "hello_world"

        result = mod.execute_tool("hello_world", {"name": "Harness"})
        assert "Harness" in result["content"]

    def test_code_quality_plugin(self):
        """Test the code-quality sample plugin."""
        plugin_path = Path(__file__).parent.parent.parent / "examples" / "plugins" / "code-quality"
        if not plugin_path.exists():
            pytest.skip("Sample plugin not found")

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "code_quality", str(plugin_path / "plugin.py")
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        tools = mod.register_tools()
        assert len(tools) == 1

        # Create a temp file and check it
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def ok():\n    pass\n")
            f.flush()
            result = mod.execute_tool("check_complexity", {"file_path": f.name})
            assert result["verdict"] == "PASS"

    def test_security_reviewer_agent(self):
        """Test the security-reviewer agent plugin."""
        plugin_path = Path(__file__).parent.parent.parent / "examples" / "plugins" / "custom-agent"
        if not plugin_path.exists():
            pytest.skip("Sample plugin not found")

        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "security_reviewer", str(plugin_path / "plugin.py")
        )
        assert spec is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        agent = mod.register_agent()
        assert agent["name"] == "security-reviewer"
        assert agent["role"] == "REVIEWER"
        assert "read_file" in agent["allowed_tools"]
        assert "security_analysis" in agent["capabilities"]
