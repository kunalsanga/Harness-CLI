"""MCP manager — discovers MCP server configurations, manages client lifecycle, and exposes tools."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from harness_core.mcp.adapter import MCPToolAdapter
from harness_core.mcp.client import MCPClient, MCPServerConfig
from harness_core.tools.base import Tool

logger = logging.getLogger(__name__)

# Core tools and built-in aliases that must never be collided with
CANONICAL_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "write_file", "edit_file", "list_files", "glob", "grep",
    "run_command", "git_status", "git_diff", "git_log", "git_identity",
    "git_add", "git_commit", "git_push", "git_remote",
    "bash", "sh", "terminal", "exec", "read", "view", "edit", "update",
    "replace", "write", "create", "list", "ls", "find", "search",
})


def load_mcp_server_configs(
    workspace_path: Path | str,
    config_data: dict[str, Any] | None = None,
) -> list[MCPServerConfig]:
    """Discover and parse MCP server configurations for a workspace.

    Searches:
    1. Project config (passed dict or `.harness/config.yaml` under `mcp:`)
    2. Standard `.mcp.json` / `mcp.json` in workspace root

    Never raises; returns an empty list on missing or invalid files.
    """
    ws = Path(workspace_path)
    servers: dict[str, MCPServerConfig] = {}

    # 1. Check .harness/config.yaml (or passed config_data)
    harness_cfg = config_data
    if harness_cfg is None:
        cfg_file = ws / ".harness" / "config.yaml"
        if cfg_file.exists():
            try:
                import yaml
                harness_cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
            except Exception as e:
                logger.debug("Failed reading .harness/config.yaml for MCP: %s", e)
                harness_cfg = {}

    if isinstance(harness_cfg, dict) and "mcp" in harness_cfg:
        mcp_section = harness_cfg["mcp"]
        if isinstance(mcp_section, dict):
            # Check global enable switch
            if mcp_section.get("enabled") is False:
                return []

            # Check servers dictionary under mcp.servers
            server_dict = mcp_section.get("servers")
            if not isinstance(server_dict, dict):
                # Flat format: mcp: { server_name: { command: ... } }
                server_dict = {
                    k: v for k, v in mcp_section.items()
                    if k not in ("enabled", "timeout", "timeout_seconds") and isinstance(v, dict)
                }

            for name, s_data in server_dict.items():
                if isinstance(s_data, dict) and s_data.get("command"):
                    cfg = _parse_server_config(name, s_data, ws)
                    if cfg:
                        servers[cfg.name] = cfg

    # 2. Check standard .mcp.json / mcp.json in workspace
    for json_filename in (".mcp.json", "mcp.json"):
        json_path = ws / json_filename
        if json_path.exists():
            try:
                raw_json = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(raw_json, dict):
                    mcp_servers = raw_json.get("mcpServers") or raw_json.get("servers") or {}
                    if isinstance(mcp_servers, dict):
                        for name, s_data in mcp_servers.items():
                            if name not in servers and isinstance(s_data, dict) and s_data.get("command"):
                                cfg = _parse_server_config(name, s_data, ws)
                                if cfg:
                                    servers[cfg.name] = cfg
            except Exception as e:
                logger.debug("Failed reading %s for MCP: %s", json_filename, e)

    return list(servers.values())


def _parse_server_config(name: str, data: dict[str, Any], workspace: Path) -> MCPServerConfig | None:
    """Safely parse a single server configuration dict."""
    command = str(data.get("command", "")).strip()
    if not command:
        return None

    args = [str(a) for a in (data.get("args") or [])]
    env = {str(k): str(v) for k, v in (data.get("env") or {}).items()}
    cwd = str(data.get("cwd", "")).strip()
    if cwd:
        try:
            cwd_path = (workspace / cwd).resolve()
            # Enforce workspace confinement on server working directory
            cwd_path.relative_to(workspace.resolve())
            cwd = str(cwd_path)
        except Exception:
            cwd = str(workspace)
    else:
        cwd = str(workspace)

    enabled = bool(data.get("enabled", True))
    timeout = float(data.get("timeout_seconds", data.get("timeout", 30.0)))
    transport = str(data.get("transport", "stdio")).strip()

    return MCPServerConfig(
        name=name,
        command=command,
        args=args,
        env=env,
        cwd=cwd,
        enabled=enabled,
        timeout_seconds=timeout,
        transport=transport,
    )


def init_mcp_client_from_workspace(
    workspace_path: Path | str,
    config_data: dict[str, Any] | None = None,
) -> MCPClient:
    """Initialize an MCPClient populated with discovered server configurations."""
    client = MCPClient()
    configs = load_mcp_server_configs(workspace_path, config_data=config_data)
    for cfg in configs:
        client.add_server(cfg)
    return client


async def discover_mcp_tools(
    client: MCPClient,
    existing_tool_names: set[str] | None = None,
) -> tuple[list[Tool], dict[str, str]]:
    """Start enabled MCP servers, discover their tools, and build adapters with namespacing.

    Returns:
        tuple of (adapters_list, alias_mapping_dict)
    """
    reserved_names = CANONICAL_TOOL_NAMES if existing_tool_names is None else (CANONICAL_TOOL_NAMES | existing_tool_names)
    servers = [s for s in client.list_servers() if s.enabled]
    if not servers:
        return [], {}

    # Start all enabled servers and discover tools
    for server in servers:
        try:
            started = await client.start_server(server.name)
            if not started:
                logger.warning("Could not start MCP server '%s'", server.name)
        except Exception as e:
            logger.warning("Exception starting MCP server '%s': %s", server.name, e)

    # Count tool names across all servers to detect collisions between servers
    all_discovered = client.list_tools()
    name_counts: dict[str, int] = {}
    for t in all_discovered:
        name_counts[t.name] = name_counts.get(t.name, 0) + 1

    adapters: list[Tool] = []
    aliases: dict[str, str] = {}

    for server in servers:
        server_tools = client.list_tools(server.name)
        for t in server_tools:
            # Deterministic primary name
            if t.name.startswith(f"mcp_{server.name}_"):
                primary_name = t.name
            else:
                primary_name = f"mcp_{server.name}_{t.name}"

            tool_aliases: list[str] = []
            # An alias to the bare tool name is permitted only when:
            # 1. It does not collide with any native/canonical tool name or alias
            # 2. It is uniquely provided by this server (no collision with other MCP servers)
            if t.name not in reserved_names and name_counts.get(t.name, 0) == 1:
                tool_aliases.append(t.name)
                aliases[t.name] = primary_name

            adapter = MCPToolAdapter(
                client=client,
                server_name=server.name,
                tool=t,
                tool_name=primary_name,
                timeout_seconds=server.timeout_seconds,
                aliases=tool_aliases,
            )
            adapters.append(adapter)

    return adapters, aliases


async def init_mcp_tools_from_workspace(
    workspace_path: Path | str,
    config_data: dict[str, Any] | None = None,
    client: MCPClient | None = None,
    existing_tool_names: set[str] | None = None,
) -> tuple[list[Tool], MCPClient | None, dict[str, str]]:
    """Bootstrap MCP tools for a workspace session.

    Returns:
        (tools, client, aliases) or ([], None, {}) if no MCP servers configured.
    """
    if client is None:
        client = init_mcp_client_from_workspace(workspace_path, config_data=config_data)

    if not client.list_servers():
        return [], None, {}

    adapters, aliases = await discover_mcp_tools(client, existing_tool_names=existing_tool_names)
    return adapters, client, aliases
