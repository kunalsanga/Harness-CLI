"""MCP (Model Context Protocol) integration for Harness."""

from .adapter import MCPToolAdapter
from .client import MCPClient, MCPServerConfig, MCPTool
from .manager import (
    discover_mcp_tools,
    init_mcp_client_from_workspace,
    init_mcp_tools_from_workspace,
    load_mcp_server_configs,
)

__all__ = [
    "MCPClient",
    "MCPServerConfig",
    "MCPTool",
    "MCPToolAdapter",
    "load_mcp_server_configs",
    "init_mcp_client_from_workspace",
    "init_mcp_tools_from_workspace",
    "discover_mcp_tools",
]
