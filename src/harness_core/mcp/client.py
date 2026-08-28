"""MCP client — discovers and invokes tools from MCP servers.

Supports stdio-based MCP servers. Architecture prepared for
HTTP/SSE/streamable HTTP where supported.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""

    name: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    enabled: bool = True
    timeout_seconds: float = 30.0
    transport: str = "stdio"  # stdio, http, sse

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "args": self.args,
            "enabled": self.enabled,
            "transport": self.transport,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class MCPTool:
    """A tool discovered from an MCP server."""

    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server": self.server_name,
        }


class MCPClient:
    """Client for interacting with MCP servers.

    Manages server lifecycle:
    - Start MCP server process
    - Initialize connection
    - Discover tools
    - Invoke tools
    - Shutdown cleanly

    MCP protocol uses JSON-RPC over stdio.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerConfig] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._tools: dict[str, list[MCPTool]] = {}
        self._initialized: dict[str, bool] = {}

    def add_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration."""
        self._servers[config.name] = config

    def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration."""
        if name in self._servers:
            self.shutdown_server(name)
            del self._servers[name]
            self._tools.pop(name, None)
            self._initialized.pop(name, None)
            return True
        return False

    def list_servers(self) -> list[MCPServerConfig]:
        """List all configured MCP servers."""
        return list(self._servers.values())

    def get_server(self, name: str) -> MCPServerConfig | None:
        """Get server config by name."""
        return self._servers.get(name)

    async def start_server(self, name: str) -> bool:
        """Start an MCP server process and initialize connection."""
        config = self._servers.get(name)
        if config is None or not config.enabled:
            return False

        if name in self._processes and self._processes[name].poll() is None:
            return True  # Already running

        try:
            env = {**dict(__import__("os").environ), **config.env}
            process = subprocess.Popen(
                [config.command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=config.cwd or None,
            )
            self._processes[name] = process

            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "harness",
                        "version": "0.1.0",
                    },
                },
            }

            response = await self._send_request(name, init_request)
            if response and "result" in response:
                self._initialized[name] = True

                # Send initialized notification
                await self._send_notification(name, {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                })

                # Discover tools
                await self._discover_tools(name)
                return True

            return False

        except Exception:
            return False

    async def _discover_tools(self, server_name: str) -> list[MCPTool]:
        """Discover tools from an MCP server."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = await self._send_request(server_name, request)
        tools = []

        if response and "result" in response:
            for tool_data in response["result"].get("tools", []):
                tool = MCPTool(
                    name=tool_data.get("name", ""),
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                    server_name=server_name,
                )
                tools.append(tool)

        self._tools[server_name] = tools
        return tools

    def list_tools(self, server_name: str | None = None) -> list[MCPTool]:
        """List discovered tools, optionally filtered by server."""
        if server_name:
            return self._tools.get(server_name, [])
        all_tools = []
        for tools in self._tools.values():
            all_tools.extend(tools)
        return all_tools

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Call a tool on an MCP server."""
        if server_name not in self._initialized:
            return None

        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        response = await self._send_request(server_name, request)
        if response and "result" in response:
            return response["result"]
        return None

    async def _send_request(
        self, server_name: str, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC request to an MCP server via stdio."""
        process = self._processes.get(server_name)
        if process is None or process.poll() is not None:
            return None

        try:
            message = json.dumps(request) + "\n"
            process.stdin.write(message.encode("utf-8"))
            process.stdin.flush()

            # Read response (line-delimited JSON)
            response_line = process.stdout.readline().decode("utf-8").strip()
            if response_line:
                return json.loads(response_line)

        except Exception:
            return None

        return None

    async def _send_notification(
        self, server_name: str, notification: dict[str, Any]
    ) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        process = self._processes.get(server_name)
        if process is None or process.poll() is not None:
            return

        try:
            message = json.dumps(notification) + "\n"
            process.stdin.write(message.encode("utf-8"))
            process.stdin.flush()
        except Exception:
            pass

    def shutdown_server(self, name: str) -> bool:
        """Shutdown an MCP server process."""
        process = self._processes.get(name)
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            self._initialized.pop(name, None)
            return True
        return False

    def shutdown_all(self) -> None:
        """Shutdown all MCP server processes."""
        for name in list(self._processes.keys()):
            self.shutdown_server(name)

    def get_status(self) -> dict[str, Any]:
        """Get status of all MCP servers."""
        status = {}
        for name, config in self._servers.items():
            process = self._processes.get(name)
            running = process is not None and process.poll() is None
            status[name] = {
                "enabled": config.enabled,
                "running": running,
                "initialized": self._initialized.get(name, False),
                "tools": len(self._tools.get(name, [])),
            }
        return status
