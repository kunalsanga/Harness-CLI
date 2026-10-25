"""MCP tool adapter — wraps MCPTool into the canonical Harness Tool interface."""

from __future__ import annotations

import json
import logging
from typing import Any

from harness_core.agent.types import ToolResult, ToolResults
from harness_core.mcp.client import MCPClient, MCPTool
from harness_core.tools.base import Tool, ToolSchema

logger = logging.getLogger(__name__)


class MCPToolAdapter(Tool):
    """Adapts an MCPTool from an MCP server into the canonical Harness Tool interface.

    Enables standard MCP servers to be exposed to Harness agent loops,
    passing through the same lifecycle, validation, timeout, and permission
    mechanisms as native tools.
    """

    def __init__(
        self,
        client: MCPClient,
        server_name: str,
        tool: MCPTool,
        *,
        tool_name: str | None = None,
        timeout_seconds: float = 30.0,
        permission_required: str = "allow",
        aliases: list[str] | None = None,
    ) -> None:
        self.client = client
        self.server_name = server_name
        self.mcp_tool = tool
        self.raw_tool_name = tool.name
        self._name = tool_name or tool.name
        self._timeout_seconds = timeout_seconds
        self._permission_required = permission_required
        self.aliases: list[str] = aliases or []

        raw_schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
        parameters = self._sanitize_input_schema(raw_schema)
        desc = tool.description or f"MCP tool '{self.raw_tool_name}' from server '{server_name}'"

        self._schema = ToolSchema(
            name=self._name,
            description=desc,
            parameters=parameters,
            permission_required=self._permission_required,
            timeout_seconds=self._timeout_seconds,
            tags=["mcp", server_name],
        )

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    @staticmethod
    def _sanitize_input_schema(raw: dict[str, Any]) -> dict[str, Any]:
        """Normalize JSON schema for LLM function calling compatibility."""
        if not isinstance(raw, dict) or not raw:
            return {"type": "object", "properties": {}}
        schema = dict(raw)
        if "type" not in schema:
            schema["type"] = "object"
        raw_props = schema.get("properties")
        if not isinstance(raw_props, dict):
            schema["properties"] = {}
        else:
            clean_props = {}
            for k, v in raw_props.items():
                if isinstance(v, dict):
                    clean_props[k] = v
                else:
                    clean_props[k] = {"type": "string", "description": str(v)}
            schema["properties"] = clean_props

        if "required" in schema and not isinstance(schema["required"], list):
            schema["required"] = []
        return schema

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the MCP tool via the MCP client."""
        # Check server health / auto-reconnect if needed
        if self.server_name not in self.client._initialized:
            started = await self.client.start_server(self.server_name)
            if not started:
                return ToolResults.error(
                    f"MCP server '{self.server_name}' is not running or failed to initialize.",
                    retryable=False,
                )

        try:
            raw_res = await self.client.call_tool(
                self.server_name, self.raw_tool_name, arguments
            )
        except Exception as exc:
            return ToolResults.error(
                f"MCP tool '{self._name}' execution error: {self._sanitize_text(str(exc))}",
                retryable=True,
            )

        if raw_res is None:
            # Check if process terminated
            proc = self.client._processes.get(self.server_name)
            if proc and proc.poll() is not None:
                err_detail = ""
                try:
                    if proc.stderr:
                        raw_err = proc.stderr.read()
                        if raw_err:
                            err_detail = f" Stderr: {self._sanitize_text(raw_err.decode('utf-8', errors='replace')[:400])}"
                except Exception:
                    pass
                return ToolResults.error(
                    f"MCP server '{self.server_name}' terminated with exit code {proc.poll()}.{err_detail}",
                    exit_code=proc.poll(),
                    retryable=False,
                )
            return ToolResults.error(
                f"MCP tool '{self._name}' call failed or timed out.",
                retryable=True,
            )

        # Handle MCP protocol errors
        if "error" in raw_res and isinstance(raw_res["error"], dict):
            err_data = raw_res["error"]
            err_msg = self._sanitize_text(err_data.get("message") or str(err_data))
            return ToolResults.error(
                f"MCP error from '{self.server_name}': {err_msg}",
                metadata={"server": self.server_name, "tool": self.raw_tool_name},
                retryable=False,
            )

        is_error = bool(raw_res.get("isError", False))
        output = self._extract_output(raw_res)
        sanitized_output = self._sanitize_text(output)

        if is_error:
            return ToolResults.error(
                sanitized_output or f"MCP tool '{self._name}' returned an error",
                output=sanitized_output,
                metadata={"server": self.server_name, "tool": self.raw_tool_name},
                retryable=True,
            )

        return ToolResults.success(
            sanitized_output,
            metadata={"server": self.server_name, "tool": self.raw_tool_name},
        )

    def _extract_output(self, result: dict[str, Any], max_chars: int = 500_000) -> str:
        """Extract textual output from an MCP tool call result with bounded size."""
        content = result.get("content")
        raw_text = ""
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        chunks.append(str(item.get("text", "")))
                    elif "text" in item:
                        chunks.append(str(item["text"]))
                    else:
                        chunks.append(json.dumps(item))
                elif isinstance(item, str):
                    chunks.append(item)
                else:
                    chunks.append(str(item))
            raw_text = "\n".join(chunks)
        elif "text" in result:
            raw_text = str(result["text"])
        elif "output" in result:
            raw_text = str(result["output"])
        else:
            raw_text = json.dumps(result)

        if len(raw_text) > max_chars:
            notice = f"\n\n[... MCP tool output truncated to {max_chars:,} characters ...]"
            return raw_text[: max_chars - len(notice)] + notice
        return raw_text

    def _sanitize_text(self, text: str) -> str:
        """Scrub potential credentials, server arguments, or environment secrets."""
        if not text:
            return ""
        import re

        server_cfg = self.client.get_server(self.server_name)
        if server_cfg:
            # 1. Scrub server env values
            if server_cfg.env:
                for val in server_cfg.env.values():
                    if isinstance(val, str) and len(val) >= 4 and val in text:
                        text = text.replace(val, "[REDACTED]")

            # 2. Scrub sensitive arguments in command line
            if server_cfg.args:
                sensitive_flags = {
                    "--token", "-t", "--key", "-k", "--api-key", "--apikey",
                    "--secret", "--password", "-p", "--pass", "--auth", "--bearer",
                }
                for i, arg in enumerate(server_cfg.args):
                    if arg.lower() in sensitive_flags and i + 1 < len(server_cfg.args):
                        secret_val = server_cfg.args[i + 1]
                        if len(secret_val) >= 4 and secret_val in text:
                            text = text.replace(secret_val, "[REDACTED]")
                    for flag in sensitive_flags:
                        prefix = f"{flag}="
                        if arg.lower().startswith(prefix):
                            secret_val = arg[len(prefix):]
                            if len(secret_val) >= 4 and secret_val in text:
                                text = text.replace(secret_val, "[REDACTED]")

        # 3. Scrub ambient system environment secrets
        import os
        sensitive_env_keys = {"TOKEN", "KEY", "SECRET", "PASSWORD", "AUTH", "CREDENTIAL"}
        for k, val in os.environ.items():
            if any(s in k.upper() for s in sensitive_env_keys):
                if isinstance(val, str) and len(val) >= 8 and val in text:
                    text = text.replace(val, "[REDACTED]")

        # 4. Scrub standard API key & bearer token patterns
        text = re.sub(r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{16,}", r"\1[REDACTED]", text)
        text = re.sub(r"\b(?:sk|ghp|gho|pat|xoxb|xoxp|live)[_\-][a-zA-Z0-9_\-]{16,}\b", "[REDACTED]", text)
        return text
