"""Manifest-bound native Microsoft Agent Framework MCP tools."""

from __future__ import annotations

from typing import Any

from agent_framework import MCPStreamableHTTPTool

from databricks_agentkit.runtime.tool_manifest import ToolManifestError, load_tools, sandbox_meta
from databricks_agentkit.runtime.workspace import workspace_client, workspace_headers


class _ConfiguredMCPTool(MCPStreamableHTTPTool):
    def __init__(self, *args: Any, protected_meta: dict[str, Any], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._protected_meta = protected_meta

    def _prepare_call_kwargs(self, tool_name: str, kwargs: dict[str, Any]):
        arguments, meta = super()._prepare_call_kwargs(tool_name, kwargs)
        # Manifest policy overrides both model input and server-advertised metadata.
        return arguments, {**(meta or {}), **self._protected_meta}


def mcp_tools() -> list[MCPStreamableHTTPTool]:
    """Build MCP connections from agent.toml; enter each for the lifetime of the run.

    This adapter supports App authentication. Request-user tool bindings are rejected until
    native authorization-error propagation is available; no caller credentials are forwarded.
    """
    tools = []
    for record in load_tools(expected_framework="maf"):
        if record.auth == "user":
            raise ToolManifestError(
                "Microsoft Agent Framework currently supports only App-authenticated tool bindings."
            )
        if record.kind not in {"mcp", "sandbox", "uc_function", "genie_one"}:
            continue
        client = workspace_client()
        host = client.config.host.rstrip("/")
        if record.kind == "uc_function":
            url = f"{host}/api/2.0/mcp/functions/{(record.function or '').replace('.', '/')}"
        elif record.kind == "genie_one":
            url = f"{host}/api/2.0/mcp/genie"
        else:
            url = f"{host}/ai-gateway/mcp-services/{record.service}"

        def headers(_: dict[str, Any], bound_client=client) -> dict[str, str]:
            return {**bound_client.config.authenticate(), **workspace_headers()}

        tools.append(
            _ConfiguredMCPTool(
                name=record.id,
                url=url,
                header_provider=headers,
                load_prompts=False,
                request_timeout=120,
                protected_meta=sandbox_meta(record) if record.kind == "sandbox" else {},
            )
        )
    return tools
