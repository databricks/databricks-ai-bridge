"""Declare Databricks managed MCP servers (e.g. Genie One) as Managed Agents custom tools.

The worker inside the sandbox is the MCP client (see ``worker.py``), so the agent's
declaration only needs each tool's name, description and input schema. Run this where
you create agents, with your own Databricks identity. Requires ``databricks-anthropic[mcp]``.
"""

from __future__ import annotations

import json
from typing import Any

from databricks.sdk import WorkspaceClient

from databricks_anthropic._useragent import tag
from databricks_anthropic.sandbox.launcher import SandboxMcpServer

GENIE_ONE = "system.ai.genie_one_mcp"
# view_ask opens an interactive View and is only offered to MCP Apps clients.
GENIE_ONE_EXCLUDE = ("view_ask",)
# With the wait tool, the raw ask/poll/cancel tools stay worker-internal.
GENIE_ONE_WAIT_EXCLUDE = ("view_ask", "genie_ask", "genie_poll_response", "genie_cancel_response")

GENIE_WAIT_DECLARATION: dict[str, Any] = {
    "type": "custom",
    "name": "genie_ask_and_wait",
    "description": (
        "Ask Databricks Genie One a natural-language data question and wait for its final answer. "
        "Returns JSON with status, final_answer, deep_link, conversation_id and the SQL Genie ran. "
        "Pass conversation_id for a follow-up question in the same Genie conversation."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The natural-language data question."},
            "conversation_id": {
                "type": "string",
                "description": "Optional earlier Genie conversation id.",
            },
        },
        "required": ["question"],
    },
}


def genie_one_mcp_url(workspace_client: WorkspaceClient) -> str:
    """Unity Gateway URL of the Genie One managed MCP server (needs EXECUTE on it)."""
    host = workspace_client.config.host.rstrip("/")
    return f"{host}/ai-gateway/mcp-services/{GENIE_ONE}?o={workspace_client.get_workspace_id()}"


def genie_one_server(
    workspace_client: WorkspaceClient, *, wait_tool: bool = True
) -> SandboxMcpServer:
    """Genie One server config. ``wait_tool`` (default) serves ``genie_ask_and_wait`` and hides the
    raw ask/poll tools, so polling happens inside the sandbox instead of one model call per poll."""
    return SandboxMcpServer(
        name="genie-one",
        url=genie_one_mcp_url(workspace_client),
        exclude=GENIE_ONE_WAIT_EXCLUDE if wait_tool else GENIE_ONE_EXCLUDE,
        wait_tool=wait_tool,
    )


def _schema(tool: Any) -> dict:
    # mcp 2.x renamed inputSchema -> input_schema.
    return getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}


def check_custom_tool_rules(name: str, schema: dict) -> list[str]:
    """Problems that would make the Managed Agents API reject a custom tool declaration."""
    problems = []
    if not (1 <= len(name) <= 128) or not all(c.isalnum() or c in "_-" for c in name):
        problems.append("name must be 1-128 letters, digits, '_' or '-'")
    if name.startswith(("mcp__", "ant__")):
        problems.append("name uses a reserved prefix")
    if "$ref" in json.dumps(schema):
        problems.append("input_schema uses $ref (inline it)")
    problems += [
        f"top-level {k} is not allowed" for k in ("oneOf", "anyOf", "allOf") if k in schema
    ]
    return problems


async def alist_mcp_custom_tools(
    workspace_client: WorkspaceClient, server: SandboxMcpServer
) -> list[dict]:
    """List a Databricks MCP server's tools as ``{"type": "custom", ...}`` declarations."""
    from importlib.metadata import version

    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client  # mcp >= 1.24

    # The transport is built on httpx2 in mcp 2.x and on httpx in 1.x; the client must match.
    if int(version("mcp").split(".")[0]) >= 2:
        import httpx2 as http_lib
    else:
        import httpx as http_lib  # ty: ignore[unresolved-import] - mcp 1.x only

    headers = {
        **workspace_client.config.authenticate(),
        "User-Agent": tag(workspace_client).config.user_agent,
    }
    async with http_lib.AsyncClient(headers=headers, timeout=http_lib.Timeout(60.0)) as http:
        async with streamable_http_client(server.url, http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()

    declarations = []
    for tool in listed.tools:
        if tool.name in server.exclude:
            continue
        schema = _schema(tool)
        problems = check_custom_tool_rules(tool.name, schema)
        if problems:
            raise ValueError(
                f"MCP tool {tool.name!r} can't be declared as-is: {'; '.join(problems)}"
            )
        declarations.append(
            {
                "type": "custom",
                "name": tool.name,
                "description": tool.description or tool.name,
                "input_schema": schema,
            }
        )
    if server.wait_tool:
        declarations.insert(0, GENIE_WAIT_DECLARATION)
    return declarations
