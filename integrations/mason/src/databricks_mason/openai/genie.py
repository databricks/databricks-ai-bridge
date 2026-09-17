"""Native Genie Agent tools for OpenAI Agents, configured through ``agent.toml``."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from agents import FunctionTool, function_tool

from databricks_mason.runtime.auth import AuthError
from databricks_mason.runtime.genie import GenieAgent
from databricks_mason.runtime.tool_manifest import load_tools

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


def genie_tools(
    *, workspace_client_for: Callable[[str], WorkspaceClient] | None = None
) -> list[FunctionTool]:
    """Build ask, poll and query-result tools for each configured Genie Agent.

    Space IDs are bound from the manifest, not exposed as model arguments. Construction performs
    no authentication or network requests; an agent with no Genie Agent bindings gets no tools.
    """
    tools = []
    for record in load_tools(expected_framework="openai"):
        if record.kind != "genie_agent":
            continue
        if (
            record.auth == "user"
            and workspace_client_for is None
            and os.getenv("DATABRICKS_APP_NAME")
        ):
            raise AuthError(
                "MCP_USER_AUTHORIZATION_MISSING",
                "This deployed Genie integration requires request-user authorization.",
                integration_id=record.id,
            )
        agent = GenieAgent(
            record.space_id or "",
            auth=record.auth or "app",
            workspace_client_for=workspace_client_for,
        )
        tools.extend(
            function_tool(method, name_override=f"{record.id}_{method.__name__}")
            for method in (agent.ask, agent.poll, agent.query_result)
        )
    return tools
