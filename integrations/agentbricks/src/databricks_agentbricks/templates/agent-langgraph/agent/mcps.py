"""MCP servers to offer the agent — this is where you configure them.

Empty by default: the agent runs with only the MCP servers declared in ``agent.toml``. Add servers
to ``build_mcp_servers`` to offer more; ``agent.py`` passes them to ``mcp_tools``, which joins them
with the ``agent.toml`` servers and fetches their tools each request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks_langchain import DatabricksMCPServer


def build_mcp_servers() -> list[DatabricksMCPServer]:
    """Return the MCP servers to offer the agent. Empty by default — add your own.

    Example (a Databricks-managed MCP, authed as the app service principal)::

        from databricks.sdk import WorkspaceClient
        from databricks_langchain import DatabricksMCPServer

        host = WorkspaceClient().config.host
        return [
            DatabricksMCPServer(
                name="system-ai",
                url=f"{host}/api/2.0/mcp/functions/system/ai",
                workspace_client=WorkspaceClient(),
            ),
        ]
    """
    return []
