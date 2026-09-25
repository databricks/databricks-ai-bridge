"""MCP servers to offer the agent — this is where you configure them.

Empty by default: the agent runs with only the MCP servers declared in ``agent.toml``. Add servers
to ``build_mcp_servers`` to offer more; ``agent.py`` joins them with the ``agent.toml`` servers and
omits any that fail to connect or list tools.
"""

from databricks_openai.agents import McpServer


def build_mcp_servers() -> list[McpServer]:
    """Return the MCP servers to offer the agent. Empty by default — add your own.

    Example (a Databricks-managed MCP, authed as the app service principal)::

        from databricks.sdk import WorkspaceClient

        client = WorkspaceClient()
        host = client.config.host.rstrip("/")
        return [
            McpServer(
                url=f"{host}/api/2.0/mcp/functions/system/ai",
                name="system-ai",
                workspace_client=client,
            ),
        ]
    """
    return []
