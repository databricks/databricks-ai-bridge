"""Customer-authored MCP servers to offer alongside ``DATABRICKS_TOOLS``.

Empty by default. Add servers to ``build_mcp_servers`` when you need a custom MCP connection;
``agent.py`` passes them to the same explicit tool-loading seam used by Mason integrations.
"""

from databricks_langchain import DatabricksMCPServer


def build_mcp_servers() -> list[DatabricksMCPServer]:
    """Return the MCP servers to offer the agent. Empty by default — add your own.

    Example (a Databricks-managed MCP, authed as the app service principal)::

        from databricks.sdk import WorkspaceClient

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
