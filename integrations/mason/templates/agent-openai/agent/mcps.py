"""MCP servers to offer the agent — this is where you configure them.

Empty by default. Add customer-authored servers to ``build_mcp_servers``; ``agent.py`` connects
them alongside the Databricks integrations selected in ``agent/databricks_tools.py``. The Agents
SDK lists each server's tools lazily during the run.
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
