"""
Databricks MCP integration for Google ADK.

This module provides a toolset that connects Databricks MCP servers to Google ADK agents.
"""

from typing import Any, Optional, Union

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset, ToolPredicate


class DatabricksMcpToolset(BaseToolset):
    """
    A Google ADK toolset that connects to Databricks MCP servers.

    This toolset wraps the DatabricksMCPClient to expose Databricks MCP tools
    (UC Functions, Vector Search, Genie) to Google ADK agents.

    Supported Databricks MCP server types:
    - UC Functions: `/api/2.0/mcp/functions/<catalog>/<schema>`
    - Vector Search: `/api/2.0/mcp/vector-search/<catalog>/<schema>`
    - Genie: `/api/2.0/mcp/genie/<genie-space-id>`

    Example:
        ```python
        from databricks_google_adk import DatabricksMcpToolset
        from google.adk.agents import Agent

        # Connect to a Databricks UC Functions MCP server
        toolset = DatabricksMcpToolset(
            server_url="https://your-workspace.databricks.com/api/2.0/mcp/functions/catalog/schema"
        )

        # Use with an ADK agent
        agent = Agent(
            name="function_caller",
            model="gemini-2.0-flash",
            instruction="You help users by calling Databricks functions.",
            tools=[toolset],
        )
        ```
    """

    def __init__(
        self,
        server_url: str,
        workspace_client: Optional[WorkspaceClient] = None,
        tool_filter: Optional[Union[ToolPredicate, list[str]]] = None,
        tool_name_prefix: Optional[str] = None,
    ):
        """
        Initialize the DatabricksMcpToolset.

        Args:
            server_url: URL of the Databricks MCP server. Supported formats:
                - UC Functions: `https://<workspace>/api/2.0/mcp/functions/<catalog>/<schema>`
                - Vector Search: `https://<workspace>/api/2.0/mcp/vector-search/<catalog>/<schema>`
                - Genie: `https://<workspace>/api/2.0/mcp/genie/<genie-space-id>`
            workspace_client: Optional WorkspaceClient for authentication.
                If not provided, will be created automatically.
            tool_filter: Optional filter to select specific tools by name or predicate.
            tool_name_prefix: Optional prefix to add to all tool names.
        """
        super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)

        self._server_url = server_url
        self._workspace_client = workspace_client
        self._mcp_client = DatabricksMCPClient(
            server_url=server_url,
            workspace_client=workspace_client,
        )
        self._tools: list[BaseTool] | None = None

    @classmethod
    def for_uc_functions(
        cls,
        catalog: str,
        schema: str,
        workspace_client: Optional[WorkspaceClient] = None,
        workspace_url: Optional[str] = None,
        **kwargs,
    ) -> "DatabricksMcpToolset":
        """
        Create a toolset for Unity Catalog functions.

        Args:
            catalog: The catalog name.
            schema: The schema name.
            workspace_client: Optional WorkspaceClient for authentication.
            workspace_url: Optional workspace URL. If not provided, will be
                inferred from workspace_client.
            **kwargs: Additional arguments passed to DatabricksMcpToolset.

        Returns:
            A DatabricksMcpToolset configured for UC functions.

        Example:
            ```python
            toolset = DatabricksMcpToolset.for_uc_functions(
                catalog="my_catalog",
                schema="my_schema",
            )
            ```
        """
        client = workspace_client or WorkspaceClient()
        base_url = workspace_url or client.config.host
        server_url = f"{base_url}/api/2.0/mcp/functions/{catalog}/{schema}"
        return cls(server_url=server_url, workspace_client=client, **kwargs)

    @classmethod
    def for_vector_search(
        cls,
        catalog: str,
        schema: str,
        workspace_client: Optional[WorkspaceClient] = None,
        workspace_url: Optional[str] = None,
        **kwargs,
    ) -> "DatabricksMcpToolset":
        """
        Create a toolset for Vector Search.

        Args:
            catalog: The catalog name.
            schema: The schema name.
            workspace_client: Optional WorkspaceClient for authentication.
            workspace_url: Optional workspace URL.
            **kwargs: Additional arguments passed to DatabricksMcpToolset.

        Returns:
            A DatabricksMcpToolset configured for Vector Search.

        Example:
            ```python
            toolset = DatabricksMcpToolset.for_vector_search(
                catalog="my_catalog",
                schema="my_schema",
            )
            ```
        """
        client = workspace_client or WorkspaceClient()
        base_url = workspace_url or client.config.host
        server_url = f"{base_url}/api/2.0/mcp/vector-search/{catalog}/{schema}"
        return cls(server_url=server_url, workspace_client=client, **kwargs)

    @classmethod
    def for_genie(
        cls,
        space_id: str,
        workspace_client: Optional[WorkspaceClient] = None,
        workspace_url: Optional[str] = None,
        **kwargs,
    ) -> "DatabricksMcpToolset":
        """
        Create a toolset for Genie.

        Args:
            space_id: The Genie space ID.
            workspace_client: Optional WorkspaceClient for authentication.
            workspace_url: Optional workspace URL.
            **kwargs: Additional arguments passed to DatabricksMcpToolset.

        Returns:
            A DatabricksMcpToolset configured for Genie.

        Example:
            ```python
            toolset = DatabricksMcpToolset.for_genie(
                space_id="my-genie-space-id",
            )
            ```
        """
        client = workspace_client or WorkspaceClient()
        base_url = workspace_url or client.config.host
        server_url = f"{base_url}/api/2.0/mcp/genie/{space_id}"
        return cls(server_url=server_url, workspace_client=client, **kwargs)

    def _load_tools(self) -> list[BaseTool]:
        """Load tools from the MCP server."""
        mcp_tools = self._mcp_client.list_tools()
        adk_tools = []

        for mcp_tool in mcp_tools:
            # Create a closure to capture the tool name
            def make_tool_fn(tool_name: str, tool_desc: str, input_schema: dict):
                def tool_fn(**kwargs) -> dict[str, Any]:
                    """Execute the MCP tool."""
                    result = self._mcp_client.call_tool(tool_name, kwargs)
                    # Extract content from CallToolResult
                    if hasattr(result, "content"):
                        # MCP returns content as a list of content items
                        contents = []
                        for item in result.content:
                            if hasattr(item, "text"):
                                contents.append(item.text)
                            elif hasattr(item, "data"):
                                contents.append(item.data)
                            else:
                                contents.append(str(item))
                        return {"result": "\n".join(contents) if contents else "Success"}
                    return {"result": str(result)}

                tool_fn.__name__ = tool_name.replace(".", "__")
                tool_fn.__doc__ = tool_desc
                return tool_fn

            tool_name = mcp_tool.name
            tool_description = mcp_tool.description or f"Call {tool_name}"
            input_schema = mcp_tool.inputSchema if hasattr(mcp_tool, "inputSchema") else {}

            fn = make_tool_fn(tool_name, tool_description, input_schema)
            adk_tools.append(FunctionTool(fn))

        return adk_tools

    async def get_tools(self, readonly_context=None) -> list[BaseTool]:
        """
        Return all tools from the Databricks MCP server.

        Args:
            readonly_context: Optional context for filtering tools.

        Returns:
            List of BaseTool instances.
        """
        if self._tools is None:
            self._tools = self._load_tools()

        # Apply filtering if tool_filter is set
        if self.tool_filter is not None:
            return [
                tool
                for tool in self._tools
                if self._is_tool_selected(
                    tool.func.__name__ if hasattr(tool, "func") else str(tool)
                )
            ]
        return self._tools

    def get_databricks_resources(self) -> list:
        """
        Get Databricks resources for MLflow model logging.

        This is useful when deploying agents that use Databricks MCP tools
        to ensure proper authorization in Model Serving.

        Returns:
            List of Databricks resource objects for MLflow.
        """
        return self._mcp_client.get_databricks_resources()

    async def close(self) -> None:
        """Clean up resources."""
        # DatabricksMCPClient doesn't have a close method currently
        pass


def create_databricks_mcp_toolset(
    server_type: str,
    *,
    catalog: Optional[str] = None,
    schema: Optional[str] = None,
    space_id: Optional[str] = None,
    workspace_client: Optional[WorkspaceClient] = None,
    **kwargs,
) -> DatabricksMcpToolset:
    """
    Factory function to create a DatabricksMcpToolset.

    Args:
        server_type: Type of MCP server: "uc_functions", "vector_search", or "genie".
        catalog: Catalog name (required for uc_functions and vector_search).
        schema: Schema name (required for uc_functions and vector_search).
        space_id: Genie space ID (required for genie).
        workspace_client: Optional WorkspaceClient for authentication.
        **kwargs: Additional arguments passed to DatabricksMcpToolset.

    Returns:
        A configured DatabricksMcpToolset.

    Example:
        ```python
        # For UC Functions
        toolset = create_databricks_mcp_toolset(
            "uc_functions",
            catalog="my_catalog",
            schema="my_schema",
        )

        # For Genie
        toolset = create_databricks_mcp_toolset(
            "genie",
            space_id="my-genie-space-id",
        )
        ```
    """
    if server_type == "uc_functions":
        if not catalog or not schema:
            raise ValueError("catalog and schema are required for uc_functions")
        return DatabricksMcpToolset.for_uc_functions(
            catalog=catalog,
            schema=schema,
            workspace_client=workspace_client,
            **kwargs,
        )
    elif server_type == "vector_search":
        if not catalog or not schema:
            raise ValueError("catalog and schema are required for vector_search")
        return DatabricksMcpToolset.for_vector_search(
            catalog=catalog,
            schema=schema,
            workspace_client=workspace_client,
            **kwargs,
        )
    elif server_type == "genie":
        if not space_id:
            raise ValueError("space_id is required for genie")
        return DatabricksMcpToolset.for_genie(
            space_id=space_id,
            workspace_client=workspace_client,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown server_type: {server_type}. "
            "Must be one of: uc_functions, vector_search, genie"
        )
