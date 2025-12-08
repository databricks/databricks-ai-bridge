from typing import Any, Callable, List, Union

from databricks.sdk import WorkspaceClient
from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, ConfigDict, Field


class MCPServer(BaseModel):
    """
    Base configuration for an MCP server connection using streamable HTTP transport.

    Accepts any additional keyword arguments which are automatically passed through
    to LangChain's Connection type, making this forward-compatible with future updates.

    Common optional parameters:
        - headers: dict[str, str] - Custom HTTP headers
        - timeout: float - Request timeout in seconds
        - sse_read_timeout: float - SSE read timeout in seconds
        - auth: httpx.Auth - Authentication handler
        - httpx_client_factory: Callable - Custom httpx client factory
        - terminate_on_close: bool - Terminate connection on close
        - session_kwargs: dict - Additional session kwargs
        - handle_tool_error: bool | str | Callable - Error handling strategy

    Example:
        ```python
        from databricks_langchain import DatabricksMultiServerMCPClient, MCPServer

        # Generic server with custom params - flat API for easy configuration
        server = MCPServer(
            name="other-server",
            url="https://other-server.com/mcp",
            headers={"X-API-Key": "secret"},
            timeout=15.0,
            handle_tool_error="An error occurred. Please try again.",
        )

        client = DatabricksMultiServerMCPClient([server])
        tools = await client.get_tools()
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(..., exclude=True, description="Name to identify this server connection")
    url: str
    handle_tool_error: Union[bool, str, Callable[[Exception], str], None] = Field(
        default=None,
        exclude=True,
        description=(
            "If True, return the error message as the output. If False, raise the error. "
            "If a string, return the string as the error message. "
            "If a callable, return the result of the callable as the error message."
        ),
    )

    def to_connection_dict(self) -> dict[str, Any]:
        """
        Convert to connection dictionary for LangChain MultiServerMCPClient.

        Automatically includes all extra fields passed to the constructor,
        allowing forward compatibility with new LangChain connection fields.
        """
        # Get all model fields including extra fields (name is auto-excluded)
        data = self.model_dump()

        # Add transport type (hardcoded to streamable_http)
        data["transport"] = "streamable_http"

        return data


class DatabricksMCPServer(MCPServer):
    """
    MCP server configuration with Databricks authentication.

    Automatically sets up OAuth authentication using the provided WorkspaceClient.
    Also accepts any additional connection parameters as keyword arguments.

    Example:
        ```python
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import DatabricksMultiServerMCPClient, DatabricksMCPServer

        # Databricks server with automatic OAuth - just pass params as kwargs!
        server = DatabricksMCPServer(
            name="databricks-prod",
            url="https://your-workspace.databricks.com/mcp",
            workspace_client=WorkspaceClient(),
            timeout=30.0,
            sse_read_timeout=60.0,
            handle_tool_error=True,  # Return errors as strings instead of raising
        )

        client = DatabricksMultiServerMCPClient([server])
        tools = await client.get_tools()
        ```
    """

    workspace_client: WorkspaceClient | None = Field(
        default=None,
        description="Databricks WorkspaceClient for authentication. If None, will be auto-initialized.",
        exclude=True,
    )

    @classmethod
    def from_uc_function(
        cls,
        catalog: str,
        schema: str,
        function_name: str = None,
        name: str = None,
        workspace_client: WorkspaceClient = None,
        **kwargs,
    ):
        """Alternative constructor that builds URL from Unity Catalog function path.

        Args:
            catalog: The catalog name (e.g., "system", "main").
            schema: The schema name (e.g., "ai", "default").
            function_name: Optional UC function name to include in the URL path.
            name: Optional server name used as prefix for tool names. If not provided,
                defaults to function_name if specified, otherwise schema.
            workspace_client: Databricks WorkspaceClient for authentication.
            **kwargs: Additional connection parameters (timeout, headers, etc.)

        Returns:
            DatabricksMCPServer instance with Databricks auth configured.

        Example:
            .. code-block:: python

                # Schema-level - server named "my_tools"
                server = DatabricksMCPServer.from_uc_function(
                    catalog="system", schema="ai", name="my_tools", timeout=30.0
                )

                # Specific function - server inherits function name
                server = DatabricksMCPServer.from_uc_function(
                    catalog="main", schema="default", function_name="duplicate_id"
                )

                # Specific function with custom server name
                server = DatabricksMCPServer.from_uc_function(
                    catalog="main",
                    schema="default",
                    function_name="duplicate_id",
                    name="my_duplicate_checker",
                )
        """
        ws_client = workspace_client or WorkspaceClient()
        base_url = ws_client.config.host

        if function_name:
            url = f"{base_url}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}"
            server_name = name or function_name
        else:
            url = f"{base_url}/api/2.0/mcp/functions/{catalog}/{schema}"
            server_name = name or schema

        return cls(name=server_name, url=url, workspace_client=ws_client, **kwargs)

    @classmethod
    def from_vector_search(
        cls,
        catalog: str,
        schema: str,
        index_name: str = None,
        name: str = None,
        workspace_client: WorkspaceClient = None,
        **kwargs,
    ):
        """Alternative constructor that builds URL from Unity Catalog vector search index path.

        Args:
            catalog: The catalog name (e.g., "main").
            schema: The schema name (e.g., "default").
            index_name: Optional vector search index name to include in the URL path.
            name: Optional server name used as prefix for tool names. If not provided,
                defaults to index_name if specified, otherwise schema.
            workspace_client: Databricks WorkspaceClient for authentication.
            **kwargs: Additional connection parameters (timeout, headers, etc.)

        Returns:
            DatabricksMCPServer instance with Databricks auth configured.

        Example:
            .. code-block:: python

                # Schema-level with custom server name
                server = DatabricksMCPServer.from_vector_search(
                    catalog="main", schema="default", name="my_search", timeout=30.0
                )

                # Specific index - server inherits index name
                server = DatabricksMCPServer.from_vector_search(
                    catalog="main", schema="default", index_name="en_wiki_index"
                )

                # Specific index with custom server name
                server = DatabricksMCPServer.from_vector_search(
                    catalog="main", schema="default", index_name="en_wiki_index", name="wikipedia"
                )
        """
        ws_client = workspace_client or WorkspaceClient()
        base_url = ws_client.config.host

        if index_name:
            url = f"{base_url}/api/2.0/mcp/vector-search/{catalog}/{schema}/{index_name}"
            server_name = name or index_name
        else:
            url = f"{base_url}/api/2.0/mcp/vector-search/{catalog}/{schema}"
            server_name = name or schema

        return cls(name=server_name, url=url, workspace_client=ws_client, **kwargs)

    def model_post_init(self, context: Any) -> None:
        """Initialize DatabricksServer with auth setup."""
        super().model_post_init(context)

        # Set up Databricks OAuth authentication after initialization
        if self.workspace_client is None:
            self.workspace_client = WorkspaceClient()

        # Store the auth provider internally
        self._auth_provider = DatabricksOAuthClientProvider(self.workspace_client)

    def to_connection_dict(self) -> dict[str, Any]:
        """
        Convert to connection dictionary, including Databricks auth.
        """
        # Get base connection dict
        data = super().to_connection_dict()

        # Add Databricks auth provider
        data["auth"] = self._auth_provider

        return data


class DatabricksMultiServerMCPClient(MultiServerMCPClient):
    """
    MultiServerMCPClient with simplified configuration for Databricks servers.

    This wrapper provides an ergonomic interface similar to LangChain's API while
    remaining forward-compatible with future connection parameters.

    Example:
        ```python
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import (
            DatabricksMultiServerMCPClient,
            DatabricksMCPServer,
            MCPServer,
        )

        client = DatabricksMultiServerMCPClient(
            [
                # Databricks server with automatic OAuth - just pass params as kwargs!
                DatabricksMCPServer(
                    name="databricks-prod",
                    url="https://your-workspace.databricks.com/mcp",
                    workspace_client=WorkspaceClient(),
                    timeout=30.0,
                    sse_read_timeout=60.0,
                    handle_tool_error=True,  # Return errors as strings instead of raising
                ),
                # Generic server with custom params - same flat API
                MCPServer(
                    name="other-server",
                    url="https://other-server.com/mcp",
                    headers={"X-API-Key": "secret"},
                    timeout=15.0,
                    handle_tool_error="An error occurred. Please try again.",
                ),
            ]
        )

        tools = await client.get_tools()
        ```
    """

    def __init__(self, servers: List[MCPServer], **kwargs):
        """
        Initialize the client with a list of server configurations.

        Args:
            servers: List of MCPServer or DatabricksMCPServer configurations
            **kwargs: Additional arguments to pass to MultiServerMCPClient
        """
        # Store server configs for later use (e.g., handle_tool_errors)
        self._server_configs = {server.name: server for server in servers}

        # Create connections dict (excluding tool-level params like handle_tool_errors)
        connections = {server.name: server.to_connection_dict() for server in servers}
        super().__init__(connections=connections, **kwargs)

    async def get_tools(self, server_name: str | None = None):
        """
        Get tools from MCP servers, applying handle_tool_error configuration.

        Args:
            server_name: Optional server name to get tools from. If None, gets tools from all servers.

        Returns:
            List of LangChain tools with handle_tool_error configurations applied.
        """
        import asyncio

        # Determine which servers to load from
        server_names = [server_name] if server_name is not None else list(self.connections.keys())

        # Load tools from servers in parallel
        load_tool_tasks = [
            asyncio.create_task(
                super(DatabricksMultiServerMCPClient, self).get_tools(server_name=name)
            )
            for name in server_names
        ]
        tools_list = await asyncio.gather(*load_tool_tasks)

        # Apply handle_tool_error configurations and collect tools
        all_tools = []
        for name, tools in zip(server_names, tools_list, strict=True):
            if name in self._server_configs:
                server_config = self._server_configs[name]
                if server_config.handle_tool_error is not None:
                    for tool in tools:
                        tool.handle_tool_error = server_config.handle_tool_error
            all_tools.extend(tools)

        return all_tools
