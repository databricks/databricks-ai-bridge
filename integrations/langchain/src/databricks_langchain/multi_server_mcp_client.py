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
            "How to handle errors raised by tools from this server. Options:\n"
            "- None/False: Raise the error\n"
            "- True: Return error message as string\n"
            "- str: Return this string when errors occur\n"
            "- Callable: Function that takes error and returns error message string"
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

    def model_post_init(self, __context: Any) -> None:
        """Initialize DatabricksServer with auth setup."""
        super().model_post_init(__context)

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
