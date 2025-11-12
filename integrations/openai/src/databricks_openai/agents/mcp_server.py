from contextlib import AbstractAsyncContextManager
from typing import Any

import mlflow
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams, ToolFilter
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import OAuthCredentialsProvider
from databricks_mcp import DatabricksOAuthClientProvider
from mcp.client.session import MessageHandlerFnT
from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
from mcp.shared.message import SessionMessage
from mcp.types import CallToolResult
from mlflow.entities import SpanType


class McpServer(MCPServerStreamableHttp):
    """Databricks MCP server implementation that extends MCPServerStreamableHttp.

    This class provides convenient access to MCP servers in the Databricks ecosystem,
    supporting connections via direct URL, Databricks Connections, or Databricks Apps.
    It automatically handles Databricks authentication and integrates with MLflow tracing.
    """

    def __init__(
        self,
        url: str = None,
        connection_name: str = None,
        app_name: str = None,
        workspace_client: WorkspaceClient = None,
        # Parameters for MCPServerStreamableHttp that can be optionally configured by the users
        params: MCPServerStreamableHttpParams = None,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
    ):
        """Create a new Databricks MCP server.

        Args:
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Databricks-Specific Parameters:
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            url: Direct URL to the MCP server. Exactly one of `url`, `connection_name`,
                or `app_name` must be provided.

            connection_name: Name of a Databricks Connection to use. The URL will be
                automatically constructed as `{workspace_host}/api/2.0/mcp/external/{connection_name}`.
                Exactly one of `url`, `connection_name`, or `app_name` must be provided.

            app_name: Name of a Databricks App to connect to. The URL will be automatically
                constructed from the app's deployed URL. Requires OAuth authentication.
                Exactly one of `url`, `connection_name`, or `app_name` must be provided.

            workspace_client: Databricks WorkspaceClient to use for authentication and API calls.
                Pass a custom WorkspaceClient to set up your own authentication method. If not
                provided, a default WorkspaceClient will be created using standard Databricks
                authentication resolution.

            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Parameters Inherited from MCPServerStreamableHttp:
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            params: Additional parameters to configure the underlying MCPServerStreamableHttp.
                This can include custom headers, timeouts, and httpx_client_factory. See
                MCPServerStreamableHttpParams for available options. If not provided, default
                parameters will be used.

            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time). Defaults to False.

            name: A readable name for the server. If not provided, a name will be automatically
                generated based on the connection type (url, connection_name, or app_name).

            client_session_timeout_seconds: The read timeout passed to the MCP ClientSession.
                Defaults to 5 seconds.

            tool_filter: The tool filter to use for filtering tools. Can be a static filter
                (dict with `allowed_tool_names` and/or `blocked_tool_names`) or a callable
                for dynamic filtering.

            use_structured_content: Whether to use `tool_result.structured_content` when calling
                an MCP tool. Defaults to False for backwards compatibility - most MCP servers
                still include the structured content in the `tool_result.content`, and using it
                by default will cause duplicate content. You can set this to True if you know the
                server will not duplicate the structured content in the `tool_result.content`.

            max_retry_attempts: Number of times to retry failed list_tools/call_tool calls.
                Defaults to 0 (no retries).

            retry_backoff_seconds_base: The base delay, in seconds, used for exponential
                backoff between retries. Defaults to 1.0.

            message_handler: Optional handler invoked for session messages as delivered by the
                ClientSession.

        Raises:
            ValueError: If not exactly one of `url`, `connection_name`, or `app_name` is provided,
                or if the app_name is invalid or requires OAuth authentication that is not available.

        Example:
            >>> # Connect using a Databricks Connection
            >>> server = McpServer(connection_name="my-mcp-connection")
            >>> # Connect to a Databricks App
            >>> server = McpServer(app_name="my-mcp-app")
            >>> # Connect using a direct URL with custom params
            >>> params = MCPServerStreamableHttpParams(
            ...     url="https://example.com/mcp", headers={"X-Custom-Header": "value"}, timeout=10
            ... )
            >>> server = McpServer(params=params, cache_tools_list=True)
        """
        # Ensure exactly one of url, connection_name, or app_name is provided
        provided_params = [param for param in [url, connection_name, app_name] if param is not None]
        if len(provided_params) == 0:
            raise ValueError(
                "Exactly one of 'url', 'connection_name', or 'app_name' must be provided"
            )
        if len(provided_params) > 1:
            raise ValueError(
                "Only one of 'url', 'connection_name', or 'app_name' can be provided at a time"
            )
        # Configure Workspace Client
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        self.workspace_client = workspace_client

        if params is None:
            params = MCPServerStreamableHttpParams()

        # Configure URL in Params
        if url is not None:
            params["url"] = url
        elif connection_name is not None:
            current_host = workspace_client.config.host
            params["url"] = f"{current_host}/api/2.0/mcp/external/{connection_name}"
        elif app_name is not None:
            # Check if the current Workspace Client has an OAuth Token
            if not isinstance(workspace_client.config._header_factory, OAuthCredentialsProvider):
                raise ValueError(
                    f"Error setting up MCP Server for Databricks App: {app_name}. Querying MCP Servers on Databricks Apps requires an OAuth Token. Please ensure the workspace client is configured with an OAuth Token. Refer to documentation at https://docs.databricks.com/aws/en/dev-tools/databricks-apps/connect-local?language=Python for more information"
                )
            try:
                app = workspace_client.apps.get(app_name)
            except Exception as e:
                raise ValueError(f"App {app_name} not found") from e

            if app.url is None:
                raise ValueError(
                    f"App {app_name} does not have a valid URL. Please ensure the app is deployed and is running."
                )
            params["url"] = f"{app.url}/mcp"

        super().__init__(
            params=params,
            cache_tools_list=cache_tools_list,
            name=name,
            client_session_timeout_seconds=client_session_timeout_seconds,
            tool_filter=tool_filter,
            use_structured_content=use_structured_content,
            max_retry_attempts=max_retry_attempts,
            retry_backoff_seconds_base=retry_backoff_seconds_base,
            message_handler=message_handler,
        )

    @mlflow.trace(span_type=SpanType.TOOL)
    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        return await super().call_tool(tool_name, arguments)

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        kwargs = {
            "url": self.params["url"],
            "headers": self.params.get("headers", None),
            "auth": DatabricksOAuthClientProvider(self.workspace_client),
            "timeout": self.params.get("timeout", 5),
            "sse_read_timeout": self.params.get("sse_read_timeout", 60 * 5),
            "terminate_on_close": self.params.get("terminate_on_close", True),
        }

        if "httpx_client_factory" in self.params:
            kwargs["httpx_client_factory"] = self.params["httpx_client_factory"]

        return streamablehttp_client(**kwargs)
