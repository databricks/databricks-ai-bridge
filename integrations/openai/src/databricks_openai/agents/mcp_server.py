from contextlib import AbstractAsyncContextManager

from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams, ToolFilter
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import OAuthCredentialsProvider
from databricks_mcp import DatabricksOAuthClientProvider
from mcp.client.session import MessageHandlerFnT
from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
from mcp.shared.message import SessionMessage


class McpServer(MCPServerStreamableHttp):
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
                    f"Error settuping MCP Server for Databricks App: {app_name}. Querying MCP Servers on Databricks Apps requires an OAuth Token. Please ensure the workspace client is configured with an OAuth Token. Refer to documentation at https://docs.databricks.com/aws/en/dev-tools/databricks-apps/connect-local?language=Python for more information"
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

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        if "httpx_client_factory" in self.params:
            return streamablehttp_client(
                url=self.params["url"],
                headers=self.params.get("headers", None),
                auth=DatabricksOAuthClientProvider(self.workspace_client),
                timeout=self.params.get("timeout", 5),
                sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
                terminate_on_close=self.params.get("terminate_on_close", True),
                httpx_client_factory=self.params["httpx_client_factory"],
            )
        else:
            return streamablehttp_client(
                url=self.params["url"],
                headers=self.params.get("headers", None),
                auth=DatabricksOAuthClientProvider(self.workspace_client),
                timeout=self.params.get("timeout", 5),
                sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
                terminate_on_close=self.params.get("terminate_on_close", True),
            )
