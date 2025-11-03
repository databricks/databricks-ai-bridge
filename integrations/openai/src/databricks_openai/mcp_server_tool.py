from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams, ToolFilter
from databricks.sdk import WorkspaceClient
from mcp.client.session import MessageHandlerFnT


class _McpServerUrlTool(MCPServerStreamableHttp):
    def __init__(
        self,
        url: str,
        headers: dict[str, str] = None,
        workspace_client: WorkspaceClient = None,
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
    ):
        params = MCPServerStreamableHttpParams(
            url=url,
            headers=headers,
            workspace_client=workspace_client,
        )
        super().__init__(params, cache_tools_list, name, client_session_timeout_seconds, tool_filter, use_structured_content, max_retry_attemps, retry_backoff_seconds_base, message_handler)


class McpServerTool(MCPServerStreamableHttp):
    def __new__(
        cls,
        url: str = None,
        connection_name: str = None,
        app_name: str = None,
        headers: dict[str, str] = None,
        workspace_client: WorkspaceClient = None,
        # Parameters for MCPServerStreamableHttp that can be optionally configured by the users
        cache_tools_list: bool = False,
        name: str | None = None,
        client_session_timeout_seconds: float | None = 5,
        tool_filter: ToolFilter = None,
        use_structured_content: bool = False,
        max_retry_attempts: int = 0,
        retry_backoff_seconds_base: float = 1.0,
        message_handler: MessageHandlerFnT | None = None,
    ):
        provided_params = [param for param in [url, connection_name, app_name] if param is not None]
        if len(provided_params) == 0:
            raise ValueError(
                "Exactly one of 'url', 'connection_name', or 'app_name' must be provided"
            )
        if len(provided_params) > 1:
            raise ValueError(
                "Only one of 'url', 'connection_name', or 'app_name' can be provided at a time"
            )

        if workspace_client is None:
            workspace_client = WorkspaceClient()

        if url is not None:
            return _McpServerUrlTool(
                url=url,
                headers=headers,
                workspace_client=workspace_client,
                cache_tools_list=cache_tools_list,
                name=name,
                client_session_timeout_seconds=client_session_timeout_seconds,
                tool_filter=tool_filter,
                use_structured_content=use_structured_content,
                max_retry_attempts=max_retry_attempts,
                retry_backoff_seconds_base=retry_backoff_seconds_base,
                message_handler=message_handler,
            )
        elif connection_name is not None:
            current_host = workspace_client.config.host
            return _McpServerUrlTool(
                url=f"https://{current_host}/api/2.0/mcp/external/{connection_name}",
                headers=headers,
                workspace_client=workspace_client,
            )
        elif app_name is not None:
            try:
                app = workspace_client.apps.get(app_name)
            except Exception as e:
                raise ValueError(f"App {app_name} not found") from e

            if app.url is None:
                raise ValueError(
                    f"App {app_name} does not have a valid URL. Please ensure the app is deployed and is running."
                )
            return _McpServerUrlTool(
                url=app.url,
                headers=headers,
                workspace_client=workspace_client,
                cache_tools_list=cache_tools_list,
                name=name,
                client_session_timeout_seconds=client_session_timeout_seconds,
                tool_filter=tool_filter,
                use_structured_content=use_structured_content,
                max_retry_attempts=max_retry_attempts,
                retry_backoff_seconds_base=retry_backoff_seconds_base,
                message_handler=message_handler,
            )
