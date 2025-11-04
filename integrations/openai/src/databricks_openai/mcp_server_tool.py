from contextlib import AbstractAsyncContextManager
from typing import Any

from agents.agent import AgentBase
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams, ToolFilter
from agents.run_context import RunContextWrapper
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksOAuthClientProvider
from mcp import Tool as MCPTool
from mcp.client.session import MessageHandlerFnT
from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
from mcp.shared.message import SessionMessage
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult


class _McpServerUrlTool(MCPServerStreamableHttp):
    def __init__(
        self,
        url: str,
        authentication_headers: dict[str, str] = None,
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
        self.workspace_client = workspace_client
        self.authentication_headers = authentication_headers
        params = MCPServerStreamableHttpParams(
            url=url,
            headers=authentication_headers,
        )
        super().__init__(
            params,
            cache_tools_list,
            name,
            client_session_timeout_seconds,
            tool_filter,
            use_structured_content,
            max_retry_attempts,
            retry_backoff_seconds_base,
            message_handler,
        )
        print(self.name)

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
            GetSessionIdCallback | None,
        ]
    ]:
        if self.authentication_headers is not None:
            return streamablehttp_client(
                url=self.params["url"], headers=self.params.get("headers", None)
            )
        else:
            return streamablehttp_client(
                url=self.params["url"], auth=DatabricksOAuthClientProvider(self.workspace_client)
            )

    async def list_tools(
        self,
        run_context: RunContextWrapper[Any] | None = None,
        agent: AgentBase | None = None,
    ) -> list[MCPTool]:
        if self.session is None:
            await self.connect()

        super().list_tools(run_context, agent)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        if self.session is None:
            await self.connect()
        super().call_tool(tool_name, arguments)

    async def list_prompts(
        self,
    ) -> ListPromptsResult:
        if self.session is None:
            await self.connect()
        super().list_prompts()

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> GetPromptResult:
        if self.session is None:
            await self.connect()

        super().get_prompt(name, arguments)


class McpServerTool(MCPServerStreamableHttp):
    def __new__(
        cls,
        url: str = None,
        connection_name: str = None,
        app_name: str = None,
        authentication_headers: dict[str, str] = None,
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
                authentication_headers=authentication_headers,
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
                authentication_headers=authentication_headers,
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
                authentication_headers=authentication_headers,
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
