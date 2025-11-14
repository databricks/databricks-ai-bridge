from contextlib import AbstractAsyncContextManager
from typing import Any

import mlflow
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksOAuthClientProvider
from mcp.client.streamable_http import GetSessionIdCallback, streamablehttp_client
from mcp.shared.message import SessionMessage
from mcp.types import CallToolResult
from mlflow.entities import SpanType


class McpServer(MCPServerStreamableHttp):
    """Databricks MCP server implementation that extends MCPServerStreamableHttp.

    This class provides convenient access to MCP servers in the Databricks ecosystem.
    It automatically handles Databricks authentication and integrates with MLflow tracing.
    """

    def __init__(
        self,
        url: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        # Parameters for MCPServerStreamableHttp that can be optionally configured by the users
        params: MCPServerStreamableHttpParams | None = None,
        **mcpserver_kwargs: object,
    ):
        """Create a new Databricks MCP server.

        Args:
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            Databricks-Specific Parameters:
            ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

            url: Direct URL to the MCP server

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

            **mcpserver_kwargs: Additional keyword arguments to pass to the parent
                MCPServerStreamableHttp class. Supports:
                - cache_tools_list (bool): Cache tools list to avoid repeated fetches. Defaults to False.
                - name (str): Readable name for the server. Auto-generated from URL if not provided.
                - client_session_timeout_seconds (float): Read timeout for MCP ClientSession. Defaults to 5.
                - tool_filter (ToolFilter): Static filter (dict) or callable for filtering tools.
                - use_structured_content (bool): Use tool_result.structured_content. Defaults to False.
                - max_retry_attempts (int): Retry attempts for failed calls. Defaults to 0.
                - retry_backoff_seconds_base (float): Base delay for exponential backoff. Defaults to 1.0.
                - message_handler (MessageHandlerFnT): Handler for session messages.

        Example:
            Using MCP servers with an OpenAI Agent:

            .. code-block:: python

                from agents import Agent, Runner
                from databricks_openai.agents import McpServer
                from agents.mcp import MCPServerStreamableHttpParams

                async with (
                    McpServer(
                        url="https://<workspace-url>/api/2.0/mcp/functions/system/ai",
                        name="system-ai",
                        params=MCPServerStreamableHttpParams(timeout=20.0),
                    ) as mcp_server,
                ):
                    agent = Agent(
                        name="my-agent",
                        instructions="You are a helpful assistant",
                        model="databricks-meta-llama-3-1-70b-instruct",
                        mcp_servers=[mcp_server],
                    )
                    result = await Runner.run(agent, user_messages)
                    return result
        """
        # Configure Workspace Client
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        self.workspace_client = workspace_client

        if params is None:
            params = MCPServerStreamableHttpParams()

        if url is not None and params.get("url") is not None and url != params.get("url"):
            raise ValueError(
                "Different URLs provided in url and the MCPServerStreamableHttpParams. Please provide only one of them."
            )

        # Configure URL in Params
        if url is not None:
            params["url"] = url

        super().__init__(params=params, **mcpserver_kwargs)

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
