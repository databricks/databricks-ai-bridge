from typing import Callable, List

from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import OAuthCredentialsProvider
from databricks.sdk.errors.platform import NotFound
from databricks_mcp import DatabricksMCPClient
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel


class ToolInfo(BaseModel):
    name: str
    spec: ChatCompletionToolParam
    exec_fn: Callable


class McpServerToolkit:
    def __init__(
        self,
        url: str = None,
        app_name: str = None,
        connection_name: str = None,
        name: str = None,
        workspace_client: WorkspaceClient = None,
    ):
        provided_params = [param for param in [url, connection_name, app_name] if param is not None]
        if len(provided_params) != 1:
            raise ValueError(
                "Exactly one of 'url', 'connection_name', or 'app_name' must be provided"
            )

        self.workspace_client = workspace_client or WorkspaceClient()
        self.name = name
        self.url = url

        if connection_name is not None:
            self.name = self.name or connection_name
            self.url = f"{self.workspace_client.config.host}/api/2.0/mcp/external/{connection_name}"
        elif app_name is not None:
            if self.name is None:
                self.name = app_name
            if not isinstance(
                self.workspace_client.config._header_factory, OAuthCredentialsProvider
            ):
                raise ValueError(
                    f"Error setting up MCP Server for Databricks App: {app_name}. Querying MCP Servers on Databricks Apps requires an OAuth Token. Please ensure the workspace client is configured with an OAuth Token. Refer to documentation at https://docs.databricks.com/aws/en/dev-tools/databricks-apps/connect-local?language=Python for more information"
                )
            try:
                app = self.workspace_client.apps.get(app_name)
            except NotFound as e:
                raise ValueError(f"App {app_name} not found") from e

            if app.url is None:
                raise ValueError(
                    f"App {app_name} does not have a valid URL. Please ensure the app is deployed and is running."
                )
            self.url = f"{app.url}/mcp"

        self.databricks_mcp_client = DatabricksMCPClient(self.url, self.workspace_client)

    def get_tools(self) -> List[ToolInfo]:
        try:
            all_tools = self.databricks_mcp_client.list_tools()
        except Exception as e:
            raise ValueError(f"Error listing tools from {self.name} MCP Server: {e}") from e

        tool_infos = []
        for tool in all_tools:
            unique_tool_name = f"{self.name}__{tool.name}" if self.name else tool.name
            tool_spec = {
                "type": "function",
                "function": {
                    "name": unique_tool_name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "parameters": tool.inputSchema,
                },
            }
            tool_infos.append(
                ToolInfo(
                    name=unique_tool_name, spec=tool_spec, exec_fn=self._create_exec_fn(tool.name)
                )
            )
        return tool_infos

    def _create_exec_fn(self, tool_name: str) -> Callable:
        def exec_fn(**kwargs):
            response = self.databricks_mcp_client.call_tool(tool_name, kwargs)
            return "".join(c.text for c in (response.content or []))

        return exec_fn
