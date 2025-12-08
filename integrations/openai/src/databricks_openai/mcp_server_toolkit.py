import asyncio
from typing import Callable, List

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel


class ToolInfo(BaseModel):
    name: str
    spec: ChatCompletionToolParam
    execute: Callable


class McpServerToolkit:
    """Toolkit for accessing MCP server tools with the OpenAI SDK.

    This class provides a simplified interface to MCP (Model Context Protocol) servers,
    automatically converting MCP tools into tool specifications for the OpenAI SDK. It's
    designed for easy integration with OpenAI clients and agents that use function calling.

    The toolkit handles authentication with Databricks, fetches available tools from the
    MCP server, and provides execution functions for each tool.

    Args:
        url: The URL of the MCP server to connect to. (Required parameter)

        name: A readable name for the MCP server. This name will be used as a prefix for
            tool names to avoid conflicts when using multiple MCP servers (e.g., "server_name__tool_name").
            If not provided, tool names will not be prefixed.

        workspace_client: Databricks WorkspaceClient to use for authentication. Pass a custom
            WorkspaceClient to set up your own authentication method. If not provided, a default
            WorkspaceClient will be created using standard Databricks authentication resolution.

    Example:
        Step 1: Create toolkit and get tools from MCP server

        .. code-block:: python

            from databricks_openai import McpServerToolkit
            from openai import OpenAI

            toolkit = McpServerToolkit(url="https://my-mcp-server.com/mcp", name="my_tools")
            tools = toolkit.get_tools()
            tool_specs = [tool.spec for tool in tools]

        Step 2: Call model with MCP tools defined

        .. code-block:: python

            client = OpenAI()
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Help me search for information about Databricks."},
            ]
            first_response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=tool_specs
            )

        Step 3: Execute function code – parse the model's response and handle tool calls

        .. code-block:: python

            import json

            tool_call = first_response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)

            # Find and execute the appropriate tool
            tool_to_execute = next(t for t in tools if t.name == tool_call.function.name)
            result = tool_to_execute.execute(**args)

        Step 4: Supply model with results – so it can incorporate them into its final response

        .. code-block:: python

            messages.append(first_response.choices[0].message)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
            second_response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=tool_specs
            )
    """

    def __init__(
        self,
        url: str,
        name: str = None,
        workspace_client: WorkspaceClient = None,
    ):
        self.workspace_client = workspace_client or WorkspaceClient()
        self.name = name
        self.url = url

        self.databricks_mcp_client = DatabricksMCPClient(self.url, self.workspace_client)

    @classmethod
    def from_uc_function(
        cls,
        catalog: str,
        schema: str,
        function_name: str = None,
        name: str = None,
        workspace_client: WorkspaceClient = None,
    ):
        """Alternative constructor that builds URL from Unity Catalog function path.

        Args:
            catalog: The catalog name (e.g., "system", "main").
            schema: The schema name (e.g., "ai", "default").
            function_name: Optional UC function name to include in the URL path.
            name: Optional toolkit name used as prefix for tool names. If not provided,
                defaults to function_name if specified, otherwise schema.
            workspace_client: Databricks WorkspaceClient for authentication.

        Returns:
            McpServerToolkit instance.

        Example:
            .. code-block:: python

                # Schema-level - toolkit named "my_tools"
                toolkit = McpServerToolkit.from_uc_function(
                    catalog="system", 
                    schema="ai",
                    name="my_tools"
                )
                
                # Specific function - toolkit inherits function name
                toolkit = McpServerToolkit.from_uc_function(
                    catalog="main", 
                    schema="default", 
                    function_name="duplicate_id"
                )
                
                # Specific function with custom toolkit name
                toolkit = McpServerToolkit.from_uc_function(
                    catalog="main", 
                    schema="default", 
                    function_name="duplicate_id",
                    name="my_duplicate_checker"
                )
        """
        ws_client = workspace_client or WorkspaceClient()
        base_url = ws_client.config.host

        if function_name:
            url = f"{base_url}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}"
            toolkit_name = name or function_name
        else:
            url = f"{base_url}/api/2.0/mcp/functions/{catalog}/{schema}"
            toolkit_name = name or schema

        return cls(url=url, name=toolkit_name, workspace_client=ws_client)

    @classmethod
    def from_vector_search(
        cls,
        catalog: str,
        schema: str,
        index_name: str = None,
        name: str = None,
        workspace_client: WorkspaceClient = None,
    ):
        """Alternative constructor that builds URL from Unity Catalog vector search index path.

        Args:
            catalog: The catalog name (e.g., "main").
            schema: The schema name (e.g., "default").
            index_name: Optional vector search index name to include in the URL path.
            name: Optional toolkit name used as prefix for tool names. If not provided,
                defaults to index_name if specified, otherwise schema.
            workspace_client: Databricks WorkspaceClient for authentication.

        Returns:
            McpServerToolkit instance.

        Example:
            .. code-block:: python

                # Schema-level with custom toolkit name
                toolkit = McpServerToolkit.from_vector_search(
                    catalog="main", 
                    schema="default",
                    name="my_search"
                )
                
                # Specific index - toolkit inherits index name
                toolkit = McpServerToolkit.from_vector_search(
                    catalog="main", 
                    schema="default", 
                    index_name="en_wiki_index"
                )
                
                # Specific index with custom toolkit name
                toolkit = McpServerToolkit.from_vector_search(
                    catalog="main", 
                    schema="default", 
                    index_name="en_wiki_index",
                    name="wikipedia"
                )
        """
        ws_client = workspace_client or WorkspaceClient()
        base_url = ws_client.config.host

        if index_name:
            url = f"{base_url}/api/2.0/mcp/vector-search/{catalog}/{schema}/{index_name}"
            toolkit_name = name or index_name
        else:
            url = f"{base_url}/api/2.0/mcp/vector-search/{catalog}/{schema}"
            toolkit_name = name or schema

        return cls(url=url, name=toolkit_name, workspace_client=ws_client)

    def get_tools(self) -> List[ToolInfo]:
        return asyncio.run(self.aget_tools())

    async def aget_tools(self) -> List[ToolInfo]:
        try:
            all_tools = await self.databricks_mcp_client._get_tools_async()
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
                    name=unique_tool_name, spec=tool_spec, execute=self._create_exec_fn(tool.name)
                )
            )
        return tool_infos

    def _create_exec_fn(self, tool_name: str) -> Callable:
        def exec_fn(**kwargs):
            response = self.databricks_mcp_client.call_tool(tool_name, kwargs)
            return "".join(c.text for c in (response.content or []))

        return exec_fn