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
        function_name: str | None = None,
        name: str | None = None,
        workspace_client: WorkspaceClient | None = None,
    ) -> "McpServerToolkit":
        """Create an MCP server toolkit from Unity Catalog function path.

        Convenience method to create a toolkit for UC functions by specifying Unity Catalog
        components instead of constructing the full URL manually.

        Args:
            catalog: Unity Catalog catalog name.
            schema: Schema name within the catalog.
            function_name: Optional UC function name. If omitted, provides access to all
                functions in the schema.
            name: Readable name for the MCP server. See __init__ for details.
            workspace_client: WorkspaceClient for authentication. See __init__ for details.

        Returns:
            McpServerToolkit instance for the specified Unity Catalog function.

        Example:
            .. code-block:: python

                from databricks_openai import McpServerToolkit
                from openai import OpenAI

                # Create toolkit from UC function
                toolkit = McpServerToolkit.from_uc_function(
                    catalog="main", schema="tools", function_name="send_email", name="email_tools"
                )

                # Get tools and use with OpenAI
                tools = toolkit.get_tools()
                tool_specs = [tool.spec for tool in tools]
        """
        ws_client = workspace_client or WorkspaceClient()
        base_url = ws_client.config.host

        if function_name:
            url = f"{base_url}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}"
        else:
            url = f"{base_url}/api/2.0/mcp/functions/{catalog}/{schema}"

        return cls(url=url, name=name, workspace_client=ws_client)

    @classmethod
    def from_vector_search(
        cls,
        catalog: str,
        schema: str,
        index_name: str | None = None,
        name: str | None = None,
        workspace_client: WorkspaceClient | None = None,
    ) -> "McpServerToolkit":
        """Create an MCP server toolkit from Unity Catalog vector search index path.

        Convenience method to create a toolkit for vector search by specifying Unity Catalog
        components instead of constructing the full URL manually.

        Args:
            catalog: Unity Catalog catalog name.
            schema: Schema name within the catalog.
            index_name: Optional vector search index name. If omitted, provides access to all
                indexes in the schema.
            name: Readable name for the MCP server. See __init__ for details.
            workspace_client: WorkspaceClient for authentication. See __init__ for details.

        Returns:
            McpServerToolkit instance for the specified Unity Catalog vector search index.

        Example:
            .. code-block:: python

                from databricks_openai import McpServerToolkit
                from openai import OpenAI

                # Create toolkit from vector search index
                toolkit = McpServerToolkit.from_vector_search(
                    catalog="main", schema="embeddings", index_name="product_docs", name="search_tools"
                )

                # Get tools and use with OpenAI
                tools = toolkit.get_tools()
                tool_specs = [tool.spec for tool in tools]
        """
        ws_client = workspace_client or WorkspaceClient()
        base_url = ws_client.config.host

        if index_name:
            url = f"{base_url}/api/2.0/mcp/vector-search/{catalog}/{schema}/{index_name}"
        else:
            url = f"{base_url}/api/2.0/mcp/vector-search/{catalog}/{schema}"

        return cls(url=url, name=name, workspace_client=ws_client)

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
