from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient


@pytest.fixture
def mock_workspace_client():
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config._header_factory = MagicMock()
    return mock_client


@pytest.fixture
def mock_mcp_tool():
    tool = MagicMock()
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.inputSchema = {
        "type": "object",
        "properties": {"param1": {"type": "string"}, "param2": {"type": "integer"}},
    }
    return tool


@pytest.fixture
def mock_mcp_response():
    response = MagicMock()
    content1, content2 = MagicMock(), MagicMock()
    content1.text, content2.text = "Hello ", "World"
    response.content = [content1, content2]
    return response


class TestMcpServerToolkitInit:
    @pytest.mark.parametrize(
        "init_kwargs,expected_url,expected_name",
        [
            ({"url": "https://test.com/mcp"}, "https://test.com/mcp", None),
            (
                {"url": "https://test.com/mcp", "name": "custom-name"},
                "https://test.com/mcp",
                "custom-name",
            ),
        ],
    )
    def test_init_with_url(self, mock_workspace_client, init_kwargs, expected_url, expected_name):
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client:
                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(**init_kwargs)
                assert toolkit.url == expected_url
                assert toolkit.name == expected_name
                assert toolkit.workspace_client == mock_workspace_client
                mock_mcp_client.assert_called_once_with(expected_url, mock_workspace_client)

    def test_init_with_custom_workspace_client(self, mock_workspace_client):
        with patch("databricks_openai.mcp_server_toolkit.DatabricksMCPClient"):
            from databricks_openai.mcp_server_toolkit import McpServerToolkit

            toolkit = McpServerToolkit(
                url="https://test.com/mcp", workspace_client=mock_workspace_client
            )
            assert toolkit.workspace_client == mock_workspace_client


class TestMcpServerToolkitGetTools:
    def test_get_tools_with_single_tool(self, mock_workspace_client, mock_mcp_tool):
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return [mock_mcp_tool]

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()

                assert len(tools) == 1
                assert tools[0].name == "test-server__test_tool"
                assert tools[0].spec["type"] == "function"
                assert tools[0].spec["function"]["name"] == "test-server__test_tool"
                assert tools[0].spec["function"]["description"] == "A test tool"
                assert tools[0].spec["function"]["parameters"] == mock_mcp_tool.inputSchema
                assert callable(tools[0].execute)

    def test_get_tools_with_multiple_tools(self, mock_workspace_client):
        tools_data = [("tool_one", "First tool"), ("tool_two", "Second tool")]
        mock_tools = []
        for name, desc in tools_data:
            tool = MagicMock()
            tool.name, tool.description, tool.inputSchema = (
                name,
                desc,
                {"type": "object", "properties": {}},
            )
            mock_tools.append(tool)

        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return mock_tools

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()

                assert len(tools) == 2
                assert tools[0].name == "test-server__tool_one"
                assert tools[1].name == "test-server__tool_two"

    @pytest.mark.parametrize(
        "has_name,expected_tool_name", [(True, "test-server__test_tool"), (False, "test_tool")]
    )
    def test_get_tools_name_prefix(
        self, mock_workspace_client, mock_mcp_tool, has_name, expected_tool_name
    ):
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return [mock_mcp_tool]

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(
                    url="https://test.com/mcp", name="test-server" if has_name else None
                )
                tools = toolkit.get_tools()
                assert tools[0].name == expected_tool_name

    def test_get_tools_with_no_description(self, mock_workspace_client):
        tool = MagicMock()
        tool.name, tool.description, tool.inputSchema = "test_tool", None, {"type": "object"}

        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return [tool]

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()
                assert tools[0].spec["function"]["description"] == "Tool: test_tool"

    @pytest.mark.parametrize("tools_list,expected_count", [([], 0)])
    def test_get_tools_edge_cases(self, mock_workspace_client, tools_list, expected_count):
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return tools_list

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()
                assert len(tools) == expected_count

    def test_get_tools_list_error(self, mock_workspace_client):
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_error():
                    raise Exception("Connection error")

                mock_mcp_client_instance._get_tools_async = mock_error
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                with pytest.raises(
                    ValueError,
                    match="Error listing tools from test-server MCP Server: Connection error",
                ):
                    toolkit.get_tools()


class TestMcpServerToolkitExecFn:
    @pytest.mark.parametrize(
        "call_kwargs,expected_result,expected_call_args",
        [
            ({"param1": "value1", "param2": 42}, "Hello World", {"param1": "value1", "param2": 42}),
            ({}, "Hello World", {}),
        ],
    )
    def test_exec_fn(
        self,
        mock_workspace_client,
        mock_mcp_tool,
        mock_mcp_response,
        call_kwargs,
        expected_result,
        expected_call_args,
    ):
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return [mock_mcp_tool]

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_instance.call_tool.return_value = mock_mcp_response
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()
                result = tools[0].execute(**call_kwargs)

                assert result == expected_result
                mock_mcp_client_instance.call_tool.assert_called_once_with(
                    "test_tool", expected_call_args, meta=None
                )

    def test_exec_fn_with_empty_response(self, mock_workspace_client, mock_mcp_tool):
        empty_response = MagicMock()
        empty_response.content = []

        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return [mock_mcp_tool]

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_instance.call_tool.return_value = empty_response
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp")
                tools = toolkit.get_tools()
                assert tools[0].execute() == ""

    def test_exec_fn_multiple_calls(self, mock_workspace_client, mock_mcp_tool):
        responses = []
        for text in ["Response 1", "Response 2"]:
            response, content = MagicMock(), MagicMock()
            content.text, response.content = text, [content]
            responses.append(response)

        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return [mock_mcp_tool]

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_instance.call_tool.side_effect = responses
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp")
                tools = toolkit.get_tools()

                assert tools[0].execute(param="first") == "Response 1"
                assert tools[0].execute(param="second") == "Response 2"
                assert mock_mcp_client_instance.call_tool.call_count == 2

    def test_exec_fn_with_meta_params(
        self, mock_workspace_client, mock_mcp_tool, mock_mcp_response
    ):
        """Test that _meta is extracted from kwargs and passed separately to call_tool."""
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return [mock_mcp_tool]

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_instance.call_tool.return_value = mock_mcp_response
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()

                meta_params = {
                    "num_results": 5,
                    "query_type": "HYBRID",
                    "filters": '{"status": "active"}',
                }
                result = tools[0].execute(query="search text", _meta=meta_params)

                assert result == "Hello World"
                mock_mcp_client_instance.call_tool.assert_called_once_with(
                    "test_tool",
                    {"query": "search text"},  # _meta should NOT be in regular args
                    meta=meta_params,  # _meta should be passed as separate parameter
                )

    def test_exec_fn_without_meta_params(
        self, mock_workspace_client, mock_mcp_tool, mock_mcp_response
    ):
        """Test that when _meta is not provided, meta=None is passed to call_tool."""
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_async():
                    return [mock_mcp_tool]

                mock_mcp_client_instance._get_tools_async = mock_async
                mock_mcp_client_instance.call_tool.return_value = mock_mcp_response
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp")
                tools = toolkit.get_tools()

                result = tools[0].execute(query="search text")

                assert result == "Hello World"
                mock_mcp_client_instance.call_tool.assert_called_once_with(
                    "test_tool", {"query": "search text"}, meta=None
                )


class TestToolInfo:
    def test_tool_info_creation_and_execution(self):
        from databricks_openai.mcp_server_toolkit import ToolInfo

        def dummy_fn(x: int = 0):
            return x * 2 if x else "test"

        tool_spec = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object"},
            },
        }

        tool_info = ToolInfo(name="test_tool", spec=tool_spec, execute=dummy_fn)
        assert tool_info.name == "test_tool"
        assert tool_info.spec == tool_spec
        assert tool_info.execute == dummy_fn
        assert tool_info.execute() == "test"
        assert tool_info.execute(5) == 10


class TestMcpServerToolkitAsyncGetTools:
    @pytest.mark.asyncio
    async def test_async_get_tools_with_single_tool(self, mock_workspace_client, mock_mcp_tool):
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                # Use an async mock for the async method
                async def mock_get_tools_async():
                    return [mock_mcp_tool]

                mock_mcp_client_instance._get_tools_async = mock_get_tools_async
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = await toolkit.aget_tools()

                assert len(tools) == 1
                assert tools[0].name == "test-server__test_tool"
                assert tools[0].spec["type"] == "function"
                assert tools[0].spec["function"]["name"] == "test-server__test_tool"
                assert tools[0].spec["function"]["description"] == "A test tool"
                assert tools[0].spec["function"]["parameters"] == mock_mcp_tool.inputSchema
                assert callable(tools[0].execute)

    @pytest.mark.asyncio
    async def test_async_get_tools_error(self, mock_workspace_client):
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_toolkit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()

                async def mock_error():
                    raise Exception("Connection error")

                mock_mcp_client_instance._get_tools_async = mock_error
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                with pytest.raises(
                    ValueError,
                    match="Error listing tools from test-server MCP Server: Connection error",
                ):
                    await toolkit.aget_tools()


class TestMcpServerToolkitFromUCResource:
    def test_from_uc_function(self, mock_workspace_client):
        """Test from_uc_function constructs correct URL."""
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch("databricks_openai.mcp_server_toolkit.DatabricksMCPClient"):
                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit.from_uc_function(
                    catalog="system",
                    schema="ai",
                    function_name="test_tool",
                    workspace_client=mock_workspace_client,
                )

                assert (
                    toolkit.url
                    == "https://test.databricks.com/api/2.0/mcp/functions/system/ai/test_tool"
                )
                assert toolkit.name is None
                assert toolkit.workspace_client == mock_workspace_client

    def test_from_vector_search(self, mock_workspace_client):
        """Test from_vector_search constructs correct URL."""
        with patch(
            "databricks_openai.mcp_server_toolkit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch("databricks_openai.mcp_server_toolkit.DatabricksMCPClient"):
                from databricks_openai.mcp_server_toolkit import McpServerToolkit

                toolkit = McpServerToolkit.from_vector_search(
                    catalog="system",
                    schema="ai",
                    index_name="test_index",
                    workspace_client=mock_workspace_client,
                )

                assert (
                    toolkit.url
                    == "https://test.databricks.com/api/2.0/mcp/vector-search/system/ai/test_index"
                )
                assert toolkit.name is None
                assert toolkit.workspace_client == mock_workspace_client
