from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import OAuthCredentialsProvider
from databricks.sdk.service.apps import App


@pytest.fixture
def mock_workspace_client():
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config._header_factory = MagicMock()
    return mock_client


@pytest.fixture
def mock_oauth_workspace_client():
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config._header_factory = MagicMock(spec=OAuthCredentialsProvider)
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
        "kwargs",
        [
            {},
            {"url": "https://test.com/mcp", "connection_name": "test-conn"},
            {"url": "https://test.com/mcp", "connection_name": "test-conn", "app_name": "test-app"},
        ],
    )
    def test_parameter_validation_errors(self, kwargs):
        with patch("databricks_openai.mcp_server_tookit.WorkspaceClient"):
            from databricks_openai.mcp_server_tookit import McpServerToolkit

            with pytest.raises(
                ValueError,
                match="Exactly one of 'url', 'connection_name', or 'app_name' must be provided",
            ):
                McpServerToolkit(**kwargs)

    @pytest.mark.parametrize(
        "init_kwargs,expected_url,expected_name",
        [
            ({"url": "https://test.com/mcp"}, "https://test.com/mcp", None),
            (
                {"url": "https://test.com/mcp", "name": "custom-name"},
                "https://test.com/mcp",
                "custom-name",
            ),
            (
                {"connection_name": "test-connection"},
                "https://test.databricks.com/api/2.0/mcp/external/test-connection",
                "test-connection",
            ),
            (
                {"connection_name": "test-connection", "name": "custom-name"},
                "https://test.databricks.com/api/2.0/mcp/external/test-connection",
                "custom-name",
            ),
        ],
    )
    def test_init_with_url_and_connection(
        self, mock_workspace_client, init_kwargs, expected_url, expected_name
    ):
        with patch(
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client:
                from databricks_openai.mcp_server_tookit import McpServerToolkit

                toolkit = McpServerToolkit(**init_kwargs)
                assert toolkit.url == expected_url
                assert toolkit.name == expected_name
                assert toolkit.workspace_client == mock_workspace_client
                if "url" in init_kwargs:
                    mock_mcp_client.assert_called_once_with(expected_url, mock_workspace_client)

    @pytest.mark.parametrize(
        "custom_name,expected_name", [(None, "test-app"), ("custom-name", "custom-name")]
    )
    def test_init_with_app_name_success(
        self, mock_oauth_workspace_client, custom_name, expected_name
    ):
        mock_app = App(name="test-app", url="https://test-app.databricks.com")
        mock_oauth_workspace_client.apps.get.return_value = mock_app

        with patch("databricks_openai.mcp_server_tookit.DatabricksMCPClient") as mock_mcp_client:
            from databricks_openai.mcp_server_tookit import McpServerToolkit

            kwargs = {"app_name": "test-app", "workspace_client": mock_oauth_workspace_client}
            if custom_name:
                kwargs["name"] = custom_name
            toolkit = McpServerToolkit(**kwargs)
            assert toolkit.url == "https://test-app.databricks.com/mcp"
            assert toolkit.name == expected_name
            mock_mcp_client.assert_called_once_with(
                "https://test-app.databricks.com/mcp", mock_oauth_workspace_client
            )

    @pytest.mark.parametrize(
        "client_fixture,setup_fn,expected_error",
        [
            (
                "mock_workspace_client",
                None,
                "Error setting up MCP Server for Databricks App.*requires an OAuth Token",
            ),
            (
                "mock_oauth_workspace_client",
                lambda c: setattr(c.apps.get, "side_effect", Exception("App not found")),
                "App test-app not found",
            ),
            (
                "mock_oauth_workspace_client",
                lambda c: setattr(
                    c.apps, "get", MagicMock(return_value=App(name="test-app", url=None))
                ),
                "App test-app does not have a valid URL.*deployed and is running",
            ),
        ],
    )
    def test_app_name_errors(self, request, client_fixture, setup_fn, expected_error):
        client = request.getfixturevalue(client_fixture)
        if setup_fn:
            setup_fn(client)

        from databricks_openai.mcp_server_tookit import McpServerToolkit

        with pytest.raises(ValueError, match=expected_error):
            McpServerToolkit(app_name="test-app", workspace_client=client)

    def test_init_with_custom_workspace_client(self, mock_workspace_client):
        with patch("databricks_openai.mcp_server_tookit.DatabricksMCPClient"):
            from databricks_openai.mcp_server_tookit import McpServerToolkit

            toolkit = McpServerToolkit(
                url="https://test.com/mcp", workspace_client=mock_workspace_client
            )
            assert toolkit.workspace_client == mock_workspace_client


class TestMcpServerToolkitGetTools:
    def test_get_tools_with_single_tool(self, mock_workspace_client, mock_mcp_tool):
        with patch(
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.return_value = [mock_mcp_tool]
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()

                assert len(tools) == 1
                assert tools[0].name == "test-server__test_tool"
                assert tools[0].spec["type"] == "function"
                assert tools[0].spec["function"]["name"] == "test-server__test_tool"
                assert tools[0].spec["function"]["description"] == "A test tool"
                assert tools[0].spec["function"]["parameters"] == mock_mcp_tool.inputSchema
                assert callable(tools[0].exec_fn)

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
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.return_value = mock_tools
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

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
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.return_value = [mock_mcp_tool]
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

                toolkit = McpServerToolkit(
                    url="https://test.com/mcp", name="test-server" if has_name else None
                )
                tools = toolkit.get_tools()
                assert tools[0].name == expected_tool_name

    def test_get_tools_with_no_description(self, mock_workspace_client):
        tool = MagicMock()
        tool.name, tool.description, tool.inputSchema = "test_tool", None, {"type": "object"}

        with patch(
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.return_value = [tool]
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()
                assert tools[0].spec["function"]["description"] == "Tool: test_tool"

    @pytest.mark.parametrize("tools_list,expected_count", [([], 0)])
    def test_get_tools_edge_cases(self, mock_workspace_client, tools_list, expected_count):
        with patch(
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.return_value = tools_list
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()
                assert len(tools) == expected_count

    def test_get_tools_list_error(self, mock_workspace_client):
        with patch(
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.side_effect = Exception("Connection error")
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

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
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.return_value = [mock_mcp_tool]
                mock_mcp_client_instance.call_tool.return_value = mock_mcp_response
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp", name="test-server")
                tools = toolkit.get_tools()
                result = tools[0].exec_fn(**call_kwargs)

                assert result == expected_result
                mock_mcp_client_instance.call_tool.assert_called_once_with(
                    "test_tool", expected_call_args
                )

    def test_exec_fn_with_empty_response(self, mock_workspace_client, mock_mcp_tool):
        empty_response = MagicMock()
        empty_response.content = []

        with patch(
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.return_value = [mock_mcp_tool]
                mock_mcp_client_instance.call_tool.return_value = empty_response
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp")
                tools = toolkit.get_tools()
                assert tools[0].exec_fn() == ""

    def test_exec_fn_multiple_calls(self, mock_workspace_client, mock_mcp_tool):
        responses = []
        for text in ["Response 1", "Response 2"]:
            response, content = MagicMock(), MagicMock()
            content.text, response.content = text, [content]
            responses.append(response)

        with patch(
            "databricks_openai.mcp_server_tookit.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.mcp_server_tookit.DatabricksMCPClient"
            ) as mock_mcp_client_class:
                mock_mcp_client_instance = MagicMock()
                mock_mcp_client_instance.list_tools.return_value = [mock_mcp_tool]
                mock_mcp_client_instance.call_tool.side_effect = responses
                mock_mcp_client_class.return_value = mock_mcp_client_instance

                from databricks_openai.mcp_server_tookit import McpServerToolkit

                toolkit = McpServerToolkit(url="https://test.com/mcp")
                tools = toolkit.get_tools()

                assert tools[0].exec_fn(param="first") == "Response 1"
                assert tools[0].exec_fn(param="second") == "Response 2"
                assert mock_mcp_client_instance.call_tool.call_count == 2


class TestToolInfo:
    def test_tool_info_creation_and_execution(self):
        from databricks_openai.mcp_server_tookit import ToolInfo

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

        tool_info = ToolInfo(name="test_tool", spec=tool_spec, exec_fn=dummy_fn)
        assert tool_info.name == "test_tool"
        assert tool_info.spec == tool_spec
        assert tool_info.exec_fn == dummy_fn
        assert tool_info.exec_fn() == "test"
        assert tool_info.exec_fn(5) == 10
