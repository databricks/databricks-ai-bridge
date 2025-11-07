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


class TestMcpServerInit:
    @pytest.mark.parametrize(
        "kwargs,expected_error",
        [
            ({}, "Exactly one of 'url', 'connection_name', or 'app_name' must be provided"),
            (
                {"url": "https://test.com/mcp", "connection_name": "test-conn"},
                "Only one of 'url', 'connection_name', or 'app_name' can be provided at a time",
            ),
        ],
    )
    def test_parameter_validation_errors(self, kwargs, expected_error):
        with patch("databricks.sdk.WorkspaceClient"):
            from databricks_openai.agents import McpServer

            with pytest.raises(ValueError, match=expected_error):
                McpServer(**kwargs)

    def test_init_with_connection_name(self, mock_workspace_client):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            server = McpServer(connection_name="test-connection")
            assert server.workspace_client == mock_workspace_client
            assert (
                server.params["url"]
                == "https://test.databricks.com/api/2.0/mcp/external/test-connection"
            )

    def test_init_with_app_name_success(self, mock_oauth_workspace_client):
        mock_app = App(name="test-app", url="https://test-app.databricks.com")
        mock_oauth_workspace_client.apps.get.return_value = mock_app

        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_oauth_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            server = McpServer(app_name="test-app")
            assert server.workspace_client == mock_oauth_workspace_client
            assert server.params["url"] == "https://test-app.databricks.com/mcp"
            mock_oauth_workspace_client.apps.get.assert_called_once_with("test-app")

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

        with patch("databricks_openai.agents.mcp_server.WorkspaceClient", return_value=client):
            from databricks_openai.agents.mcp_server import McpServer

            with pytest.raises(ValueError, match=expected_error):
                McpServer(app_name="test-app")

    def test_init_with_custom_workspace_client(self):
        custom_client = MagicMock(spec=WorkspaceClient)
        custom_client.config.host = "https://custom.databricks.com"
        from databricks_openai.agents.mcp_server import McpServer

        server = McpServer(url="https://test.com/mcp", workspace_client=custom_client)
        assert server.workspace_client == custom_client

    def test_init_with_custom_params(self, mock_workspace_client):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            custom_params = {"headers": {"Custom-Header": "value"}, "timeout": 10}
            server = McpServer(url="https://test.com/mcp", params=custom_params)
            assert server.params["url"] == "https://test.com/mcp"
            assert server.params["headers"] == {"Custom-Header": "value"}
            assert server.params["timeout"] == 10

    def test_init_with_optional_parameters(self, mock_workspace_client):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            server = McpServer(
                url="https://test.com/mcp",
                cache_tools_list=True,
                name="test-server",
                client_session_timeout_seconds=10.0,
                use_structured_content=True,
                max_retry_attempts=3,
                retry_backoff_seconds_base=2.0,
            )
            assert server.workspace_client == mock_workspace_client


class TestMcpServerCreateStreams:
    @pytest.mark.parametrize(
        "params,expected_values",
        [
            (None, {"timeout": 5, "sse_read_timeout": 300, "terminate_on_close": True}),
            (
                {
                    "headers": {"Custom-Header": "test-value"},
                    "timeout": 10,
                    "sse_read_timeout": 120,
                    "terminate_on_close": False,
                },
                {
                    "headers": {"Custom-Header": "test-value"},
                    "timeout": 10,
                    "sse_read_timeout": 120,
                    "terminate_on_close": False,
                },
            ),
        ],
    )
    def test_create_streams(self, mock_workspace_client, params, expected_values):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.agents.mcp_server.streamablehttp_client"
            ) as mock_streamable:
                from databricks_openai.agents.mcp_server import McpServer

                server = (
                    McpServer(url="https://test.com/mcp", params=params)
                    if params
                    else McpServer(url="https://test.com/mcp")
                )
                server.create_streams()

                mock_streamable.assert_called_once()
                call_kwargs = mock_streamable.call_args.kwargs
                assert call_kwargs["url"] == "https://test.com/mcp"
                for key, value in expected_values.items():
                    assert call_kwargs[key] == value
                if params is None:
                    assert "httpx_client_factory" not in call_kwargs
