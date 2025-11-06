from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.credentials_provider import OAuthCredentialsProvider
from databricks.sdk.service.apps import App


@pytest.fixture
def mock_workspace_client():
    """Create a mock WorkspaceClient for testing."""
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config._header_factory = MagicMock()
    return mock_client


@pytest.fixture
def mock_oauth_workspace_client():
    """Create a mock WorkspaceClient with OAuth credentials for testing."""
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config._header_factory = MagicMock(spec=OAuthCredentialsProvider)
    return mock_client


class TestMcpServerInit:
    """Tests for McpServer initialization."""

    def test_no_parameters_raises_error(self):
        """Test that providing no parameters raises ValueError."""
        with patch("databricks.sdk.WorkspaceClient"):
            from databricks_openai.agents import McpServer

            with pytest.raises(
                ValueError,
                match="Exactly one of 'url', 'connection_name', or 'app_name' must be provided",
            ):
                McpServer()

    def test_multiple_parameters_raises_error(self):
        """Test that providing multiple parameters raises ValueError."""
        with patch("databricks.sdk.WorkspaceClient"):
            from databricks_openai.agents import McpServer

            with pytest.raises(
                ValueError,
                match="Only one of 'url', 'connection_name', or 'app_name' can be provided at a time",
            ):
                McpServer(url="https://test.com/mcp", connection_name="test-conn")

    def test_init_with_connection_name(self, mock_workspace_client):
        """Test initialization with connection_name parameter."""
        mock_workspace_client.config.host = "https://test.databricks.com"

        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            server = McpServer(connection_name="test-connection")
            assert server.workspace_client == mock_workspace_client
            expected_url = "https://test.databricks.com/api/2.0/mcp/external/test-connection"
            assert server.params["url"] == expected_url

    def test_init_with_app_name_success(self, mock_oauth_workspace_client):
        """Test successful initialization with app_name parameter."""
        # Mock the app.get() call
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

    def test_init_with_app_name_no_oauth_raises_error(self, mock_workspace_client):
        """Test that app_name without OAuth credentials raises ValueError."""
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            with pytest.raises(
                ValueError,
                match="Error settuping MCP Server for Databricks App.*requires an OAuth Token",
            ):
                McpServer(app_name="test-app")

    def test_init_with_app_name_not_found(self, mock_oauth_workspace_client):
        """Test that app_name with non-existent app raises ValueError."""
        mock_oauth_workspace_client.apps.get.side_effect = Exception("App not found")

        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_oauth_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            with pytest.raises(ValueError, match="App test-app not found"):
                McpServer(app_name="test-app")

    def test_init_with_app_name_no_url(self, mock_oauth_workspace_client):
        """Test that app_name with app without URL raises ValueError."""
        mock_app = App(name="test-app", url=None)
        mock_oauth_workspace_client.apps.get.return_value = mock_app

        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_oauth_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            with pytest.raises(
                ValueError, match="App test-app does not have a valid URL.*deployed and is running"
            ):
                McpServer(app_name="test-app")

    def test_init_with_custom_workspace_client(self):
        """Test initialization with custom workspace_client parameter."""
        custom_client = MagicMock(spec=WorkspaceClient)
        custom_client.config.host = "https://custom.databricks.com"

        from databricks_openai.agents.mcp_server import McpServer

        server = McpServer(url="https://test.com/mcp", workspace_client=custom_client)
        assert server.workspace_client == custom_client

    def test_init_with_custom_params(self, mock_workspace_client):
        """Test initialization with custom MCPServerStreamableHttpParams."""
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            custom_params = {
                "headers": {"Custom-Header": "value"},
                "timeout": 10,
            }
            server = McpServer(url="https://test.com/mcp", params=custom_params)
            assert server.params["url"] == "https://test.com/mcp"
            assert server.params["headers"] == {"Custom-Header": "value"}
            assert server.params["timeout"] == 10

    def test_init_with_optional_parameters(self, mock_workspace_client):
        """Test initialization with optional parameters."""
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
    """Tests for McpServer.create_streams method."""

    def test_create_streams_without_httpx_client_factory(self, mock_workspace_client):
        """Test create_streams without httpx_client_factory in params."""
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.agents.mcp_server.streamablehttp_client"
            ) as mock_streamable:
                from databricks_openai.agents.mcp_server import McpServer

                server = McpServer(url="https://test.com/mcp")

                # Call create_streams
                result = server.create_streams()

                # Verify streamablehttp_client was called with correct parameters
                mock_streamable.assert_called_once()
                call_kwargs = mock_streamable.call_args.kwargs
                assert call_kwargs["url"] == "https://test.com/mcp"
                assert call_kwargs["timeout"] == 5
                assert call_kwargs["sse_read_timeout"] == 60 * 5
                assert call_kwargs["terminate_on_close"] is True
                assert "httpx_client_factory" not in call_kwargs

    def test_create_streams_with_custom_headers_and_timeouts(self, mock_workspace_client):
        """Test create_streams with custom headers and timeout values."""
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "databricks_openai.agents.mcp_server.streamablehttp_client"
            ) as mock_streamable:
                from databricks_openai.agents.mcp_server import McpServer

                params = {
                    "headers": {"Custom-Header": "test-value"},
                    "timeout": 10,
                    "sse_read_timeout": 120,
                    "terminate_on_close": False,
                }
                server = McpServer(url="https://test.com/mcp", params=params)

                # Call create_streams
                result = server.create_streams()

                # Verify streamablehttp_client was called with custom parameters
                mock_streamable.assert_called_once()
                call_kwargs = mock_streamable.call_args.kwargs
                assert call_kwargs["headers"] == {"Custom-Header": "test-value"}
                assert call_kwargs["timeout"] == 10
                assert call_kwargs["sse_read_timeout"] == 120
                assert call_kwargs["terminate_on_close"] is False
