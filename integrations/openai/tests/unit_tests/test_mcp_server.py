from unittest.mock import MagicMock, patch

import pytest
from agents.mcp import MCPServerStreamableHttpParams
from databricks.sdk import WorkspaceClient


@pytest.fixture
def mock_workspace_client():
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config._header_factory = MagicMock()
    return mock_client


class TestMcpServerInit:
    def test_init_with_url(self, mock_workspace_client):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            server = McpServer(url="https://test.com/mcp")
            assert server.workspace_client == mock_workspace_client
            assert server.params["url"] == "https://test.com/mcp"

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

            custom_params: MCPServerStreamableHttpParams = {
                "url": "https://test.com/mcp",
                "headers": {"Custom-Header": "value"},
                "timeout": 10,
            }
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

    @pytest.mark.parametrize(
        "url,params_dict,expected_url,expected_extra",
        [
            # URL in params dict only
            (None, {"url": "https://from-params.com/mcp"}, "https://from-params.com/mcp", {}),
            # URL param only
            ("https://test.com/mcp", None, "https://test.com/mcp", {}),
            # URL param with same URL in params
            ("https://test.com/mcp", {"url": "https://test.com/mcp"}, "https://test.com/mcp", {}),
            # URL param with params dict (no URL in dict)
            (
                "https://test.com/mcp",
                {"headers": {"Custom-Header": "value"}},
                "https://test.com/mcp",
                {"headers": {"Custom-Header": "value"}},
            ),
            # Complete params dict with URL, headers, timeout
            (
                None,
                {
                    "url": "https://test.com/mcp",
                    "headers": {"Custom-Header": "value"},
                    "timeout": 15,
                },
                "https://test.com/mcp",
                {"headers": {"Custom-Header": "value"}, "timeout": 15},
            ),
        ],
    )
    def test_init_url_and_params_combinations(
        self, mock_workspace_client, url, params_dict, expected_url, expected_extra
    ):
        """Test various combinations of url and params initialization"""
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            params: MCPServerStreamableHttpParams | None = params_dict
            server = McpServer(url=url, params=params)
            assert server.params["url"] == expected_url
            for key, value in expected_extra.items():
                assert server.params[key] == value
            assert server.workspace_client == mock_workspace_client

    def test_client_session_timeout_propagation(self, mock_workspace_client):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            with patch(
                "agents.mcp.MCPServerStreamableHttp.__init__", return_value=None
            ) as mock_super_init:
                from databricks_openai.agents.mcp_server import McpServer

                # Case 1: Timeout provided, defaults client_session_timeout_seconds
                McpServer(url="https://test.com/mcp", timeout=30.0)

                _, kwargs = mock_super_init.call_args
                assert kwargs["client_session_timeout_seconds"] == 30.0

                # Case 2: client_session_timeout_seconds explicitly provided
                McpServer(
                    url="https://test.com/mcp",
                    timeout=30.0,
                    client_session_timeout_seconds=10.0,
                )

                _, kwargs = mock_super_init.call_args
                assert kwargs["client_session_timeout_seconds"] == 10.0


class TestMcpServerCreateStreams:
    @pytest.mark.parametrize(
        "params,expected_values",
        [
            (None, {"timeout": 20, "sse_read_timeout": 300, "terminate_on_close": True}),
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


class TestMcpServerFromUCResource:
    """Tests for from_uc_function and from_vector_search class methods."""

    def test_from_uc_function(self, mock_workspace_client):
        """Test from_uc_function constructs correct URL."""
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            server = McpServer.from_uc_function(
                catalog="system",
                schema="ai",
                function_name="test_tool",
                workspace_client=mock_workspace_client,
            )

            assert (
                server.params["url"]
                == "https://test.databricks.com/api/2.0/mcp/functions/system/ai/test_tool"
            )
            assert server.workspace_client == mock_workspace_client

    def test_from_vector_search(self, mock_workspace_client):
        """Test from_vector_search constructs correct URL."""
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import McpServer

            server = McpServer.from_vector_search(
                catalog="system",
                schema="ai",
                index_name="test_index",
                workspace_client=mock_workspace_client,
            )

            assert (
                server.params["url"]
                == "https://test.databricks.com/api/2.0/mcp/vector-search/system/ai/test_index"
            )
            assert server.workspace_client == mock_workspace_client
