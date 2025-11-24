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


class TestDatabricksMCPServerInit:
    def test_init_with_url(self, mock_workspace_client):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import DatabricksMCPServer

            server = DatabricksMCPServer(url="https://test.com/mcp")
            assert server.workspace_client == mock_workspace_client
            assert server.params["url"] == "https://test.com/mcp"

    def test_init_with_custom_workspace_client(self):
        custom_client = MagicMock(spec=WorkspaceClient)
        custom_client.config.host = "https://custom.databricks.com"
        from databricks_openai.agents.mcp_server import DatabricksMCPServer

        server = DatabricksMCPServer(url="https://test.com/mcp", workspace_client=custom_client)
        assert server.workspace_client == custom_client

    def test_init_with_custom_params(self, mock_workspace_client):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import DatabricksMCPServer

            custom_params = {"headers": {"Custom-Header": "value"}, "timeout": 10}
            server = DatabricksMCPServer(url="https://test.com/mcp", params=custom_params)
            assert server.params["url"] == "https://test.com/mcp"
            assert server.params["headers"] == {"Custom-Header": "value"}
            assert server.params["timeout"] == 10

    def test_init_with_optional_parameters(self, mock_workspace_client):
        with patch(
            "databricks_openai.agents.mcp_server.WorkspaceClient",
            return_value=mock_workspace_client,
        ):
            from databricks_openai.agents.mcp_server import DatabricksMCPServer

            server = DatabricksMCPServer(
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
            from databricks_openai.agents.mcp_server import DatabricksMCPServer

            params: MCPServerStreamableHttpParams | None = params_dict  # type: ignore
            server = DatabricksMCPServer(url=url, params=params)
            assert server.params["url"] == expected_url
            for key, value in expected_extra.items():
                assert server.params[key] == value
            assert server.workspace_client == mock_workspace_client


class TestDatabricksMCPServerCreateStreams:
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
                from databricks_openai.agents.mcp_server import DatabricksMCPServer

                server = (
                    DatabricksMCPServer(url="https://test.com/mcp", params=params)
                    if params
                    else DatabricksMCPServer(url="https://test.com/mcp")
                )
                server.create_streams()

                mock_streamable.assert_called_once()
                call_kwargs = mock_streamable.call_args.kwargs
                assert call_kwargs["url"] == "https://test.com/mcp"
                for key, value in expected_values.items():
                    assert call_kwargs[key] == value
                if params is None:
                    assert "httpx_client_factory" not in call_kwargs
