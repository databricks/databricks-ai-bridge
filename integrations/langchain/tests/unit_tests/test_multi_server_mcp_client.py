"""Unit tests for DatabricksMultiServerMCPClient and related classes."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from databricks.sdk import WorkspaceClient

from databricks_langchain.multi_server_mcp_client import (
    DatabricksMcpHttpClientFactory,
    DatabricksMCPServer,
    DatabricksMultiServerMCPClient,
    MCPServer,
)


class TestMCPServer:
    """Tests for the MCPServer class."""

    def test_basic_server_creation(self):
        """Test creating a basic server with minimal parameters."""
        server = MCPServer(name="test-server", url="https://example.com/mcp")

        assert server.name == "test-server"
        assert server.url == "https://example.com/mcp"
        assert server.handle_tool_error is None

    @pytest.mark.parametrize(
        "extra_params",
        [
            {"timeout": 30.0},
            {"headers": {"X-API-Key": "secret"}},
            {"sse_read_timeout": 60.0},
            {"timeout": 15.0, "headers": {"Authorization": "Bearer token"}},
            {"session_kwargs": {"some_param": "value"}},
        ],
    )
    def test_server_accepts_extra_params(self, extra_params: dict[str, Any]):
        """Test that MCPServer accepts and preserves extra parameters."""
        server = MCPServer(
            name="test-server",
            url="https://example.com/mcp",
            handle_tool_error=True,
            **extra_params,
        )

        connection_dict = server.to_connection_dict()

        # Check that extra params are in connection dict
        for key, value in extra_params.items():
            assert connection_dict[key] == value
            assert "name" not in connection_dict
            assert "handle_tool_error" not in connection_dict

    @pytest.mark.parametrize(
        "handle_tool_error_value",
        [
            True,
            False,
            "Custom error message",
            lambda e: f"Error: {e}",
            None,
        ],
    )
    def test_server_handle_tool_error_types(self, handle_tool_error_value: Any):
        """Test that handle_tool_error accepts various types."""
        server = MCPServer(
            name="test-server",
            url="https://example.com/mcp",
            handle_tool_error=handle_tool_error_value,
        )

        assert server.handle_tool_error == handle_tool_error_value


class TestDatabricksMCPServer:
    """Tests for the DatabricksMCPServer class."""

    def test_databricks_server_without_workspace_client(self):
        """Test DatabricksMCPServer creates WorkspaceClient automatically."""
        with (
            patch("databricks_langchain.multi_server_mcp_client.WorkspaceClient") as mock_ws,
            patch(
                "databricks_langchain.multi_server_mcp_client.DatabricksOAuthClientProvider"
            ) as mock_auth,
        ):
            mock_ws_instance = MagicMock()
            mock_ws.return_value = mock_ws_instance
            mock_auth_instance = MagicMock()
            mock_auth.return_value = mock_auth_instance

            server = DatabricksMCPServer(name="databricks", url="https://databricks.com/mcp")

            # Should have created WorkspaceClient
            mock_ws.assert_called_once()
            # Should have created auth provider
            mock_auth.assert_called_once_with(mock_ws_instance)

    def test_databricks_server_with_workspace_client(self):
        """Test DatabricksMCPServer uses provided WorkspaceClient."""
        mock_workspace_client = create_autospec(WorkspaceClient, instance=True)

        with patch(
            "databricks_langchain.multi_server_mcp_client.DatabricksOAuthClientProvider"
        ) as mock_auth:
            mock_auth_instance = MagicMock()
            mock_auth.return_value = mock_auth_instance

            server = DatabricksMCPServer(
                name="databricks",
                url="https://databricks.com/mcp",
                workspace_client=mock_workspace_client,
            )

            # Should have used provided client
            mock_auth.assert_called_once_with(mock_workspace_client)
            assert server.workspace_client is mock_workspace_client

            connection_dict = server.to_connection_dict()
            assert "workspace_client" not in connection_dict
            assert "auth" in connection_dict
            assert connection_dict["auth"] is mock_auth_instance

    def test_databricks_server_accepts_extra_params(self):
        """Test that DatabricksMCPServer accepts extra connection params."""
        mock_workspace_client = create_autospec(WorkspaceClient, instance=True)

        with patch(
            "databricks_langchain.multi_server_mcp_client.DatabricksOAuthClientProvider"
        ) as mock_auth:
            mock_auth_instance = MagicMock()
            mock_auth.return_value = mock_auth_instance

            server = DatabricksMCPServer(
                name="databricks",
                url="https://databricks.com/mcp",
                workspace_client=mock_workspace_client,
                timeout=45.0,
                headers={"X-Custom": "header"},
            )

            connection_dict = server.to_connection_dict()

            assert connection_dict["timeout"] == 45.0
            assert connection_dict["headers"] == {"X-Custom": "header"}

    def test_databricks_server_includes_http_factory(self):
        """Test that DatabricksMCPServer includes the custom HTTP client factory."""
        mock_workspace_client = create_autospec(WorkspaceClient, instance=True)

        with patch(
            "databricks_langchain.multi_server_mcp_client.DatabricksOAuthClientProvider"
        ) as mock_auth:
            mock_auth_instance = MagicMock()
            mock_auth.return_value = mock_auth_instance

            server = DatabricksMCPServer(
                name="databricks",
                url="https://databricks.com/mcp",
                workspace_client=mock_workspace_client,
            )

            connection_dict = server.to_connection_dict()

            assert "httpx_client_factory" in connection_dict
            assert isinstance(
                connection_dict["httpx_client_factory"], DatabricksMcpHttpClientFactory
            )


class TestDatabricksMultiServerMCPClient:
    """Tests for the DatabricksMultiServerMCPClient class."""

    def test_client_initialization_with_multiple_servers(self):
        """Test client initialization with multiple servers."""
        with patch(
            "databricks_langchain.multi_server_mcp_client.MultiServerMCPClient.__init__"
        ) as mock_init:
            mock_init.return_value = None

            servers = [
                MCPServer(name="server1", url="https://server1.com/mcp"),
                MCPServer(name="server2", url="https://server2.com/mcp"),
            ]
            client = DatabricksMultiServerMCPClient(servers)

            # Check that parent __init__ was called
            mock_init.assert_called_once()

            # Check connections dict structure
            call_kwargs = mock_init.call_args[1]
            connections = call_kwargs["connections"]

            assert len(connections) == 2
            assert "server1" in connections
            assert "server2" in connections

            assert hasattr(client, "_server_configs")
            assert len(client._server_configs) == 2
            assert "server1" in client._server_configs
            assert "server2" in client._server_configs

    @pytest.mark.asyncio
    async def test_get_tools_all_servers(self):
        """Test get_tools without server_name (all servers)."""
        servers = [
            MCPServer(name="server1", url="https://server1.com/mcp", handle_tool_error=True),
            MCPServer(
                name="server2", url="https://server2.com/mcp", handle_tool_error="Custom error"
            ),
        ]

        # Create mock tools for each server
        mock_tool1 = MagicMock()
        mock_tool2 = MagicMock()
        mock_tool3 = MagicMock()

        # Mock parent get_tools to return different tools for different servers
        async def mock_get_tools_side_effect(server_name=None):
            if server_name == "server1":
                return [mock_tool1, mock_tool2]
            elif server_name == "server2":
                return [mock_tool3]
            return []

        with (
            patch(
                "databricks_langchain.multi_server_mcp_client.MultiServerMCPClient.__init__"
            ) as mock_init,
            patch(
                "databricks_langchain.multi_server_mcp_client.MultiServerMCPClient.get_tools",
                new_callable=AsyncMock,
                side_effect=mock_get_tools_side_effect,
            ) as mock_parent_get_tools,
        ):
            mock_init.return_value = None

            client = DatabricksMultiServerMCPClient(servers)
            client.connections = {
                "server1": servers[0].to_connection_dict(),
                "server2": servers[1].to_connection_dict(),
            }

            tools = await client.get_tools()

            # Should call parent get_tools for each server
            assert mock_parent_get_tools.call_count == 2

            # Should apply handle_tool_error from respective servers
            assert mock_tool1.handle_tool_error is True
            assert mock_tool2.handle_tool_error is True
            assert mock_tool3.handle_tool_error == "Custom error"

            # Should return all tools
            assert len(tools) == 3
            assert mock_tool1 in tools
            assert mock_tool2 in tools
            assert mock_tool3 in tools

    @pytest.mark.asyncio
    async def test_get_tools_parallel_execution(self):
        """Test that get_tools executes server requests in parallel."""
        servers = [MCPServer(name=f"server{i}", url=f"https://server{i}.com/mcp") for i in range(5)]

        call_count = 0
        call_times = []

        async def mock_get_tools_with_delay(server_name=None):
            nonlocal call_count
            call_count += 1
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate async work
            return [MagicMock()]

        with (
            patch(
                "databricks_langchain.multi_server_mcp_client.MultiServerMCPClient.__init__"
            ) as mock_init,
            patch(
                "databricks_langchain.multi_server_mcp_client.MultiServerMCPClient.get_tools",
                new_callable=AsyncMock,
                side_effect=mock_get_tools_with_delay,
            ) as mock_parent_get_tools,
        ):
            mock_init.return_value = None

            client = DatabricksMultiServerMCPClient(servers)
            client.connections = {server.name: server.to_connection_dict() for server in servers}

            start_time = asyncio.get_event_loop().time()
            tools = await client.get_tools()
            end_time = asyncio.get_event_loop().time()

            # All 5 servers should be called
            assert call_count == 5

            # Should return tools from all servers
            assert len(tools) == 5

    @pytest.mark.asyncio
    async def test_get_tools_with_databricks_server(self):
        """Test get_tools with DatabricksMCPServer."""
        mock_workspace_client = create_autospec(WorkspaceClient, instance=True)
        mock_tool = MagicMock()

        with (
            patch(
                "databricks_langchain.multi_server_mcp_client.MultiServerMCPClient.__init__"
            ) as mock_init,
            patch(
                "databricks_langchain.multi_server_mcp_client.DatabricksOAuthClientProvider"
            ) as mock_auth,
            patch(
                "databricks_langchain.multi_server_mcp_client.MultiServerMCPClient.get_tools",
                new_callable=AsyncMock,
            ) as mock_parent_get_tools,
        ):
            mock_init.return_value = None
            mock_auth_instance = MagicMock()
            mock_auth.return_value = mock_auth_instance
            mock_parent_get_tools.return_value = [mock_tool]

            server = DatabricksMCPServer(
                name="databricks",
                url="https://databricks.com/mcp",
                workspace_client=mock_workspace_client,
                handle_tool_error=True,
            )
            client = DatabricksMultiServerMCPClient([server])
            client.connections = {"databricks": server.to_connection_dict()}

            tools = await client.get_tools(server_name="databricks")

            # Should apply handle_tool_error
            assert mock_tool.handle_tool_error is True

            # Connection should have auth
            assert "auth" in client.connections["databricks"]


class TestDatabricksMCPServerFromUCResource:
    """Tests for from_uc_function and from_vector_search class methods."""

    def test_from_uc_function(self):
        """Test from_uc_function constructs correct URL."""
        mock_workspace_client = create_autospec(WorkspaceClient, instance=True)
        mock_workspace_client.config.host = "https://test.databricks.com"

        with patch("databricks_langchain.multi_server_mcp_client.DatabricksOAuthClientProvider"):
            server = DatabricksMCPServer.from_uc_function(
                catalog="system",
                schema="ai",
                function_name="test_tool",
                name="my_server",
                workspace_client=mock_workspace_client,
            )

            assert (
                server.url
                == "https://test.databricks.com/api/2.0/mcp/functions/system/ai/test_tool"
            )
            assert server.name == "my_server"
            assert server.workspace_client == mock_workspace_client

    def test_from_vector_search(self):
        """Test from_vector_search constructs correct URL."""
        mock_workspace_client = create_autospec(WorkspaceClient, instance=True)
        mock_workspace_client.config.host = "https://test.databricks.com"

        with patch("databricks_langchain.multi_server_mcp_client.DatabricksOAuthClientProvider"):
            server = DatabricksMCPServer.from_vector_search(
                catalog="system",
                schema="ai",
                index_name="test_index",
                name="my_search",
                workspace_client=mock_workspace_client,
            )

            assert (
                server.url
                == "https://test.databricks.com/api/2.0/mcp/vector-search/system/ai/test_index"
            )
            assert server.name == "my_search"
            assert server.workspace_client == mock_workspace_client
