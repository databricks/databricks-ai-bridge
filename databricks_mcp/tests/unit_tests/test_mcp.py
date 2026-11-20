import re
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient
from mcp.types import CallToolResult, TextContent, Tool

from databricks_mcp.mcp import (
    EXTERNAL_MCP,
    GENIE_MCP,
    MCP_URL_PATTERNS,
    UC_FUNCTIONS_MCP,
    VECTOR_SEARCH_MCP,
    DatabricksMCPClient,
    _is_databricks_apps_url,
    _is_oauth_auth,
)
from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider


def _patch_mcp_session(mock_session):
    """Patch the version-agnostic ``_open_mcp_session`` helper to yield ``mock_session``.

    Works regardless of whether mcp 1.x or 2.x is installed, since both code
    paths funnel through ``_open_mcp_session``.
    """

    @asynccontextmanager
    async def _fake_session(*args, **kwargs):
        yield mock_session

    return patch("databricks_mcp.mcp._open_mcp_session", _fake_session)


def _make_tool(name, description=""):
    """Construct an mcp ``Tool`` compatibly across mcp 1.x and 2.x.

    mcp 2.x renamed the ``inputSchema`` field to ``input_schema`` (keeping
    ``inputSchema`` as a validation alias), so build via ``model_validate`` with
    the wire name, which is accepted on both versions and avoids a type error.
    """
    return Tool.model_validate({"name": name, "description": description, "inputSchema": {}})


class TestDatabricksMCPClient:
    """Test cases for DatabricksMCPClient class."""

    def test_init_with_workspace_client(self):
        """Test initialization with provided workspace client."""
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient(
            "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
        )

        assert client.server_url == "https://test.com/api/2.0/mcp/functions/catalog/schema"
        assert client.client == workspace_client

    @patch("databricks_mcp.mcp.WorkspaceClient")
    def test_init_without_workspace_client(self, mock_workspace_client):
        """Test initialization without workspace client (should create default)."""
        mock_client_instance = MagicMock()
        mock_workspace_client.return_value = mock_client_instance

        client = DatabricksMCPClient("https://test.com/api/2.0/mcp/functions/catalog/schema")

        assert client.server_url == "https://test.com/api/2.0/mcp/functions/catalog/schema"
        assert client.client == mock_client_instance
        mock_workspace_client.assert_called_once()

    @pytest.mark.parametrize(
        "url,expected_mcp_type",
        [
            (
                "https://test.com/api/2.0/mcp/functions/catalog/schema",
                UC_FUNCTIONS_MCP,
            ),
            (
                "https://test.com/api/2.0/mcp/vector-search/catalog/schema",
                VECTOR_SEARCH_MCP,
            ),
            (
                "https://test.com/api/2.0/mcp/genie/space-id",
                GENIE_MCP,
            ),
            (
                "https://test.com/api/2.0/mcp/external/my-connection",
                EXTERNAL_MCP,
            ),
            ("https://test.com/invalid/path", None),
        ],
    )
    def test_get_databricks_managed_mcp_url_type(self, url, expected_mcp_type):
        """Test URL type detection for different MCP types."""
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient(url, workspace_client)
        mcp_type = client._get_databricks_managed_mcp_url_type()

        assert mcp_type == expected_mcp_type

    @pytest.mark.parametrize(
        "url,expected_genie_id",
        [
            ("https://test.com/api/2.0/mcp/genie/my-space-id", "my-space-id"),
            ("https://test.com/api/2.0/mcp/genie/another-space", "another-space"),
        ],
    )
    def test_extract_genie_id_valid(self, url, expected_genie_id):
        """Test extraction of Genie ID from valid URLs."""
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient(url, workspace_client)
        genie_id = client._extract_genie_id()

        assert genie_id == expected_genie_id

    @pytest.mark.parametrize(
        "url,expected_error",
        [
            (
                "https://test.com/api/2.0/mcp/functions/catalog/schema",
                "Missing /genie/ segment in:",
            ),
            (
                "https://test.com/api/2.0/mcp/genie/",
                "Genie ID not found in:",
            ),
        ],
    )
    def test_extract_genie_id_errors(self, url, expected_error):
        """Test extraction of Genie ID from invalid URLs."""
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient(url, workspace_client)

        with pytest.raises(ValueError, match=expected_error):
            client._extract_genie_id()

    @pytest.mark.parametrize(
        "url,expected_connection_name",
        [
            ("https://test.com/api/2.0/mcp/external/my-connection", "my-connection"),
            ("https://test.com/api/2.0/mcp/external/tavily_mcp", "tavily_mcp"),
        ],
    )
    def test_extract_connection_name_valid(self, url, expected_connection_name):
        """Test extraction of connection name from valid external MCP URLs."""
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient(url, workspace_client)
        assert client._extract_connection_name() == expected_connection_name

    @pytest.mark.parametrize(
        "url,expected_error",
        [
            (
                "https://test.com/api/2.0/mcp/functions/catalog/schema",
                "Missing /external/ segment in:",
            ),
            (
                "https://test.com/api/2.0/mcp/external/",
                "Connection name not found in:",
            ),
        ],
    )
    def test_extract_connection_name_errors(self, url, expected_error):
        """Test extraction of connection name from invalid URLs."""
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient(url, workspace_client)
        with pytest.raises(ValueError, match=expected_error):
            client._extract_connection_name()

    def test_get_databricks_resources_external(self):
        """Test getting Databricks resources for external MCP."""
        with (
            patch.object(
                DatabricksMCPClient,
                "_get_databricks_managed_mcp_url_type",
                return_value=EXTERNAL_MCP,
            ),
            patch.object(
                DatabricksMCPClient, "_extract_connection_name", return_value="my-connection"
            ),
        ):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/external/my-connection", workspace_client
            )
            resources = client.get_databricks_resources()

            assert len(resources) == 1
            assert resources[0].name == "my-connection"

    @pytest.mark.parametrize(
        "input_name,expected_name",
        [
            ("tool__name", "tool.name"),
            ("tool_name", "tool_name"),
            ("tool__name__with__multiple", "tool.name.with.multiple"),
            ("function__one", "function.one"),
            ("index__search", "index.search"),
        ],
    )
    def test_normalize_tool_name(self, input_name, expected_name):
        """Test tool name normalization (double underscores to dots)."""
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient(
            "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
        )

        assert client._normalize_tool_name(input_name) == expected_name

    @pytest.mark.asyncio
    async def test_get_tools_async(self):
        """Test asynchronous tool fetching."""
        mock_tools = [_make_tool("test_tool", "Test tool")]
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=mock_tools))

        with _patch_mcp_session(mock_session):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            tools = await client._get_tools_async()

            assert tools == mock_tools
            mock_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tools_async(self):
        """Test asynchronous tool calling."""
        mock_result = CallToolResult(content=[TextContent(type="text", text="test result")])
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        with _patch_mcp_session(mock_session):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            result = await client._call_tools_async("test_tool", {"arg": "value"})

            assert result == mock_result
            mock_session.call_tool.assert_called_once_with("test_tool", {"arg": "value"})

    @pytest.mark.asyncio
    async def test_get_tools_async_forwards_timeout(self):
        """Test that _get_tools_async forwards its timeout to _open_mcp_session."""
        mock_tools = [_make_tool("test_tool", "Test tool")]
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=mock_tools))

        @asynccontextmanager
        async def _fake_session(*args, **kwargs):
            assert kwargs.get("timeout") == 42.0
            yield mock_session

        with patch("databricks_mcp.mcp._open_mcp_session", _fake_session):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            tools = await client._get_tools_async(timeout=42.0)

            assert tools == mock_tools

    @pytest.mark.asyncio
    async def test_call_tools_async_forwards_timeout(self):
        """Test that _call_tools_async forwards its timeout to _open_mcp_session."""
        mock_result = CallToolResult(content=[TextContent(type="text", text="test result")])
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def _fake_session(*args, **kwargs):
            assert kwargs.get("timeout") == 120.0
            yield mock_session

        with patch("databricks_mcp.mcp._open_mcp_session", _fake_session):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            result = await client._call_tools_async("test_tool", {"arg": "value"}, timeout=120.0)

            assert result == mock_result

    def test_list_tools(self):
        """Test synchronous tool listing."""
        mock_tools = [_make_tool("test_tool", "Test tool")]

        with patch.object(DatabricksMCPClient, "_get_tools_async", return_value=mock_tools):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            tools = client.list_tools()

            assert tools == mock_tools

    def test_list_tools_forwards_timeout(self):
        """Test that list_tools() forwards an explicit timeout to _get_tools_async."""
        mock_tools = [_make_tool("test_tool", "Test tool")]

        with patch.object(
            DatabricksMCPClient, "_get_tools_async", return_value=mock_tools
        ) as mock_get_tools:
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            tools = client.list_tools(timeout=90.0)

            assert tools == mock_tools
            mock_get_tools.assert_called_once_with(
                timeout=90.0, terminate_on_close=None, client_kwargs=None
            )

    def test_list_tools_forwards_terminate_on_close_and_client_kwargs(self):
        """Test that list_tools() forwards terminate_on_close and client_kwargs (e.g. headers)."""
        mock_tools = [_make_tool("test_tool", "Test tool")]
        headers = {"Accept-Encoding": "identity"}

        with patch.object(
            DatabricksMCPClient, "_get_tools_async", return_value=mock_tools
        ) as mock_get_tools:
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            tools = client.list_tools(terminate_on_close=False, client_kwargs={"headers": headers})

            assert tools == mock_tools
            mock_get_tools.assert_called_once_with(
                timeout=None, terminate_on_close=False, client_kwargs={"headers": headers}
            )

    def test_call_tool(self):
        """Test synchronous tool calling."""
        mock_result = CallToolResult(content=[TextContent(type="text", text="test result")])

        with patch.object(DatabricksMCPClient, "_call_tools_async", return_value=mock_result):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            result = client.call_tool("test_tool", {"arg": "value"})

            assert result == mock_result

    def test_call_tool_forwards_timeout(self):
        """Test that call_tool() forwards an explicit timeout to _call_tools_async."""
        mock_result = CallToolResult(content=[TextContent(type="text", text="test result")])

        with patch.object(
            DatabricksMCPClient, "_call_tools_async", return_value=mock_result
        ) as mock_call:
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            result = client.call_tool("test_tool", {"arg": "value"}, timeout=60.0)

            assert result == mock_result
            mock_call.assert_called_once_with(
                "test_tool",
                {"arg": "value"},
                timeout=60.0,
                terminate_on_close=None,
                client_kwargs=None,
            )

    def test_call_tool_forwards_terminate_on_close_and_client_kwargs(self):
        """Test that call_tool() forwards terminate_on_close and client_kwargs (e.g. headers)."""
        mock_result = CallToolResult(content=[TextContent(type="text", text="test result")])
        headers = {"Accept-Encoding": "identity"}

        with patch.object(
            DatabricksMCPClient, "_call_tools_async", return_value=mock_result
        ) as mock_call:
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            result = client.call_tool(
                "test_tool",
                {"arg": "value"},
                terminate_on_close=False,
                client_kwargs={"headers": headers},
            )

            assert result == mock_result
            mock_call.assert_called_once_with(
                "test_tool",
                {"arg": "value"},
                timeout=None,
                terminate_on_close=False,
                client_kwargs={"headers": headers},
            )

    @pytest.mark.asyncio
    async def test_alist_tools_forwards_timeout(self):
        """Test that alist_tools() forwards an explicit timeout to _get_tools_async."""
        mock_tools = [_make_tool("test_tool", "Test tool")]

        with patch.object(
            DatabricksMCPClient, "_get_tools_async", return_value=mock_tools
        ) as mock_get_tools:
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            tools = await client.alist_tools(timeout=15.0)

            assert tools == mock_tools
            mock_get_tools.assert_called_once_with(
                timeout=15.0, terminate_on_close=None, client_kwargs=None
            )

    @pytest.mark.asyncio
    async def test_acall_tool_forwards_timeout(self):
        """Test that acall_tool() forwards an explicit timeout to _call_tools_async."""
        mock_result = CallToolResult(content=[TextContent(type="text", text="test result")])

        with patch.object(
            DatabricksMCPClient, "_call_tools_async", return_value=mock_result
        ) as mock_call:
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            result = await client.acall_tool("test_tool", {"arg": "value"}, timeout=75.0)

            assert result == mock_result
            mock_call.assert_called_once_with(
                "test_tool",
                {"arg": "value"},
                timeout=75.0,
                terminate_on_close=None,
                client_kwargs=None,
            )

    @pytest.mark.asyncio
    async def test_open_mcp_session_client_kwargs_reach_httpx_client(self):
        """Test that client_kwargs (e.g. headers) reach the underlying httpx2 client,
        without clobbering the explicit auth/follow_redirects/timeout kwargs, on mcp>=2.0.0."""
        from databricks_mcp.mcp import _MCP_V2, _open_mcp_session

        if not _MCP_V2:
            pytest.skip("This assertion targets the mcp>=2.0.0 (httpx2) code path")

        captured_client_kwargs = {}
        captured_transport_kwargs = {}
        mock_session = AsyncMock()

        class _FakeAsyncClient:
            def __init__(self, **kwargs):
                captured_client_kwargs.update(kwargs)

            async def __aenter__(self):
                return MagicMock()

            async def __aexit__(self, *exc_info):
                return False

        @asynccontextmanager
        async def _fake_client_session(*args, **kwargs):
            yield mock_session

        def _fake_streamable_http_client(server_url, **kwargs):
            captured_transport_kwargs.update(kwargs)
            return MagicMock()

        with (
            patch("databricks_mcp.mcp.httpx2.AsyncClient", _FakeAsyncClient),
            patch("databricks_mcp.mcp.streamable_http_client", _fake_streamable_http_client),
            patch("databricks_mcp.mcp.Client", _fake_client_session),
        ):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            auth = DatabricksOAuthClientProvider(workspace_client)
            async with _open_mcp_session(
                "https://test.com/api/2.0/mcp/functions/catalog/schema",
                auth,
                timeout=42.0,
                terminate_on_close=False,
                client_kwargs={"headers": {"Accept-Encoding": "identity"}},
            ) as session:
                assert session is mock_session

        assert captured_client_kwargs["auth"] is auth
        assert captured_client_kwargs["follow_redirects"] is True
        assert captured_client_kwargs["timeout"] == 42.0
        assert captured_client_kwargs["headers"] == {"Accept-Encoding": "identity"}
        assert captured_transport_kwargs["terminate_on_close"] is False

    @pytest.mark.parametrize(
        "mcp_type,tool_names,expected_resource_names",
        [
            (
                UC_FUNCTIONS_MCP,
                ["function__one", "function__two"],
                ["function.one", "function.two"],
            ),
            (
                VECTOR_SEARCH_MCP,
                ["index__one", "index__two"],
                ["index.one", "index.two"],
            ),
        ],
    )
    def test_get_databricks_resources_with_tools(
        self, mcp_type, tool_names, expected_resource_names
    ):
        """Test getting Databricks resources for MCP types that require tool listing."""
        mock_tools = [_make_tool(name, f"Tool {name}") for name in tool_names]

        with (
            patch.object(DatabricksMCPClient, "list_tools", return_value=mock_tools),
            patch.object(
                DatabricksMCPClient,
                "_get_databricks_managed_mcp_url_type",
                return_value=mcp_type,
            ),
        ):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            resources = client.get_databricks_resources()

            assert len(resources) == len(expected_resource_names)
            for i, expected_name in enumerate(expected_resource_names):
                assert resources[i].name == expected_name

    def test_get_databricks_resources_genie(self):
        """Test getting Databricks resources for Genie MCP."""
        with (
            patch.object(
                DatabricksMCPClient, "_get_databricks_managed_mcp_url_type", return_value=GENIE_MCP
            ),
            patch.object(DatabricksMCPClient, "_extract_genie_id", return_value="my-genie-space"),
        ):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/genie/my-genie-space", workspace_client
            )
            resources = client.get_databricks_resources()

            assert len(resources) == 1
            assert resources[0].name == "my-genie-space"

    def test_get_databricks_resources_invalid_url(self):
        """Test getting Databricks resources for invalid URL."""
        with patch.object(
            DatabricksMCPClient, "_get_databricks_managed_mcp_url_type", return_value=None
        ):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient("https://test.com/invalid/path", workspace_client)

            resources = client.get_databricks_resources()
            assert resources == []

    def test_get_databricks_resources_unknown_mcp_type(self):
        """Test getting Databricks resources for unknown MCP type."""
        mock_tools = [_make_tool("test_tool", "Test tool")]

        with (
            patch.object(DatabricksMCPClient, "list_tools", return_value=mock_tools),
            patch.object(
                DatabricksMCPClient,
                "_get_databricks_managed_mcp_url_type",
                return_value="unknown_type",
            ),
        ):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/unknown/path", workspace_client
            )
            resources = client.get_databricks_resources()

            assert resources == []

    def test_get_databricks_resources_exception_handling(self):
        """Test exception handling in get_databricks_resources."""
        with patch.object(
            DatabricksMCPClient,
            "_get_databricks_managed_mcp_url_type",
            side_effect=Exception("Test error"),
        ):
            workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
            client = DatabricksMCPClient(
                "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
            )
            resources = client.get_databricks_resources()

            assert resources == []


class TestMCPURLPatterns:
    """Test cases for MCP URL patterns."""

    @pytest.mark.parametrize(
        "pattern_name,valid_urls,invalid_urls",
        [
            (
                UC_FUNCTIONS_MCP,
                [
                    "/api/2.0/mcp/functions/catalog/schema",
                    "/api/2.0/mcp/functions/my_catalog/my_schema",
                ],
                [
                    "/api/2.0/mcp/functions/catalog",
                    "/api/2.0/mcp/functions/catalog/schema/extra",
                    "/api/2.0/mcp/vector-search/catalog/schema",
                ],
            ),
            (
                VECTOR_SEARCH_MCP,
                [
                    "/api/2.0/mcp/vector-search/catalog/schema",
                    "/api/2.0/mcp/vector-search/my_catalog/my_schema",
                ],
                [
                    "/api/2.0/mcp/vector-search/catalog",
                    "/api/2.0/mcp/vector-search/catalog/schema/extra",
                    "/api/2.0/mcp/functions/catalog/schema",
                ],
            ),
            (
                GENIE_MCP,
                [
                    "/api/2.0/mcp/genie/space-id",
                    "/api/2.0/mcp/genie/my-genie-space",
                ],
                [
                    "/api/2.0/mcp/genie",
                    "/api/2.0/mcp/genie/space-id/extra",
                    "/api/2.0/mcp/functions/catalog/schema",
                ],
            ),
            (
                EXTERNAL_MCP,
                [
                    "/api/2.0/mcp/external/my-connection",
                    "/api/2.0/mcp/external/tavily_mcp",
                ],
                [
                    "/api/2.0/mcp/external",
                    "/api/2.0/mcp/external/my-connection/extra",
                    "/api/2.0/mcp/functions/catalog/schema",
                ],
            ),
        ],
    )
    def test_mcp_url_patterns(self, pattern_name, valid_urls, invalid_urls):
        """Test MCP URL pattern matching for all types."""
        pattern = MCP_URL_PATTERNS[pattern_name]

        # Test valid URLs
        for url in valid_urls:
            assert re.match(pattern, url), f"URL should match: {url}"

        # Test invalid URLs
        for url in invalid_urls:
            assert not re.match(pattern, url), f"URL should not match: {url}"

    @pytest.mark.parametrize(
        "status_code,expected_exc,expected_msg,method_name",
        [
            (302, PermissionError, "Access denied to the MCP server", "list_tools"),
            (302, PermissionError, "Access denied to the MCP server", "call_tool"),
            (404, ValueError, "MCP Server not found at the provided server url", "list_tools"),
            (404, ValueError, "MCP Server not found at the provided server url", "call_tool"),
            # Any non-302/404 should re-raise the original error
            (500, Exception, "Original connection error", "list_tools"),
            (500, Exception, "Original connection error", "call_tool"),
        ],
    )
    def test_error_decorator_paths(self, status_code, expected_exc, expected_msg, method_name):
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient("https://custom-mcp-server.com", workspace_client)

        original_error = Exception("Original connection error")

        with (
            patch.object(client, "_get_databricks_managed_mcp_url_type", return_value=None),
            patch("databricks_mcp.mcp.DatabricksOAuthClientProvider") as mock_auth_provider,
            patch("requests.request") as mock_request,
            patch("databricks_mcp.mcp._open_mcp_session") as mock_session,
            patch.object(
                client.client.config,
                "authenticate",
                return_value={"Authorization": "Bearer test-token"},
            ) as mock_auth,
        ):
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_request.return_value = mock_response

            # Trigger decorator by failing the MCP call
            mock_session.side_effect = original_error

            method = getattr(client, method_name)
            if expected_exc is Exception:
                with pytest.raises(Exception, match=expected_msg):
                    if method_name == "list_tools":
                        method()
                    else:
                        method("test_tool", {"arg": "value"})
            else:
                with pytest.raises(expected_exc, match=expected_msg):
                    if method_name == "list_tools":
                        method()
                    else:
                        method("test_tool", {"arg": "value"})

            # Verify probe request was attempted with correct headers and payload
            expected_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": "Bearer test-token",
            }
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[0] == ("POST", "https://custom-mcp-server.com")
            assert call_args[1]["headers"] == expected_headers
            assert "data" in call_args[1]
            assert "allow_redirects" in call_args[1]
            assert call_args[1]["allow_redirects"] == False

    def test_error_decorator_managed_server_reraises_original(self):
        workspace_client = WorkspaceClient(host="https://test.com", token="test-token")
        client = DatabricksMCPClient(
            "https://test.com/api/2.0/mcp/functions/catalog/schema", workspace_client
        )

        original_error = Exception("Databricks server error")

        with (
            patch.object(
                client, "_get_databricks_managed_mcp_url_type", return_value=UC_FUNCTIONS_MCP
            ),
            patch("databricks_mcp.mcp._open_mcp_session") as mock_session,
            patch("databricks_mcp.mcp.DatabricksOAuthClientProvider"),
            patch("requests.request") as mock_request,
        ):
            mock_session.side_effect = original_error

            with pytest.raises(Exception, match="Databricks server error"):
                client.list_tools()

            mock_request.assert_not_called()


class TestIsDatabricksAppsUrl:
    """Test cases for _is_databricks_apps_url helper function."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://my-app.staging.aws.databricksapps.com/mcp", True),
            ("https://my-app.prod.azure.databricksapps.com/mcp", True),
            ("https://my-app.databricksapps.com", True),
            ("https://test.cloud.databricks.com/api/2.0/mcp/functions/a/b", False),
            ("https://custom-server.example.com/mcp", False),
            ("https://databricksapps.com.evil.com/mcp", False),
            ("https://notdatabricksapps.com/mcp", False),
        ],
    )
    def test_is_databricks_apps_url(self, url, expected):
        assert _is_databricks_apps_url(url) == expected


class TestIsOauthAuth:
    @pytest.mark.parametrize(
        "side_effect,expected",
        [
            (None, True),  # oauth_token succeeds
            (ValueError("not available"), False),  # oauth_token raises
        ],
    )
    def test_is_oauth_auth(self, side_effect, expected):
        mock_client = MagicMock(spec=WorkspaceClient)
        if side_effect:
            mock_client.config.oauth_token.side_effect = side_effect
        assert _is_oauth_auth(mock_client) is expected


class TestDatabricksMCPClientOAuthValidation:
    @pytest.mark.parametrize("auth_type", ["pat", "runtime"])
    def test_raises_error_for_non_oauth_with_databricks_apps(self, auth_type):
        mock_client = MagicMock(spec=WorkspaceClient)
        mock_client.config.oauth_token.side_effect = ValueError(f"not available for {auth_type}")
        with pytest.raises(ValueError, match="OAuth authentication is required"):
            DatabricksMCPClient("https://my-app.databricksapps.com/mcp", mock_client)

    def test_allows_oauth_with_databricks_apps(self):
        mock_client = MagicMock(spec=WorkspaceClient)
        client = DatabricksMCPClient("https://my-app.databricksapps.com/mcp", mock_client)
        assert client.server_url == "https://my-app.databricksapps.com/mcp"

    def test_allows_non_oauth_with_non_databricks_apps(self):
        mock_client = MagicMock(spec=WorkspaceClient)
        mock_client.config.oauth_token.side_effect = ValueError("not available")
        client = DatabricksMCPClient("https://test.com/api/2.0/mcp/functions/a/b", mock_client)
        assert client.server_url == "https://test.com/api/2.0/mcp/functions/a/b"
