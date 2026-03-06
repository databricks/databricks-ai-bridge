"""
Core DatabricksMCPClient integration tests.

Tests list_tools, call_tool, and auth paths against a live Databricks MCP server
backed by UC functions, Vector Search indexes, and Genie spaces.
"""

from __future__ import annotations

import os

import pytest
from conftest import _skip_if_not_found
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult

from databricks_mcp import DatabricksMCPClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MCP_INTEGRATION_TESTS") != "1",
    reason="MCP integration tests disabled. Set RUN_MCP_INTEGRATION_TESTS=1 to enable.",
)

CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_mcp_test"
FUNCTION_NAME = "echo_message"

# =============================================================================
# UC Functions
# =============================================================================


@pytest.mark.integration
class TestMCPClientUCFunctions:
    """Verify list_tools() and call_tool() against a live UC function MCP server."""

    def test_list_tools_returns_valid_tools(self, cached_tools_list):
        assert len(cached_tools_list) > 0
        for tool in cached_tools_list:
            assert tool.name, "Tool name should be non-empty"
            assert tool.inputSchema, "Tool should have an inputSchema"
            assert "properties" in tool.inputSchema, "inputSchema should have 'properties'"

    def test_tool_name_contains_function_identifier(self, cached_tools_list):
        tool_names = [t.name for t in cached_tools_list]
        assert any(FUNCTION_NAME in name or "echo_message" in name for name in tool_names), (
            f"Expected a tool name containing '{FUNCTION_NAME}', got: {tool_names}"
        )

    def test_schema_level_lists_multiple_tools(self, schema_mcp_client):
        """Schema-level URL should list at least the echo_message function."""
        try:
            tools = schema_mcp_client.list_tools()
        except ExceptionGroup as e:  # ty: ignore[unresolved-reference]
            _skip_if_not_found(e, "Schema-level UC endpoint not available")
        assert len(tools) >= 1, "Schema-level listing should return at least one tool"

    def test_call_tool_echoes_input(self, cached_call_result):
        assert isinstance(cached_call_result, CallToolResult)
        assert cached_call_result.content, "call_tool result should have content"
        assert len(cached_call_result.content) > 0
        first_item = cached_call_result.content[0]
        assert hasattr(first_item, "text"), "First content item should have .text"
        assert "hello" in first_item.text, (
            f"Echo function should return input 'hello', got: {first_item.text}"
        )

    def test_call_tool_with_different_args(self, mcp_client, cached_tools_list):
        tool_name = cached_tools_list[0].name
        result = mcp_client.call_tool(tool_name, {"message": "integration_test_42"})
        text = result.content[0].text
        assert "integration_test_42" in text, (
            f"Echo should return 'integration_test_42', got: {text}"
        )


# =============================================================================
# Vector Search
# =============================================================================


@pytest.mark.integration
class TestMCPClientVectorSearch:
    """Verify list_tools() and call_tool() against a live VS MCP server."""

    def test_list_tools_returns_valid_tools(self, cached_vs_tools_list):
        assert len(cached_vs_tools_list) > 0
        for tool in cached_vs_tools_list:
            assert tool.name, "Tool name should be non-empty"
            assert tool.inputSchema, "Tool should have an inputSchema"
            assert "properties" in tool.inputSchema, "inputSchema should have 'properties'"

    def test_call_tool_returns_result_with_content(self, cached_vs_call_result):
        assert isinstance(cached_vs_call_result, CallToolResult)
        assert cached_vs_call_result.content, "call_tool result should have content"
        assert len(cached_vs_call_result.content) > 0

    def test_schema_level_lists_tools(self, vs_schema_mcp_client):
        """Schema-level VS URL should list at least one tool."""
        try:
            tools = vs_schema_mcp_client.list_tools()
        except ExceptionGroup as e:  # ty: ignore[unresolved-reference]
            _skip_if_not_found(e, "Schema-level VS endpoint not available")
        assert len(tools) >= 1, "Schema-level VS listing should return at least one tool"


# =============================================================================
# Genie
# =============================================================================


@pytest.mark.integration
class TestMCPClientGenie:
    """Verify list_tools() and call_tool() against a live Genie MCP server."""

    def test_list_tools_returns_valid_tools(self, cached_genie_tools_list):
        assert len(cached_genie_tools_list) > 0
        for tool in cached_genie_tools_list:
            assert tool.name, "Tool name should be non-empty"
            assert tool.inputSchema, "Tool should have an inputSchema"
            assert "properties" in tool.inputSchema, "inputSchema should have 'properties'"

    def test_call_tool_returns_result_with_content(self, cached_genie_call_result):
        assert isinstance(cached_genie_call_result, CallToolResult)
        assert cached_genie_call_result.content, "call_tool result should have content"
        assert len(cached_genie_call_result.content) > 0


# =============================================================================
# DBSQL
# =============================================================================


@pytest.mark.integration
class TestMCPClientDBSQL:
    """Verify list_tools() and call_tool() against a live DBSQL MCP server."""

    def test_list_tools_returns_expected_tools(self, cached_dbsql_tools_list):
        tool_names = [t.name for t in cached_dbsql_tools_list]
        for expected in ["execute_sql", "execute_sql_read_only", "poll_sql_result"]:
            assert expected in tool_names, f"Expected tool '{expected}' not found in {tool_names}"

    def test_call_tool_execute_sql_read_only(self, dbsql_mcp_client, cached_dbsql_tools_list):
        """execute_sql_read_only with SHOW CATALOGS should return results."""
        result = dbsql_mcp_client.call_tool("execute_sql_read_only", {"query": "SHOW CATALOGS"})
        assert isinstance(result, CallToolResult)
        assert result.content, "SHOW CATALOGS should return content"
        assert len(result.content) > 0


# =============================================================================
# Raw streamable_http_client
# =============================================================================


@pytest.mark.integration
class TestRawStreamableHttpClient:
    """Verify DatabricksOAuthClientProvider works with the raw MCP SDK streamable_http_client.

    This tests the low-level path: httpx.AsyncClient + DatabricksOAuthClientProvider
    + streamable_http_client + ClientSession, without going through DatabricksMCPClient.
    """

    def test_uc_function_list_and_call(self, uc_function_url, workspace_client):
        """list_tools + call_tool via raw streamable_http_client for UC functions."""
        import asyncio

        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        from databricks_mcp import DatabricksOAuthClientProvider

        async def _test():
            async with httpx.AsyncClient(
                auth=DatabricksOAuthClientProvider(workspace_client),
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, read=120.0),
            ) as http_client:
                async with streamable_http_client(uc_function_url, http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()

                        # list_tools
                        tools_response = await session.list_tools()
                        tools = tools_response.tools
                        assert len(tools) > 0
                        tool_names = [t.name for t in tools]
                        assert any("echo_message" in name for name in tool_names)

                        # call_tool
                        tool_name = next(n for n in tool_names if "echo_message" in n)
                        result = await session.call_tool(tool_name, {"message": "raw_client_test"})
                        assert result.content
                        assert "raw_client_test" in str(result.content[0].text)

        asyncio.run(_test())

    def test_vs_list_tools(self, vs_mcp_url, workspace_client):
        """list_tools via raw streamable_http_client for Vector Search."""
        import asyncio

        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        from databricks_mcp import DatabricksOAuthClientProvider

        async def _test():
            async with httpx.AsyncClient(
                auth=DatabricksOAuthClientProvider(workspace_client),
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, read=120.0),
            ) as http_client:
                async with streamable_http_client(vs_mcp_url, http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools_response = await session.list_tools()
                        assert len(tools_response.tools) > 0

        asyncio.run(_test())

    def test_dbsql_list_and_call(self, dbsql_mcp_url, workspace_client):
        """list_tools + call_tool via raw streamable_http_client for DBSQL."""
        import asyncio

        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        from databricks_mcp import DatabricksOAuthClientProvider

        async def _test():
            async with httpx.AsyncClient(
                auth=DatabricksOAuthClientProvider(workspace_client),
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, read=120.0),
            ) as http_client:
                async with streamable_http_client(dbsql_mcp_url, http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()

                        tools_response = await session.list_tools()
                        tools = tools_response.tools
                        tool_names = [t.name for t in tools]
                        assert "execute_sql_read_only" in tool_names

                        result = await session.call_tool(
                            "execute_sql_read_only", {"query": "SHOW CATALOGS"}
                        )
                        assert result.content
                        assert len(result.content) > 0

        asyncio.run(_test())

    def test_genie_list_and_call(self, genie_mcp_url, workspace_client):
        """list_tools + call_tool via raw streamable_http_client for Genie."""
        import asyncio

        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        from databricks_mcp import DatabricksOAuthClientProvider

        async def _test():
            async with httpx.AsyncClient(
                auth=DatabricksOAuthClientProvider(workspace_client),
                follow_redirects=True,
                timeout=httpx.Timeout(120.0, read=120.0),
            ) as http_client:
                async with streamable_http_client(genie_mcp_url, http_client=http_client) as (
                    read_stream,
                    write_stream,
                    _,
                ):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()

                        tools_response = await session.list_tools()
                        tools = tools_response.tools
                        assert len(tools) > 0

                        # Call the first tool (query_space_*)
                        tool = tools[0]
                        properties = tool.inputSchema.get("properties", {})
                        param_name = next(iter(properties), "query")
                        result = await session.call_tool(
                            tool.name, {param_name: "How many rows are there?"}
                        )
                        assert result.content
                        assert len(result.content) > 0

        asyncio.run(_test())


# =============================================================================
# Error paths
# =============================================================================


@pytest.mark.integration
class TestMCPClientErrorPaths:
    """Verify DatabricksMCPClient returns helpful errors for invalid inputs.

    Tests the 3-layer customer journey: bad function → bad tool → bad arguments.
    Each test constructs its own client inline (no shared fixtures).
    """

    def test_nonexistent_function_raises_error(self, workspace_client):
        """list_tools() on a nonexistent function raises McpError(BAD_REQUEST: ... not found)."""
        host = workspace_client.config.host
        bad_url = f"{host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}/nonexistent_fn_xyz"
        client = DatabricksMCPClient(bad_url, workspace_client)

        with pytest.raises(ExceptionGroup) as exc_info:  # ty: ignore[unresolved-reference]
            client.list_tools()

        # McpError is nested inside ExceptionGroup(s) from asyncio.run + TaskGroup
        mcp_errors = exc_info.value.exceptions
        while mcp_errors and isinstance(mcp_errors[0], ExceptionGroup):  # ty: ignore[unresolved-reference]
            mcp_errors = mcp_errors[0].exceptions
        assert len(mcp_errors) == 1 and isinstance(mcp_errors[0], McpError), (
            f"Expected McpError, got: {mcp_errors}"
        )
        assert "BAD_REQUEST" in str(mcp_errors[0]), f"Expected BAD_REQUEST, got: {mcp_errors[0]}"
        assert "not found" in str(mcp_errors[0]).lower(), (
            f"Expected 'not found', got: {mcp_errors[0]}"
        )

    def test_nonexistent_tool_name_raises_error(self, workspace_client):
        """call_tool() with a malformed tool name raises McpError(BAD_REQUEST: ... malformed)."""
        host = workspace_client.config.host
        url = f"{host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}/{FUNCTION_NAME}"
        client = DatabricksMCPClient(url, workspace_client)

        with pytest.raises(ExceptionGroup) as exc_info:  # ty: ignore[unresolved-reference]
            client.call_tool("nonexistent_tool_xyz", {"message": "test"})

        mcp_errors = exc_info.value.exceptions
        while mcp_errors and isinstance(mcp_errors[0], ExceptionGroup):  # ty: ignore[unresolved-reference]
            mcp_errors = mcp_errors[0].exceptions
        assert len(mcp_errors) == 1 and isinstance(mcp_errors[0], McpError), (
            f"Expected McpError, got: {mcp_errors}"
        )
        assert "BAD_REQUEST" in str(mcp_errors[0]), f"Expected BAD_REQUEST, got: {mcp_errors[0]}"
        assert "malformed" in str(mcp_errors[0]).lower(), (
            f"Expected 'malformed', got: {mcp_errors[0]}"
        )

    def test_wrong_arguments_raises_error(self, workspace_client):
        """call_tool() with wrong arguments raises McpError(BAD_REQUEST: Missing parameter)."""
        host = workspace_client.config.host
        url = f"{host}/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}/{FUNCTION_NAME}"
        client = DatabricksMCPClient(url, workspace_client)

        tools = client.list_tools()
        assert len(tools) > 0
        tool_name = tools[0].name

        # echo_message expects "message", we pass "completely_wrong_arg"
        with pytest.raises(ExceptionGroup) as exc_info:  # ty: ignore[unresolved-reference]
            client.call_tool(tool_name, {"completely_wrong_arg": "test"})

        mcp_errors = exc_info.value.exceptions
        while mcp_errors and isinstance(mcp_errors[0], ExceptionGroup):  # ty: ignore[unresolved-reference]
            mcp_errors = mcp_errors[0].exceptions
        assert len(mcp_errors) == 1 and isinstance(mcp_errors[0], McpError), (
            f"Expected McpError, got: {mcp_errors}"
        )
        assert "BAD_REQUEST" in str(mcp_errors[0]), f"Expected BAD_REQUEST, got: {mcp_errors[0]}"
        assert "missing parameter value" in str(mcp_errors[0]).lower(), (
            f"Expected 'Missing parameter value', got: {mcp_errors[0]}"
        )


# =============================================================================
# Auth paths
# =============================================================================


@pytest.mark.integration
class TestMCPClientAuth:
    """Verify different auth methods work through the MCP layer."""

    def test_oauth_m2m_auth_works(self, uc_function_url):
        """OAuth M2M auth with explicit client_id/client_secret should work."""
        from databricks.sdk import WorkspaceClient

        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
        host = os.environ.get("DATABRICKS_HOST")
        if not (client_id and client_secret and host):
            pytest.skip(
                "OAuth-M2M credentials not available "
                "(need DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET)"
            )

        oauth_wc = WorkspaceClient(
            host=host,
            client_id=client_id,
            client_secret=client_secret,
        )
        client = DatabricksMCPClient(uc_function_url, oauth_wc)
        tools = client.list_tools()
        assert len(tools) > 0

    def test_pat_auth_works(self, workspace_client, uc_function_url):
        """PAT auth (extracted from current auth) should work."""
        from databricks.sdk import WorkspaceClient

        headers = workspace_client.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        assert token, "Could not extract bearer token from workspace client"

        pat_wc = WorkspaceClient(host=workspace_client.config.host, token=token, auth_type="pat")
        client = DatabricksMCPClient(uc_function_url, pat_wc)
        tools = client.list_tools()
        assert len(tools) > 0
