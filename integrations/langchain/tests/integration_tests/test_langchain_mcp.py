"""
Integration tests for LangChain MCP wrappers.

Tests DatabricksMCPServer, DatabricksMultiServerMCPClient against a live
Databricks MCP server backed by a UC function (echo_message).

Prerequisites:
    Run databricks_mcp/tests/integration_tests/setup_workspace.py once to create
    the test UC function.
"""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MCP_INTEGRATION_TESTS") != "1",
    reason="MCP integration tests disabled. Set RUN_MCP_INTEGRATION_TESTS=1 to enable.",
)

CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_mcp_test"
FUNCTION_NAME = "echo_message"

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def workspace_client():
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


@pytest.fixture(scope="session")
def mcp_server(workspace_client):
    from databricks_langchain import DatabricksMCPServer

    return DatabricksMCPServer.from_uc_function(
        catalog=CATALOG,
        schema=SCHEMA,
        function_name=FUNCTION_NAME,
        name="mcp-test",
        workspace_client=workspace_client,
    )


@pytest.fixture(scope="session")
def cached_langchain_tools(mcp_server):
    """Cache get_tools() result to minimize API calls.

    As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient no longer supports
    context manager usage. Use client.get_tools() directly instead.
    """
    from databricks_langchain import DatabricksMultiServerMCPClient

    async def _get():
        client = DatabricksMultiServerMCPClient([mcp_server])
        return await client.get_tools()

    try:
        return asyncio.run(_get())
    except Exception as e:
        pytest.skip(
            f"Could not get tools from MCP server — is the test function set up? "
            f"Run setup_workspace.py first. Error: {e}"
        )


# =============================================================================
# DatabricksMCPServer Init Tests
# =============================================================================


@pytest.mark.integration
class TestDatabricksMCPServerInit:
    """Test DatabricksMCPServer construction and URL generation."""

    def test_from_uc_function_creates_server(self, workspace_client):
        from databricks_langchain import DatabricksMCPServer

        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="mcp-init-test",
            workspace_client=workspace_client,
        )
        assert server is not None

    def test_server_url_matches_pattern(self, mcp_server):
        expected_suffix = f"/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}/{FUNCTION_NAME}"
        assert mcp_server.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {mcp_server.url}"
        )

    def test_to_connection_dict_has_transport_and_auth(self, mcp_server):
        conn = mcp_server.to_connection_dict()
        assert conn["transport"] == "streamable_http"
        assert "auth" in conn, "Connection dict should have 'auth' key"


# =============================================================================
# DatabricksMultiServerMCPClient Tools Tests
# =============================================================================


@pytest.mark.integration
class TestDatabricksMultiServerMCPClientTools:
    """Test DatabricksMultiServerMCPClient tool listing with live MCP server."""

    def test_get_tools_returns_langchain_tools(self, cached_langchain_tools):
        assert len(cached_langchain_tools) > 0

    def test_tools_are_langchain_base_tools(self, cached_langchain_tools):
        from langchain_core.tools import BaseTool

        for tool in cached_langchain_tools:
            assert isinstance(tool, BaseTool), f"Expected BaseTool, got {type(tool)}"

    def test_tool_has_name_and_description(self, cached_langchain_tools):
        for tool in cached_langchain_tools:
            assert tool.name, "Tool should have a non-empty name"
            assert tool.description, "Tool should have a non-empty description"


# =============================================================================
# DatabricksMultiServerMCPClient Execution Tests
# =============================================================================


@pytest.mark.integration
class TestDatabricksMultiServerMCPClientExecution:
    """Test DatabricksMultiServerMCPClient tool invocation with live MCP server."""

    def test_tool_invoke_returns_result(self, workspace_client, mcp_server):
        from databricks_langchain import DatabricksMultiServerMCPClient

        async def _test():
            client = DatabricksMultiServerMCPClient([mcp_server])
            tools = await client.get_tools()
            assert len(tools) > 0
            result = await tools[0].ainvoke({"message": "hello"})
            assert result is not None

        asyncio.run(_test())

    def test_tool_invoke_result_contains_input(self, workspace_client, mcp_server):
        from databricks_langchain import DatabricksMultiServerMCPClient

        async def _test():
            client = DatabricksMultiServerMCPClient([mcp_server])
            tools = await client.get_tools()
            assert len(tools) > 0
            result = await tools[0].ainvoke({"message": "hello"})
            # LangChain MCP tools return raw content (list of dicts or string)
            result_str = str(result)
            assert "hello" in result_str, f"Echo should return 'hello', got: {result}"

        asyncio.run(_test())


# =============================================================================
# Kwargs Pass-Through Tests
# =============================================================================


@pytest.mark.integration
class TestLangChainMCPKwargsPassThrough:
    """Verify kwargs like handle_tool_error and timeout are passed through correctly."""

    def test_handle_tool_error_string_applied_to_tools(self, workspace_client):
        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="error-test",
            workspace_client=workspace_client,
            handle_tool_error="Custom error occurred",
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            tools = await client.get_tools()
            assert len(tools) > 0
            for tool in tools:
                assert tool.handle_tool_error == "Custom error occurred", (
                    f"Expected handle_tool_error='Custom error occurred', "
                    f"got: {tool.handle_tool_error}"
                )

        asyncio.run(_test())

    def test_handle_tool_error_true_applied_to_tools(self, workspace_client):
        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="error-bool-test",
            workspace_client=workspace_client,
            handle_tool_error=True,
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            tools = await client.get_tools()
            assert len(tools) > 0
            for tool in tools:
                assert tool.handle_tool_error is True, (
                    f"Expected handle_tool_error=True, got: {tool.handle_tool_error}"
                )

        asyncio.run(_test())

    def test_timeout_kwarg_accepted(self, workspace_client):
        from databricks_langchain import DatabricksMCPServer

        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="timeout-test",
            workspace_client=workspace_client,
            timeout=60.0,
        )
        conn = server.to_connection_dict()
        assert conn["timeout"] == timedelta(seconds=60.0), (
            f"Expected timedelta(seconds=60.0), got: {conn['timeout']}"
        )


# =============================================================================
# Auth Path Tests
# =============================================================================


@pytest.mark.integration
class TestLangChainMCPAuthPaths:
    """Verify auth credentials are correctly forwarded through the LangChain MCP wrappers."""

    def test_current_auth_produces_working_client(self, workspace_client):
        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="auth-current",
            workspace_client=workspace_client,
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            tools = await client.get_tools()
            assert len(tools) > 0

        asyncio.run(_test())

    def test_pat_auth_produces_working_client(self, workspace_client):
        from databricks.sdk import WorkspaceClient

        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        headers = workspace_client.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        assert token, "Could not extract bearer token from workspace client"

        pat_wc = WorkspaceClient(host=workspace_client.config.host, token=token, auth_type="pat")
        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="auth-pat",
            workspace_client=pat_wc,
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            tools = await client.get_tools()
            assert len(tools) > 0

        asyncio.run(_test())

    def test_oauth_m2m_auth_produces_working_client(self):
        from databricks.sdk import WorkspaceClient

        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

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
        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="auth-oauth",
            workspace_client=oauth_wc,
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            tools = await client.get_tools()
            assert len(tools) > 0

        asyncio.run(_test())
