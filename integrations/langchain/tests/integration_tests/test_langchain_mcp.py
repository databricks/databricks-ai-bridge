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

VS_SCHEMA = "databricks_ai_bridge_vs_test"
VS_INDEX = "delta_sync_managed"

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
# DatabricksMCPServer — UC Functions
# =============================================================================


@pytest.mark.integration
class TestDatabricksMCPServerUCFunctions:
    """Test DatabricksMCPServer with UC function MCP endpoints (init, exec, auth, kwargs, schema-level)."""

    # -- Init / listing --

    def test_server_url_matches_pattern(self, mcp_server):
        expected_suffix = f"/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}/{FUNCTION_NAME}"
        assert mcp_server.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {mcp_server.url}"
        )

    def test_to_connection_dict_has_transport_and_auth(self, mcp_server):
        conn = mcp_server.to_connection_dict()
        assert conn["transport"] == "streamable_http"
        assert "auth" in conn, "Connection dict should have 'auth' key"

    def test_get_tools_returns_valid_base_tools(self, cached_langchain_tools):
        from langchain_core.tools import BaseTool

        assert len(cached_langchain_tools) > 0
        for tool in cached_langchain_tools:
            assert isinstance(tool, BaseTool), f"Expected BaseTool, got {type(tool)}"
            assert tool.name, "Tool should have a non-empty name"
            assert tool.description, "Tool should have a non-empty description"

    # -- Execution --

    def test_tool_invoke_echoes_input(self, workspace_client, mcp_server):
        from databricks_langchain import DatabricksMultiServerMCPClient

        async def _test():
            client = DatabricksMultiServerMCPClient([mcp_server])
            tools = await client.get_tools()
            assert len(tools) > 0
            result = await tools[0].ainvoke({"message": "hello"})
            assert result is not None
            result_str = str(result)
            assert "hello" in result_str, f"Echo should return 'hello', got: {result}"

        asyncio.run(_test())

    # -- Kwargs pass-through --

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

    # -- Auth paths --

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

    # -- Schema-level --

    def test_from_uc_function_schema_level_url(self, workspace_client):
        from databricks_langchain import DatabricksMCPServer

        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            name="schema-level-test",
            workspace_client=workspace_client,
        )
        expected_suffix = f"/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"
        assert server.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {server.url}"
        )

    def test_schema_level_get_tools_returns_tools(self, workspace_client):
        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            name="schema-level-tools",
            workspace_client=workspace_client,
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            tools = await client.get_tools()
            assert len(tools) >= 1, "Schema-level listing should return at least one tool"

        asyncio.run(_test())


# =============================================================================
# DatabricksMCPServer — Vector Search
# =============================================================================


@pytest.mark.integration
class TestDatabricksMCPServerVectorSearch:
    """Test DatabricksMCPServer.from_vector_search() (index-level + schema-level)."""

    def test_from_vector_search_url_pattern(self, workspace_client):
        from databricks_langchain import DatabricksMCPServer

        server = DatabricksMCPServer.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            index_name=VS_INDEX,
            name="vs-test",
            workspace_client=workspace_client,
        )
        expected_suffix = f"/api/2.0/mcp/vector-search/{CATALOG}/{VS_SCHEMA}/{VS_INDEX}"
        assert server.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {server.url}"
        )

    def test_from_vector_search_tool_invoke(self, workspace_client):
        """Get tools (verify BaseTool type) and invoke a VS tool."""
        from langchain_core.tools import BaseTool

        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        server = DatabricksMCPServer.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            index_name=VS_INDEX,
            name="vs-invoke-test",
            workspace_client=workspace_client,
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            try:
                tools = await client.get_tools()
            except Exception as e:
                pytest.skip(f"VS MCP endpoint not available: {e}")
            assert len(tools) > 0
            for tool in tools:
                assert isinstance(tool, BaseTool), f"Expected BaseTool, got {type(tool)}"
            tool = tools[0]
            # Dynamically extract first required param from tool schema
            # args_schema may be a Pydantic model class or a plain dict
            raw_schema = tool.args_schema
            if isinstance(raw_schema, dict):
                schema = raw_schema
            elif raw_schema is not None and hasattr(raw_schema, "schema"):
                schema = raw_schema.schema()
            else:
                schema = {}
            properties = schema.get("properties", {})
            param_name = next(iter(properties), "query")
            result = await tool.ainvoke({param_name: "test"})
            assert result is not None

        asyncio.run(_test())

    def test_from_vector_search_schema_level(self, workspace_client):
        """Schema-level VS: verify URL pattern and listing returns ≥1 tool."""
        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        server = DatabricksMCPServer.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            name="vs-schema-level-test",
            workspace_client=workspace_client,
        )
        expected_suffix = f"/api/2.0/mcp/vector-search/{CATALOG}/{VS_SCHEMA}"
        assert server.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {server.url}"
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            try:
                tools = await client.get_tools()
            except Exception as e:
                pytest.skip(f"VS schema-level MCP endpoint not available: {e}")
            assert len(tools) >= 1, "Schema-level VS listing should return at least one tool"

        asyncio.run(_test())


# =============================================================================
# DatabricksMcpHttpClientFactory Token Refresh Tests
# =============================================================================


@pytest.mark.integration
class TestDatabricksMcpHttpClientFactoryTokenRefresh:
    """Verify DatabricksMcpHttpClientFactory re-creates auth providers for token refresh."""

    def test_sequential_invocations_succeed(self, workspace_client):
        """Two sequential get_tools() calls succeed — proves token refresh works."""
        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        server = DatabricksMCPServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="token-refresh-test",
            workspace_client=workspace_client,
        )

        async def _test():
            client = DatabricksMultiServerMCPClient([server])
            tools_1 = await client.get_tools()
            assert len(tools_1) > 0
            tools_2 = await client.get_tools()
            assert len(tools_2) > 0

        asyncio.run(_test())

    def test_factory_creates_new_oauth_provider(self, workspace_client):
        """DatabricksMcpHttpClientFactory re-creates DatabricksOAuthClientProvider when called."""
        import httpx
        from databricks_mcp import DatabricksOAuthClientProvider

        from databricks_langchain.multi_server_mcp_client import (
            DatabricksMcpHttpClientFactory,
        )

        factory = DatabricksMcpHttpClientFactory()
        original_auth = DatabricksOAuthClientProvider(workspace_client)

        client_1 = factory(timeout=httpx.Timeout(10), auth=original_auth)
        client_2 = factory(timeout=httpx.Timeout(10), auth=original_auth)

        # Each call should produce a client with a *different* auth provider instance
        assert client_1.auth is not original_auth, (
            "Factory should create a new DatabricksOAuthClientProvider, not reuse the original"
        )
        assert client_1.auth is not client_2.auth, (
            "Each factory call should produce a distinct auth provider instance"
        )
