"""
Integration tests for OpenAI MCP wrappers (McpServerToolkit + McpServer).

Tests McpServerToolkit (vanilla OpenAI SDK) and McpServer (OpenAI Agents SDK)
against a live Databricks MCP server backed by a UC function (echo_message).

Prerequisites:
    Run databricks_mcp/tests/integration_tests/setup_workspace.py once to create
    the test UC function.
"""

from __future__ import annotations

import asyncio
import os

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
def toolkit(workspace_client):
    from databricks_openai import McpServerToolkit

    return McpServerToolkit.from_uc_function(
        catalog=CATALOG,
        schema=SCHEMA,
        function_name=FUNCTION_NAME,
        workspace_client=workspace_client,
    )


@pytest.fixture(scope="session")
def toolkit_tools(toolkit):
    """Cache get_tools() result to minimize API calls."""
    try:
        return toolkit.get_tools()
    except Exception as e:
        pytest.skip(
            f"Could not get tools from MCP server — is the test function set up? "
            f"Run setup_workspace.py first. Error: {e}"
        )


# =============================================================================
# McpServerToolkit Init Tests
# =============================================================================


@pytest.mark.integration
class TestMcpServerToolkitInit:
    """Test McpServerToolkit initialization and tool listing with live MCP server."""

    def test_from_uc_function_creates_toolkit(self, workspace_client):
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=workspace_client,
        )
        assert toolkit is not None
        assert toolkit.url is not None

    def test_get_tools_returns_tool_infos(self, toolkit_tools):
        from databricks_openai.mcp_server_toolkit import ToolInfo

        assert len(toolkit_tools) > 0
        for tool in toolkit_tools:
            assert isinstance(tool, ToolInfo)

    def test_tool_spec_is_valid_openai_format(self, toolkit_tools):
        for tool in toolkit_tools:
            spec = tool.spec
            assert spec["type"] == "function"
            assert "function" in spec
            func = spec["function"]
            assert "name" in func
            assert "parameters" in func
            assert "properties" in func["parameters"]

    def test_tool_name_prefixed_when_name_provided(self, workspace_client):
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            name="mcp_test",
            workspace_client=workspace_client,
        )
        tools = toolkit.get_tools()
        assert len(tools) > 0
        for tool in tools:
            assert tool.name.startswith("mcp_test__"), (
                f"Tool name should start with 'mcp_test__', got: {tool.name}"
            )


# =============================================================================
# McpServerToolkit Execution Tests
# =============================================================================


@pytest.mark.integration
class TestMcpServerToolkitExecution:
    """Test McpServerToolkit tool execution with live MCP server."""

    def test_execute_returns_string(self, toolkit_tools):
        tool = toolkit_tools[0]
        result = tool.execute(message="hello")
        assert isinstance(result, str)

    def test_execute_result_contains_input(self, toolkit_tools):
        tool = toolkit_tools[0]
        result = tool.execute(message="hello")
        assert "hello" in result, f"Echo should return 'hello', got: {result}"


# =============================================================================
# McpServer (Agents SDK) Init Tests
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsInit:
    """Test McpServer (Agents SDK) context manager and tool listing."""

    def test_from_uc_function_context_manager(self, workspace_client):
        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name=FUNCTION_NAME,
                workspace_client=workspace_client,
            ) as server:
                assert server is not None

        asyncio.run(_test())

    def test_server_lists_tools(self, workspace_client):
        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name=FUNCTION_NAME,
                workspace_client=workspace_client,
            ) as server:
                tools = await server.list_tools()
                assert len(tools) > 0

        asyncio.run(_test())


# =============================================================================
# McpServer (Agents SDK) Execution Tests
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsExecution:
    """Test McpServer (Agents SDK) tool execution."""

    def test_call_tool_returns_call_tool_result(self, workspace_client):
        from mcp.types import CallToolResult

        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name=FUNCTION_NAME,
                workspace_client=workspace_client,
            ) as server:
                tools = await server.list_tools()
                tool_name = tools[0].name
                result = await server.call_tool(tool_name, {"message": "hello"})
                assert isinstance(result, CallToolResult)

        asyncio.run(_test())

    def test_call_tool_result_has_content(self, workspace_client):
        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name=FUNCTION_NAME,
                workspace_client=workspace_client,
            ) as server:
                tools = await server.list_tools()
                tool_name = tools[0].name
                result = await server.call_tool(tool_name, {"message": "hello"})
                assert result.content, "call_tool result should have content"
                assert len(result.content) > 0

        asyncio.run(_test())


# =============================================================================
# Auth Path Tests
# =============================================================================


@pytest.mark.integration
class TestOpenAIMCPAuthPaths:
    """Verify auth credentials are correctly forwarded through the OpenAI MCP wrappers."""

    def test_current_auth_produces_working_toolkit(self, workspace_client):
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=workspace_client,
        )
        tools = toolkit.get_tools()
        assert len(tools) > 0

    def test_pat_auth_produces_working_toolkit(self, workspace_client):
        from databricks.sdk import WorkspaceClient

        from databricks_openai import McpServerToolkit

        headers = workspace_client.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        assert token, "Could not extract bearer token from workspace client"

        pat_wc = WorkspaceClient(host=workspace_client.config.host, token=token, auth_type="pat")
        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=pat_wc,
        )
        tools = toolkit.get_tools()
        assert len(tools) > 0

    def test_oauth_m2m_auth_produces_working_toolkit(self):
        from databricks.sdk import WorkspaceClient

        from databricks_openai import McpServerToolkit

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
        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=oauth_wc,
        )
        tools = toolkit.get_tools()
        assert len(tools) > 0


# =============================================================================
# Schema-Level Listing Tests
# =============================================================================


@pytest.mark.integration
class TestMcpServerToolkitSchemaLevel:
    """Test schema-level URL construction and listing through OpenAI wrapper."""

    def test_from_uc_function_schema_level_url(self, workspace_client):
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            workspace_client=workspace_client,
        )
        expected_suffix = f"/api/2.0/mcp/functions/{CATALOG}/{SCHEMA}"
        assert toolkit.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {toolkit.url}"
        )

    def test_schema_level_get_tools_returns_tools(self, workspace_client):
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            workspace_client=workspace_client,
        )
        try:
            tools = toolkit.get_tools()
        except Exception as e:
            pytest.skip(f"Schema-level listing failed: {e}")
        assert len(tools) >= 1, "Schema-level listing should return at least one tool"


# =============================================================================
# Vector Search MCP Tests (McpServerToolkit)
# =============================================================================


@pytest.mark.integration
class TestMcpServerToolkitVectorSearch:
    """Test McpServerToolkit.from_vector_search() with live VS MCP server."""

    def test_from_vector_search_url_pattern(self, workspace_client):
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            index_name=VS_INDEX,
            workspace_client=workspace_client,
        )
        expected_suffix = f"/api/2.0/mcp/vector-search/{CATALOG}/{VS_SCHEMA}/{VS_INDEX}"
        assert toolkit.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {toolkit.url}"
        )

    def test_from_vector_search_get_tools_returns_tool_infos(self, workspace_client):
        from databricks_openai import McpServerToolkit
        from databricks_openai.mcp_server_toolkit import ToolInfo

        toolkit = McpServerToolkit.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            index_name=VS_INDEX,
            workspace_client=workspace_client,
        )
        try:
            tools = toolkit.get_tools()
        except Exception as e:
            pytest.skip(f"VS MCP endpoint not available: {e}")
        assert len(tools) > 0
        for tool in tools:
            assert isinstance(tool, ToolInfo), f"Expected ToolInfo, got {type(tool)}"

    def test_from_vector_search_execute(self, workspace_client):
        """Execute a VS tool via McpServerToolkit and verify string result."""
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            index_name=VS_INDEX,
            workspace_client=workspace_client,
        )
        try:
            tools = toolkit.get_tools()
        except Exception as e:
            pytest.skip(f"VS MCP endpoint not available: {e}")
        assert len(tools) > 0
        tool = tools[0]
        # Dynamically extract first param from tool spec
        properties = tool.spec["function"]["parameters"].get("properties", {})
        param_name = next(iter(properties), "query")
        result = tool.execute(**{param_name: "test"})
        assert isinstance(result, str)


# =============================================================================
# Vector Search MCP Tests (McpServer / Agents SDK)
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsVectorSearch:
    """Test McpServer.from_vector_search() with live VS MCP server (Agents SDK)."""

    def test_from_vector_search_lists_tools(self, workspace_client):
        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_vector_search(
                catalog=CATALOG,
                schema=VS_SCHEMA,
                index_name=VS_INDEX,
                workspace_client=workspace_client,
            ) as server:
                try:
                    tools = await server.list_tools()
                except Exception as e:
                    pytest.skip(f"VS MCP endpoint not available: {e}")
                assert len(tools) > 0

        asyncio.run(_test())

    def test_from_vector_search_call_tool(self, workspace_client):
        """call_tool on a VS tool via Agents SDK context manager."""
        from mcp.types import CallToolResult

        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_vector_search(
                catalog=CATALOG,
                schema=VS_SCHEMA,
                index_name=VS_INDEX,
                workspace_client=workspace_client,
            ) as server:
                try:
                    tools = await server.list_tools()
                except Exception as e:
                    pytest.skip(f"VS MCP endpoint not available: {e}")
                assert len(tools) > 0
                tool = tools[0]
                # Dynamically extract first param from tool's inputSchema
                properties = tool.inputSchema.get("properties", {})
                param_name = next(iter(properties), "query")
                result = await server.call_tool(tool.name, {param_name: "test"})
                assert isinstance(result, CallToolResult)
                assert result.content, "VS call_tool result should have content"

        asyncio.run(_test())


# =============================================================================
# Vector Search Schema-Level Tests (McpServerToolkit)
# =============================================================================


@pytest.mark.integration
class TestMcpServerToolkitVSSchemaLevel:
    """Test schema-level VS URL construction and listing through McpServerToolkit."""

    def test_from_vector_search_schema_level_url(self, workspace_client):
        """from_vector_search(catalog, schema) without index_name → schema-level URL."""
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            workspace_client=workspace_client,
        )
        expected_suffix = f"/api/2.0/mcp/vector-search/{CATALOG}/{VS_SCHEMA}"
        assert toolkit.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {toolkit.url}"
        )

    def test_schema_level_vs_get_tools_returns_tools(self, workspace_client):
        """Schema-level VS listing returns ≥1 ToolInfo."""
        from databricks_openai import McpServerToolkit
        from databricks_openai.mcp_server_toolkit import ToolInfo

        toolkit = McpServerToolkit.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            workspace_client=workspace_client,
        )
        try:
            tools = toolkit.get_tools()
        except Exception as e:
            pytest.skip(f"VS schema-level MCP endpoint not available: {e}")
        assert len(tools) >= 1, "Schema-level VS listing should return at least one tool"
        for tool in tools:
            assert isinstance(tool, ToolInfo), f"Expected ToolInfo, got {type(tool)}"


# =============================================================================
# Vector Search Schema-Level Tests (McpServer / Agents SDK)
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsVSSchemaLevel:
    """Test schema-level VS listing via Agents SDK context manager."""

    def test_from_vector_search_schema_level_lists_tools(self, workspace_client):
        """Schema-level VS listing via Agents SDK context manager returns ≥1 tool."""
        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_vector_search(
                catalog=CATALOG,
                schema=VS_SCHEMA,
                workspace_client=workspace_client,
            ) as server:
                try:
                    tools = await server.list_tools()
                except Exception as e:
                    pytest.skip(f"VS schema-level MCP endpoint not available: {e}")
                assert len(tools) >= 1, "Schema-level VS listing should return at least one tool"

        asyncio.run(_test())


# =============================================================================
# Agents SDK Auth Path Tests
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsAuthPaths:
    """Verify auth credentials are correctly forwarded through the Agents SDK MCP wrapper."""

    def test_pat_auth_agents_sdk(self, workspace_client):
        """McpServer with PAT-authenticated WorkspaceClient lists tools."""
        from databricks.sdk import WorkspaceClient

        from databricks_openai.agents import McpServer

        headers = workspace_client.config.authenticate()
        token = headers.get("Authorization", "").replace("Bearer ", "")
        assert token, "Could not extract bearer token from workspace client"

        pat_wc = WorkspaceClient(host=workspace_client.config.host, token=token, auth_type="pat")

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name=FUNCTION_NAME,
                workspace_client=pat_wc,
            ) as server:
                tools = await server.list_tools()
                assert len(tools) > 0

        asyncio.run(_test())

    def test_oauth_m2m_auth_agents_sdk(self):
        """McpServer with OAuth M2M WorkspaceClient lists tools."""
        from databricks.sdk import WorkspaceClient

        from databricks_openai.agents import McpServer

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

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name=FUNCTION_NAME,
                workspace_client=oauth_wc,
            ) as server:
                tools = await server.list_tools()
                assert len(tools) > 0

        asyncio.run(_test())


# =============================================================================
# MLflow Tracing Tests (McpServer / Agents SDK)
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsMLflowTracing:
    """Verify McpServer.call_tool() creates MLflow traces."""

    def test_call_tool_creates_mlflow_trace(self, workspace_client):
        """McpServer.call_tool() creates an MLflow trace with SpanType.TOOL."""
        import mlflow
        from mlflow.entities import SpanType

        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name=FUNCTION_NAME,
                workspace_client=workspace_client,
            ) as server:
                tools = await server.list_tools()
                tool_name = tools[0].name
                await server.call_tool(tool_name, {"message": "tracing-test"})

            trace = mlflow.get_last_active_trace()
            assert trace is not None, "call_tool should produce an MLflow trace"
            spans = trace.data.spans
            assert len(spans) > 0, "Trace should contain at least one span"
            root_span = spans[0]
            assert root_span.span_type == SpanType.TOOL, (
                f"Expected span_type={SpanType.TOOL}, got: {root_span.span_type}"
            )

        asyncio.run(_test())


# =============================================================================
# Agents SDK Kwargs Pass-Through Tests
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsKwargsPassThrough:
    """Verify kwargs like timeout are correctly handled by McpServer."""

    def test_timeout_kwarg_accepted(self, workspace_client):
        """McpServer.from_uc_function(timeout=30.0) stores correct timeout in params."""
        from databricks_openai.agents import McpServer

        server = McpServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=workspace_client,
            timeout=30.0,
        )
        assert server.params["timeout"] == 30.0, (
            f"Expected timeout=30.0, got: {server.params['timeout']}"
        )

    def test_timeout_default_is_20(self, workspace_client):
        """McpServer without explicit timeout defaults to 20.0 seconds."""
        from databricks_openai.agents import McpServer

        server = McpServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=workspace_client,
        )
        assert server.params["timeout"] == 20.0, (
            f"Expected default timeout=20.0, got: {server.params['timeout']}"
        )
