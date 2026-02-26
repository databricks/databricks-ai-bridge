"""
Integration tests for OpenAI MCP wrappers (McpServerToolkit + McpServer).

Tests McpServerToolkit (vanilla OpenAI SDK) and McpServer (OpenAI Agents SDK)
against a live Databricks MCP server backed by a UC function (echo_message).

Prerequisites:
    The test UC function must exist in the workspace.
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
            f"Is the test function set up? Error: {e}"
        )


# =============================================================================
# McpServerToolkit — UC Functions
# =============================================================================


@pytest.mark.integration
class TestMcpServerToolkitUCFunctions:
    """Test McpServerToolkit with UC function MCP endpoints (init, exec, auth, schema-level)."""

    # -- Init / listing --

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

    # -- Execution --

    def test_execute_echoes_input(self, toolkit_tools):
        tool = toolkit_tools[0]
        result = tool.execute(message="hello")
        assert isinstance(result, str)
        assert "hello" in result, f"Echo should return 'hello', got: {result}"

    # -- Schema-level --

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

    # -- Auth paths --

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
# McpServerToolkit — Vector Search
# =============================================================================


@pytest.mark.integration
class TestMcpServerToolkitVectorSearch:
    """Test McpServerToolkit.from_vector_search() (index-level + schema-level)."""

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

    def test_from_vector_search_execute(self, workspace_client):
        """Get tools (verify ToolInfo type) and execute a VS tool."""
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
        tool = tools[0]
        # Dynamically extract first param from tool spec
        properties = tool.spec["function"]["parameters"].get("properties", {})
        param_name = next(iter(properties), "query")  # ty:ignore[no-matching-overload]
        result = tool.execute(**{param_name: "test"})
        assert isinstance(result, str)

    def test_from_vector_search_schema_level(self, workspace_client):
        """Schema-level VS: verify URL pattern and listing returns ≥1 ToolInfo."""
        from databricks_openai import McpServerToolkit
        from databricks_openai.mcp_server_toolkit import ToolInfo

        toolkit = McpServerToolkit.from_vector_search(
            catalog=CATALOG,
            schema=VS_SCHEMA,
            workspace_client=workspace_client,
        )
        expected_suffix = f"/api/2.0/mcp/vector-search/{CATALOG}/{VS_SCHEMA}"
        assert toolkit.url.endswith(expected_suffix), (
            f"URL should end with '{expected_suffix}', got: {toolkit.url}"
        )
        try:
            tools = toolkit.get_tools()
        except Exception as e:
            pytest.skip(f"VS schema-level MCP endpoint not available: {e}")
        assert len(tools) >= 1, "Schema-level VS listing should return at least one tool"
        for tool in tools:
            assert isinstance(tool, ToolInfo), f"Expected ToolInfo, got {type(tool)}"


# =============================================================================
# McpServer (Agents SDK) — UC Functions
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsUCFunctions:
    """Test McpServer (Agents SDK) with UC function MCP endpoints (init, exec, auth, tracing, timeout)."""

    # -- Listing --

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

    # -- Execution --

    def test_call_tool_returns_result_with_content(self, workspace_client):
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
                assert result.content, "call_tool result should have content"
                assert len(result.content) > 0

        asyncio.run(_test())

    # -- Auth paths --

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

    # -- MLflow tracing --

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

            trace_id = mlflow.get_last_active_trace_id()
            assert trace_id is not None, "call_tool should produce an MLflow trace"
            trace = mlflow.get_trace(trace_id)
            assert trace is not None, "Should be able to retrieve the trace by ID"
            spans = trace.data.spans
            assert len(spans) > 0, "Trace should contain at least one span"
            root_span = spans[0]
            assert root_span.span_type == SpanType.TOOL, (
                f"Expected span_type={SpanType.TOOL}, got: {root_span.span_type}"
            )

        asyncio.run(_test())

    # -- Timeout kwargs --

    def test_timeout_defaults_and_override(self, workspace_client):
        """McpServer defaults to 20.0s timeout; explicit timeout=30.0 overrides it."""
        from databricks_openai.agents import McpServer

        default_server = McpServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=workspace_client,
        )
        assert default_server.params["timeout"] == 20.0, (
            f"Expected default timeout=20.0, got: {default_server.params['timeout']}"
        )

        custom_server = McpServer.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=workspace_client,
            timeout=30.0,
        )
        assert custom_server.params["timeout"] == 30.0, (
            f"Expected timeout=30.0, got: {custom_server.params['timeout']}"
        )


# =============================================================================
# McpServer (Agents SDK) — Vector Search
# =============================================================================


@pytest.mark.integration
class TestMcpServerAgentsVectorSearch:
    """Test McpServer.from_vector_search() via Agents SDK (index-level + schema-level)."""

    def test_from_vector_search_call_tool(self, workspace_client):
        """List tools and call_tool on a VS tool via Agents SDK context manager."""
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
# Error Paths
# =============================================================================


@pytest.mark.integration
class TestMcpServerToolkitErrorPaths:
    """Test error handling in McpServerToolkit wrapper."""

    def test_nonexistent_function_raises_value_error(self, workspace_client):
        """get_tools() on a nonexistent function wraps the error in ValueError with server name."""
        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name="nonexistent_fn_xyz",
            name="error-test",
            workspace_client=workspace_client,
        )
        with pytest.raises(ValueError, match="Error listing tools from error-test MCP Server"):
            toolkit.get_tools()

    def test_execute_nonexistent_tool_raises_error(self, workspace_client):
        """call_tool with a nonexistent tool name raises ExceptionGroup > McpError(BAD_REQUEST)."""
        from mcp.shared.exceptions import McpError

        from databricks_openai import McpServerToolkit

        toolkit = McpServerToolkit.from_uc_function(
            catalog=CATALOG,
            schema=SCHEMA,
            function_name=FUNCTION_NAME,
            workspace_client=workspace_client,
        )
        with pytest.raises(ExceptionGroup) as exc_info:  # ty: ignore[unresolved-reference]
            toolkit.databricks_mcp_client.call_tool("nonexistent_tool_xyz", {"message": "test"})

        # Unwrap to find McpError
        def find_mcp_error(eg):
            for exc in eg.exceptions:
                if isinstance(exc, McpError):
                    return exc
                if isinstance(exc, ExceptionGroup):  # ty: ignore[unresolved-reference]
                    found = find_mcp_error(exc)
                    if found:
                        return found
            return None

        mcp_error = find_mcp_error(exc_info.value)
        assert mcp_error is not None, f"Expected McpError, got: {exc_info.value}"
        assert "BAD_REQUEST" in str(mcp_error), f"Expected BAD_REQUEST, got: {mcp_error}"

    def test_execute_wrong_arguments_raises_error(self, toolkit_tools):
        """execute() with wrong arguments raises ExceptionGroup > McpError(BAD_REQUEST)."""
        tool = toolkit_tools[0]
        # echo_message expects "message", we pass "wrong_arg"
        with pytest.raises(ExceptionGroup):  # ty: ignore[unresolved-reference]
            tool.execute(wrong_arg="test")


@pytest.mark.integration
class TestMcpServerAgentsErrorPaths:
    """Test error handling in McpServer (Agents SDK) wrapper."""

    def test_nonexistent_function_raises_mcp_error(self, workspace_client):
        """McpServer with a nonexistent function raises McpError(BAD_REQUEST: ... not found)."""
        from mcp.shared.exceptions import McpError

        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name="nonexistent_fn_xyz",
                workspace_client=workspace_client,
            ) as server:
                await server.list_tools()

        with pytest.raises(McpError, match="BAD_REQUEST"):
            asyncio.run(_test())

    def test_call_tool_nonexistent_tool_raises_mcp_error(self, workspace_client):
        """call_tool() with a malformed tool name raises McpError(BAD_REQUEST: ... malformed)."""
        from mcp.shared.exceptions import McpError

        from databricks_openai.agents import McpServer

        async def _test():
            async with McpServer.from_uc_function(
                catalog=CATALOG,
                schema=SCHEMA,
                function_name=FUNCTION_NAME,
                workspace_client=workspace_client,
            ) as server:
                await server.call_tool("nonexistent_tool_xyz", {"message": "test"})

        with pytest.raises(McpError, match="BAD_REQUEST"):
            asyncio.run(_test())

    def test_call_tool_wrong_arguments_raises_user_error(self, workspace_client):
        """call_tool() with wrong arguments raises UserError about missing parameters."""
        from agents.exceptions import UserError

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
                await server.call_tool(tool_name, {"wrong_arg": "test"})

        with pytest.raises(UserError, match="missing required parameters"):
            asyncio.run(_test())
