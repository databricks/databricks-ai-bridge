from unittest.mock import MagicMock, patch

import pytest

from databricks_google_adk import DatabricksMcpToolset, create_databricks_mcp_toolset


@pytest.fixture
def mock_mcp_client():
    """Mock the DatabricksMCPClient."""
    with patch("databricks_google_adk.mcp.DatabricksMCPClient") as mock:
        mock_instance = MagicMock()

        # Create mock tools
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "First tool"
        mock_tool1.inputSchema = {"type": "object", "properties": {}}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Second tool"
        mock_tool2.inputSchema = {"type": "object", "properties": {}}

        mock_instance.list_tools.return_value = [mock_tool1, mock_tool2]

        # Mock call_tool result
        mock_result = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Tool result"
        mock_result.content = [mock_content]
        mock_instance.call_tool.return_value = mock_result

        mock_instance.get_databricks_resources.return_value = []

        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_workspace_client():
    """Mock the WorkspaceClient."""
    with patch("databricks_google_adk.mcp.WorkspaceClient") as mock:
        mock_instance = MagicMock()
        mock_instance.config.host = "https://test-workspace.databricks.com"
        mock.return_value = mock_instance
        yield mock_instance


class TestDatabricksMcpToolset:
    """Tests for DatabricksMcpToolset class."""

    def test_init(self, mock_mcp_client):
        """Test DatabricksMcpToolset initialization."""
        toolset = DatabricksMcpToolset(
            server_url="https://workspace.databricks.com/api/2.0/mcp/functions/cat/schema"
        )
        assert toolset._server_url == "https://workspace.databricks.com/api/2.0/mcp/functions/cat/schema"

    @pytest.mark.asyncio
    async def test_get_tools(self, mock_mcp_client):
        """Test get_tools returns tools from MCP server."""
        toolset = DatabricksMcpToolset(
            server_url="https://workspace.databricks.com/api/2.0/mcp/functions/cat/schema"
        )
        tools = await toolset.get_tools()

        assert len(tools) == 2
        mock_mcp_client.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_tools_caching(self, mock_mcp_client):
        """Test that tools are cached after first load."""
        toolset = DatabricksMcpToolset(
            server_url="https://workspace.databricks.com/api/2.0/mcp/functions/cat/schema"
        )

        await toolset.get_tools()
        await toolset.get_tools()

        # list_tools should only be called once due to caching
        assert mock_mcp_client.list_tools.call_count == 1

    @pytest.mark.asyncio
    async def test_get_tools_with_filter(self, mock_mcp_client):
        """Test get_tools with tool_filter."""
        toolset = DatabricksMcpToolset(
            server_url="https://workspace.databricks.com/api/2.0/mcp/functions/cat/schema",
            tool_filter=["tool1"],
        )
        tools = await toolset.get_tools()

        # Only tool1 should be returned
        assert len(tools) == 1

    def test_get_databricks_resources(self, mock_mcp_client):
        """Test get_databricks_resources delegates to MCP client."""
        toolset = DatabricksMcpToolset(
            server_url="https://workspace.databricks.com/api/2.0/mcp/functions/cat/schema"
        )
        resources = toolset.get_databricks_resources()

        mock_mcp_client.get_databricks_resources.assert_called_once()
        assert resources == []

    @pytest.mark.asyncio
    async def test_close(self, mock_mcp_client):
        """Test close method doesn't raise."""
        toolset = DatabricksMcpToolset(
            server_url="https://workspace.databricks.com/api/2.0/mcp/functions/cat/schema"
        )
        await toolset.close()  # Should not raise


class TestDatabricksMcpToolsetFactories:
    """Tests for DatabricksMcpToolset factory methods."""

    def test_for_uc_functions(self, mock_mcp_client, mock_workspace_client):
        """Test for_uc_functions factory method."""
        toolset = DatabricksMcpToolset.for_uc_functions(
            catalog="my_catalog",
            schema="my_schema",
        )

        expected_url = "https://test-workspace.databricks.com/api/2.0/mcp/functions/my_catalog/my_schema"
        assert toolset._server_url == expected_url

    def test_for_vector_search(self, mock_mcp_client, mock_workspace_client):
        """Test for_vector_search factory method."""
        toolset = DatabricksMcpToolset.for_vector_search(
            catalog="my_catalog",
            schema="my_schema",
        )

        expected_url = "https://test-workspace.databricks.com/api/2.0/mcp/vector-search/my_catalog/my_schema"
        assert toolset._server_url == expected_url

    def test_for_genie(self, mock_mcp_client, mock_workspace_client):
        """Test for_genie factory method."""
        toolset = DatabricksMcpToolset.for_genie(
            space_id="my-genie-space",
        )

        expected_url = "https://test-workspace.databricks.com/api/2.0/mcp/genie/my-genie-space"
        assert toolset._server_url == expected_url


class TestCreateDatabricksMcpToolset:
    """Tests for create_databricks_mcp_toolset factory function."""

    def test_create_uc_functions(self, mock_mcp_client, mock_workspace_client):
        """Test creating UC functions toolset."""
        toolset = create_databricks_mcp_toolset(
            "uc_functions",
            catalog="cat",
            schema="sch",
        )
        assert "mcp/functions/cat/sch" in toolset._server_url

    def test_create_vector_search(self, mock_mcp_client, mock_workspace_client):
        """Test creating Vector Search toolset."""
        toolset = create_databricks_mcp_toolset(
            "vector_search",
            catalog="cat",
            schema="sch",
        )
        assert "mcp/vector-search/cat/sch" in toolset._server_url

    def test_create_genie(self, mock_mcp_client, mock_workspace_client):
        """Test creating Genie toolset."""
        toolset = create_databricks_mcp_toolset(
            "genie",
            space_id="space-123",
        )
        assert "mcp/genie/space-123" in toolset._server_url

    def test_create_uc_functions_missing_params(self, mock_mcp_client):
        """Test that missing catalog/schema raises ValueError."""
        with pytest.raises(ValueError, match="catalog and schema are required"):
            create_databricks_mcp_toolset("uc_functions")

    def test_create_genie_missing_space_id(self, mock_mcp_client):
        """Test that missing space_id raises ValueError."""
        with pytest.raises(ValueError, match="space_id is required"):
            create_databricks_mcp_toolset("genie")

    def test_create_unknown_type(self, mock_mcp_client):
        """Test that unknown server_type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown server_type"):
            create_databricks_mcp_toolset("unknown_type")
