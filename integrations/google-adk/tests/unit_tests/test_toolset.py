from unittest.mock import MagicMock, patch

import pytest
from databricks_ai_bridge.genie import GenieResponse
from databricks_ai_bridge.test_utils.vector_search import (
    DELTA_SYNC_INDEX,
    mock_vs_client,  # noqa: F401
    mock_workspace_client,  # noqa: F401
)
from google.adk.tools import BaseTool, FunctionTool

from databricks_google_adk import DatabricksToolset


@pytest.fixture
def mock_genie():
    """Mock the Genie class for toolset tests."""
    with patch("databricks_google_adk.genie.Genie") as mock:
        mock_instance = MagicMock()
        mock_instance.description = "Test Genie Space"
        mock_instance.ask_question.return_value = GenieResponse(
            result="Test result",
            query="SELECT * FROM table",
            description="Query description",
            conversation_id="conv-123",
        )
        mock.return_value = mock_instance
        yield mock_instance


class TestDatabricksToolset:
    """Tests for DatabricksToolset class."""

    def test_empty_toolset(self, mock_genie):
        """Test creating an empty toolset."""
        toolset = DatabricksToolset()
        assert toolset._tools == []

    def test_toolset_with_vector_search(self, mock_genie):
        """Test creating a toolset with Vector Search indexes."""
        toolset = DatabricksToolset(
            vector_search_indexes=[DELTA_SYNC_INDEX],
        )
        assert len(toolset._tools) == 1
        assert isinstance(toolset._tools[0], FunctionTool)

    def test_toolset_with_genie(self, mock_genie):
        """Test creating a toolset with Genie spaces."""
        toolset = DatabricksToolset(
            genie_space_ids=["test-space-1"],
        )
        assert len(toolset._tools) == 1
        assert isinstance(toolset._tools[0], FunctionTool)

    def test_toolset_with_multiple_tools(self, mock_genie):
        """Test creating a toolset with multiple tools."""
        toolset = DatabricksToolset(
            vector_search_indexes=[DELTA_SYNC_INDEX],
            genie_space_ids=["test-space-1", "test-space-2"],
        )
        assert len(toolset._tools) == 3  # 1 VS + 2 Genie

    @pytest.mark.asyncio
    async def test_get_tools(self, mock_genie):
        """Test get_tools returns all tools."""
        toolset = DatabricksToolset(
            vector_search_indexes=[DELTA_SYNC_INDEX],
            genie_space_ids=["test-space-1"],
        )
        tools = await toolset.get_tools()
        assert len(tools) == 2

    def test_add_vector_search_tool(self, mock_genie):
        """Test add_vector_search_tool method."""
        toolset = DatabricksToolset()
        result = toolset.add_vector_search_tool(
            index_name=DELTA_SYNC_INDEX,
            tool_name="custom_search",
        )

        # Should return self for chaining
        assert result is toolset
        assert len(toolset._tools) == 1

    def test_add_genie_tool(self, mock_genie):
        """Test add_genie_tool method."""
        toolset = DatabricksToolset()
        result = toolset.add_genie_tool(
            space_id="test-space",
            tool_name="custom_genie",
        )

        # Should return self for chaining
        assert result is toolset
        assert len(toolset._tools) == 1

    def test_add_custom_tool(self, mock_genie):
        """Test add_custom_tool method."""

        def custom_func(x: str) -> str:
            return x

        custom_tool = FunctionTool(custom_func)

        toolset = DatabricksToolset()
        result = toolset.add_custom_tool(custom_tool)

        assert result is toolset
        assert len(toolset._tools) == 1
        assert toolset._tools[0] is custom_tool

    def test_method_chaining(self, mock_genie):
        """Test that builder methods can be chained."""
        toolset = (
            DatabricksToolset()
            .add_vector_search_tool(index_name=DELTA_SYNC_INDEX)
            .add_genie_tool(space_id="test-space")
        )

        assert len(toolset._tools) == 2

    @pytest.mark.asyncio
    async def test_get_tools_with_filter(self, mock_genie):
        """Test get_tools with tool_filter."""
        toolset = DatabricksToolset(
            vector_search_indexes=[DELTA_SYNC_INDEX],
            genie_space_ids=["test-space"],
            tool_filter=["genie_test_space"],  # Only include genie tool
        )

        tools = await toolset.get_tools()
        # Only the genie tool should be returned (filtered by name)
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_close(self, mock_genie):
        """Test close method doesn't raise."""
        toolset = DatabricksToolset()
        await toolset.close()  # Should not raise

    def test_genie_tool_name_formatting(self, mock_genie):
        """Test that Genie tool names handle dashes correctly."""
        toolset = DatabricksToolset(
            genie_space_ids=["space-with-dashes"],
        )

        # The tool name should have dashes replaced with underscores
        tool = toolset._tools[0]
        assert tool.func.__name__ == "genie_space_with_dashes"
