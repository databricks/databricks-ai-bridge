from unittest.mock import MagicMock, patch

import pytest
from databricks_ai_bridge.genie import GenieResponse
from google.adk.tools import FunctionTool

from databricks_google_adk import GenieTool, create_genie_tool


@pytest.fixture
def mock_genie():
    """Mock the Genie class."""
    with patch("databricks_google_adk.genie.Genie") as mock:
        mock_instance = MagicMock()
        mock_instance.description = "Test Genie Space"
        mock_instance.ask_question.return_value = GenieResponse(
            result="| Column1 | Column2 |\n|---------|---------|\\n| Value1  | Value2  |",
            query="SELECT * FROM table",
            description="Query description",
            conversation_id="conv-123",
        )
        mock.return_value = mock_instance
        yield mock_instance


class TestCreateGenieTool:
    """Tests for create_genie_tool function."""

    def test_create_genie_tool_returns_function_tool(self, mock_genie):
        """Test that create_genie_tool returns a FunctionTool."""
        tool = create_genie_tool(space_id="test-space")
        assert isinstance(tool, FunctionTool)

    def test_create_genie_tool_custom_name(self, mock_genie):
        """Test that custom tool name is respected."""
        tool = create_genie_tool(space_id="test-space", tool_name="my_genie")
        assert tool.func.__name__ == "my_genie"

    def test_create_genie_tool_custom_description(self, mock_genie):
        """Test that custom description is respected."""
        custom_desc = "Custom description"
        tool = create_genie_tool(space_id="test-space", tool_description=custom_desc)
        assert tool.func.__doc__ == custom_desc

    def test_create_genie_tool_uses_space_description(self, mock_genie):
        """Test that space description is used when no custom description provided."""
        tool = create_genie_tool(space_id="test-space")
        # Should use the mock's description
        assert tool.func.__doc__ == "Test Genie Space"


class TestGenieTool:
    """Tests for GenieTool class."""

    def test_genie_tool_init(self, mock_genie):
        """Test GenieTool initialization."""
        genie = GenieTool(space_id="test-space")
        assert genie.space_id == "test-space"
        assert genie.conversation_id is None

    def test_genie_tool_description_property(self, mock_genie):
        """Test description property."""
        genie = GenieTool(space_id="test-space")
        assert genie.description == "Test Genie Space"

    def test_genie_tool_as_tool(self, mock_genie):
        """Test as_tool method returns FunctionTool."""
        genie = GenieTool(space_id="test-space")
        tool = genie.as_tool()
        assert isinstance(tool, FunctionTool)

    def test_genie_tool_as_tool_caching(self, mock_genie):
        """Test that as_tool returns the same instance on repeated calls."""
        genie = GenieTool(space_id="test-space")
        tool1 = genie.as_tool()
        tool2 = genie.as_tool()
        assert tool1 is tool2

    def test_genie_tool_ask(self, mock_genie):
        """Test ask method returns expected format."""
        genie = GenieTool(space_id="test-space")
        result = genie.ask("What is the total?")

        assert "result" in result
        assert "query" in result
        assert "description" in result
        assert "conversation_id" in result
        assert result["conversation_id"] == "conv-123"

    def test_genie_tool_ask_updates_conversation_id(self, mock_genie):
        """Test that ask updates conversation_id."""
        genie = GenieTool(space_id="test-space")
        assert genie.conversation_id is None

        genie.ask("First question")
        assert genie.conversation_id == "conv-123"

    def test_genie_tool_reset_conversation(self, mock_genie):
        """Test reset_conversation clears conversation_id."""
        genie = GenieTool(space_id="test-space")
        genie.ask("First question")
        assert genie.conversation_id is not None

        genie.reset_conversation()
        assert genie.conversation_id is None

    def test_genie_tool_new_conversation_flag(self, mock_genie):
        """Test that new_conversation=True starts fresh conversation."""
        genie = GenieTool(space_id="test-space")
        genie._conversation_id = "old-conv"

        # With new_conversation=True, should pass None to ask_question
        genie.ask("New question", new_conversation=True)

        mock_genie.ask_question.assert_called_with("New question", conversation_id=None)

    def test_genie_tool_continues_conversation(self, mock_genie):
        """Test that conversation continues by default."""
        genie = GenieTool(space_id="test-space")
        genie._conversation_id = "existing-conv"

        genie.ask("Follow-up question", new_conversation=False)

        mock_genie.ask_question.assert_called_with(
            "Follow-up question", conversation_id="existing-conv"
        )

    def test_genie_tool_custom_tool_name(self, mock_genie):
        """Test custom tool name."""
        genie = GenieTool(space_id="test-space", tool_name="custom_genie")
        tool = genie.as_tool()
        assert tool.func.__name__ == "custom_genie"

    def test_genie_tool_handles_dataframe_result(self, mock_genie):
        """Test that DataFrame results are converted to markdown."""
        import pandas as pd

        mock_genie.ask_question.return_value = GenieResponse(
            result=pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]}),
            query="SELECT * FROM table",
            description="Query description",
            conversation_id="conv-123",
        )

        genie = GenieTool(space_id="test-space")
        result = genie.ask("Query with DataFrame")

        # Result should be markdown string, not DataFrame
        assert isinstance(result["result"], str)
        assert "col1" in result["result"]
        assert "col2" in result["result"]
