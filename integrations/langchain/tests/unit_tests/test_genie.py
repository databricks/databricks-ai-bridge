from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.service.dashboards import GenieSpace
from databricks_ai_bridge.genie import Genie, GenieResponse
from langchain_core.messages import AIMessage
from mcp.types import CallToolResult

from databricks_langchain.genie import (
    GenieAgent,
    _concat_messages_array,
    _query_genie_as_agent,
)


@pytest.fixture(autouse=True)
def mock_databricks_oauth_provider():
    """Auto-mock DatabricksOAuthClientProvider for all tests to avoid OAuth validation errors."""
    with patch("databricks_mcp.mcp.DatabricksOAuthClientProvider") as mock_auth:
        # Return a MagicMock instance that won't try to get OAuth tokens
        mock_auth_instance = MagicMock()
        mock_auth.return_value = mock_auth_instance
        yield mock_auth


def test_concat_messages_array():
    # Test a simple case with multiple messages
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    result = _concat_messages_array(messages)
    expected = "user: What is the weather?\nassistant: It is sunny."
    assert result == expected

    # Test case with missing content
    messages = [{"role": "user"}, {"role": "assistant", "content": "I don't know."}]
    result = _concat_messages_array(messages)
    expected = "user: \nassistant: I don't know."
    assert result == expected

    # Test case with non-dict message objects
    class Message:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    messages = [
        Message("user", "Tell me a joke."),
        Message("assistant", "Why did the chicken cross the road?"),
    ]
    result = _concat_messages_array(messages)
    expected = "user: Tell me a joke.\nassistant: Why did the chicken cross the road?"
    assert result == expected


@patch("databricks.sdk.WorkspaceClient")
def test_query_genie_as_agent(MockWorkspaceClient):
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space

    # Create a proper GenieResponse instance with conversation_id
    mock_genie_response = GenieResponse(
        result="It is sunny.",
        query="SELECT * FROM weather",
        description="This is the reasoning for the query",
        conversation_id="conv-123",  # Add this
    )

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}
    genie = Genie("space-id", MockWorkspaceClient)

    # Mock the ask_question method at the module level to avoid mlflow tracing issues
    with patch("databricks_ai_bridge.genie.Genie.ask_question", return_value=mock_genie_response):
        # Test with include_context=False (default)
        result = _query_genie_as_agent(input_data, genie, "Genie")
        expected_message = {
            "messages": [AIMessage(content="It is sunny.", name="query_result")],
            "conversation_id": "conv-123",  # Add this
        }
        assert result == expected_message

        # Test with include_context=True
        result = _query_genie_as_agent(input_data, genie, "Genie", include_context=True)
        expected_messages = {
            "messages": [
                AIMessage(content="This is the reasoning for the query", name="query_reasoning"),
                AIMessage(content="SELECT * FROM weather", name="query_sql"),
                AIMessage(content="It is sunny.", name="query_result"),
            ],
            "conversation_id": "conv-123",  # Add this
        }
        assert result == expected_messages


@patch("databricks.sdk.WorkspaceClient")
@patch("langchain_core.runnables.RunnableLambda")
def test_create_genie_agent(MockRunnableLambda, MockWorkspaceClient):
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    mock_client = MockWorkspaceClient.return_value
    mock_client.genie.get_space.return_value = mock_space

    agent = GenieAgent("space-id", "Genie", client=mock_client)
    assert agent.description == "description"

    mock_client.genie.get_space.assert_called_once()
    assert agent == MockRunnableLambda.return_value


@patch("databricks.sdk.WorkspaceClient")
@patch("langchain_core.runnables.RunnableLambda")
def test_create_genie_agent_with_description(MockRunnableLambda, MockWorkspaceClient):
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description=None,
    )
    mock_client = MockWorkspaceClient.return_value
    mock_client.genie.get_space.return_value = mock_space

    agent = GenieAgent("space-id", "Genie", "this is a description", client=mock_client)
    assert agent.description == "this is a description"

    mock_client.genie.get_space.assert_called_once()
    assert agent == MockRunnableLambda.return_value


@patch("databricks.sdk.WorkspaceClient")
def test_query_genie_with_client(mock_workspace_client):
    mock_workspace_client.genie.get_space.return_value = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )

    # Create a proper GenieResponse instance
    mock_genie_response = GenieResponse(
        result="It is sunny.",
        query="SELECT weather FROM data",
        description="Query reasoning",
        conversation_id="conv-456",
    )

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}
    genie = Genie("space-id", mock_workspace_client)

    # Mock the ask_question method at the module level to avoid mlflow tracing issues
    with patch("databricks_ai_bridge.genie.Genie.ask_question", return_value=mock_genie_response):
        result = _query_genie_as_agent(input_data, genie, "Genie")
        expected_message = {
            "messages": [AIMessage(content="It is sunny.", name="query_result")],
            "conversation_id": "conv-456",
        }
        assert result == expected_message


@patch("databricks.sdk.WorkspaceClient")
def test_create_genie_agent_with_include_context(MockWorkspaceClient):
    """Test creating a GenieAgent with include_context parameter and verify it propagates correctly"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    mock_client = MockWorkspaceClient.return_value
    mock_client.genie.get_space.return_value = mock_space

    # Create a proper GenieResponse instance
    mock_genie_response = GenieResponse(
        result="It is sunny.",
        query="SELECT * FROM weather",
        description="This is the reasoning for the query",
        conversation_id="conv-789",
    )

    # Test with include_context=True
    agent = GenieAgent("space-id", "Genie", include_context=True, client=mock_client)
    assert agent.description == "description"

    # Create test input
    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}

    # Mock the ask_question method on the Genie instance
    with patch("databricks_ai_bridge.genie.Genie.ask_question", return_value=mock_genie_response):
        # Invoke the agent and verify include_context=True behavior
        result = agent.invoke(input_data)

        # Should include reasoning, SQL, and result when include_context=True
        expected_messages = [
            AIMessage(content="This is the reasoning for the query", name="query_reasoning"),
            AIMessage(content="SELECT * FROM weather", name="query_sql"),
            AIMessage(content="It is sunny.", name="query_result"),
        ]
        assert result["messages"] == expected_messages
        assert result["conversation_id"] == "conv-789"  # Add this

    mock_client.genie.get_space.assert_called_once()


def test_create_genie_agent_no_space_id():
    with pytest.raises(ValueError):
        GenieAgent("", "Genie")


@patch("databricks.sdk.WorkspaceClient")
def test_message_processor_functionality(MockWorkspaceClient):
    """Test message_processor parameter in both _query_genie_as_agent and GenieAgent"""
    mock_space = GenieSpace(space_id="space-id", title="Sales Space", description="description")
    MockWorkspaceClient.genie.get_space.return_value = mock_space

    mock_genie_response = GenieResponse(
        result="It is sunny.",
        query="SELECT * FROM weather",
        description="Query reasoning",
        conversation_id="conv-abc",  # Add this
    )

    # Test data with multiple messages
    input_data = {
        "messages": [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Assistant response"},
            {"role": "user", "content": "Second message"},
        ]
    }

    genie = Genie("space-id", MockWorkspaceClient)

    # Test 1: Custom message processor that concatenates all content
    def custom_processor(messages):
        contents = []
        for msg in messages:
            if isinstance(msg, dict):
                contents.append(msg.get("content", ""))
            else:
                contents.append(msg.content)
        return " | ".join(contents)

    with patch.object(genie, "ask_question", return_value=mock_genie_response) as mock_ask:
        result = _query_genie_as_agent(
            input_data, genie, "Genie", message_processor=custom_processor
        )
        expected_query = "First message | Assistant response | Second message"
        mock_ask.assert_called_with(expected_query, conversation_id=None)  # Update this
        assert result == {
            "messages": [AIMessage(content="It is sunny.", name="query_result")],
            "conversation_id": "conv-abc",  # Add this
        }

    # Test 2: Last message processor (as requested in the user's example)
    def last_message_processor(messages):
        if not messages:
            return ""
        last_msg = messages[-1]
        if isinstance(last_msg, dict):
            return last_msg.get("content", "")
        else:
            return last_msg.content

    with patch.object(genie, "ask_question", return_value=mock_genie_response) as mock_ask:
        result = _query_genie_as_agent(
            input_data, genie, "Genie", message_processor=last_message_processor
        )
        expected_query = "Second message"
        mock_ask.assert_called_with(expected_query, conversation_id=None)  # Update this
        assert result == {
            "messages": [AIMessage(content="It is sunny.", name="query_result")],
            "conversation_id": "conv-abc",  # Add this
        }

    # Test 4: GenieAgent end-to-end with message_processor
    with patch.object(genie, "ask_question", return_value=mock_genie_response) as mock_ask:
        agent = GenieAgent(
            "space-id",
            "Genie",
            message_processor=last_message_processor,
            client=MockWorkspaceClient,
        )

        with patch(
            "databricks_ai_bridge.genie.Genie.ask_question", return_value=mock_genie_response
        ) as mock_ask_agent:
            result = agent.invoke(input_data)
            expected_query = "Second message"
            mock_ask_agent.assert_called_once_with(
                expected_query, conversation_id=None
            )  # Update this
            assert result["messages"] == [AIMessage(content="It is sunny.", name="query_result")]
            assert result["conversation_id"] == "conv-abc"  # Add this


@patch("databricks.sdk.WorkspaceClient")
def test_conversation_continuity(MockWorkspaceClient):
    """Test that conversation_id is passed through correctly for conversation continuity"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space

    # First response creates a conversation
    mock_genie_response_1 = GenieResponse(
        result="First response",
        query="SELECT * FROM data",
        description="Query reasoning",
        conversation_id="conv-new-123",
    )

    # Second response continues the conversation
    mock_genie_response_2 = GenieResponse(
        result="Follow-up response",
        query="SELECT * FROM data WHERE region='NA'",
        description="Follow-up query",
        conversation_id="conv-new-123",
    )

    genie = Genie("space-id", MockWorkspaceClient)

    # First query - no conversation_id in input
    input_data_1 = {"messages": [{"role": "user", "content": "Show me data"}]}

    with patch.object(genie, "ask_question", return_value=mock_genie_response_1) as mock_ask:
        result_1 = _query_genie_as_agent(input_data_1, genie, "Genie")
        # Should be called with conversation_id=None
        mock_ask.assert_called_with(
            "I will provide you a chat history, where your name is Genie. Please help with the described information in the chat history.\nuser: Show me data",
            conversation_id=None,
        )
        assert result_1["conversation_id"] == "conv-new-123"

    # Second query - pass conversation_id from previous response
    input_data_2 = {
        "messages": [{"role": "user", "content": "Now filter by region"}],
        "conversation_id": result_1["conversation_id"],
    }

    with patch.object(genie, "ask_question", return_value=mock_genie_response_2) as mock_ask:
        result_2 = _query_genie_as_agent(input_data_2, genie, "Genie")
        # Should be called with the conversation_id from previous response
        mock_ask.assert_called_with(
            "I will provide you a chat history, where your name is Genie. Please help with the described information in the chat history.\nuser: Now filter by region",
            conversation_id="conv-new-123",
        )
        assert result_2["conversation_id"] == "conv-new-123"


@patch("databricks.sdk.WorkspaceClient")
def test_dataframe_return(MockWorkspaceClient):
    """Test that DataFrames are returned correctly with markdown conversion"""
    import pandas as pd

    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space

    # Create a DataFrame result
    test_df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

    mock_genie_response = GenieResponse(
        result=test_df,  # DataFrame result
        query="SELECT * FROM users",
        description="Query reasoning",
        conversation_id="conv-df-123",
    )

    input_data = {"messages": [{"role": "user", "content": "Show me users"}]}
    genie = Genie("space-id", MockWorkspaceClient, return_pandas=True)

    with patch.object(genie, "ask_question", return_value=mock_genie_response):
        result = _query_genie_as_agent(input_data, genie, "Genie")

        # Should have dataframe field
        assert "dataframe" in result
        assert isinstance(result["dataframe"], pd.DataFrame)
        assert result["dataframe"].equals(test_df)

        # Message content should be markdown
        assert isinstance(result["messages"][0].content, str)
        assert "Alice" in result["messages"][0].content
        assert "Bob" in result["messages"][0].content

        # Should have conversation_id
        assert result["conversation_id"] == "conv-df-123"


@patch("databricks.sdk.WorkspaceClient")
def test_string_return_no_dataframe_field(MockWorkspaceClient):
    """Test that string results don't include dataframe field"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space

    mock_genie_response = GenieResponse(
        result="String result",  # String, not DataFrame
        query="SELECT * FROM data",
        description="Query reasoning",
        conversation_id="conv-str-123",
    )

    input_data = {"messages": [{"role": "user", "content": "Show me data"}]}
    genie = Genie("space-id", MockWorkspaceClient)

    with patch.object(genie, "ask_question", return_value=mock_genie_response):
        result = _query_genie_as_agent(input_data, genie, "Genie")

        # Should NOT have dataframe field
        assert "dataframe" not in result

        # Message content should be the string
        assert result["messages"][0].content == "String result"
        assert result["conversation_id"] == "conv-str-123"
