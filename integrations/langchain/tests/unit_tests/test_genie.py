from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.genie import GenieResponse
from langchain_core.messages import AIMessage

from databricks_langchain.genie import (
    GenieAgent,
    _concat_messages_array,
    _query_genie_as_agent,
)


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


@patch("databricks_langchain.genie.Genie")
def test_query_genie_as_agent(MockGenie):
    # Mock the Genie class and its response
    mock_genie = MockGenie.return_value
    mock_genie.ask_question.return_value = GenieResponse(result="It is sunny.")

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}
    result = _query_genie_as_agent(input_data, "space-id", "Genie", None)

    expected_message = {"messages": [AIMessage(content="It is sunny.")]}
    assert result == expected_message

    # Test the case when genie_response is empty
    mock_genie.ask_question.return_value = GenieResponse(result=None)
    result = _query_genie_as_agent(input_data, "space-id", "Genie", None)

    expected_message = {"messages": [AIMessage(content="")]}
    assert result == expected_message


@patch("langchain_core.runnables.RunnableLambda")
def test_create_genie_agent(MockRunnableLambda):
    mock_runnable = MockRunnableLambda.return_value

    agent = GenieAgent("space-id", "Genie")
    assert agent == mock_runnable

    # Check that the partial function is created with the correct arguments
    MockRunnableLambda.assert_called()


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
def test_query_genie_with_client(mock_mcp_client_class):
    # Create a properly mocked workspace client
    mock_workspace_client = MagicMock(spec=WorkspaceClient)
    mock_workspace_client.config.host = "https://test.databricks.com"
    mock_workspace_client.config._header_factory = MagicMock()
    mock_workspace_client.genie.get_space.return_value = MagicMock(
        space_id="space-id", description="Test space"
    )

    # Mock the MCP client instance and its call_tool method
    mock_mcp_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = '{"content": "It is sunny.", "status": "COMPLETED"}'
    mock_response.content = [mock_content]
    mock_mcp_client_instance.call_tool.return_value = mock_response
    mock_mcp_client_class.return_value = mock_mcp_client_instance

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}
    result = _query_genie_as_agent(input_data, "space-id", "Genie", mock_workspace_client)

    expected_message = {"messages": [AIMessage(content="It is sunny.")]}
    assert result == expected_message


def test_create_genie_agent_no_space_id():
    with pytest.raises(ValueError):
        GenieAgent("", "Genie")
