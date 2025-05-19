from unittest.mock import patch

import pytest
from databricks.sdk.service.dashboards import GenieSpace
from databricks_ai_bridge.genie import Genie
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


@patch("databricks.sdk.WorkspaceClient")
def test_query_genie_as_agent(MockWorkspaceClient):
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space
    MockWorkspaceClient.genie._api.do.side_effect = [
        {"conversation_id": "123", "message_id": "abc"},
        {"status": "COMPLETED", "attachments": [{"text": {"content": "It is sunny."}}]},
    ]

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}
    genie = Genie("space-id", MockWorkspaceClient)
    result = _query_genie_as_agent(input_data, genie, "Genie")

    expected_message = {"messages": [AIMessage(content="It is sunny.")]}
    assert result == expected_message


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
    mock_workspace_client.genie._api.do.side_effect = [
        {"conversation_id": "123", "message_id": "abc"},
        {"status": "COMPLETED", "attachments": [{"text": {"content": "It is sunny."}}]},
    ]

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}
    genie = Genie("space-id", mock_workspace_client)
    result = _query_genie_as_agent(input_data, genie, "Genie")

    expected_message = {"messages": [AIMessage(content="It is sunny.")]}
    assert result == expected_message


def test_create_genie_agent_no_space_id():
    with pytest.raises(ValueError):
        GenieAgent("", "Genie")
