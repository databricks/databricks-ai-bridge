from unittest.mock import patch

from databricks_ai_bridge.genie import GenieResponse
from langchain_core.messages import AIMessage, HumanMessage

from databricks_langchain.genie import GenieAgent

import pytest


@pytest.fixture
def agent():
    return GenieAgent("id-1", "Genie")


@pytest.fixture
def agent_with_metadata():
    return GenieAgent("id-1", "Genie", return_metadata=True)


def test_concat_messages_array_base_messages(agent):
    messages = [HumanMessage("What is the weather?"), AIMessage("It is sunny.")]

    result = agent._concat_messages_array(messages)

    expected_result = "human: What is the weather?\nai: It is sunny."

    assert result == expected_result


def test_concat_messages_array(agent):
    # Test a simple case with multiple messages
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    result = agent._concat_messages_array(messages)
    expected = "user: What is the weather?\nassistant: It is sunny."
    assert result == expected

    # Test case with missing content
    messages = [{"role": "user"}, {"role": "assistant", "content": "I don't know."}]
    result = agent._concat_messages_array(messages)
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
    result = agent._concat_messages_array(messages)
    expected = "user: Tell me a joke.\nassistant: Why did the chicken cross the road?"
    assert result == expected


@patch("databricks_ai_bridge.genie.Genie.ask_question")
def test_query_genie_as_agent(mock_ask_question, agent):

    genie_response = GenieResponse(result="It is sunny.")

    mock_ask_question.return_value = genie_response

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}

    result = agent._query_genie_as_agent(input_data)

    expected_message = {"messages": [AIMessage(content="It is sunny.")]}

    assert result == expected_message

    # Test the case when genie_response is empty
    genie_empty_response = GenieResponse(result=None)

    mock_ask_question.return_value = genie_empty_response

    result = agent._query_genie_as_agent(input_data)

    expected_message = {"messages": [AIMessage(content="")]}

    assert result == expected_message


@patch("databricks_ai_bridge.genie.Genie.ask_question")
def test_query_genie_as_agent_with_metadata(mock_ask_question, agent_with_metadata):

    genie_response = GenieResponse(result="It is sunny.", query="select a from data_table", description="description")

    mock_ask_question.return_value = genie_response

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}

    result = agent_with_metadata._query_genie_as_agent(input_data)

    expected_message = {"messages": [AIMessage(content="It is sunny.")], "metadata": genie_response}

    assert result == expected_message

    # Test the case when genie_response is empty
    genie_empty_response = GenieResponse(result=None)

    mock_ask_question.return_value = genie_empty_response

    result = agent_with_metadata._query_genie_as_agent(input_data)

    expected_message = {"messages": [AIMessage(content="")], "metadata": None}

    assert result == expected_message


@patch("databricks_ai_bridge.genie.Genie.ask_question")
def test_query_genie_as_agent_invoke(mock_ask_question, agent):

    genie_response = GenieResponse(result="It is sunny.", query="select a from data_table", description="description")

    mock_ask_question.return_value = genie_response

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}

    result = agent.invoke(input_data)

    expected_message = {"messages": [AIMessage(content="It is sunny.")]}

    assert result == expected_message
