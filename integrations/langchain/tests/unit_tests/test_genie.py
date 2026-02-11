from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk.service.dashboards import GenieSpace
from databricks_ai_bridge.genie import Genie, GenieResponse, QueryAttachment
from langchain_core.messages import AIMessage

from databricks_langchain.genie import (
    GenieAgent,
    _concat_messages_array,
    _query_genie_as_agent,
)


def _mock_mcp_client():
    """Helper to create a mock MCP client for Genie tests"""
    mock_tool_query = MagicMock()
    mock_tool_query.name = "query_space_space-id"

    mock_tool_poll = MagicMock()
    mock_tool_poll.name = "poll_response_space-id"

    mock_mcp_client = MagicMock()
    mock_mcp_client.list_tools.return_value = [mock_tool_query, mock_tool_poll]

    return mock_mcp_client


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


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_query_genie_as_agent(MockWorkspaceClient, MockMCPClient):
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    # Create a proper GenieResponse instance
    mock_genie_response = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT weather_condition FROM weather_data WHERE date = TODAY()",
                description="Retrieving today's weather condition from the weather_data table",
                result="| weather_condition |\n|-------------------|\n| sunny             |",
            )
        ],
        text_attachments=["Based on the data, today's weather is sunny."],
        suggested_questions=["What about tomorrow?", "Show by city"],
        conversation_id="conv-123",
        message_id="msg-123",
    )

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}
    genie = Genie("space-id", MockWorkspaceClient)

    # Mock the ask_question method to return our mock response
    with patch.object(genie, "ask_question", return_value=mock_genie_response):
        # Test with include_context=False (default) - no suggested_questions
        result = _query_genie_as_agent(input_data, genie, "Genie")
        expected_result = {
            "messages": [
                AIMessage(
                    content="| weather_condition |\n|-------------------|\n| sunny             |\n\nBased on the data, today's weather is sunny.",
                    name="query_result",
                ),
            ],
            "conversation_id": "conv-123",
            "message_id": "msg-123",
        }
        assert result == expected_result
        # Verify no suggested_questions message when flag is off
        assert not any(m.name == "suggested_questions" for m in result["messages"])

        # Test with include_suggested_questions=True
        result = _query_genie_as_agent(input_data, genie, "Genie", include_suggested_questions=True)
        assert any(m.name == "suggested_questions" for m in result["messages"])
        sq_msg = [m for m in result["messages"] if m.name == "suggested_questions"][0]
        assert sq_msg.content == "What about tomorrow?\n\nShow by city"

        # Test with include_context=True
        result = _query_genie_as_agent(input_data, genie, "Genie", include_context=True)
        expected_result_with_context = {
            "messages": [
                AIMessage(
                    content="Retrieving today's weather condition from the weather_data table",
                    name="query_reasoning",
                ),
                AIMessage(
                    content="SELECT weather_condition FROM weather_data WHERE date = TODAY()",
                    name="query_sql",
                ),
                AIMessage(
                    content="| weather_condition |\n|-------------------|\n| sunny             |\n\nBased on the data, today's weather is sunny.",
                    name="query_result",
                ),
            ],
            "conversation_id": "conv-123",
            "message_id": "msg-123",
        }
        assert result == expected_result_with_context


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
@patch("langchain_core.runnables.RunnableLambda")
def test_create_genie_agent(MockRunnableLambda, MockWorkspaceClient, MockMCPClient):
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    mock_client = MockWorkspaceClient.return_value
    mock_client.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    agent = GenieAgent("space-id", "Genie", client=mock_client)
    assert agent.description == "description"

    mock_client.genie.get_space.assert_called_once()
    assert agent == MockRunnableLambda.return_value


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
@patch("langchain_core.runnables.RunnableLambda")
def test_create_genie_agent_with_description(
    MockRunnableLambda, MockWorkspaceClient, MockMCPClient
):
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description=None,
    )
    mock_client = MockWorkspaceClient.return_value
    mock_client.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    agent = GenieAgent("space-id", "Genie", "this is a description", client=mock_client)
    assert agent.description == "this is a description"

    mock_client.genie.get_space.assert_called_once()
    assert agent == MockRunnableLambda.return_value


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_query_genie_with_client(mock_workspace_client, MockMCPClient):
    mock_workspace_client.genie.get_space.return_value = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockMCPClient.return_value = _mock_mcp_client()

    # Create a proper GenieResponse instance
    mock_genie_response = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT weather FROM data",
                description="Query reasoning",
                result="| weather |\n|---------|\n| sunny   |",
            )
        ],
        text_attachments=["The current weather is sunny."],
        conversation_id="conv-456",
        message_id="msg-456",
    )

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}
    genie = Genie("space-id", mock_workspace_client)

    # Mock the ask_question method to return our mock response
    with patch.object(genie, "ask_question", return_value=mock_genie_response):
        result = _query_genie_as_agent(input_data, genie, "Genie")
        expected_result = {
            "messages": [
                AIMessage(
                    content="| weather |\n|---------|\n| sunny   |\n\nThe current weather is sunny.",
                    name="query_result",
                ),
            ],
            "conversation_id": "conv-456",
            "message_id": "msg-456",
        }
        assert result == expected_result


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_create_genie_agent_with_include_context(MockWorkspaceClient, MockMCPClient):
    """Test creating a GenieAgent with include_context parameter and verify it propagates correctly"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    mock_client = MockWorkspaceClient.return_value
    mock_client.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    # Create a proper GenieResponse instance
    mock_genie_response = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT * FROM weather",
                description="This is the reasoning for the query",
                result="| condition |\n|-----------|\n| sunny     |",
            )
        ],
        text_attachments=["Today's weather is sunny."],
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

        expected_result = {
            "messages": [
                AIMessage(
                    content="This is the reasoning for the query",
                    name="query_reasoning",
                ),
                AIMessage(
                    content="SELECT * FROM weather",
                    name="query_sql",
                ),
                AIMessage(
                    content="| condition |\n|-----------|\n| sunny     |\n\nToday's weather is sunny.",
                    name="query_result",
                ),
            ],
            "conversation_id": "conv-789",
            "message_id": "",
        }
        assert result == expected_result

    mock_client.genie.get_space.assert_called_once()


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_create_genie_agent_with_include_suggested_questions(MockWorkspaceClient, MockMCPClient):
    """Test creating a GenieAgent with include_suggested_questions and verify it propagates"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    mock_client = MockWorkspaceClient.return_value
    mock_client.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    mock_genie_response = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT * FROM weather",
                description="This is the reasoning for the query",
                result="| condition |\n|-----------|\n| sunny     |",
            )
        ],
        text_attachments=["Today's weather is sunny."],
        suggested_questions=["What about tomorrow?", "Show by region"],
        conversation_id="conv-789",
    )

    agent = GenieAgent("space-id", "Genie", include_suggested_questions=True, client=mock_client)
    assert agent.description == "description"

    input_data = {"messages": [{"role": "user", "content": "What is the weather?"}]}

    with patch("databricks_ai_bridge.genie.Genie.ask_question", return_value=mock_genie_response):
        result = agent.invoke(input_data)

        # Should have suggested_questions message since flag is True
        assert any(m.name == "suggested_questions" for m in result["messages"])
        sq_msg = [m for m in result["messages"] if m.name == "suggested_questions"][0]
        assert sq_msg.content == "What about tomorrow?\n\nShow by region"

    mock_client.genie.get_space.assert_called_once()


def test_create_genie_agent_no_space_id():
    with pytest.raises(ValueError):
        GenieAgent("", "Genie")


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_message_processor_functionality(MockWorkspaceClient, MockMCPClient):
    """Test message_processor parameter in both _query_genie_as_agent and GenieAgent"""
    mock_space = GenieSpace(space_id="space-id", title="Sales Space", description="description")
    MockWorkspaceClient.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    mock_genie_response = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT * FROM weather",
                description="Query reasoning",
                result="| weather |\n|---------|\n| sunny   |",
            )
        ],
        text_attachments=["It is sunny today."],
        conversation_id="conv-abc",
        message_id="msg-abc",
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
        mock_ask.assert_called_with(expected_query, conversation_id=None)
        assert result == {
            "messages": [
                AIMessage(
                    content="| weather |\n|---------|\n| sunny   |\n\nIt is sunny today.",
                    name="query_result",
                ),
            ],
            "conversation_id": "conv-abc",
            "message_id": "msg-abc",
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
        mock_ask.assert_called_with(expected_query, conversation_id=None)
        assert result == {
            "messages": [
                AIMessage(
                    content="| weather |\n|---------|\n| sunny   |\n\nIt is sunny today.",
                    name="query_result",
                ),
            ],
            "conversation_id": "conv-abc",
            "message_id": "msg-abc",
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
            mock_ask_agent.assert_called_once_with(expected_query, conversation_id=None)
            assert result == {
                "messages": [
                    AIMessage(
                        content="| weather |\n|---------|\n| sunny   |\n\nIt is sunny today.",
                        name="query_result",
                    ),
                ],
                "conversation_id": "conv-abc",
                "message_id": "msg-abc",
            }


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_conversation_continuity(MockWorkspaceClient, MockMCPClient):
    """Test that conversation_id is passed through correctly for conversation continuity"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    # First response creates a conversation
    mock_genie_response_1 = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT region, SUM(revenue) FROM sales GROUP BY region",
                description="Aggregating revenue by region",
                result="| region | revenue |\n|--------|--------|\n| NA     | 5000   |\n| EU     | 3000   |",
            )
        ],
        text_attachments=["Here is the revenue breakdown by region."],
        conversation_id="conv-new-123",
    )

    # Second response continues the conversation
    mock_genie_response_2 = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT * FROM sales WHERE region='NA'",
                description="Filtering sales data for North America",
                result="| product | revenue |\n|---------|--------|\n| Widget  | 3000   |\n| Gadget  | 2000   |",
            )
        ],
        text_attachments=["Here are the NA sales details."],
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


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_dataframe_return(MockWorkspaceClient, MockMCPClient):
    """Test that DataFrames are returned correctly with markdown conversion"""
    import pandas as pd

    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    # Create DataFrame results
    test_df1 = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    test_df2 = pd.DataFrame({"city": ["NYC", "LA"], "population": [8000000, 4000000]})

    mock_genie_response = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT * FROM users",
                description="Fetching user data",
                result=test_df1,
            ),
            QueryAttachment(
                query="SELECT * FROM cities",
                description="Fetching city data",
                result=test_df2,
            ),
        ],
        text_attachments=["Here are the users and cities from the database."],
        conversation_id="conv-df-123",
    )

    input_data = {"messages": [{"role": "user", "content": "Show me users and cities"}]}
    genie = Genie("space-id", MockWorkspaceClient, return_pandas=True)

    with patch.object(genie, "ask_question", return_value=mock_genie_response):
        result = _query_genie_as_agent(input_data, genie, "Genie")

        # Should have dataframes field (plural, list)
        assert "dataframes" in result
        assert isinstance(result["dataframes"], list)
        assert len(result["dataframes"]) == 2
        assert result["dataframes"][0].equals(test_df1)
        assert result["dataframes"][1].equals(test_df2)

        # Message content should be markdown with all data + summary
        assert isinstance(result["messages"][0].content, str)
        assert "Alice" in result["messages"][0].content
        assert "Bob" in result["messages"][0].content
        assert "NYC" in result["messages"][0].content
        assert "Here are the users" in result["messages"][0].content

        # Should have conversation_id
        assert result["conversation_id"] == "conv-df-123"


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_string_return_no_dataframes_field(MockWorkspaceClient, MockMCPClient):
    """Test that string results don't include dataframes field"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    mock_genie_response = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT * FROM data",
                description="Query reasoning",
                result="| col1 | col2 |\n|------|------|\n| a    | b    |",
            )
        ],
        text_attachments=["Here is the data you requested."],
        conversation_id="conv-str-123",
    )

    input_data = {"messages": [{"role": "user", "content": "Show me data"}]}
    genie = Genie("space-id", MockWorkspaceClient)

    with patch.object(genie, "ask_question", return_value=mock_genie_response):
        result = _query_genie_as_agent(input_data, genie, "Genie")

        # Should NOT have dataframes field (string results, not DataFrames)
        assert "dataframes" not in result

        # Message content should contain the table and summary
        assert "col1" in result["messages"][0].content
        assert "Here is the data" in result["messages"][0].content
        assert result["conversation_id"] == "conv-str-123"


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_multiple_attachments_concatenated(MockWorkspaceClient, MockMCPClient):
    """Test that multiple query and text attachments are concatenated into messages"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    # Create a GenieResponse with multiple attachments
    # query_attachment.result = actual query data, text_attachments = Genie's summaries
    mock_genie_response = GenieResponse(
        query_attachments=[
            QueryAttachment(
                query="SELECT * FROM sales",
                description="Fetching sales data for Q1",
                result="| quarter | revenue |\n|---------|--------|\n| Q1      | 1000   |",
            ),
            QueryAttachment(
                query="SELECT * FROM expenses",
                description="Fetching expense data for Q1",
                result="| quarter | cost |\n|---------|------|\n| Q1      | 500  |",
            ),
        ],
        text_attachments=[
            "Based on the data, Q1 had $1000 in revenue.",
            "Expenses for Q1 were $500, resulting in $500 profit.",
        ],
        suggested_questions=["What about Q2?", "Break down by department"],
        conversation_id="conv-multi-123",
        message_id="msg-multi-123",
    )

    input_data = {"messages": [{"role": "user", "content": "Show me Q1 financials"}]}
    genie = Genie("space-id", MockWorkspaceClient)

    with patch.object(genie, "ask_question", return_value=mock_genie_response):
        # Test without include_context - only result
        result = _query_genie_as_agent(input_data, genie, "Genie")

        # Result should be concatenated (1 message): results first, then summaries
        assert len(result["messages"]) == 1
        assert result["messages"][0].name == "query_result"
        assert "revenue" in result["messages"][0].content
        assert "cost" in result["messages"][0].content
        assert "$1000 in revenue" in result["messages"][0].content
        assert "$500 profit" in result["messages"][0].content

        # Test with include_context - reasoning + sql + result
        result_with_context = _query_genie_as_agent(
            input_data, genie, "Genie", include_context=True
        )

        # Should have 3 messages: reasoning, sql, result
        assert len(result_with_context["messages"]) == 3

        # Reasoning should be concatenated
        reasoning_msg = result_with_context["messages"][0]
        assert reasoning_msg.name == "query_reasoning"
        assert "Fetching sales data" in reasoning_msg.content
        assert "Fetching expense data" in reasoning_msg.content

        # SQL should be concatenated
        sql_msg = result_with_context["messages"][1]
        assert sql_msg.name == "query_sql"
        assert "SELECT * FROM sales" in sql_msg.content
        assert "SELECT * FROM expenses" in sql_msg.content

        # Result should be concatenated
        result_msg = result_with_context["messages"][2]
        assert result_msg.name == "query_result"

        # Test include_suggested_questions=False (default) — no suggested_questions message
        assert not any(m.name == "suggested_questions" for m in result["messages"])

        # Test include_suggested_questions=True
        result_with_sq = _query_genie_as_agent(
            input_data, genie, "Genie", include_suggested_questions=True
        )
        assert any(m.name == "suggested_questions" for m in result_with_sq["messages"])
        sq_msg = [m for m in result_with_sq["messages"] if m.name == "suggested_questions"][0]
        assert sq_msg.content == "What about Q2?\n\nBreak down by department"


@patch("databricks_ai_bridge.genie.DatabricksMCPClient")
@patch("databricks.sdk.WorkspaceClient")
def test_text_only_response(MockWorkspaceClient, MockMCPClient):
    """Test handling of text-only responses (no query attachments) - Genie couldn't run a query"""
    mock_space = GenieSpace(
        space_id="space-id",
        title="Sales Space",
        description="description",
    )
    MockWorkspaceClient.genie.get_space.return_value = mock_space
    MockMCPClient.return_value = _mock_mcp_client()

    # Create a GenieResponse with only text attachments (Genie couldn't find relevant data)
    mock_genie_response = GenieResponse(
        query_attachments=[],
        text_attachments=[
            "I cannot answer that question based on the available data in this space.",
        ],
        suggested_questions=["Try asking about sales", "Ask about revenue"],
        conversation_id="conv-text-123",
        message_id="msg-text-123",
    )

    input_data = {"messages": [{"role": "user", "content": "What is the meaning of life?"}]}
    genie = Genie("space-id", MockWorkspaceClient)

    with patch.object(genie, "ask_question", return_value=mock_genie_response):
        result = _query_genie_as_agent(input_data, genie, "Genie")

        # Should have 1 result message with Genie's text response
        assert len(result["messages"]) == 1
        assert result["messages"][0].name == "query_result"
        assert "cannot answer that question" in result["messages"][0].content

        # No dataframes field (no query results)
        assert "dataframes" not in result

        # Test with include_context - should not have reasoning/sql since no query attachments
        result_with_context = _query_genie_as_agent(
            input_data, genie, "Genie", include_context=True
        )
        # Only result message (no reasoning or sql since query_attachments is empty)
        assert len(result_with_context["messages"]) == 1
        assert result_with_context["messages"][0].name == "query_result"

        # Test include_suggested_questions=False (default) — no suggested_questions message
        assert not any(m.name == "suggested_questions" for m in result["messages"])

        # Test include_suggested_questions=True
        result_with_sq = _query_genie_as_agent(
            input_data, genie, "Genie", include_suggested_questions=True
        )
        assert any(m.name == "suggested_questions" for m in result_with_sq["messages"])
        sq_msg = [m for m in result_with_sq["messages"] if m.name == "suggested_questions"][0]
        assert sq_msg.content == "Try asking about sales\n\nAsk about revenue"
