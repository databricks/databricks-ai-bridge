import json
import random
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

import mlflow
import pandas as pd
import pytest
from mcp.types import CallToolResult, Tool

from databricks_ai_bridge.genie import Genie, _count_tokens, _parse_query_result


@pytest.fixture
def mock_workspace_client():
    with patch("databricks_ai_bridge.genie.WorkspaceClient") as MockWorkspaceClient:
        mock_client = MockWorkspaceClient.return_value
        yield mock_client


@pytest.fixture
def genie(mock_workspace_client):
    mock_query_tool = Mock(spec=Tool)
    mock_query_tool.name = "query_space_test_space_id"

    mock_poll_tool = Mock(spec=Tool)
    mock_poll_tool.name = "poll_response_test_space_id"

    with patch("databricks_ai_bridge.genie.DatabricksMCPClient") as MockMCPClient:
        mock_mcp_instance = MockMCPClient.return_value
        mock_mcp_instance.list_tools.return_value = [mock_query_tool, mock_poll_tool]

        genie_instance = Genie(space_id="test_space_id")
        genie_instance._mcp_client = mock_mcp_instance
        return genie_instance


def test_start_conversation(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.return_value = {"conversation_id": "123"}
    response = genie.start_conversation("Hello")
    assert response == {"conversation_id": "123"}
    mock_workspace_client.genie._api.do.assert_called_once_with(
        "POST",
        "/api/2.0/genie/spaces/test_space_id/start-conversation",
        body={"content": "Hello"},
        headers=genie.headers,
    )


def test_create_message(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.return_value = {"message_id": "456"}
    response = genie.create_message("123", "Hello again")
    assert response == {"message_id": "456"}
    mock_workspace_client.genie._api.do.assert_called_once_with(
        "POST",
        "/api/2.0/genie/spaces/test_space_id/conversations/123/messages",
        body={"content": "Hello again"},
        headers=genie.headers,
    )


def test_poll_for_result_completed_with_text(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Result"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "COMPLETED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        genie_result = genie.poll_for_result("123", "456")
        assert genie_result.result == "Result"
        assert genie_result.message_id == "456"


def test_poll_for_result_completed_with_query(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [
                                {
                                    "query": "SELECT *",
                                    "description": "Test query",
                                    "statement_response": {
                                        "status": {"state": "SUCCEEDED"},
                                        "manifest": {"schema": {"columns": []}},
                                        "result": {"data_array": []},
                                    },
                                }
                            ],
                            "textAttachments": [],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "COMPLETED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        genie_result = genie.poll_for_result("123", "456")
        assert genie_result.result == pd.DataFrame().to_markdown()
        assert genie_result.query == "SELECT *"
        assert genie_result.description == "Test query"


def test_poll_for_result_failed(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Message processing failed: Test error"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "FAILED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        genie_result = genie.poll_for_result("123", "456")
        assert genie_result.result == "Message processing failed: Test error"


def test_poll_for_result_cancelled(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Message processing failed: Cancelled"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "CANCELLED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        genie_result = genie.poll_for_result("123", "456")
        assert genie_result.result == "Message processing failed: Cancelled"


def test_poll_for_result_expired(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Message processing failed: Expired"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "QUERY_RESULT_EXPIRED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        genie_result = genie.poll_for_result("123", "456")
        assert genie_result.result == "Message processing failed: Expired"


def test_poll_for_result_max_iterations(genie, mock_workspace_client):
    # patch MAX_ITERATIONS to 2 for this test and sleep to avoid delays
    with (
        patch("databricks_ai_bridge.genie.MAX_ITERATIONS", 2),
        patch("databricks_ai_bridge.genie.ITERATION_FREQUENCY", 0.1),
        patch("time.sleep", return_value=None),
    ):
        mock_mcp_result = CallToolResult(
            content=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "content": {
                                "queryAttachments": [],
                                "textAttachments": ["Query is still processing"],
                            },
                            "conversationId": "123",
                            "messageId": "456",
                            "status": "RUNNING",
                        }
                    ),
                }
            ]
        )

        with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
            result = genie.poll_for_result("123", "456")
            assert "timed out" in result.result.lower()
            assert "2 iterations of 0.1 seconds" in result.result


def test_ask_question(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Answer"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "COMPLETED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        genie_result = genie.ask_question("What is the meaning of life?")
        assert genie_result.result == "Answer"
        assert genie_result.conversation_id == "123"


def test_ask_question_continued_conversation(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["42"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "COMPLETED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        genie_result = genie.ask_question("What is the meaning of life?", "123")
        assert genie_result.result == "42"
        assert genie_result.conversation_id == "123"


def test_ask_question_calls_mcp_without_conversation_id(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Answer"],
                        },
                        "conversationId": "new-123",
                        "messageId": "456",
                        "status": "COMPLETED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result) as mock_call:
        genie.ask_question("What is the meaning of life?")

        # Verify MCP client was called with correct args (no conversation_id)
        mock_call.assert_called_once_with(
            "query_space_test_space_id", {"query": "What is the meaning of life?"}
        )


def test_ask_question_calls_mcp_with_conversation_id(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Answer"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "COMPLETED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result) as mock_call:
        genie.ask_question("What is the meaning of life?", "123")

        # Verify MCP client was called with conversation_id included
        mock_call.assert_called_once_with(
            "query_space_test_space_id",
            {"query": "What is the meaning of life?", "conversation_id": "123"},
        )


def test_ask_question_returns_message_id(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Answer"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "COMPLETED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        result = genie.ask_question("What is the meaning of life?")
        assert result.message_id == "456"
        assert result.conversation_id == "123"
        assert result.result == "Answer"


def test_poll_for_result_returns_message_id(genie, mock_workspace_client):
    mock_mcp_result = CallToolResult(
        content=[
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "content": {
                            "queryAttachments": [],
                            "textAttachments": ["Result"],
                        },
                        "conversationId": "123",
                        "messageId": "456",
                        "status": "COMPLETED",
                    }
                ),
            }
        ]
    )

    with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
        result = genie.poll_for_result("123", "456")
        assert result.message_id == "456"
        assert result.conversation_id == "123"
        assert result.result == "Result"


def test_parse_query_result_empty():
    resp = {"manifest": {"schema": {"columns": []}}, "result": None}
    result = _parse_query_result(resp, truncate_results=True)
    assert result == "EMPTY"


def test_parse_query_result_with_data():
    resp = {
        "manifest": {
            "schema": {
                "columns": [
                    {"name": "id", "type_name": "INT"},
                    {"name": "name", "type_name": "STRING"},
                    {"name": "created_at", "type_name": "TIMESTAMP"},
                ]
            }
        },
        "result": {
            "data_array": [
                {
                    "values": [
                        {"string_value": "1"},
                        {"string_value": "Alice"},
                        {"string_value": "2023-10-01T00:00:00Z"},
                    ]
                },
                {
                    "values": [
                        {"string_value": "2"},
                        {"string_value": "Bob"},
                        {"string_value": "2023-10-02T00:00:00Z"},
                    ]
                },
            ]
        },
    }
    result = _parse_query_result(resp, truncate_results=True)
    expected_df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "created_at": [datetime(2023, 10, 1), datetime(2023, 10, 2)],
        }
    )
    assert result == expected_df.to_markdown()


def test_parse_query_result_with_null_values():
    resp = {
        "manifest": {
            "schema": {
                "columns": [
                    {"name": "id", "type_name": "INT"},
                    {"name": "name", "type_name": "STRING"},
                    {"name": "created_at", "type_name": "TIMESTAMP"},
                ]
            }
        },
        "result": {
            "data_array": [
                {"values": [{"string_value": "1"}, None, {"string_value": "2023-10-01T00:00:00Z"}]},
                {"values": [{"string_value": "2"}, {"string_value": "Bob"}, None]},
            ]
        },
    }
    result = _parse_query_result(resp, truncate_results=True)
    expected_df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": [None, "Bob"],
            "created_at": [datetime(2023, 10, 1), None],
        }
    )
    assert result == expected_df.to_markdown()


@pytest.mark.parametrize("truncate_results", [True, False])
def test_parse_query_result_trims_data(truncate_results):
    # patch MAX_TOKENS_OF_DATA to 100 for this test
    with patch("databricks_ai_bridge.genie.MAX_TOKENS_OF_DATA", 120):
        resp = {
            "manifest": {
                "schema": {
                    "columns": [
                        {"name": "id", "type_name": "INT"},
                        {"name": "name", "type_name": "STRING"},
                        {"name": "created_at", "type_name": "TIMESTAMP"},
                    ]
                }
            },
            "result": {
                "data_array": [
                    {
                        "values": [
                            {"string_value": "1"},
                            {"string_value": "Alice"},
                            {"string_value": "2023-10-01T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "2"},
                            {"string_value": "Bob"},
                            {"string_value": "2023-10-02T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "3"},
                            {"string_value": "Charlie"},
                            {"string_value": "2023-10-03T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "4"},
                            {"string_value": "David"},
                            {"string_value": "2023-10-04T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "5"},
                            {"string_value": "Eve"},
                            {"string_value": "2023-10-05T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "6"},
                            {"string_value": "Frank"},
                            {"string_value": "2023-10-06T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "7"},
                            {"string_value": "Grace"},
                            {"string_value": "2023-10-07T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "8"},
                            {"string_value": "Hank"},
                            {"string_value": "2023-10-08T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "9"},
                            {"string_value": "Ivy"},
                            {"string_value": "2023-10-09T00:00:00Z"},
                        ]
                    },
                    {
                        "values": [
                            {"string_value": "10"},
                            {"string_value": "Jack"},
                            {"string_value": "2023-10-10T00:00:00Z"},
                        ]
                    },
                ]
            },
        }
        result = _parse_query_result(resp, truncate_results=truncate_results)

        if truncate_results:
            assert (
                result
                == pd.DataFrame(
                    {
                        "id": [1, 2, 3],
                        "name": ["Alice", "Bob", "Charlie"],
                        "created_at": [
                            datetime(2023, 10, 1),
                            datetime(2023, 10, 2),
                            datetime(2023, 10, 3),
                        ],
                    }
                ).to_markdown()
            )
            assert _count_tokens(result) <= 120
        else:
            assert (
                result
                == pd.DataFrame(
                    {
                        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "name": [
                            "Alice",
                            "Bob",
                            "Charlie",
                            "David",
                            "Eve",
                            "Frank",
                            "Grace",
                            "Hank",
                            "Ivy",
                            "Jack",
                        ],
                        "created_at": [
                            datetime(2023, 10, 1),
                            datetime(2023, 10, 2),
                            datetime(2023, 10, 3),
                            datetime(2023, 10, 4),
                            datetime(2023, 10, 5),
                            datetime(2023, 10, 6),
                            datetime(2023, 10, 7),
                            datetime(2023, 10, 8),
                            datetime(2023, 10, 9),
                            datetime(2023, 10, 10),
                        ],
                    }
                ).to_markdown()
            )


def markdown_to_dataframe(markdown_str: str) -> pd.DataFrame:
    if markdown_str == "":
        return pd.DataFrame()

    lines = markdown_str.strip().splitlines()

    # Remove Markdown separator row (2nd line)
    lines = [line.strip().strip("|") for i, line in enumerate(lines) if i != 1]

    # Re-join cleaned lines and parse
    cleaned_markdown = "\n".join(lines)
    df = pd.read_csv(StringIO(cleaned_markdown), sep="|")

    # Strip whitespace from column names and values
    df.columns = [col.strip() for col in df.columns]
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Drop the first column
    df = df.drop(columns=[df.columns[0]])

    return df


@pytest.mark.parametrize("max_tokens", [1, 100, 1000, 2000, 8000, 10000, 15000, 19000, 100000])
def test_parse_query_result_trims_large_data(max_tokens):
    """
    Ensure _parse_query_result trims output to stay within token limits.
    """
    with patch("databricks_ai_bridge.genie.MAX_TOKENS_OF_DATA", max_tokens):
        base_date = datetime(2023, 1, 1)
        names = [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
            "Frank",
            "Grace",
            "Hank",
            "Ivy",
            "Jack",
        ]

        # Generate data in MCP format
        data_array = [
            {
                "values": [
                    {"string_value": str(i + 1)},
                    {"string_value": random.choice(names)},
                    {
                        "string_value": (
                            base_date + timedelta(days=random.randint(0, 365))
                        ).strftime("%Y-%m-%dT%H:%M:%SZ")
                    },
                ]
            }
            for i in range(1000)
        ]

        response = {
            "manifest": {
                "schema": {
                    "columns": [
                        {"name": "id", "type_name": "INT"},
                        {"name": "name", "type_name": "STRING"},
                        {"name": "created_at", "type_name": "TIMESTAMP"},
                    ]
                }
            },
            "result": {"data_array": data_array},
        }

        markdown_result = _parse_query_result(response, truncate_results=True)
        result_df = markdown_to_dataframe(markdown_result)

        expected_df = pd.DataFrame(
            {
                "id": [int(row["values"][0]["string_value"]) for row in data_array],
                "name": [row["values"][1]["string_value"] for row in data_array],
                "created_at": [
                    datetime.strptime(row["values"][2]["string_value"], "%Y-%m-%dT%H:%M:%SZ")
                    for row in data_array
                ],
            }
        )

        expected_markdown = (
            "" if len(result_df) == 0 else expected_df[: len(result_df)].to_markdown()
        )
        # Ensure result matches expected subset and respects token limit
        assert markdown_result == expected_markdown
        assert _count_tokens(markdown_result) <= max_tokens
        # Ensure adding one more row would exceed token limit or we're at full length
        next_row_exceeds = (
            _count_tokens(expected_df.iloc[: len(result_df) + 1].to_markdown()) > max_tokens
        )
        assert len(result_df) == len(expected_df) or next_row_exceeds


def test_poll_query_results_max_iterations(genie, mock_workspace_client):
    # patch MAX_ITERATIONS to 2 for this test and sleep to avoid delays
    with patch("databricks_ai_bridge.genie.MAX_ITERATIONS", 2):
        with patch("time.sleep", return_value=None):
            mock_responses = [
                CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {
                                        "queryAttachments": [],
                                        "textAttachments": ["Processing"],
                                    },
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                ),
                CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {
                                        "queryAttachments": [],
                                        "textAttachments": ["Still processing"],
                                    },
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                ),
                CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {
                                        "queryAttachments": [],
                                        "textAttachments": ["Should not reach this"],
                                    },
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                ),
            ]

            with patch.object(genie._mcp_client, "call_tool", side_effect=mock_responses):
                result = genie.poll_for_result("123", "456")

            assert result.result == "Genie query timed out after 2 iterations of 0.5 seconds"


def test_parse_query_result_with_timestamp_formats():
    resp = {
        "manifest": {"schema": {"columns": [{"name": "created_at", "type_name": "TIMESTAMP"}]}},
        "result": {
            "data_array": [
                {"values": [{"string_value": "2023-10-01T14:30:45"}]},  # %Y-%m-%dT%H:%M:%S
                {"values": [{"string_value": "2023-10-02 09:15:22"}]},  # %Y-%m-%d %H:%M:%S
                {"values": [{"string_value": "2023-10-03T16:45"}]},  # %Y-%m-%dT%H:%M
                {"values": [{"string_value": "2023-10-04 11:20"}]},  # %Y-%m-%d %H:%M
                {"values": [{"string_value": "2023-10-05T08"}]},  # %Y-%m-%dT%H
                {"values": [{"string_value": "2023-10-06 19"}]},  # %Y-%m-%d %H
                {"values": [{"string_value": "2023-10-07"}]},  # %Y-%m-%d
            ]
        },
    }
    result = _parse_query_result(resp, truncate_results=True)
    assert (
        result
        == pd.DataFrame(
            {
                "created_at": [
                    datetime(2023, 10, 1, 14, 30, 45),  # full timestamp
                    datetime(2023, 10, 2, 9, 15, 22),  # full timestamp with space
                    datetime(2023, 10, 3, 16, 45),  # hour and minute only
                    datetime(2023, 10, 4, 11, 20),  # hour and minute with space
                    datetime(2023, 10, 5, 8),  # hour only
                    datetime(2023, 10, 6, 19),  # hour only with space
                    datetime(2023, 10, 7),  # date only
                ],
            }
        ).to_markdown()
    )


def test_poll_for_result_creates_genie_timeline_span(genie, mock_workspace_client):
    with patch("mlflow.start_span") as mock_start_span:
        with patch("mlflow.tracking.MlflowClient") as MockClient:
            mock_span = MagicMock()
            mock_span.trace_id = "trace_123"
            mock_span.span_id = "span_456"
            mock_start_span.return_value.__enter__.return_value = mock_span

            mock_mcp_result = CallToolResult(
                content=[
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "content": {
                                    "queryAttachments": [],
                                    "textAttachments": ["Done"],
                                },
                                "conversationId": "123",
                                "messageId": "456",
                                "status": "COMPLETED",
                            }
                        ),
                    }
                ]
            )

            with patch.object(genie._mcp_client, "call_tool", return_value=mock_mcp_result):
                genie.poll_for_result("123", "456")

            mock_start_span.assert_called_once_with(name="genie_timeline", span_type="CHAIN")


def test_poll_for_result_span_create_on_status_change(genie, mock_workspace_client):
    with patch("mlflow.start_span") as mock_start_span:
        with patch("mlflow.tracking.MlflowClient") as MockClient:
            with patch("time.sleep", return_value=None):
                mock_client = MockClient.return_value
                mock_parent_span = MagicMock()
                mock_parent_span.trace_id = "trace_123"
                mock_parent_span.span_id = "parent_456"
                mock_start_span.return_value.__enter__.return_value = mock_parent_span

                mock_child_span = MagicMock()
                mock_child_span.span_id = "child_789"
                mock_client.start_span.return_value = mock_child_span

                mock_mcp_result1 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                )
                mock_mcp_result2 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Done"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "COMPLETED",
                                }
                            ),
                        }
                    ]
                )

                with patch.object(
                    genie._mcp_client, "call_tool", side_effect=[mock_mcp_result1, mock_mcp_result2]
                ):
                    genie.poll_for_result("123", "456")

                # Verify span was created for EXECUTING_QUERY
                mock_client.start_span.assert_called_once()
                start_kwargs = mock_client.start_span.call_args[1]
                assert start_kwargs["name"] == "executing_query"
                assert start_kwargs["trace_id"] == "trace_123"
                assert start_kwargs["parent_id"] == "parent_456"
                assert start_kwargs["span_type"] == "CHAIN"


def test_poll_for_result_span_close_on_status_change(genie, mock_workspace_client):
    with patch("mlflow.start_span") as mock_start_span:
        with patch("mlflow.tracking.MlflowClient") as MockClient:
            with patch("time.sleep", return_value=None):
                mock_client = MockClient.return_value
                mock_parent_span = MagicMock()
                mock_parent_span.trace_id = "trace_123"
                mock_parent_span.span_id = "parent_456"
                mock_start_span.return_value.__enter__.return_value = mock_parent_span

                mock_child_span = MagicMock()
                mock_child_span.span_id = "child_789"
                mock_client.start_span.return_value = mock_child_span

                mock_mcp_result1 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                )
                mock_mcp_result2 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Done"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "COMPLETED",
                                }
                            ),
                        }
                    ]
                )

                with patch.object(
                    genie._mcp_client, "call_tool", side_effect=[mock_mcp_result1, mock_mcp_result2]
                ):
                    genie.poll_for_result("123", "456")

                # Verify span was closed when transitioning to COMPLETED
                mock_client.end_span.assert_called_once()
                end_kwargs = mock_client.end_span.call_args[1]
                assert end_kwargs["trace_id"] == "trace_123"
                assert end_kwargs["span_id"] == "child_789"
                assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_no_duplicate_span_on_same_status(genie, mock_workspace_client):
    with patch("mlflow.start_span") as mock_start_span:
        with patch("mlflow.tracking.MlflowClient") as MockClient:
            with patch("time.sleep", return_value=None):
                mock_client = MockClient.return_value
                mock_parent_span = MagicMock()
                mock_parent_span.trace_id = "trace_123"
                mock_parent_span.span_id = "parent_456"
                mock_start_span.return_value.__enter__.return_value = mock_parent_span

                mock_child_span = MagicMock()
                mock_child_span.span_id = "child_789"
                mock_client.start_span.return_value = mock_child_span

                mock_mcp_result1 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                )
                mock_mcp_result2 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",  # duplicate status
                                }
                            ),
                        }
                    ]
                )
                mock_mcp_result3 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",  # duplicate status
                                }
                            ),
                        }
                    ]
                )
                mock_mcp_result4 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Done"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "COMPLETED",
                                }
                            ),
                        }
                    ]
                )

                with patch.object(
                    genie._mcp_client,
                    "call_tool",
                    side_effect=[
                        mock_mcp_result1,
                        mock_mcp_result2,
                        mock_mcp_result3,
                        mock_mcp_result4,
                    ],
                ):
                    genie.poll_for_result("123", "456")

                # should only create span once for EXECUTING_QUERY despite 3 status changes
                mock_client.start_span.assert_called_once()


def test_poll_for_result_cancelled_terminal_state(genie, mock_workspace_client):
    with patch("mlflow.start_span") as mock_start_span:
        with patch("mlflow.tracking.MlflowClient") as MockClient:
            with patch("time.sleep", return_value=None):
                mock_client = MockClient.return_value
                mock_parent_span = MagicMock()
                mock_parent_span.trace_id = "trace_123"
                mock_parent_span.span_id = "parent_456"
                mock_start_span.return_value.__enter__.return_value = mock_parent_span

                mock_child_span = MagicMock()
                mock_child_span.span_id = "child_789"
                mock_client.start_span.return_value = mock_child_span

                mock_mcp_result1 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                )
                mock_mcp_result2 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Message processing failed: Query cancelled"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "CANCELLED",
                                }
                            ),
                        }
                    ]
                )

                with patch.object(
                    genie._mcp_client, "call_tool", side_effect=[mock_mcp_result1, mock_mcp_result2]
                ):
                    result = genie.poll_for_result("123", "456")

                assert result.result == "Message processing failed: Query cancelled"
                mock_client.end_span.assert_called_once()
                end_kwargs = mock_client.end_span.call_args[1]
                assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_failed_terminal_state(genie, mock_workspace_client):
    with patch("mlflow.start_span") as mock_start_span:
        with patch("mlflow.tracking.MlflowClient") as MockClient:
            with patch("time.sleep", return_value=None):
                mock_client = MockClient.return_value
                mock_parent_span = MagicMock()
                mock_parent_span.trace_id = "trace_123"
                mock_parent_span.span_id = "parent_456"
                mock_start_span.return_value.__enter__.return_value = mock_parent_span

                mock_child_span = MagicMock()
                mock_child_span.span_id = "child_789"
                mock_client.start_span.return_value = mock_child_span

                mock_mcp_result1 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                )
                mock_mcp_result2 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Message processing failed: some error"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "FAILED",
                                }
                            ),
                        }
                    ]
                )

                with patch.object(
                    genie._mcp_client, "call_tool", side_effect=[mock_mcp_result1, mock_mcp_result2]
                ):
                    result = genie.poll_for_result("123", "456")

                assert result.result == "Message processing failed: some error"
                mock_client.end_span.assert_called_once()
                end_kwargs = mock_client.end_span.call_args[1]
                assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_query_result_expired_terminal_state(genie, mock_workspace_client):
    with patch("mlflow.start_span") as mock_start_span:
        with patch("mlflow.tracking.MlflowClient") as MockClient:
            with patch("time.sleep", return_value=None):
                mock_client = MockClient.return_value
                mock_parent_span = MagicMock()
                mock_parent_span.trace_id = "trace_123"
                mock_parent_span.span_id = "parent_456"
                mock_start_span.return_value.__enter__.return_value = mock_parent_span

                mock_child_span = MagicMock()
                mock_child_span.span_id = "child_789"
                mock_client.start_span.return_value = mock_child_span

                mock_mcp_result1 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "EXECUTING_QUERY",
                                }
                            ),
                        }
                    ]
                )
                mock_mcp_result2 = CallToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "content": {"queryAttachments": [], "textAttachments": ["Message processing failed: Result expired"]},
                                    "conversationId": "123",
                                    "messageId": "456",
                                    "status": "QUERY_RESULT_EXPIRED",
                                }
                            ),
                        }
                    ]
                )

                with patch.object(
                    genie._mcp_client, "call_tool", side_effect=[mock_mcp_result1, mock_mcp_result2]
                ):
                    result = genie.poll_for_result("123", "456")

                assert result.result == "Message processing failed: Result expired"
                mock_client.end_span.assert_called_once()
                end_kwargs = mock_client.end_span.call_args[1]
                assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_timeout_includes_timeout_attribute(genie, mock_workspace_client):
    with patch("databricks_ai_bridge.genie.MAX_ITERATIONS", 2):
        with patch("mlflow.start_span") as mock_start_span:
            with patch("mlflow.tracking.MlflowClient") as MockClient:
                with patch("time.sleep", return_value=None):
                    mock_client = MockClient.return_value
                    mock_parent_span = MagicMock()
                    mock_parent_span.trace_id = "trace_123"
                    mock_parent_span.span_id = "parent_456"
                    mock_start_span.return_value.__enter__.return_value = mock_parent_span

                    mock_child_span = MagicMock()
                    mock_child_span.span_id = "child_789"
                    mock_client.start_span.return_value = mock_child_span

                    mock_responses = [
                        CallToolResult(
                            content=[
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                            "conversationId": "123",
                                            "messageId": "456",
                                            "status": "EXECUTING_QUERY",
                                        }
                                    ),
                                }
                            ]
                        ),
                        CallToolResult(
                            content=[
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "content": {"queryAttachments": [], "textAttachments": ["Still processing"]},
                                            "conversationId": "123",
                                            "messageId": "456",
                                            "status": "EXECUTING_QUERY",
                                        }
                                    ),
                                }
                            ]
                        ),
                        CallToolResult(
                            content=[
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "content": {"queryAttachments": [], "textAttachments": ["Should not reach"]},
                                            "conversationId": "123",
                                            "messageId": "456",
                                            "status": "EXECUTING_QUERY",
                                        }
                                    ),
                                }
                            ]
                        ),
                    ]

                    with patch.object(genie._mcp_client, "call_tool", side_effect=mock_responses):
                        result = genie.poll_for_result("123", "456")

                    assert "timed out" in result.result
                    mock_client.end_span.assert_called_once()
                    end_kwargs = mock_client.end_span.call_args[1]
                    assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_continues_on_mlflow_tracing_exceptions(genie, mock_workspace_client):
    with (
        patch("mlflow.start_span") as mock_start_span,
        patch("mlflow.tracking.MlflowClient") as MockClient,
        patch("time.sleep", return_value=None),
    ):
        mock_client = MockClient.return_value
        mock_parent_span = MagicMock()
        mock_parent_span.trace_id = "trace_123"
        mock_parent_span.span_id = "parent_456"
        mock_start_span.return_value.__enter__.return_value = mock_parent_span

        mock_child_span = MagicMock()
        mock_child_span.span_id = "child_789"

        # make both start_span and end_span raise exceptions for comprehensiveness
        mock_client.start_span.side_effect = mlflow.exceptions.MlflowTracingException(
            "Tracing failed"
        )
        mock_client.end_span.side_effect = mlflow.exceptions.MlflowTracingException(
            "End span failed"
        )

        mock_responses = [
            CallToolResult(
                content=[
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "content": {"queryAttachments": [], "textAttachments": ["Processing"]},
                                "conversationId": "123",
                                "messageId": "456",
                                "status": "EXECUTING_QUERY",
                            }
                        ),
                    }
                ]
            ),
            CallToolResult(
                content=[
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "content": {"queryAttachments": [], "textAttachments": ["Success"]},
                                "conversationId": "123",
                                "messageId": "456",
                                "status": "COMPLETED",
                            }
                        ),
                    }
                ]
            ),
        ]

        with patch.object(genie._mcp_client, "call_tool", side_effect=mock_responses):
            result = genie.poll_for_result("123", "456")

        # should still complete successfully despite tracing failures
        assert result.result == "Success"
