from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from databricks_ai_bridge.genie import Genie, _count_tokens, _parse_query_result


@pytest.fixture
def mock_workspace_client():
    with patch("databricks_ai_bridge.genie.WorkspaceClient") as MockWorkspaceClient:
        mock_client = MockWorkspaceClient.return_value
        yield mock_client


@pytest.fixture
def genie(mock_workspace_client):
    return Genie(space_id="test_space_id")


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
    mock_workspace_client.genie._api.do.side_effect = [
        {
            "status": "COMPLETED",
            "attachments": [{"attachment_id": "123", "text": {"content": "Result"}}],
        },
    ]
    genie_result = genie.poll_for_result("123", "456")
    assert genie_result.result == "Result"


def test_poll_for_result_completed_with_query(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {
            "status": "COMPLETED",
            "attachments": [{"attachment_id": "123", "query": {"query": "SELECT *"}}],
        },
        {
            "statement_response": {
                "status": {"state": "SUCCEEDED"},
                "manifest": {"schema": {"columns": []}},
                "result": {
                    "data_array": [],
                },
            }
        },
    ]
    genie_result = genie.poll_for_result("123", "456")
    assert genie_result.result == pd.DataFrame().to_markdown()


def test_poll_for_result_failed(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "FAILED", "error": "Test error"},
    ]
    genie_result = genie.poll_for_result("123", "456")
    assert genie_result.result == "Genie query failed with error: Test error"


def test_poll_for_result_cancelled(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "CANCELLED"},
    ]
    genie_result = genie.poll_for_result("123", "456")
    assert genie_result.result == "Genie query cancelled."


def test_poll_for_result_expired(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "QUERY_RESULT_EXPIRED"},
    ]
    genie_result = genie.poll_for_result("123", "456")
    assert genie_result.result == "Genie query query_result_expired."


def test_poll_for_result_max_iterations(genie, mock_workspace_client):
    # patch MAX_ITERATIONS to 2 for this test and sleep to avoid delays
    with (
        patch("databricks_ai_bridge.genie.MAX_ITERATIONS", 2),
        patch("time.sleep", return_value=None),
    ):
        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "EXECUTING_QUERY"},
            {"status": "EXECUTING_QUERY"},
        ]
        result = genie.poll_for_result("123", "456")
        assert result.result == "Genie query timed out after 2 iterations of 5 seconds"


def test_ask_question(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"conversation_id": "123", "message_id": "456"},
        {"status": "COMPLETED", "attachments": [{"text": {"content": "Answer"}}]},
    ]
    genie_result = genie.ask_question("What is the meaning of life?")
    assert genie_result.result == "Answer"


def test_parse_query_result_empty():
    resp = {"manifest": {"schema": {"columns": []}}, "result": None}
    result = _parse_query_result(resp)
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
                ["1", "Alice", "2023-10-01T00:00:00Z"],
                ["2", "Bob", "2023-10-02T00:00:00Z"],
            ]
        },
    }
    result = _parse_query_result(resp)
    expected_df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "created_at": [datetime(2023, 10, 1).date(), datetime(2023, 10, 2).date()],
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
                ["1", None, "2023-10-01T00:00:00Z"],
                ["2", "Bob", None],
            ]
        },
    }
    result = _parse_query_result(resp)
    expected_df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": [None, "Bob"],
            "created_at": [datetime(2023, 10, 1).date(), None],
        }
    )
    assert result == expected_df.to_markdown()


def test_parse_query_result_trims_large_data():
    # patch MAX_TOKENS_OF_DATA to 100 for this test
    with patch("databricks_ai_bridge.genie.MAX_TOKENS_OF_DATA", 100):
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
                    ["1", "Alice", "2023-10-01T00:00:00Z"],
                    ["2", "Bob", "2023-10-02T00:00:00Z"],
                    ["3", "Charlie", "2023-10-03T00:00:00Z"],
                    ["4", "David", "2023-10-04T00:00:00Z"],
                    ["5", "Eve", "2023-10-05T00:00:00Z"],
                    ["6", "Frank", "2023-10-06T00:00:00Z"],
                    ["7", "Grace", "2023-10-07T00:00:00Z"],
                    ["8", "Hank", "2023-10-08T00:00:00Z"],
                    ["9", "Ivy", "2023-10-09T00:00:00Z"],
                    ["10", "Jack", "2023-10-10T00:00:00Z"],
                ]
            },
        }
        result = _parse_query_result(resp)
        assert (
            result
            == pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "name": ["Alice", "Bob", "Charlie"],
                    "created_at": [
                        datetime(2023, 10, 1).date(),
                        datetime(2023, 10, 2).date(),
                        datetime(2023, 10, 3).date(),
                    ],
                }
            ).to_markdown()
        )
        assert _count_tokens(result) <= 100


def test_poll_query_results_max_iterations(genie, mock_workspace_client):
    # patch MAX_ITERATIONS to 2 for this test and sleep to avoid delays
    with (
        patch("databricks_ai_bridge.genie.MAX_ITERATIONS", 2),
        patch("time.sleep", return_value=None),
    ):
        mock_workspace_client.genie._api.do.side_effect = [
            {
                "status": "COMPLETED",
                "attachments": [{"attachment_id": "123", "query": {"query": "SELECT *"}}],
            },
            {"statement_response": {"status": {"state": "PENDING"}}},
            {"statement_response": {"status": {"state": "PENDING"}}},
            {"statement_response": {"status": {"state": "PENDING"}}},
        ]
        result = genie.poll_for_result("123", "456")
        assert result.result == "Genie query for result timed out after 2 iterations of 5 seconds"
