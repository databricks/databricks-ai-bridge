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
        {"status": "COMPLETED", "attachments": [{"text": {"content": "Result"}}]},
    ]
    result = genie.poll_for_result("123", "456")
    assert result == "Result"


def test_poll_for_result_completed_with_query(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "COMPLETED", "attachments": [{"query": {"query": "SELECT *"}}]},
        {
            "statement_response": {
                "status": {"state": "SUCCEEDED"},
                "manifest": {"schema": {"columns": []}},
                "result": {
                    "data_typed_array": [],
                },
            }
        },
    ]
    result = genie.poll_for_result("123", "456")
    assert result == pd.DataFrame().to_markdown()


def test_poll_for_result_executing_query(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "EXECUTING_QUERY", "attachments": [{"query": {"query": "SELECT *"}}]},
        {
            "statement_response": {
                "status": {"state": "SUCCEEDED"},
                "manifest": {"schema": {"columns": []}},
                "result": {
                    "data_typed_array": [],
                },
            }
        },
    ]
    result = genie.poll_for_result("123", "456")
    assert result == pd.DataFrame().to_markdown()


def test_poll_for_result_failed(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "FAILED"},
    ]
    result = genie.poll_for_result("123", "456")
    assert result is None


def test_poll_for_result_cancelled(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "CANCELLED"},
    ]
    result = genie.poll_for_result("123", "456")
    assert result is None


def test_poll_for_result_expired(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"status": "QUERY_RESULT_EXPIRED"},
    ]
    result = genie.poll_for_result("123", "456")
    assert result is None


def test_poll_for_result_max_iterations(genie, mock_workspace_client):
    # patch MAX_ITERATIONS to 2 for this test and sleep to avoid delays
    with (
        patch("databricks_ai_bridge.genie.MAX_ITERATIONS", 2),
        patch("time.sleep", return_value=None),
    ):
        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY", "attachments": [{"query": {"query": "SELECT *"}}]},
            {
                "statement_response": {
                    "status": {"state": "RUNNING"},
                }
            },
            {
                "statement_response": {
                    "status": {"state": "RUNNING"},
                }
            },
            {
                "statement_response": {
                    "status": {"state": "RUNNING"},
                }
            },
        ]
        result = genie.poll_for_result("123", "456")
        assert result is None


def test_ask_question(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"conversation_id": "123", "message_id": "456"},
        {"status": "COMPLETED", "attachments": [{"text": {"content": "Answer"}}]},
    ]
    result = genie.ask_question("What is the meaning of life?")
    assert result == "Answer"


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
            "data_typed_array": [
                {"values": [{"str": "1"}, {"str": "Alice"}, {"str": "2023-10-01T00:00:00Z"}]},
                {"values": [{"str": "2"}, {"str": "Bob"}, {"str": "2023-10-02T00:00:00Z"}]},
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
            "data_typed_array": [
                {"values": [{"str": "1"}, {"str": None}, {"str": "2023-10-01T00:00:00Z"}]},
                {"values": [{"str": "2"}, {"str": "Bob"}, {"str": None}]},
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
                "data_typed_array": [
                    {"values": [{"str": "1"}, {"str": "Alice"}, {"str": "2023-10-01T00:00:00Z"}]},
                    {"values": [{"str": "2"}, {"str": "Bob"}, {"str": "2023-10-02T00:00:00Z"}]},
                    {"values": [{"str": "3"}, {"str": "Charlie"}, {"str": "2023-10-03T00:00:00Z"}]},
                    {"values": [{"str": "4"}, {"str": "David"}, {"str": "2023-10-04T00:00:00Z"}]},
                    {"values": [{"str": "5"}, {"str": "Eve"}, {"str": "2023-10-05T00:00:00Z"}]},
                    {"values": [{"str": "6"}, {"str": "Frank"}, {"str": "2023-10-06T00:00:00Z"}]},
                    {"values": [{"str": "7"}, {"str": "Grace"}, {"str": "2023-10-07T00:00:00Z"}]},
                    {"values": [{"str": "8"}, {"str": "Hank"}, {"str": "2023-10-08T00:00:00Z"}]},
                    {"values": [{"str": "9"}, {"str": "Ivy"}, {"str": "2023-10-09T00:00:00Z"}]},
                    {"values": [{"str": "10"}, {"str": "Jack"}, {"str": "2023-10-10T00:00:00Z"}]},
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
