import random
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest

from databricks_ai_bridge.genie import (
    Genie,
    _count_tokens,
    _extract_suggested_questions_from_attachment,
    _extract_text_attachment_content_from_attachment,
    _parse_attachments,
    _parse_query_result,
)


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
        patch("databricks_ai_bridge.genie.ITERATION_FREQUENCY", 0.1),
        patch("time.sleep", return_value=None),
    ):
        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "EXECUTING_QUERY"},
            {"status": "EXECUTING_QUERY"},
        ]
        result = genie.poll_for_result("123", "456")
        assert (
            result.result
            == "Genie query timed out after 2 iterations of 0.1 seconds (total 0.2 seconds)"
        )


def test_ask_question(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"conversation_id": "123", "message_id": "456"},
        {"status": "COMPLETED", "attachments": [{"text": {"content": "Answer"}}]},
    ]
    genie_result = genie.ask_question("What is the meaning of life?")
    assert genie_result.result == "Answer"
    assert genie_result.conversation_id == "123"


def test_ask_question_continued_conversation(genie, mock_workspace_client):
    mock_workspace_client.genie._api.do.side_effect = [
        {"conversation_id": "123", "message_id": "456"},
        {"status": "COMPLETED", "attachments": [{"text": {"content": "42"}}]},
    ]
    genie_result = genie.ask_question("What is the meaning of life?", "123")
    assert genie_result.result == "42"
    assert genie_result.conversation_id == "123"


def test_ask_question_calls_start_once_and_not_create_on_new(genie, mock_workspace_client):
    # arrange
    with (
        patch.object(genie, "create_message") as mock_create_message,
        patch.object(genie, "start_conversation") as mock_start_conversation,
        patch.object(genie, "poll_for_result") as mock_poll_for_result,
    ):
        # act
        genie.ask_question("What is the meaning of life?")

        # assert
        mock_create_message.assert_not_called()
        mock_start_conversation.assert_called_once()


def test_ask_question_calls_create_once_and_not_start_on_continue(genie, mock_workspace_client):
    # arrange
    with (
        patch.object(genie, "create_message") as mock_create_message,
        patch.object(genie, "start_conversation") as mock_start_conversation,
        patch.object(genie, "poll_for_result") as mock_poll_for_result,
    ):
        # act
        genie.ask_question("What is the meaning of life?", "123")

        # assert
        mock_create_message.assert_called_once()
        mock_start_conversation.assert_not_called()


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
                ["1", "Alice", "2023-10-01T00:00:00Z"],
                ["2", "Bob", "2023-10-02T00:00:00Z"],
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
                ["1", None, "2023-10-01T00:00:00Z"],
                ["2", "Bob", None],
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
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

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

        data_array = [
            [
                str(i + 1),
                random.choice(names),
                (base_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ]
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
        assert isinstance(markdown_result, str)  # return_pandas=False by default
        result_df = markdown_to_dataframe(markdown_result)

        expected_df = pd.DataFrame(
            {
                "id": [int(row[0]) for row in data_array],
                "name": [row[1] for row in data_array],
                "created_at": [
                    datetime.strptime(row[2], "%Y-%m-%dT%H:%M:%SZ") for row in data_array
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


def test_parse_query_result_preserves_float_precision():
    """
    Ensure large and small float values are not converted to scientific notation.
    """
    resp = {
        "manifest": {
            "schema": {
                "columns": [
                    {"name": "id", "type_name": "INT"},
                    {"name": "large_float", "type_name": "FLOAT"},
                    {"name": "small_float", "type_name": "DOUBLE"},
                    {"name": "decimal_val", "type_name": "DECIMAL"},
                ]
            }
        },
        "result": {
            "data_array": [
                ["1", "123456789012.45", "7800000000.0", "0.000000123456"],
                ["2", "9876543210987.65", "1400000000.0", "0.000000987654"],
            ]
        },
    }
    result = _parse_query_result(resp, truncate_results=False, return_pandas=False)

    # Verify large numbers are NOT in scientific notation
    assert "123456789012.45" in result
    assert "9876543210987.65" in result
    assert "7800000000" in result
    assert "1400000000" in result

    # Verify small numbers are NOT in scientific notation
    assert "0.000000123456" in result
    assert "0.000000987654" in result

    # Verify scientific notation is NOT present
    assert "e+" not in result.lower()
    assert "e-" not in result.lower()


def test_parse_query_result_returns_floats_for_pandas():
    """
    Ensure return_pandas=True still returns proper float types for computation.
    """
    resp = {
        "manifest": {
            "schema": {
                "columns": [
                    {"name": "id", "type_name": "INT"},
                    {"name": "value", "type_name": "FLOAT"},
                ]
            }
        },
        "result": {
            "data_array": [
                ["1", "123456789012.45"],
                ["2", "0.000000123456"],
            ]
        },
    }
    result = _parse_query_result(resp, truncate_results=False, return_pandas=True)

    assert isinstance(result, pd.DataFrame)
    assert result["value"].dtype == "float64"
    assert result["value"].iloc[0] == pytest.approx(123456789012.45, rel=1e-6)
    assert result["value"].iloc[1] == pytest.approx(0.000000123456, rel=1e-6)


def test_parse_query_result_preserves_float_precision_with_truncate():
    """
    Ensure truncate_results=True path also avoids scientific notation.
    """
    resp = {
        "manifest": {
            "schema": {
                "columns": [
                    {"name": "id", "type_name": "INT"},
                    {"name": "large_float", "type_name": "FLOAT"},
                    {"name": "small_float", "type_name": "DOUBLE"},
                ]
            }
        },
        "result": {
            "data_array": [
                ["1", "123456789012.45", "0.000000123456"],
                ["2", "9876543210987.65", "0.000000987654"],
            ]
        },
    }
    result = _parse_query_result(resp, truncate_results=True, return_pandas=False)

    # Verify values are NOT in scientific notation
    assert "123456789012.45" in result
    assert "0.000000123456" in result
    assert "e+" not in result.lower()
    assert "e-" not in result.lower()


def test_parse_query_result_with_timestamp_formats():
    resp = {
        "manifest": {"schema": {"columns": [{"name": "created_at", "type_name": "TIMESTAMP"}]}},
        "result": {
            "data_array": [
                ["2023-10-01T14:30:45"],  # %Y-%m-%dT%H:%M:%S
                ["2023-10-02 09:15:22"],  # %Y-%m-%d %H:%M:%S
                ["2023-10-03T16:45"],  # %Y-%m-%dT%H:%M
                ["2023-10-04 11:20"],  # %Y-%m-%d %H:%M
                ["2023-10-05T08"],  # %Y-%m-%dT%H
                ["2023-10-06 19"],  # %Y-%m-%d %H
                ["2023-10-07"],  # %Y-%m-%d
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
    with (
        patch("mlflow.start_span") as mock_start_span,
        patch("mlflow.tracking.MlflowClient") as MockClient,
    ):
        mock_span = MagicMock()
        mock_span.trace_id = "trace_123"
        mock_span.span_id = "span_456"
        mock_start_span.return_value.__enter__.return_value = mock_span

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "COMPLETED", "attachments": [{"text": {"content": "Done"}}]},
        ]

        genie.poll_for_result("123", "456")

        # Check that genie_timeline span was created (there are 2 spans: poll_result and genie_timeline)
        mock_start_span.assert_any_call(name="genie_timeline", span_type="CHAIN")


def test_poll_for_result_span_create_on_status_change(genie, mock_workspace_client):
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
        mock_client.start_span.return_value = mock_child_span

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "COMPLETED", "attachments": [{"text": {"content": "Done"}}]},
        ]

        genie.poll_for_result("123", "456")

        # Verify span was created for EXECUTING_QUERY
        mock_client.start_span.assert_called_once()
        start_kwargs = mock_client.start_span.call_args[1]
        assert start_kwargs["name"] == "executing_query"
        assert start_kwargs["trace_id"] == "trace_123"
        assert start_kwargs["parent_id"] == "parent_456"
        assert start_kwargs["span_type"] == "CHAIN"


def test_poll_for_result_span_close_on_status_change(genie, mock_workspace_client):
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
        mock_client.start_span.return_value = mock_child_span

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "COMPLETED", "attachments": [{"text": {"content": "Done"}}]},
        ]

        genie.poll_for_result("123", "456")

        # Verify span was closed when transitioning to COMPLETED
        mock_client.end_span.assert_called_once()
        end_kwargs = mock_client.end_span.call_args[1]
        assert end_kwargs["trace_id"] == "trace_123"
        assert end_kwargs["span_id"] == "child_789"
        assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_no_duplicate_span_on_same_status(genie, mock_workspace_client):
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
        mock_client.start_span.return_value = mock_child_span

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "EXECUTING_QUERY"},  # duplicate status
            {"status": "EXECUTING_QUERY"},  # duplicate status
            {"status": "COMPLETED", "attachments": [{"text": {"content": "Done"}}]},
        ]

        genie.poll_for_result("123", "456")

        # should only create span once for EXECUTING_QUERY despite 3 status changes
        mock_client.start_span.assert_called_once()


def test_poll_for_result_cancelled_terminal_state(genie, mock_workspace_client):
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
        mock_client.start_span.return_value = mock_child_span

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "CANCELLED"},
        ]

        result = genie.poll_for_result("123", "456")

        assert result.result == "Genie query cancelled."
        mock_client.end_span.assert_called_once()
        end_kwargs = mock_client.end_span.call_args[1]
        assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_failed_terminal_state(genie, mock_workspace_client):
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
        mock_client.start_span.return_value = mock_child_span

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "FAILED", "error": "some error"},
        ]

        result = genie.poll_for_result("123", "456")

        assert result.result == "Genie query failed with error: some error"
        mock_client.end_span.assert_called_once()
        end_kwargs = mock_client.end_span.call_args[1]
        assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_query_result_expired_terminal_state(genie, mock_workspace_client):
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
        mock_client.start_span.return_value = mock_child_span

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "QUERY_RESULT_EXPIRED"},
        ]

        result = genie.poll_for_result("123", "456")

        assert result.result == "Genie query query_result_expired."
        mock_client.end_span.assert_called_once()
        end_kwargs = mock_client.end_span.call_args[1]
        assert end_kwargs["attributes"]["final_state"] == "EXECUTING_QUERY"


def test_poll_for_result_timeout_includes_timeout_attribute(genie, mock_workspace_client):
    with (
        patch("databricks_ai_bridge.genie.MAX_ITERATIONS", 2),
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
        mock_client.start_span.return_value = mock_child_span

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "EXECUTING_QUERY"},
            {"status": "EXECUTING_QUERY"},
        ]

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

        mock_workspace_client.genie._api.do.side_effect = [
            {"status": "EXECUTING_QUERY"},
            {"status": "COMPLETED", "attachments": [{"text": {"content": "Success"}}]},
        ]

        result = genie.poll_for_result("123", "456")

        # should still complete successfully despite tracing failures
        assert result.result == "Success"


# Parametrized tests for _parse_attachments
@pytest.mark.parametrize(
    "resp,exp_query,exp_text,exp_questions",
    [
        # All three attachment types
        (
            {
                "attachments": [
                    {"attachment_id": "1", "query": {"query": "SELECT *", "description": "Test"}},
                    {"attachment_id": "2", "text": {"content": "Summary text"}},
                    {
                        "attachment_id": "3",
                        "suggested_questions": {"questions": ["Q1?", "Q2?", "Q3?"]},
                    },
                ]
            },
            {"attachment_id": "1", "query": {"query": "SELECT *", "description": "Test"}},
            {"attachment_id": "2", "text": {"content": "Summary text"}},
            {"attachment_id": "3", "suggested_questions": {"questions": ["Q1?", "Q2?", "Q3?"]}},
        ),
        # Only query
        (
            {
                "attachments": [
                    {"attachment_id": "1", "query": {"query": "SELECT 1", "description": "Desc"}}
                ]
            },
            {"attachment_id": "1", "query": {"query": "SELECT 1", "description": "Desc"}},
            None,
            None,
        ),
        # Only text
        (
            {"attachments": [{"attachment_id": "2", "text": {"content": "Text only"}}]},
            None,
            {"attachment_id": "2", "text": {"content": "Text only"}},
            None,
        ),
        # Only suggested questions
        (
            {
                "attachments": [
                    {"attachment_id": "3", "suggested_questions": {"questions": ["Question?"]}}
                ]
            },
            None,
            None,
            {"attachment_id": "3", "suggested_questions": {"questions": ["Question?"]}},
        ),
        # Edge cases - all return None for all fields
        ({"attachments": []}, None, None, None),
        ({}, None, None, None),
        ({"attachments": None}, None, None, None),
        ({"attachments": "not a list"}, None, None, None),
        # Invalid items - only valid dict is parsed
        (
            {"attachments": ["string", 123, None, {"query": {"query": "SELECT 1"}}]},
            {"query": {"query": "SELECT 1"}},
            None,
            None,
        ),
    ],
)
def test_parse_attachments(resp, exp_query, exp_text, exp_questions):
    """Test parsing attachments with various input scenarios."""
    result = _parse_attachments(resp)
    assert result["query_attachment"] == exp_query
    assert result["text_attachment"] == exp_text
    assert result["suggested_questions_attachment"] == exp_questions


# Parametrized tests for _extract_suggested_questions_from_attachment
@pytest.mark.parametrize(
    "attachment,expected",
    [
        ({"suggested_questions": {"questions": ["Q1?", "Q2?", "Q3?"]}}, ["Q1?", "Q2?", "Q3?"]),
        ({"suggested_questions": {"questions": ["Only?"]}}, ["Only?"]),
        ({"suggested_questions": {"questions": []}}, None),
        ("not a dict", None),
        (None, None),
        ({"other_key": "value"}, None),
        ({"suggested_questions": "not a dict"}, None),
        ({"suggested_questions": {"other_key": "value"}}, None),
        ({"suggested_questions": {"questions": "not a list"}}, None),
        (
            {"suggested_questions": {"questions": ["Valid?", 123, None, "Another?", {}]}},
            ["Valid?", "Another?"],
        ),
        ({"suggested_questions": {"questions": [123, None, {}, []]}}, None),
    ],
)
def test_extract_suggested_questions(attachment, expected):
    """Test extracting suggested questions with various inputs."""
    assert _extract_suggested_questions_from_attachment(attachment) == expected


# Parametrized tests for _extract_text_attachment_content_from_attachment
@pytest.mark.parametrize(
    "attachment,expected",
    [
        ({"text": {"content": "Summary text"}}, "Summary text"),
        ({"text": {"content": ""}}, ""),
        ("not a dict", ""),
        (None, ""),
        ({"other_key": "value"}, ""),
        ({"text": "not a dict"}, ""),
        ({"text": {"other_key": "value"}}, ""),
        ({"text": {"content": "Line 1\nLine 2\nLine 3"}}, "Line 1\nLine 2\nLine 3"),
    ],
)
def test_extract_text_content(attachment, expected):
    """Test extracting text content with various inputs."""
    assert _extract_text_attachment_content_from_attachment(attachment) == expected


def test_poll_with_all_attachments(genie, mock_workspace_client):
    """Test with suggested questions, text, and query."""
    mock_workspace_client.genie._api.do.side_effect = [
        {
            "status": "COMPLETED",
            "conversation_id": "conv_123",
            "attachments": [
                {"attachment_id": "1", "text": {"content": "Summary"}},
                {"attachment_id": "2", "suggested_questions": {"questions": ["Q1?", "Q2?"]}},
                {"attachment_id": "3", "query": {"query": "SELECT *", "description": "Query"}},
            ],
        },
        {
            "statement_response": {
                "status": {"state": "SUCCEEDED"},
                "conversation_id": "conv_123",
                "manifest": {"schema": {"columns": [{"name": "id", "type_name": "INT"}]}},
                "result": {"data_array": [["1"], ["2"]]},
            }
        },
    ]

    result = genie.poll_for_result("conv_123", "msg_456")
    assert result.suggested_questions == ["Q1?", "Q2?"]
    assert result.text_attachment_content == "Summary"
    assert result.conversation_id == "conv_123"
    assert isinstance(result.result, str)


def test_poll_text_only_no_query(genie, mock_workspace_client):
    """Test with only text attachment and no query."""
    mock_workspace_client.genie._api.do.side_effect = [
        {
            "status": "COMPLETED",
            "conversation_id": "conv_456",
            "attachments": [
                {"attachment_id": "1", "text": {"content": "Just text"}},
                {"attachment_id": "2", "suggested_questions": {"questions": ["Follow-up?"]}},
            ],
        }
    ]

    result = genie.poll_for_result("conv_456", "msg_789")
    assert result.result == "Just text"
    assert result.text_attachment_content == "Just text"
    assert result.suggested_questions == ["Follow-up?"]


def test_poll_query_only(genie, mock_workspace_client):
    """Test with query only - no suggestions or text."""
    mock_workspace_client.genie._api.do.side_effect = [
        {
            "status": "COMPLETED",
            "attachments": [
                {"attachment_id": "1", "query": {"query": "SELECT 1", "description": "Simple"}},
            ],
        },
        {
            "statement_response": {
                "status": {"state": "SUCCEEDED"},
                "manifest": {"schema": {"columns": [{"name": "val", "type_name": "INT"}]}},
                "result": {"data_array": [["1"]]},
            }
        },
    ]

    result = genie.poll_for_result("conv_123", "msg_456")
    assert result.suggested_questions is None
    assert result.text_attachment_content == ""
    assert isinstance(result.result, str)


def test_poll_null_attachments(genie, mock_workspace_client):
    """Test with null/missing attachments."""
    mock_workspace_client.genie._api.do.side_effect = [{"status": "COMPLETED", "attachments": None}]

    result = genie.poll_for_result("conv_123", "msg_456")
    assert result.suggested_questions is None
    assert result.text_attachment_content == ""


def test_ask_question_mlflow_trace_logs_inputs_and_outputs(genie, mock_workspace_client):
    """Test that ask_question logs inputs and outputs as span attributes."""
    mock_workspace_client.genie._api.do.side_effect = [
        {"conversation_id": "conv_123", "message_id": "msg_456"},
        {
            "status": "COMPLETED",
            "conversation_id": "conv_123",
            "attachments": [
                {"attachment_id": "att_1", "query": {"query": "SELECT * FROM sales", "description": "All sales"}},
            ],
        },
        {
            "statement_response": {
                "status": {"state": "SUCCEEDED"},
                "conversation_id": "conv_123",
                "manifest": {"schema": {"columns": [{"name": "id", "type_name": "INT"}]}},
                "result": {"data_array": [["1"], ["2"]]},
            }
        },
    ]

    result = genie.ask_question("What is the meaning of life?")
    assert isinstance(result.result, str)

    # Verify the mlflow trace was created with correct span attributes
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None

    spans = trace.search_spans(name="ask_question")
    assert len(spans) == 1
    ask_span = spans[0]

    # Verify input attributes
    assert ask_span.attributes["space_id"] == "test_space_id"
    assert ask_span.attributes["input.question"] == "What is the meaning of life?"
    assert ask_span.attributes["input.conversation_id"] == ""

    # Verify output attributes (output.result is intentionally not logged)
    assert "output.result" not in ask_span.attributes
    assert ask_span.attributes["output.query"] == "SELECT * FROM sales"
    assert ask_span.attributes["output.description"] == "All sales"
    assert ask_span.attributes["output.conversation_id"] == "conv_123"


def test_ask_question_mlflow_trace_with_conversation_id(genie, mock_workspace_client):
    """Test that ask_question logs conversation_id input when continuing a conversation."""
    mock_workspace_client.genie._api.do.side_effect = [
        {"conversation_id": "conv_existing", "message_id": "msg_789"},
        {
            "status": "COMPLETED",
            "conversation_id": "conv_existing",
            "attachments": [
                {"text": {"content": "Follow-up answer"}},
                {"suggested_questions": {"questions": ["Next question?"]}},
            ],
        },
    ]

    result = genie.ask_question("Follow-up question", conversation_id="conv_existing")
    assert result.result == "Follow-up answer"

    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    assert trace is not None

    spans = trace.search_spans(name="ask_question")
    assert len(spans) == 1
    ask_span = spans[0]

    # Verify input attributes include the conversation_id
    assert ask_span.attributes["space_id"] == "test_space_id"
    assert ask_span.attributes["input.question"] == "Follow-up question"
    assert ask_span.attributes["input.conversation_id"] == "conv_existing"

    # Verify output attributes
    assert "output.result" not in ask_span.attributes
    assert ask_span.attributes["output.query"] == ""
    assert ask_span.attributes["output.description"] == ""
    assert ask_span.attributes["output.conversation_id"] == "conv_existing"
