"""
Behavior tests for the Genie bridge code.

Validates our ask_question orchestration, conversation continuity
(create_message routing), and _parse_query_result with return_pandas=True.

All tests use cached session fixtures — no additional live API calls
beyond conftest setup.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GENIE_INTEGRATION_TESTS") != "1",
    reason="Genie integration tests disabled. Set RUN_GENIE_INTEGRATION_TESTS=1 to enable.",
)

# =============================================================================
# Ask Question Behavior
# =============================================================================


@pytest.mark.behavior
class TestGenieAskQuestion:
    """Smoke test: ask_question orchestrates start_conversation + poll correctly."""

    def test_returns_nonempty_result(self, genie_response):
        # Validates our ask_question() orchestration works end-to-end
        assert isinstance(genie_response.result, str)
        assert len(genie_response.result) > 0
        # For a data question, Genie should populate all response fields
        assert genie_response.query, "Expected a non-empty SQL query for a data question"
        assert genie_response.description, "Expected a non-empty description for a data question"
        assert genie_response.text_attachment_content, "Expected non-empty text attachment content"
        assert isinstance(genie_response.suggested_questions, list), (
            "Expected suggested follow-up questions"
        )
        assert len(genie_response.suggested_questions) > 0
        assert all(isinstance(q, str) and q for q in genie_response.suggested_questions)

    def test_conversation_id_populated(self, genie_response):
        # Validates our code correctly extracts conversation_id from API response
        assert isinstance(genie_response.conversation_id, str)
        assert len(genie_response.conversation_id) > 0


# =============================================================================
# Conversation Continuity
# =============================================================================


@pytest.mark.behavior
class TestGenieConversationContinuity:
    """Validates our ask_question correctly routes to create_message when conversation_id is provided."""

    def test_continued_response_has_result(self, genie_continued_response):
        assert genie_continued_response.result is not None
        assert len(str(genie_continued_response.result)) > 0

    def test_conversation_id_preserved(self, genie_conversation_id, genie_continued_response):
        # Our code passes conversation_id through to create_message
        assert genie_continued_response.conversation_id == genie_conversation_id


# =============================================================================
# Pandas Mode
# =============================================================================


@pytest.mark.behavior
class TestGeniePandasMode:
    """Validates our _parse_query_result with return_pandas=True."""

    def test_result_is_dataframe(self, genie_pandas_response):
        assert isinstance(genie_pandas_response.result, pd.DataFrame)
        assert len(genie_pandas_response.result) > 0

    def test_dataframe_has_typed_columns(self, genie_pandas_response):
        # Our _parse_query_result converts types (INT->int, FLOAT->float, etc.)
        # At least one column should be non-object (proves our type conversion works)
        df = genie_pandas_response.result
        non_object_cols = df.select_dtypes(exclude=["object"]).columns
        assert len(non_object_cols) > 0, f"All columns are object type: {df.dtypes.to_dict()}"
