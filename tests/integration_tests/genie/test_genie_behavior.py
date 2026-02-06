"""
Behavior tests for the Genie API.

Validates end-to-end behavior: ask_question results, conversation
continuity, pandas mode, and the start_conversation + poll_for_result
low-level flow.

Most tests use cached session fixtures. Only TestGenieStartConversation
makes an additional live API call.
"""

from __future__ import annotations

import pandas as pd
import pytest


# =============================================================================
# Ask Question Behavior
# =============================================================================


@pytest.mark.behavior
class TestGenieAskQuestion:
    """Validate ask_question returns meaningful results."""

    def test_result_is_nonempty_string(self, genie_response):
        assert isinstance(genie_response.result, str)
        assert len(genie_response.result) > 0

    def test_result_has_table_structure(self, genie_response):
        # Genie returns markdown tables with pipe separators
        result = genie_response.result
        assert "|" in result, "Expected markdown table with | separators"

    def test_query_has_sql_keywords(self, genie_response):
        # Soft assertion: Genie usually generates SQL
        if genie_response.query:
            query_upper = genie_response.query.upper()
            assert "SELECT" in query_upper or "WITH" in query_upper

    def test_conversation_id_is_valid(self, genie_response):
        assert genie_response.conversation_id is not None
        assert isinstance(genie_response.conversation_id, str)
        assert len(genie_response.conversation_id) > 0


# =============================================================================
# Conversation Continuity
# =============================================================================


@pytest.mark.behavior
class TestGenieConversationContinuity:
    """Validate follow-up questions in existing conversations."""

    def test_continued_response_has_result(self, genie_continued_response):
        assert genie_continued_response.result is not None
        result_str = str(genie_continued_response.result)
        assert len(result_str) > 0

    def test_continued_response_has_conversation_id(self, genie_continued_response):
        assert genie_continued_response.conversation_id is not None
        assert isinstance(genie_continued_response.conversation_id, str)
        assert len(genie_continued_response.conversation_id) > 0

    def test_continued_response_preserves_conversation(
        self, genie_conversation_id, genie_continued_response
    ):
        # The conversation_id should match the one we passed in
        assert genie_continued_response.conversation_id == genie_conversation_id


# =============================================================================
# Pandas Mode
# =============================================================================


@pytest.mark.behavior
class TestGeniePandasMode:
    """Validate return_pandas=True behavior."""

    def test_result_is_dataframe(self, genie_pandas_response):
        assert isinstance(genie_pandas_response.result, pd.DataFrame)

    def test_dataframe_has_columns(self, genie_pandas_response):
        df = genie_pandas_response.result
        assert len(df.columns) > 0

    def test_dataframe_has_rows(self, genie_pandas_response):
        df = genie_pandas_response.result
        assert len(df) > 0

    def test_dataframe_has_numeric_column(self, genie_pandas_response):
        df = genie_pandas_response.result
        numeric_cols = df.select_dtypes(include=["number"]).columns
        assert len(numeric_cols) > 0, f"Expected at least one numeric column, got dtypes: {df.dtypes.to_dict()}"


# =============================================================================
# Start Conversation + Poll (Low-Level API)
# =============================================================================


@pytest.mark.behavior
@pytest.mark.slow
class TestGenieStartConversation:
    """Test start_conversation + poll_for_result flow. Makes 1 live API call."""

    def test_start_conversation_returns_ids(self, genie_instance):
        resp = genie_instance.start_conversation("How many orders are there?")
        assert "conversation_id" in resp
        assert "message_id" in resp
        assert isinstance(resp["conversation_id"], str)
        assert isinstance(resp["message_id"], str)

        # Poll for the result
        genie_response = genie_instance.poll_for_result(
            resp["conversation_id"], resp["message_id"]
        )

        from databricks_ai_bridge.genie import GenieResponse

        assert isinstance(genie_response, GenieResponse)
        assert genie_response.result is not None
