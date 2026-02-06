"""
Bridge-layer tests for Genie response parsing.

Validates that response parsing (markdown, SQL, DataFrame) works
correctly with live data. Uses cached session fixtures — no additional
live API calls.
"""

from __future__ import annotations

import pandas as pd
import pytest


# =============================================================================
# Response Parsing (String Mode)
# =============================================================================


@pytest.mark.integration
class TestGenieResponseParsing:
    """Validate string-mode response parsing with live data."""

    def test_result_is_nonempty_string(self, genie_response):
        assert isinstance(genie_response.result, str)
        assert len(genie_response.result) > 0

    def test_sql_contains_select(self, genie_response):
        # Soft: Genie usually generates SQL for data questions
        if genie_response.query:
            assert "SELECT" in genie_response.query.upper() or "WITH" in genie_response.query.upper()

    def test_markdown_is_parseable(self, genie_response):
        result = genie_response.result
        # Markdown tables have header row, separator row, and data rows
        lines = result.strip().split("\n")
        assert len(lines) >= 3, f"Expected at least 3 lines (header + separator + data), got {len(lines)}"
        # Separator row contains dashes
        assert any("---" in line for line in lines), "Expected markdown separator row with ---"

    def test_conversation_id_format(self, genie_response):
        cid = genie_response.conversation_id
        assert cid is not None
        assert isinstance(cid, str)
        # Conversation IDs are non-empty strings
        assert len(cid) > 0


# =============================================================================
# Pandas Response Parsing
# =============================================================================


@pytest.mark.integration
class TestGeniePandasParsing:
    """Validate pandas-mode response parsing with live data."""

    def test_dataframe_columns_are_named(self, genie_pandas_response):
        df = genie_pandas_response.result
        assert isinstance(df, pd.DataFrame)
        for col in df.columns:
            assert isinstance(col, str)
            assert len(col) > 0

    def test_not_all_dtypes_object(self, genie_pandas_response):
        df = genie_pandas_response.result
        # At least one column should be non-object (numeric or datetime)
        non_object_cols = df.select_dtypes(exclude=["object"]).columns
        assert len(non_object_cols) > 0, f"All columns are object type: {df.dtypes.to_dict()}"

    def test_not_all_nan(self, genie_pandas_response):
        df = genie_pandas_response.result
        assert not df.isna().all().all(), "All values are NaN"


# =============================================================================
# Space Metadata
# =============================================================================


@pytest.mark.integration
class TestGenieSpaceMetadata:
    """Validate Genie instance metadata from the Space."""

    def test_description_is_str_or_none(self, genie_instance):
        assert genie_instance.description is None or isinstance(genie_instance.description, str)

    def test_space_id_matches_input(self, genie_instance, genie_space_id):
        assert genie_instance.space_id == genie_space_id
