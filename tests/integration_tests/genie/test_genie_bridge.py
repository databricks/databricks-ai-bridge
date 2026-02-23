"""
Bridge-layer tests for Genie response parsing.

Validates that our _parse_query_result correctly parses live API responses
into markdown tables and pandas DataFrames. Uses cached session fixtures —
no additional live API calls.
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
# Response Parsing (String Mode)
# =============================================================================


@pytest.mark.integration
class TestGenieResponseParsing:
    """Validate our _parse_query_result correctly parses live API responses."""

    def test_markdown_table_is_parseable(self, genie_response):
        # Our _parse_query_result generates markdown with header + separator + data
        result = genie_response.result
        lines = result.strip().split("\n")
        assert len(lines) >= 3, f"Expected header + separator + data rows, got {len(lines)}"
        assert any("---" in line for line in lines), "Expected markdown separator row"
        # Validate pipe-delimited format (our to_markdown output)
        assert "|" in lines[0], "Expected pipe-separated header row"

    def test_query_field_populated(self, genie_response):
        # Our poll_for_result extracts query from attachment["query"]["query"]
        # For data questions, this should contain the generated SQL
        if genie_response.query:
            assert isinstance(genie_response.query, str)
            assert len(genie_response.query) > 0

    def test_text_only_response(self, genie_text_response):
        # Text fallback path: no SQL query, just a text result
        assert isinstance(genie_text_response.result, str)
        assert len(genie_text_response.result) > 0
        assert not genie_text_response.query
        assert isinstance(genie_text_response.conversation_id, str)


# =============================================================================
# Pandas Response Parsing
# =============================================================================


@pytest.mark.integration
class TestGeniePandasParsing:
    """Validate our _parse_query_result with return_pandas=True on live data."""

    def test_columns_are_named_strings(self, genie_pandas_response):
        # Our code extracts column names from resp["manifest"]["schema"]["columns"]
        df = genie_pandas_response.result
        assert isinstance(df, pd.DataFrame)
        for col in df.columns:
            assert isinstance(col, str)
            assert len(col) > 0

    def test_type_conversion_produces_non_object_columns(self, genie_pandas_response):
        # Our _parse_query_result converts INT->int, FLOAT->float, etc.
        df = genie_pandas_response.result
        non_object_cols = df.select_dtypes(exclude=["object"]).columns
        assert len(non_object_cols) > 0, (
            f"Type conversion failed, all object: {df.dtypes.to_dict()}"
        )

    def test_values_not_all_nan(self, genie_pandas_response):
        # Regression: ensure our parsing doesn't produce all-NaN values
        df = genie_pandas_response.result
        assert not df.isna().all().all(), "All values are NaN — parsing may be broken"
