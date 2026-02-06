"""
Contract tests for the Genie API surface.

Validates that import paths, class signatures, and response shapes
match the expected contract. Uses cached session fixtures — no
additional live API calls beyond conftest setup.
"""

from __future__ import annotations

import pytest


# =============================================================================
# Import Contract
# =============================================================================


@pytest.mark.contract
class TestGenieImportContract:
    """Verify that public symbols are importable."""

    def test_genie_importable(self):
        from databricks_ai_bridge.genie import Genie

        assert Genie is not None

    def test_genie_response_importable(self):
        from databricks_ai_bridge.genie import GenieResponse

        assert GenieResponse is not None

    def test_genie_response_has_expected_fields(self):
        from databricks_ai_bridge.genie import GenieResponse

        resp = GenieResponse(result="test")
        assert hasattr(resp, "result")
        assert hasattr(resp, "query")
        assert hasattr(resp, "description")
        assert hasattr(resp, "conversation_id")


# =============================================================================
# Class Contract
# =============================================================================


@pytest.mark.contract
class TestGenieClassContract:
    """Verify Genie class has expected attributes and methods."""

    def test_genie_has_space_id(self, genie_instance):
        assert hasattr(genie_instance, "space_id")
        assert isinstance(genie_instance.space_id, str)
        assert len(genie_instance.space_id) > 0

    def test_genie_has_description(self, genie_instance):
        assert hasattr(genie_instance, "description")

    def test_genie_has_start_conversation(self, genie_instance):
        assert hasattr(genie_instance, "start_conversation")
        assert callable(genie_instance.start_conversation)

    def test_genie_has_create_message(self, genie_instance):
        assert hasattr(genie_instance, "create_message")
        assert callable(genie_instance.create_message)

    def test_genie_has_poll_for_result(self, genie_instance):
        assert hasattr(genie_instance, "poll_for_result")
        assert callable(genie_instance.poll_for_result)

    def test_genie_has_ask_question(self, genie_instance):
        assert hasattr(genie_instance, "ask_question")
        assert callable(genie_instance.ask_question)


# =============================================================================
# Response Contract
# =============================================================================


@pytest.mark.contract
class TestGenieResponseContract:
    """Verify live GenieResponse matches expected contract."""

    def test_response_is_genie_response(self, genie_response):
        from databricks_ai_bridge.genie import GenieResponse

        assert isinstance(genie_response, GenieResponse)

    def test_response_result_not_none(self, genie_response):
        assert genie_response.result is not None

    def test_response_conversation_id_is_string(self, genie_response):
        assert isinstance(genie_response.conversation_id, str)
        assert len(genie_response.conversation_id) > 0

    def test_response_query_is_str_or_none(self, genie_response):
        assert genie_response.query is None or isinstance(genie_response.query, str)

    def test_response_description_is_str_or_none(self, genie_response):
        assert genie_response.description is None or isinstance(genie_response.description, str)
