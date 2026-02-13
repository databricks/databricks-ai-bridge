"""
Genie-specific fixtures for integration tests.

Shared fixtures (workspace_client, markers) are inherited from the
parent conftest at tests/integration_tests/conftest.py.

Environment Variables:
    RUN_GENIE_INTEGRATION_TESTS  - Set to "1" to enable these tests
    GENIE_SPACE_ID               - Required. The Genie Space ID to test against.
"""

from __future__ import annotations

import os

import pytest

# Skip all tests in this directory if genie integration tests are not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GENIE_INTEGRATION_TESTS") != "1",
    reason="Genie integration tests disabled. Set RUN_GENIE_INTEGRATION_TESTS=1 to enable.",
)


# =============================================================================
# Genie Space ID Fixture
# =============================================================================


@pytest.fixture(scope="session")
def genie_space_id():
    """Get the Genie Space ID from the GENIE_SPACE_ID environment variable."""
    space_id = os.environ.get("GENIE_SPACE_ID")
    if not space_id:
        pytest.skip("GENIE_SPACE_ID environment variable not set")
    return space_id


# =============================================================================
# Genie Instance Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def genie_instance(workspace_client, genie_space_id):
    """Create a Genie instance (string mode)."""
    from databricks_ai_bridge.genie import Genie

    return Genie(genie_space_id, client=workspace_client)


@pytest.fixture(scope="session")
def genie_pandas_instance(workspace_client, genie_space_id):
    """Create a Genie instance (pandas mode)."""
    from databricks_ai_bridge.genie import Genie

    return Genie(genie_space_id, client=workspace_client, return_pandas=True)


# =============================================================================
# Cached Genie Response Fixtures (minimize live API calls)
# =============================================================================


@pytest.fixture(scope="session")
def genie_response(genie_instance):
    """Cached response: 'What is the total amount by region?'"""
    return genie_instance.ask_question("What is the total amount by region?")


@pytest.fixture(scope="session")
def genie_conversation_id(genie_response):
    """Extract conversation_id from the cached genie_response."""
    return genie_response.conversation_id


@pytest.fixture(scope="session")
def genie_continued_response(genie_instance, genie_conversation_id):
    """Cached follow-up response using existing conversation."""
    return genie_instance.ask_question(
        "Now filter to only completed orders",
        conversation_id=genie_conversation_id,
    )


@pytest.fixture(scope="session")
def genie_pandas_response(genie_pandas_instance):
    """Cached pandas-mode response: 'What is the average amount by status?'"""
    return genie_pandas_instance.ask_question("What is the average amount by status?")
