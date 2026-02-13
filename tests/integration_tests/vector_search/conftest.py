"""
Vector Search-specific fixtures for integration tests.

Shared fixtures (workspace_client, markers) are inherited from the
parent conftest at tests/integration_tests/conftest.py.

Optional Environment Variables (for local testing):
    VS_TEST_DELTA_SYNC_INDEX     - Override default delta-sync index name
    VS_TEST_DIRECT_ACCESS_INDEX  - Override default direct-access index name
"""

from __future__ import annotations

import os

import pytest

# Skip all tests in this directory if integration tests are not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VS_INTEGRATION_TESTS") != "1",
    reason="Integration tests disabled. Set RUN_VS_INTEGRATION_TESTS=1 to enable.",
)


# =============================================================================
# Test Index Configuration
# =============================================================================

CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_vs_test"

# Delta-sync index with managed embeddings (databricks-bge-large-en)
DEFAULT_DELTA_SYNC_INDEX = f"{CATALOG}.{SCHEMA}.delta_sync_managed"

# Direct-access index (pre-computed 1024-dim vectors)
DEFAULT_DIRECT_ACCESS_INDEX = f"{CATALOG}.{SCHEMA}.direct_access_test"

# Source table for delta-sync index
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.test_documents"

# Embedding model endpoint
EMBEDDING_ENDPOINT = "databricks-bge-large-en"


# =============================================================================
# Vector Search Client Fixture
# =============================================================================


@pytest.fixture(scope="session")
def vector_search_client(workspace_client):
    """
    Create a VectorSearchClient for direct API testing.

    VectorSearchClient uses different env vars than the SDK,
    so we pass credentials explicitly.
    """
    from databricks.vector_search.client import VectorSearchClient

    workspace_url = workspace_client.config.host.rstrip("/")

    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
    token = os.environ.get("DATABRICKS_TOKEN")

    if client_id and client_secret:
        return VectorSearchClient(
            workspace_url=workspace_url,
            service_principal_client_id=client_id,
            service_principal_client_secret=client_secret,
            disable_notice=True,
        )
    elif token:
        return VectorSearchClient(
            workspace_url=workspace_url,
            personal_access_token=token,
            disable_notice=True,
        )
    else:
        # Try to get token from workspace client
        headers = workspace_client.config.authenticate()
        if "Authorization" in headers:
            token = headers["Authorization"].replace("Bearer ", "")
            return VectorSearchClient(
                workspace_url=workspace_url,
                personal_access_token=token,
                disable_notice=True,
            )
        raise RuntimeError("No credentials found for VectorSearchClient")


# =============================================================================
# Index Name Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def delta_sync_index_name() -> str:
    """Get the delta-sync managed embeddings index name."""
    return os.environ.get("VS_TEST_DELTA_SYNC_INDEX", DEFAULT_DELTA_SYNC_INDEX)


@pytest.fixture(scope="session")
def direct_access_index_name() -> str:
    """Get the direct-access index name."""
    return os.environ.get("VS_TEST_DIRECT_ACCESS_INDEX", DEFAULT_DIRECT_ACCESS_INDEX)


# =============================================================================
# VectorSearchIndex Instance Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def delta_sync_index(vector_search_client, delta_sync_index_name):
    """Get the delta-sync VectorSearchIndex instance."""
    return vector_search_client.get_index(index_name=delta_sync_index_name)


@pytest.fixture(scope="session")
def direct_access_index(vector_search_client, direct_access_index_name):
    """Get the direct-access VectorSearchIndex instance."""
    try:
        return vector_search_client.get_index(index_name=direct_access_index_name)
    except Exception:
        pytest.skip(f"Direct-access index '{direct_access_index_name}' not available")


# =============================================================================
# IndexDetails Fixtures (bridge utility)
# =============================================================================


@pytest.fixture(scope="session")
def delta_sync_index_details(delta_sync_index):
    """IndexDetails for the delta-sync index."""
    from databricks_ai_bridge.utils.vector_search import IndexDetails

    return IndexDetails(delta_sync_index)


@pytest.fixture(scope="session")
def direct_access_index_details(direct_access_index):
    """IndexDetails for the direct-access index."""
    from databricks_ai_bridge.utils.vector_search import IndexDetails

    return IndexDetails(direct_access_index)


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_query_vector():
    """Generate a deterministic 1024-dim fake query vector for direct-access testing."""
    import random

    random.seed(42)
    return [random.uniform(-1, 1) for _ in range(1024)]


@pytest.fixture(scope="session")
def embedding_endpoint() -> str:
    """Get the embedding model endpoint name."""
    return EMBEDDING_ENDPOINT
