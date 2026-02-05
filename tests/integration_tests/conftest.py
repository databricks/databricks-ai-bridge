"""
Shared fixtures and configuration for integration tests.

These tests require a live Databricks workspace. They are NOT run by default -
set RUN_INTEGRATION_TESTS=1 to enable.

Feature-specific fixtures live in their respective subdirectories:
- vector_search/conftest.py  - Vector Search indexes, clients, index details
- lakebase/conftest.py       - Lakebase instances, pools, clients

Environment Variables:
======================
Required:
    RUN_INTEGRATION_TESTS     - Set to "1" to enable integration tests
    DATABRICKS_HOST           - Workspace URL
    DATABRICKS_CLIENT_ID      - Service principal client ID
    DATABRICKS_CLIENT_SECRET  - Service principal client secret
"""

from __future__ import annotations

import pytest

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def workspace_client():
    """
    Create a WorkspaceClient using environment variables.

    The SDK auto-detects auth from env vars (e.g. DATABRICKS_HOST,
    DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET for OAuth M2M).
    For local testing, set DATABRICKS_CONFIG_PROFILE instead.
    """
    from databricks.sdk import WorkspaceClient

    return WorkspaceClient()


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line(
        "markers", "contract: mark test as contract validation (API signatures)"
    )
    config.addinivalue_line(
        "markers", "behavior: mark test as behavior validation (search results)"
    )
    config.addinivalue_line("markers", "slow: mark test as slow (may take > 30 seconds)")
