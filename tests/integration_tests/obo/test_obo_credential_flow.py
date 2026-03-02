"""
Integration tests for OBO (On-Behalf-Of) credential flows.

Verifies that identity is forwarded correctly through both the Model Serving
and Databricks Apps authentication paths by using two different service principals:
  - SP-A ("deployer"): authenticated via DATABRICKS_CLIENT_ID/SECRET
  - SP-B ("end user"): authenticated via OBO_TEST_CLIENT_ID/SECRET

The test injects SP-B's token through each OBO path, then calls a `whoami()`
UC function to assert the result is SP-B's identity and differs from SP-A's.

Environment Variables:
======================
Required:
    RUN_OBO_INTEGRATION_TESTS     - Set to "1" to enable
    DATABRICKS_HOST               - Workspace URL
    DATABRICKS_CLIENT_ID          - SP-A (deployer) client ID
    DATABRICKS_CLIENT_SECRET      - SP-A (deployer) client secret
    OBO_TEST_CLIENT_ID            - SP-B (end user) client ID
    OBO_TEST_CLIENT_SECRET        - SP-B (end user) client secret
    OBO_TEST_WAREHOUSE_ID         - SQL warehouse for statement execution
"""

from __future__ import annotations

import os
import threading

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

from databricks_ai_bridge.model_serving_obo_credential_strategy import (
    ModelServingUserCredentials,
)

# Skip all tests if not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OBO_INTEGRATION_TESTS") != "1",
    reason="OBO integration tests disabled. Set RUN_OBO_INTEGRATION_TESTS=1 to enable.",
)

# Non-sensitive resource names (same pattern as FMAPI tests)
CATALOG = "integration_testing"
SCHEMA = "databricks_ai_bridge_mcp_test"


# =============================================================================
# Helpers
# =============================================================================


def _call_whoami(client: WorkspaceClient, warehouse_id: str) -> str:
    """Execute the whoami() UC function via SQL and return the caller identity."""
    result = client.statement_execution.execute_statement(
        statement=f"SELECT {CATALOG}.{SCHEMA}.whoami() AS caller",
        warehouse_id=warehouse_id,
        wait_timeout="30s",
    )
    assert result.status.state.value == "SUCCEEDED", (
        f"SQL statement failed: {result.status}"
    )
    return result.result.data_array[0][0]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def deployer_client():
    """SP-A: the 'deployer' service principal, using default DATABRICKS_CLIENT_ID/SECRET."""
    return WorkspaceClient()


@pytest.fixture(scope="module")
def deployer_identity(deployer_client):
    """The deployer's display name, used to verify OBO clients see a different identity."""
    return deployer_client.current_user.me().display_name


@pytest.fixture(scope="module")
def end_user_client():
    """SP-B: the 'end user' service principal, using OBO_TEST_CLIENT_ID/SECRET."""
    client_id = os.environ.get("OBO_TEST_CLIENT_ID")
    client_secret = os.environ.get("OBO_TEST_CLIENT_SECRET")
    host = os.environ.get("DATABRICKS_HOST")
    if not all([client_id, client_secret, host]):
        pytest.skip(
            "OBO_TEST_CLIENT_ID, OBO_TEST_CLIENT_SECRET, and DATABRICKS_HOST must be set"
        )
    return WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret)


@pytest.fixture(scope="module")
def end_user_identity(end_user_client):
    """The end user's display name, derived dynamically (no hardcoded SP app IDs)."""
    return end_user_client.current_user.me().display_name


@pytest.fixture(scope="module")
def end_user_token(end_user_client):
    """Bearer token for SP-B, extracted from its authenticated headers."""
    headers = end_user_client.config.authenticate()
    token = headers.get("Authorization", "").replace("Bearer ", "")
    assert token, "Failed to extract Bearer token for end user SP"
    return token


@pytest.fixture(scope="module")
def warehouse_id():
    """SQL warehouse ID for statement execution."""
    wh_id = os.environ.get("OBO_TEST_WAREHOUSE_ID")
    if not wh_id:
        pytest.skip("OBO_TEST_WAREHOUSE_ID must be set")
    return wh_id


@pytest.fixture
def obo_client_model_serving(end_user_token, monkeypatch):
    """
    Simulate the Model Serving OBO environment.

    Sets env vars that ModelServingUserCredentials checks, then injects
    SP-B's token into the thread-local slot (the same slot mlflowserving
    would populate in a real serving environment).
    """
    monkeypatch.setenv("IS_IN_DB_MODEL_SERVING_ENV", "true")
    monkeypatch.setenv(
        "DB_MODEL_SERVING_HOST_URL", os.environ.get("DATABRICKS_HOST", "")
    )
    # Prevent the SDK from picking up SP-A's credentials
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", "/dev/null")
    monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    main_thread = threading.main_thread()
    main_thread.__dict__["invokers_token"] = end_user_token

    cfg = Config(credentials_strategy=ModelServingUserCredentials())
    wc = WorkspaceClient(config=cfg)
    yield wc

    main_thread.__dict__.pop("invokers_token", None)


@pytest.fixture(scope="module")
def obo_client_apps(end_user_token):
    """
    Simulate the Databricks Apps OBO path.

    This mirrors what get_user_workspace_client() does in app-templates:
    WorkspaceClient(token=<x-forwarded-access-token>, auth_type="pat")
    """
    return WorkspaceClient(
        host=os.environ.get("DATABRICKS_HOST", ""),
        token=end_user_token,
        auth_type="pat",
    )


# =============================================================================
# Tests: Model Serving OBO
# =============================================================================


@pytest.mark.obo
class TestModelServingOBO:
    """Verify identity forwarding through the ModelServingUserCredentials path."""

    def test_auth_type(self, obo_client_model_serving):
        assert (
            obo_client_model_serving.config.auth_type
            == "model_serving_user_credentials"
        )

    def test_identity_is_end_user(self, obo_client_model_serving, end_user_identity):
        me = obo_client_model_serving.current_user.me()
        assert me.display_name == end_user_identity

    def test_whoami_differs_from_deployer(
        self,
        obo_client_model_serving,
        deployer_identity,
        end_user_identity,
        warehouse_id,
    ):
        caller = _call_whoami(obo_client_model_serving, warehouse_id)
        assert caller != deployer_identity, (
            f"OBO client should NOT see deployer identity, got {caller}"
        )
        assert end_user_identity in caller


# =============================================================================
# Tests: Databricks Apps OBO
# =============================================================================


@pytest.mark.obo
class TestAppsOBO:
    """Verify identity forwarding through the Apps path (direct token injection)."""

    def test_identity_is_end_user(self, obo_client_apps, end_user_identity):
        me = obo_client_apps.current_user.me()
        assert me.display_name == end_user_identity

    def test_whoami_differs_from_deployer(
        self, obo_client_apps, deployer_identity, end_user_identity, warehouse_id
    ):
        caller = _call_whoami(obo_client_apps, warehouse_id)
        assert caller != deployer_identity, (
            f"Apps OBO client should NOT see deployer identity, got {caller}"
        )
        assert end_user_identity in caller
