"""
End-to-end integration tests for OBO (On-Behalf-Of) credential flows.

Invokes pre-deployed agents (Model Serving endpoint and Databricks App) as
two different service principals and asserts each caller sees their own identity
via the whoami() UC function tool.

  - SP-A ("deployer"): authenticated via DATABRICKS_CLIENT_ID/SECRET
  - SP-B ("end user"): authenticated via OBO_TEST_CLIENT_ID/SECRET

Environment Variables:
======================
Required:
    RUN_OBO_INTEGRATION_TESTS      - Set to "1" to enable
    DATABRICKS_HOST                - Workspace URL
    DATABRICKS_CLIENT_ID           - SP-A client ID
    DATABRICKS_CLIENT_SECRET       - SP-A client secret
    OBO_TEST_CLIENT_ID             - SP-B client ID
    OBO_TEST_CLIENT_SECRET         - SP-B client secret
    OBO_TEST_SERVING_ENDPOINT      - Pre-deployed Model Serving endpoint name
    OBO_TEST_APP_NAME              - Pre-deployed Databricks App name
"""

from __future__ import annotations

import logging
import os
import time

import pytest
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI

log = logging.getLogger(__name__)

# Skip all tests if not enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_OBO_INTEGRATION_TESTS") != "1",
    reason="OBO integration tests disabled. Set RUN_OBO_INTEGRATION_TESTS=1 to enable.",
)

_MAX_RETRIES = 3
_MAX_WARMUP_ATTEMPTS = 10
_WARMUP_INTERVAL = 30  # seconds between warmup attempts (5 min total)
_PROMPT = "Call the whoami tool and respond with ONLY the raw result. Do not add any other text."


# =============================================================================
# Helpers
# =============================================================================


def _invoke_agent(client: DatabricksOpenAI, model: str) -> str:
    """Invoke the agent and return the response text, with retry logic."""
    last_exc = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": _PROMPT}],
            )
            # Extract text from response output items
            parts = []
            for item in response.output:
                if hasattr(item, "text"):
                    parts.append(item.text)
                elif hasattr(item, "content") and isinstance(item.content, list):
                    for content_item in item.content:
                        if hasattr(content_item, "text"):
                            parts.append(content_item.text)
            text = " ".join(parts)
            assert text, f"Agent returned empty response: {response.output}"
            return text
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                log.warning("Attempt %d/%d failed: %s — retrying", attempt + 1, _MAX_RETRIES, exc)
                time.sleep(2)
    raise last_exc  # type: ignore[misc]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def sp_a_workspace_client():
    """SP-A WorkspaceClient using default DATABRICKS_CLIENT_ID/SECRET."""
    return WorkspaceClient()


@pytest.fixture(scope="module")
def sp_b_workspace_client():
    """SP-B WorkspaceClient using OBO_TEST_CLIENT_ID/SECRET."""
    client_id = os.environ.get("OBO_TEST_CLIENT_ID")
    client_secret = os.environ.get("OBO_TEST_CLIENT_SECRET")
    host = os.environ.get("DATABRICKS_HOST")
    if not all([client_id, client_secret, host]):
        pytest.skip("OBO_TEST_CLIENT_ID, OBO_TEST_CLIENT_SECRET, and DATABRICKS_HOST must be set")
    return WorkspaceClient(host=host, client_id=client_id, client_secret=client_secret)


@pytest.fixture(scope="module")
def sp_a_identity(sp_a_workspace_client):
    """SP-A's display name."""
    return sp_a_workspace_client.current_user.me().display_name


@pytest.fixture(scope="module")
def sp_b_identity(sp_b_workspace_client):
    """SP-B's display name."""
    return sp_b_workspace_client.current_user.me().display_name


@pytest.fixture(scope="module")
def sp_a_client(sp_a_workspace_client):
    """DatabricksOpenAI client authenticated as SP-A."""
    return DatabricksOpenAI(workspace_client=sp_a_workspace_client)


@pytest.fixture(scope="module")
def sp_b_client(sp_b_workspace_client):
    """DatabricksOpenAI client authenticated as SP-B."""
    return DatabricksOpenAI(workspace_client=sp_b_workspace_client)


@pytest.fixture(scope="module")
def serving_endpoint():
    """Pre-deployed Model Serving endpoint name."""
    name = os.environ.get("OBO_TEST_SERVING_ENDPOINT")
    if not name:
        pytest.skip("OBO_TEST_SERVING_ENDPOINT must be set")
    return name


@pytest.fixture(scope="module")
def serving_endpoint_ready(sp_a_client, serving_endpoint):
    """Warm up the serving endpoint (may be scaled to zero) before tests."""
    for attempt in range(_MAX_WARMUP_ATTEMPTS):
        try:
            sp_a_client.responses.create(
                model=serving_endpoint,
                input=[{"role": "user", "content": "ping"}],
            )
            log.info("Serving endpoint is warm after %d attempt(s)", attempt + 1)
            return
        except Exception as exc:
            log.info(
                "Warmup attempt %d/%d: %s — waiting %ds",
                attempt + 1,
                _MAX_WARMUP_ATTEMPTS,
                exc,
                _WARMUP_INTERVAL,
            )
            time.sleep(_WARMUP_INTERVAL)
    pytest.fail(
        f"Serving endpoint '{serving_endpoint}' did not scale up within "
        f"{_MAX_WARMUP_ATTEMPTS * _WARMUP_INTERVAL}s"
    )


@pytest.fixture(scope="module")
def app_name():
    """Pre-deployed Databricks App name."""
    name = os.environ.get("OBO_TEST_APP_NAME")
    if not name:
        pytest.skip("OBO_TEST_APP_NAME must be set")
    return name


# =============================================================================
# Tests: Model Serving OBO
# =============================================================================


@pytest.mark.obo
class TestModelServingOBO:
    """Invoke a pre-deployed Model Serving agent as two different SPs."""

    def test_sp_a_and_sp_b_see_different_identities(
        self, sp_a_client, sp_b_client, serving_endpoint, serving_endpoint_ready
    ):
        sp_a_response = _invoke_agent(sp_a_client, serving_endpoint)
        sp_b_response = _invoke_agent(sp_b_client, serving_endpoint)
        assert sp_a_response != sp_b_response, (
            "SP-A and SP-B should see different identities from whoami()"
        )

    def test_sp_b_sees_own_identity(
        self, sp_b_client, sp_b_identity, serving_endpoint, serving_endpoint_ready
    ):
        response = _invoke_agent(sp_b_client, serving_endpoint)
        assert sp_b_identity in response, (
            f"Expected SP-B identity '{sp_b_identity}' in response, got: {response}"
        )


# =============================================================================
# Tests: Databricks Apps OBO
# =============================================================================


@pytest.mark.obo
class TestAppsOBO:
    """Invoke a pre-deployed Databricks App agent as two different SPs."""

    def test_sp_a_and_sp_b_see_different_identities(self, sp_a_client, sp_b_client, app_name):
        model = f"apps/{app_name}"
        sp_a_response = _invoke_agent(sp_a_client, model)
        sp_b_response = _invoke_agent(sp_b_client, model)
        assert sp_a_response != sp_b_response, (
            "SP-A and SP-B should see different identities from whoami()"
        )

    def test_sp_b_sees_own_identity(self, sp_b_client, sp_b_identity, app_name):
        model = f"apps/{app_name}"
        response = _invoke_agent(sp_b_client, model)
        assert sp_b_identity in response, (
            f"Expected SP-B identity '{sp_b_identity}' in response, got: {response}"
        )
