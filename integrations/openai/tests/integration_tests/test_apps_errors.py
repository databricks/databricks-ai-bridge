import os

import pytest
from databricks.sdk import WorkspaceClient

from databricks_openai import DatabricksOpenAI

DATABRICKS_CONFIG_PROFILE = "dogfood"


class TestDatabricksAppsErrors:
    @pytest.fixture
    def client(self):
        ws = WorkspaceClient(profile=DATABRICKS_CONFIG_PROFILE)
        return DatabricksOpenAI(workspace_client=ws)

    @pytest.mark.parametrize(
        "app_name",
        [
            "ai-slide-generator-dev",
            "agent-builder-assistant",
            "at-observability-poc",
        ],
    )
    def test_responses_endpoint_not_found(self, client, app_name):
        with pytest.raises(ValueError, match=r"(?s)(404|405).*Hint:.*(/responses endpoint)"):
            client.responses.create(
                model=f"apps/{app_name}",
                input=[{"role": "user", "content": "Hello"}],
            )

    def test_app_compute_stopped(self, client):
        with pytest.raises(ValueError, match=r"(?s)DNS.*Hint:.*stopped"):
            client.responses.create(
                model="apps/alexmon2",
                input=[{"role": "user", "content": "Hello"}],
            )

    def test_no_permission_to_app(self):
        client_id = os.environ.get("TEST_SP_CLIENT_ID")
        client_secret = os.environ.get("TEST_SP_CLIENT_SECRET")
        if not client_id or not client_secret:
            pytest.skip("TEST_SP_CLIENT_ID and TEST_SP_CLIENT_SECRET env vars required")

        ws = WorkspaceClient(
            host="https://e2-dogfood.staging.cloud.databricks.com",
            client_id=client_id,
            client_secret=client_secret,
        )
        client = DatabricksOpenAI(workspace_client=ws)

        with pytest.raises(ValueError, match=r"(?s)(404|405).*Hint:.*(/responses endpoint)"):
            client.responses.create(
                model="apps/ai-slide-generator-dev",
                input=[{"role": "user", "content": "Hello"}],
            )
