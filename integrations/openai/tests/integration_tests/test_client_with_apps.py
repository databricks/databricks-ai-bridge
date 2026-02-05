import os

import pytest
from databricks.sdk import WorkspaceClient

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI


def _get_config_profile():
    return os.environ.get("DATABRICKS_CONFIG_PROFILE", "dogfood")


def _get_test_app_name():
    return os.environ.get("DATABRICKS_TEST_APP_NAME", "agent-customer-support")


class TestDatabricksApps:
    @pytest.fixture
    def client(self):
        ws = WorkspaceClient(profile=_get_config_profile())
        return DatabricksOpenAI(workspace_client=ws)

    @pytest.fixture
    def async_client(self):
        ws = WorkspaceClient(profile=_get_config_profile())
        return AsyncDatabricksOpenAI(workspace_client=ws)

    def test_responses_create_with_apps_prefix(self, client):
        app_name = _get_test_app_name()
        response = client.responses.create(
            model=f"apps/{app_name}",
            input=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        assert len(list(response)) > 0

    @pytest.mark.asyncio
    async def test_async_responses_create_with_apps_prefix(self, async_client):
        app_name = _get_test_app_name()
        response = await async_client.responses.create(
            model=f"apps/{app_name}",
            input=[{"role": "user", "content": "Hello"}],
        )
        assert response is not None
        assert response.output is not None

    def test_responses_create_with_direct_base_url(self):
        ws = WorkspaceClient(profile=_get_config_profile())
        app_name = _get_test_app_name()
        app = ws.apps.get(name=app_name)

        client = DatabricksOpenAI(
            workspace_client=ws,
            base_url=app.url,
        )
        response = client.responses.create(
            input=[{"role": "user", "content": "Hello"}],
        )
        assert response is not None
        assert response.output is not None
