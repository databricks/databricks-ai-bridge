"""
Integration tests for DatabricksOpenAI with use_ai_gateway_native_api=True.

Prerequisites:
- AI Gateway V2 must be enabled on the test workspace.
- Set DATABRICKS_CONFIG_PROFILE (or use default credentials).

Run with:
    RUN_AI_GATEWAY_NATIVE_API_TESTS=1 python -m pytest \
        tests/integration_tests/test_ai_gateway_native_api.py -v
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from databricks.sdk import WorkspaceClient

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AI_GATEWAY_NATIVE_API_TESTS") != "1",
    reason="AI Gateway native API tests disabled. Set RUN_AI_GATEWAY_NATIVE_API_TESTS=1 to enable.",
)

_TEST_MODEL = os.environ.get("AI_GATEWAY_NATIVE_API_MODEL", "databricks-gpt-5-4")
_TEST_INPUT = [{"role": "user", "content": "Reply with exactly the word PONG and nothing else."}]


@pytest.fixture(scope="module")
def workspace_client():
    return WorkspaceClient()


@pytest.fixture(scope="module")
def sync_client(workspace_client):
    return DatabricksOpenAI(workspace_client=workspace_client, use_ai_gateway_native_api=True)


@pytest_asyncio.fixture(scope="module")
async def async_client(workspace_client):
    return AsyncDatabricksOpenAI(workspace_client=workspace_client, use_ai_gateway_native_api=True)


class TestAIGatewayNativeAPISync:
    def test_base_url_uses_openai_path(self, sync_client):
        assert "/openai/v1" in str(sync_client.base_url)
        assert "ai-gateway" in str(sync_client.base_url)

    def test_responses(self, sync_client):
        response = sync_client.responses.create(
            model=_TEST_MODEL,
            input=_TEST_INPUT,
            max_output_tokens=50,
        )
        assert response.output_text is not None


@pytest.mark.asyncio
class TestAIGatewayNativeAPIAsync:
    async def test_base_url_uses_openai_path(self, async_client):
        assert "/openai/v1" in str(async_client.base_url)
        assert "ai-gateway" in str(async_client.base_url)

    async def test_responses(self, async_client):
        response = await async_client.responses.create(
            model=_TEST_MODEL,
            input=_TEST_INPUT,
            max_output_tokens=50,
        )
        assert response.output_text is not None
