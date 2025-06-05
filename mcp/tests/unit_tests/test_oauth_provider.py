import pytest
from databricks.sdk import WorkspaceClient

from databricks_mcp import DatabricksOAuthClientProvider


@pytest.mark.asyncio
async def test_oauth_provider():
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")
    provider = DatabricksOAuthClientProvider(workspace_client=workspace_client)
    oauth_token = await provider.storage.get_tokens()
    assert oauth_token.access_token == "test-token"
    assert oauth_token.expires_in == 60
    assert oauth_token.token_type == "bearer"
