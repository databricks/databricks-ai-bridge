from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient

from databricks_mcp import DatabricksOAuthClientProvider


@pytest.mark.asyncio
async def test_oauth_provider():
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")
    with patch.object(workspace_client.current_user, "me", return_value=MagicMock()):
        provider = DatabricksOAuthClientProvider(workspace_client=workspace_client)
        oauth_token = await provider.context.storage.get_tokens()
        assert oauth_token is not None
        assert oauth_token.access_token == "test-token"
        assert oauth_token.expires_in == 60
        assert oauth_token.token_type.lower() == "bearer"


@pytest.mark.asyncio
async def test_authenticate_raises_exception():
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")

    with patch.object(workspace_client.current_user, "me", return_value=MagicMock()):
        with patch.object(
            workspace_client.config, "authenticate", return_value={"Authorization": "Basic abc123"}
        ):
            with pytest.raises(
                ValueError, match="Invalid authentication token format. Expected Bearer token."
            ):
                provider = DatabricksOAuthClientProvider(workspace_client=workspace_client)

                oauth_token = await provider.context.storage.get_tokens()
                assert oauth_token is not None
                assert oauth_token.access_token == "test-token"
                assert oauth_token.expires_in == 60
                assert oauth_token.token_type.lower() == "bearer"
