from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import PermissionDenied

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


def test_preflight_check_raises_permission_denied():
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")

    with patch.object(
        workspace_client.current_user,
        "me",
        side_effect=PermissionDenied(
            "This API is disabled for users without the workspace-access entitlement."
        ),
    ):
        with pytest.raises(
            PermissionError,
            match="The workspace client does not have permission to access the Databricks workspace",
        ):
            DatabricksOAuthClientProvider(workspace_client=workspace_client)
