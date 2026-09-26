from unittest.mock import MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient

from databricks_mcp import DatabricksOAuthClientProvider

SERVER_URL = "https://test-databricks.com/ai-gateway/mcp-services/system.ai.slack"


def test_oauth_provider_requires_mcp_server_url():
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")

    with pytest.raises(TypeError, match="server_url"):
        DatabricksOAuthClientProvider(workspace_client=workspace_client)  # ty: ignore[missing-argument]


@pytest.mark.asyncio
async def test_oauth_provider():
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")
    with patch.object(workspace_client.current_user, "me", return_value=MagicMock()):
        provider = DatabricksOAuthClientProvider(
            workspace_client=workspace_client, server_url=SERVER_URL
        )
        oauth_token = await provider.context.storage.get_tokens()
        assert oauth_token is not None
        assert oauth_token.access_token == "test-token"
        assert oauth_token.expires_in == 60
        assert oauth_token.token_type.lower() == "bearer"


@pytest.mark.asyncio
async def test_oauth_provider_uses_mcp_server_url_for_resource_validation():
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")
    with patch.object(workspace_client.current_user, "me", return_value=MagicMock()):
        provider = DatabricksOAuthClientProvider(
            workspace_client=workspace_client,
            server_url=SERVER_URL,
        )

    assert provider.context.server_url == SERVER_URL


@pytest.mark.asyncio
async def test_oauth_provider_initializes_client_metadata_for_mcp_130():
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")

    with patch.object(workspace_client.current_user, "me", return_value=MagicMock()):
        provider = DatabricksOAuthClientProvider(
            workspace_client=workspace_client, server_url=SERVER_URL
        )

    assert provider.context.client_metadata is not None
    assert provider.context.client_metadata.redirect_uris


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
                provider = DatabricksOAuthClientProvider(
                    workspace_client=workspace_client, server_url=SERVER_URL
                )

                oauth_token = await provider.context.storage.get_tokens()
                assert oauth_token is not None
                assert oauth_token.access_token == "test-token"
                assert oauth_token.expires_in == 60
                assert oauth_token.token_type.lower() == "bearer"
