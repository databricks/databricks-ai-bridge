import asyncio
from unittest.mock import MagicMock, patch

import httpx
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
async def test_async_auth_flow_stamps_bearer_token():
    """`async_auth_flow` must add the active Databricks bearer token to the
    request as `Authorization: Bearer <token>`, and otherwise pass through.
    """
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")
    with patch.object(workspace_client.current_user, "me", return_value=MagicMock()):
        provider = DatabricksOAuthClientProvider(workspace_client=workspace_client)
        request = httpx.Request("POST", "https://test-databricks.com/api/2.0/mcp/some-path")

        flow = provider.async_auth_flow(request)
        stamped = await flow.__anext__()

        assert stamped is request
        assert stamped.headers["Authorization"] == "Bearer test-token"

        # The generator should complete after the single yield.
        with pytest.raises(StopAsyncIteration):
            await flow.__anext__()


@pytest.mark.asyncio
async def test_async_auth_flow_does_not_serialize_concurrent_requests():
    """Regression test for the streamable-HTTP deadlock.

    `mcp.client.streamable_http.streamablehttp_client` opens a long-lived GET
    (`handle_get_stream`) for server-pushed SSE events AND fires per-RPC POSTs
    through the same auth provider. The upstream `OAuthClientProvider`'s
    `async_auth_flow` acquires `self.context.lock` and holds it *across* the
    `yield request` (i.e. for the entire HTTP request lifetime). When the GET
    is long-lived (Atlassian / Jira / Confluence — UC-Connection-backed MCPs
    keep the SSE channel open indefinitely), the lock is never released and
    every POST queues behind it until `client_session_timeout_seconds` fires:

        mcp.shared.exceptions.McpError: Timed out while waiting for response
        to ClientRequest. Waited 20.0 seconds.

    Our override skips the parent's locked flow entirely (no OAuth dance is
    needed — the credential is fully managed by WorkspaceClient). This test
    pins that behavior by interleaving two `async_auth_flow` calls and
    asserting neither blocks on the other.
    """
    workspace_client = WorkspaceClient(host="https://test-databricks.com", token="test-token")
    with patch.object(workspace_client.current_user, "me", return_value=MagicMock()):
        provider = DatabricksOAuthClientProvider(workspace_client=workspace_client)

        # Simulate the streamable-HTTP transport's pattern: one long-lived
        # request (the GET) holding its auth_flow generator open while a
        # second short-lived request (the POST) needs to authenticate.
        long_lived_req = httpx.Request("GET", "https://test-databricks.com/api/2.0/mcp/some-path")
        short_lived_req = httpx.Request("POST", "https://test-databricks.com/api/2.0/mcp/some-path")

        long_flow = provider.async_auth_flow(long_lived_req)
        # Yield once to "send" the GET; do NOT close the generator — this
        # mirrors the live GET sitting on an open SSE channel.
        await long_flow.__anext__()

        # The POST's auth_flow must complete promptly even while the GET's
        # flow is still open. Wrap in a tight timeout: a regression
        # (lock-bound flow) would hang here until the GET closes.
        short_flow = provider.async_auth_flow(short_lived_req)
        stamped = await asyncio.wait_for(short_flow.__anext__(), timeout=1.0)

        assert stamped is short_lived_req
        assert stamped.headers["Authorization"] == "Bearer test-token"


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
