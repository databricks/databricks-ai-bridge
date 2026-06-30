from collections.abc import AsyncGenerator

import httpx
from databricks.sdk import WorkspaceClient
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthToken
from typing_extensions import override

TOKEN_EXPIRATION_SECONDS = 60


class DatabricksTokenStorage(TokenStorage):
    """Read-only token storage that surfaces the active Databricks bearer token.

    Retained for callers that import `DatabricksTokenStorage` directly. Note:
    `DatabricksOAuthClientProvider` no longer uses this class (it adds the
    Authorization header directly — see the class docstring there for why).
    """

    def __init__(self, workspace_client):
        self.workspace_client = workspace_client

    async def get_tokens(self) -> OAuthToken | None:
        headers = self.workspace_client.config.authenticate()
        authorization_header = headers["Authorization"]
        if not authorization_header.startswith("Bearer "):
            raise ValueError("Invalid authentication token format. Expected Bearer token.")

        token = authorization_header.split("Bearer ")[1]
        return OAuthToken(access_token=token, expires_in=TOKEN_EXPIRATION_SECONDS)


class DatabricksOAuthClientProvider(OAuthClientProvider):
    """
    An httpx auth provider for Databricks-fronted MCP servers. The credential
    is fully managed by `WorkspaceClient.config.authenticate()` (which already
    handles OBO / U2M / PAT / SP refresh), so this class is reduced to a thin
    bearer-token stamp on each outgoing request.

    Implementation note: earlier versions of this class inherited the full
    `mcp.client.auth.OAuthClientProvider.async_auth_flow` behavior, which
    acquires `self.context.lock` and holds it **across** the `yield request`
    that suspends until httpx receives the HTTP response:

        async with self.context.lock:
            ...
            response = yield request   # lock still held while awaiting response
            ...

    `mcp.client.streamable_http.streamablehttp_client` spawns two concurrent
    tasks that share the same auth provider:

      * a long-lived GET (`handle_get_stream`) that subscribes to
        server-pushed SSE events for the session; against many MCP servers
        (e.g. UC-Connection-backed Atlassian / Jira / Confluence) this GET
        stays open indefinitely waiting on the channel.
      * per-JSON-RPC POSTs (`_handle_post_request`) for `tools/list`,
        `tools/call`, …

    Both go through `auth.async_auth_flow`. Because the GET acquires the
    auth lock first and never releases it (its response never returns), every
    POST queues on that lock and eventually fails with::

        mcp.shared.exceptions.McpError: Timed out while waiting for response
        to ClientRequest. Waited 20.0 seconds.

    even though the server is healthy (a direct `httpx` call returns the
    same response in <1s).

    The Databricks scenario doesn't need any of the upstream OAuth-dance
    machinery the lock guards (no client registration, no PKCE, no
    callback handlers, no refresh-token flow) — the credential is just
    whatever `WorkspaceClient` already produced. So this override skips
    the parent's `async_auth_flow` and stamps the Authorization header
    directly, without ever taking the lock. That eliminates the deadlock
    between GET and POST tasks on the streamable-HTTP transport.

    Usage:
        .. code-block:: python

            from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
            from mcp.client.streamable_http import streamablehttp_client
            from mcp.client.session import ClientSession

            workspace_client = WorkspaceClient()

            async with streamablehttp_client(
                url="https://mcp-server-url",
                auth=DatabricksOAuthClientProvider(workspace_client),
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

    Args:
        workspace_client (databricks.sdk.WorkspaceClient): The Databricks workspace client used for authentication and requests.
    """

    def __init__(self, workspace_client: WorkspaceClient):
        self.workspace_client = workspace_client
        # Retained for backward compatibility — some callers reference this
        # attribute directly. Not used by `async_auth_flow` anymore.
        self.databricks_token_storage = DatabricksTokenStorage(workspace_client)

        super().__init__(
            server_url="",
            client_metadata=None,  # ty:ignore[invalid-argument-type]: No metadata available
            storage=self.databricks_token_storage,
            redirect_handler=None,
            callback_handler=None,
        )

    @override
    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """Stamp the current Databricks bearer token and yield. No lock.

        Overrides `OAuthClientProvider.async_auth_flow`, which holds an
        `async with self.context.lock:` across the request yield and would
        otherwise serialize all concurrent requests through this provider
        (see class docstring for the deadlock this caused on the
        streamable-HTTP transport).
        """
        headers = self.workspace_client.config.authenticate()
        authorization = headers.get("Authorization")
        if authorization:
            request.headers["Authorization"] = authorization
        yield request
