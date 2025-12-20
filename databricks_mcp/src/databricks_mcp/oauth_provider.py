from databricks.sdk import WorkspaceClient
from databricks.sdk.errors.platform import PermissionDenied
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthToken

TOKEN_EXPIRATION_SECONDS = 60


class DatabricksTokenStorage(TokenStorage):
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
    An OAuthClientProvider for Databricks. This class extends mcp.client.auth.OAuthClientProvider
    and can be used with the `mcp.client.streamable_http` to authorize the MCP Server with Databricks.

    Usage:
        .. code-block:: python

            from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
            from mcp.client.streamable_http import streamablehttp_client
            from mcp.client.session import ClientSession

            # Initialize the Databricks workspace client
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

        # Pre-flight check: verify the workspace client has basic permissions
        try:
            workspace_client.current_user.me()
        except PermissionDenied as e:
            raise PermissionError(
                f"The workspace client does not have permission to access the Databricks workspace. "
                f"Please ensure the service principal or user has the required permissions to call Databricks APIs. "
                f"Original error: {e}"
            ) from e

        self.databricks_token_storage = DatabricksTokenStorage(workspace_client)

        super().__init__(
            server_url="",
            client_metadata=None,  # ty:ignore[invalid-argument-type]: No metadata available
            storage=self.databricks_token_storage,
            redirect_handler=None,
            callback_handler=None,
        )
