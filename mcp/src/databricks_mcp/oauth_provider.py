from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthToken
from databricks_sdk import WorkspaceClient

class DatabricksTokenStorage(TokenStorage):
    def __init__(self, workspace_client):
        self.workspace_client = workspace_client

    async def get_tokens(self) -> OAuthToken | None:
        headers = self.workspace_client.config.authenticate()
        token = headers["Authorization"].split("Bearer ")[1]
        return OAuthToken(access_token=token, expires_in=60)


class DatabricksOAuthClientProvider(OAuthClientProvider):
    def __init__(self, workspace_client: WorkspaceClient):
        self.databricks_token_storage = DatabricksTokenStorage(workspace_client)

        super.__init__(
            server_url="",
            client_metadata=None,
            storage=self.databricks_token_storage,
            redirect_handler=None,
            callback_handler=None,
        )