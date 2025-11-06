from databricks.sdk import WorkspaceClient
from httpx import AsyncClient, Auth, Client, Request
from openai import AsyncOpenAI, OpenAI


def _get_authorized_http_client(workspace_client):
    class BearerAuth(Auth):
        def __init__(self, get_headers_func):
            self.get_headers_func = get_headers_func

        def auth_flow(self, request: Request) -> Request:
            auth_headers = self.get_headers_func()
            request.headers["Authorization"] = auth_headers["Authorization"]
            yield request

    databricks_token_auth = BearerAuth(workspace_client.config.authenticate)
    return Client(auth=databricks_token_auth)


def _get_authorized_async_http_client(workspace_client):
    class BearerAuth(Auth):
        def __init__(self, get_headers_func):
            self.get_headers_func = get_headers_func

        def auth_flow(self, request: Request) -> Request:
            auth_headers = self.get_headers_func()
            request.headers["Authorization"] = auth_headers["Authorization"]
            yield request

    databricks_token_auth = BearerAuth(workspace_client.config.authenticate)
    return AsyncClient(auth=databricks_token_auth)


class DatabricksOpenAI(OpenAI):
    def __init__(self, workspace_client: WorkspaceClient = None):
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        current_host = workspace_client.config.host
        super().__init__(
            base_url=f"{current_host}/serving-endpoints",
            api_key="no-token",
            http_client=_get_authorized_http_client(workspace_client),
        )


class AsyncDatabricksOpenAI(AsyncOpenAI):
    def __init__(self, workspace_client: WorkspaceClient = None):
        if workspace_client is None:
            workspace_client = WorkspaceClient()
        current_host = workspace_client.config.host
        super().__init__(
            base_url=f"{current_host}/serving-endpoints",
            api_key="no-token",
            http_client=_get_authorized_async_http_client(workspace_client),
        )
