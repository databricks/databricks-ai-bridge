from databricks.sdk import WorkspaceClient
from httpx import AsyncClient, Auth, Client, Request
from openai import AsyncOpenAI, OpenAI


class BearerAuth(Auth):
    def __init__(self, get_headers_func):
        self.get_headers_func = get_headers_func

    def auth_flow(self, request: Request) -> Request:
        auth_headers = self.get_headers_func()
        request.headers["Authorization"] = auth_headers["Authorization"]
        yield request


def _get_authorized_http_client(workspace_client):
    databricks_token_auth = BearerAuth(workspace_client.config.authenticate)
    return Client(auth=databricks_token_auth)


def _get_authorized_async_http_client(workspace_client):
    databricks_token_auth = BearerAuth(workspace_client.config.authenticate)
    return AsyncClient(auth=databricks_token_auth)


class DatabricksOpenAI(OpenAI):
    """OpenAI client authenticated with Databricks to query LLMs and agents hosted on Databricks.

    This client extends the standard OpenAI client with Databricks authentication, allowing you
    to interact with foundation models and AI agents deployed on Databricks using the familiar
    OpenAI SDK interface.

    The client automatically handles authentication using your Databricks credentials.

    Args:
        workspace_client: Databricks WorkspaceClient to use for authentication. Pass a custom
            WorkspaceClient to set up your own authentication method. If not provided, a default
            WorkspaceClient will be created using standard Databricks authentication resolution.

    Example:
        >>> # Use default Databricks authentication
        >>> client = DatabricksOpenAI()
        >>> response = client.chat.completions.create(
        ...     model="databricks-meta-llama-3-1-70b-instruct",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>> # Use custom WorkspaceClient for authentication
        >>> from databricks.sdk import WorkspaceClient
        >>> ws = WorkspaceClient(host="https://my-workspace.cloud.databricks.com", token="...")
        >>> client = DatabricksOpenAI(workspace_client=ws)
    """

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
    """Async OpenAI client authenticated with Databricks to query LLMs and agents hosted on Databricks.

    This client extends the standard AsyncOpenAI client with Databricks authentication, allowing you
    to interact with foundation models and AI agents deployed on Databricks using the familiar
    OpenAI SDK interface with async/await support.

    The client automatically handles authentication using your Databricks credentials.

    Args:
        workspace_client: Databricks WorkspaceClient to use for authentication. Pass a custom
            WorkspaceClient to set up your own authentication method. If not provided, a default
            WorkspaceClient will be created using standard Databricks authentication resolution.

    Example:
        >>> # Use default Databricks authentication
        >>> client = AsyncDatabricksOpenAI()
        >>> response = await client.chat.completions.create(
        ...     model="databricks-meta-llama-3-1-70b-instruct",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )
        >>> # Use custom WorkspaceClient for authentication
        >>> from databricks.sdk import WorkspaceClient
        >>> ws = WorkspaceClient(host="https://my-workspace.cloud.databricks.com", token="...")
        >>> client = AsyncDatabricksOpenAI(workspace_client=ws)
    """

    def __init__(self, workspace_client: WorkspaceClient = None):
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        current_host = workspace_client.config.host
        super().__init__(
            base_url=f"{current_host}/serving-endpoints",
            api_key="no-token",
            http_client=_get_authorized_async_http_client(workspace_client),
        )
