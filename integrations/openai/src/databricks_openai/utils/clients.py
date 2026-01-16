from typing import Generator

from databricks.sdk import WorkspaceClient
from httpx import AsyncClient, Auth, Client, Request, Response
from openai import AsyncOpenAI, OpenAI
from openai.resources.chat import AsyncChat, Chat
from openai.resources.chat.completions import AsyncCompletions, Completions
from typing_extensions import override


class BearerAuth(Auth):
    def __init__(self, get_headers_func):
        self.get_headers_func = get_headers_func

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        auth_headers = self.get_headers_func()
        request.headers["Authorization"] = auth_headers["Authorization"]
        yield request


def _strip_strict_from_tools(tools: list | None) -> list | None:
    """Remove 'strict' field from tool function definitions.

    Databricks model endpoints (except GPT) don't support the 'strict' field
    in tool schemas, but openai-agents SDK v0.6.4+ includes it.
    """
    if tools is None:
        return None
    for tool in tools:
        if isinstance(tool, dict) and "function" in tool:
            tool.get("function", {}).pop("strict", None)
    return tools


def _should_strip_strict(model: str | None) -> bool:
    """Determine if strict should be stripped based on model name.

    GPT models (hosted via Databricks) support the strict field.
    Non-GPT models (Claude, Llama, etc.) do not.
    """
    if model is None:
        return True  # Default to stripping if model unknown
    return "gpt" not in model.lower()


def _get_authorized_http_client(workspace_client):
    databricks_token_auth = BearerAuth(workspace_client.config.authenticate)
    return Client(auth=databricks_token_auth)


class DatabricksCompletions(Completions):
    """Completions that conditionally strips 'strict' from tools for non-GPT models."""

    def create(self, **kwargs):
        model = kwargs.get("model")
        if _should_strip_strict(model):
            _strip_strict_from_tools(kwargs.get("tools"))
        return super().create(**kwargs)


class DatabricksChat(Chat):
    """Chat resource that uses Databricks completions with strict stripping."""

    completions: DatabricksCompletions


class DatabricksOpenAI(OpenAI):
    """OpenAI client authenticated with Databricks to query LLMs and agents hosted on Databricks.

    This client extends the standard OpenAI client with Databricks authentication, allowing you
    to interact with foundation models and AI agents deployed on Databricks using the familiar
    OpenAI SDK interface.

    The client automatically handles authentication using your Databricks credentials.

    For non-GPT models (Claude, Llama, etc.), this client automatically strips the 'strict'
    field from tool definitions, as these models don't support this OpenAI-specific parameter.

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

    def __init__(self, workspace_client: WorkspaceClient | None = None):
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        current_host = workspace_client.config.host
        super().__init__(
            base_url=f"{current_host}/serving-endpoints",
            api_key="no-token",
            http_client=_get_authorized_http_client(workspace_client),
        )

    @override
    @property
    def chat(self) -> Chat:
        if not isinstance(super().chat, DatabricksChat):
            chat = super().chat
            # Replace the completions with our custom one
            chat_with_custom_completions = DatabricksChat(client=chat._client)
            chat_with_custom_completions.completions = DatabricksCompletions(
                client=chat.completions._client
            )
            return chat_with_custom_completions
        return super().chat


class AsyncDatabricksCompletions(AsyncCompletions):
    """Async completions that conditionally strips 'strict' from tools for non-GPT models."""

    async def create(self, **kwargs):
        model = kwargs.get("model")
        if _should_strip_strict(model):
            _strip_strict_from_tools(kwargs.get("tools"))
        return await super().create(**kwargs)


class AsyncDatabricksChat(AsyncChat):
    """Async chat resource that uses Databricks completions with strict stripping."""

    completions: AsyncDatabricksCompletions


def _get_authorized_async_http_client(workspace_client):
    databricks_token_auth = BearerAuth(workspace_client.config.authenticate)
    return AsyncClient(auth=databricks_token_auth)


class AsyncDatabricksOpenAI(AsyncOpenAI):
    """Async OpenAI client authenticated with Databricks to query LLMs and agents hosted on Databricks.

    This client extends the standard AsyncOpenAI client with Databricks authentication, allowing you
    to interact with foundation models and AI agents deployed on Databricks using the familiar
    OpenAI SDK interface with async/await support.

    The client automatically handles authentication using your Databricks credentials.

    For non-GPT models (Claude, Llama, etc.), this client automatically strips the 'strict'
    field from tool definitions, as these models don't support this OpenAI-specific parameter.

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

    def __init__(self, workspace_client: WorkspaceClient | None = None):
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        current_host = workspace_client.config.host
        super().__init__(
            base_url=f"{current_host}/serving-endpoints",
            api_key="no-token",
            http_client=_get_authorized_async_http_client(workspace_client),
        )

    @property
    def chat(self) -> AsyncChat:
        if not isinstance(super().chat, AsyncDatabricksChat):
            chat = super().chat
            # Replace the completions with our custom one
            chat_with_custom_completions = AsyncDatabricksChat(client=chat._client)
            chat_with_custom_completions.completions = AsyncDatabricksCompletions(
                client=chat.completions._client
            )
            return chat_with_custom_completions
        return super().chat
