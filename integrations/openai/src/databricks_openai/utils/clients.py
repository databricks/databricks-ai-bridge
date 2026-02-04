from typing import Generator

from databricks.sdk import WorkspaceClient
from httpx import AsyncClient, Auth, Client, Request, Response
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI
from openai.resources.chat import AsyncChat, Chat
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.resources.responses import AsyncResponses, Responses
from typing_extensions import override

# Prefix for routing requests to Databricks Apps
_APPS_ENDPOINT_PREFIX = "apps/"
# Domain pattern indicating a Databricks App URL
_DATABRICKS_APPS_DOMAIN = "databricksapps"


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


def _get_authorized_http_client(workspace_client: WorkspaceClient) -> Client:
    databricks_token_auth = BearerAuth(workspace_client.config.authenticate)
    return Client(auth=databricks_token_auth)


def _get_authorized_async_http_client(workspace_client: WorkspaceClient) -> AsyncClient:
    databricks_token_auth = BearerAuth(workspace_client.config.authenticate)
    return AsyncClient(auth=databricks_token_auth)


def _validate_oauth_for_apps(workspace_client: WorkspaceClient) -> None:
    """Validate that workspace_client uses OAuth (required for Apps)."""
    try:
        workspace_client.config.oauth_token()
    except Exception as e:
        raise ValueError(
            "Querying Databricks Apps requires OAuth authentication. "
            "See https://docs.databricks.com/aws/en/dev-tools/auth/oauth-u2m.html "
            "or https://docs.databricks.com/aws/en/dev-tools/auth/oauth-m2m.html"
        ) from e


def _get_app_url(workspace_client: WorkspaceClient, app_name: str) -> str:
    """Look up the URL for a Databricks App by name."""
    try:
        app = workspace_client.apps.get(name=app_name)
    except Exception as e:
        raise ValueError(
            f"Failed to get Databricks App '{app_name}'. "
            f"Make sure the app exists and you have permission. Error: {e}"
        ) from e

    if not app.url:
        raise ValueError(f"App '{app_name}' has no URL. Ensure it's deployed.")

    return app.url


def _wrap_app_error(e: Exception, app_name: str) -> ValueError:
    """Wrap OpenAI API errors with helpful hints for Databricks Apps."""
    if isinstance(e, APIStatusError):
        status_code = e.status_code
        message = e.message
        if status_code == 404 or status_code == 405:
            hint = (
                f"Hint: App '{app_name}' may not support the OpenAI Responses API. "
                f"Ensure the app implements the /responses endpoint."
            )
        elif status_code == 403:
            hint = f"Hint: Ensure you have CAN_USE permission on app '{app_name}'."
        elif status_code >= 500:
            hint = (
                f"Hint: App '{app_name}' encountered an internal error. "
                f"Check the app logs and status in the Databricks workspace."
            )
        elif "DNS" in message or "resolution" in message.lower():
            hint = (
                f"Hint: App '{app_name}' may be stopped or unavailable. "
                f"Check the app status in the Databricks workspace."
            )
        else:
            hint = None

        error_msg = f"Error querying app '{app_name}': {status_code} - {message}"
        if hint:
            error_msg = f"{error_msg}\n{hint}"
        return ValueError(error_msg)
    elif isinstance(e, APIConnectionError):
        message = str(e)
        if "DNS" in message or "resolution" in message.lower():
            hint = (
                f"Hint: App '{app_name}' may be stopped or unavailable. "
                f"Check the app status in the Databricks workspace."
            )
        else:
            hint = (
                f"Hint: App '{app_name}' may be starting up or unavailable. "
                f"Check the app status in the Databricks workspace."
            )
        return ValueError(f"Error connecting to app '{app_name}': {message}\n{hint}")
    return ValueError(f"Error querying app '{app_name}': {e}")


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


class DatabricksResponses(Responses):
    """Responses resource that handles apps/ prefix routing."""

    def __init__(self, client, workspace_client: WorkspaceClient):
        super().__init__(client)
        self._workspace_client = workspace_client
        self._app_clients_cache: dict[str, OpenAI] = {}

    def _get_app_client(self, app_name: str) -> OpenAI:
        """Get or create a client for a specific app."""
        if app_name not in self._app_clients_cache:
            _validate_oauth_for_apps(self._workspace_client)
            app_url = _get_app_url(self._workspace_client, app_name)
            # Authentication is handled via http_client, not api_key
            self._app_clients_cache[app_name] = OpenAI(
                base_url=app_url,
                api_key="no-token",
                http_client=_get_authorized_http_client(self._workspace_client),
            )
        return self._app_clients_cache[app_name]

    def create(self, **kwargs):
        model = kwargs.get("model", "")

        if isinstance(model, str) and model.startswith(_APPS_ENDPOINT_PREFIX):
            app_name = model[len(_APPS_ENDPOINT_PREFIX) :]
            app_client = self._get_app_client(app_name)
            try:
                return app_client.responses.create(**kwargs)
            except (APIStatusError, APIConnectionError) as e:
                raise _wrap_app_error(e, app_name) from e

        return super().create(**kwargs)


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
        base_url: Optional base URL to override the default serving endpoints URL. When the URL
            points to a Databricks App (contains "databricksapps"), OAuth authentication is
            required.

    Example - Query a serving endpoint:
        >>> client = DatabricksOpenAI()
        >>> response = client.chat.completions.create(
        ...     model="databricks-meta-llama-3-1-70b-instruct",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )

    Example - Query a Databricks App directly by URL (requires OAuth):
        >>> # WorkspaceClient must be configured with OAuth authentication
        >>> # See: https://docs.databricks.com/aws/en/dev-tools/auth/oauth-u2m.html
        >>> client = DatabricksOpenAI(
        ...     base_url="https://my-app.aws.databricksapps.com",
        ...     workspace_client=WorkspaceClient(),
        ... )
        >>> response = client.responses.create(
        ...     input=[{"role": "user", "content": "Hello"}],
        ... )

    Example - Query a Databricks App by name (requires OAuth):
        >>> # WorkspaceClient must be configured with OAuth authentication
        >>> # See: https://docs.databricks.com/aws/en/dev-tools/auth/oauth-u2m.html
        >>> client = DatabricksOpenAI()
        >>> response = client.responses.create(
        ...     model="apps/my-agent",  # Looks up app URL automatically
        ...     input=[{"role": "user", "content": "Hello"}],
        ... )
    """

    def __init__(
        self,
        workspace_client: WorkspaceClient | None = None,
        base_url: str | None = None,
    ):
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        self._workspace_client = workspace_client

        if base_url is not None:
            # Only validate OAuth for Databricks App URLs
            if _DATABRICKS_APPS_DOMAIN in base_url:
                _validate_oauth_for_apps(workspace_client)
            target_base_url = base_url
        else:
            # Default: Serving endpoints
            target_base_url = f"{workspace_client.config.host}/serving-endpoints"

        # Authentication is handled via http_client, not api_key
        super().__init__(
            base_url=target_base_url,
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

    @property
    def responses(self) -> Responses:
        if not hasattr(self, "_databricks_responses"):
            self._databricks_responses = DatabricksResponses(self, self._workspace_client)
        return self._databricks_responses


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


class AsyncDatabricksResponses(AsyncResponses):
    """Async Responses resource that handles apps/ prefix routing."""

    def __init__(self, client, workspace_client: WorkspaceClient):
        super().__init__(client)
        self._workspace_client = workspace_client
        self._app_clients_cache: dict[str, AsyncOpenAI] = {}

    def _get_app_client(self, app_name: str) -> AsyncOpenAI:
        """Get or create an async client for a specific app."""
        if app_name not in self._app_clients_cache:
            _validate_oauth_for_apps(self._workspace_client)
            app_url = _get_app_url(self._workspace_client, app_name)
            # Authentication is handled via http_client, not api_key
            self._app_clients_cache[app_name] = AsyncOpenAI(
                base_url=app_url,
                api_key="no-token",
                http_client=_get_authorized_async_http_client(self._workspace_client),
            )
        return self._app_clients_cache[app_name]

    async def create(self, **kwargs):
        model = kwargs.get("model", "")

        if isinstance(model, str) and model.startswith(_APPS_ENDPOINT_PREFIX):
            app_name = model[len(_APPS_ENDPOINT_PREFIX) :]
            app_client = self._get_app_client(app_name)
            try:
                return await app_client.responses.create(**kwargs)
            except (APIStatusError, APIConnectionError) as e:
                raise _wrap_app_error(e, app_name) from e

        return await super().create(**kwargs)


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
        base_url: Optional base URL to override the default serving endpoints URL. When the URL
            points to a Databricks App (contains "databricksapps"), OAuth authentication is
            required.

    Example - Query a serving endpoint:
        >>> client = AsyncDatabricksOpenAI()
        >>> response = await client.chat.completions.create(
        ...     model="databricks-meta-llama-3-1-70b-instruct",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )

    Example - Query a Databricks App directly by URL (requires OAuth):
        >>> # WorkspaceClient must be configured with OAuth authentication
        >>> # See: https://docs.databricks.com/aws/en/dev-tools/auth/oauth-u2m.html
        >>> client = AsyncDatabricksOpenAI(
        ...     base_url="https://my-app.aws.databricksapps.com",
        ...     workspace_client=WorkspaceClient(),
        ... )
        >>> response = await client.responses.create(
        ...     input=[{"role": "user", "content": "Hello"}],
        ... )

    Example - Query a Databricks App by name (requires OAuth):
        >>> # WorkspaceClient must be configured with OAuth authentication
        >>> # See: https://docs.databricks.com/aws/en/dev-tools/auth/oauth-u2m.html
        >>> client = AsyncDatabricksOpenAI()
        >>> response = await client.responses.create(
        ...     model="apps/my-agent",  # Looks up app URL automatically
        ...     input=[{"role": "user", "content": "Hello"}],
        ... )
    """

    def __init__(
        self,
        workspace_client: WorkspaceClient | None = None,
        base_url: str | None = None,
    ):
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        self._workspace_client = workspace_client

        if base_url is not None:
            # Only validate OAuth for Databricks App URLs
            if _DATABRICKS_APPS_DOMAIN in base_url:
                _validate_oauth_for_apps(workspace_client)
            target_base_url = base_url
        else:
            # Default: Serving endpoints
            target_base_url = f"{workspace_client.config.host}/serving-endpoints"

        # Authentication is handled via http_client, not api_key
        super().__init__(
            base_url=target_base_url,
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

    @property
    def responses(self) -> AsyncResponses:
        if not hasattr(self, "_databricks_responses"):
            self._databricks_responses = AsyncDatabricksResponses(self, self._workspace_client)
        return self._databricks_responses
