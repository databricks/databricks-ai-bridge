import os
from typing import Any, Generator
from urllib.parse import urlparse

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


def _get_openai_api_key():
    """Return OPENAI_API_KEY from env if set, otherwise 'no-token'.

    Passed through to the OpenAI client so that the agents SDK tracing
    client can authenticate with the OpenAI API.
    """
    return os.environ.get("OPENAI_API_KEY") or "no-token"


class BearerAuth(Auth):
    def __init__(self, get_headers_func):
        self.get_headers_func = get_headers_func

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        auth_headers = self.get_headers_func()
        request.headers["Authorization"] = auth_headers["Authorization"]
        yield request


def _strip_strict_from_tools(tools: Any) -> Any:
    """Remove 'strict' field from tool function definitions.

    Databricks model endpoints (except GPT) don't support the 'strict' field
    in tool schemas, but openai-agents SDK v0.6.4+ includes it.
    """
    # Handle None or OpenAI's NOT_GIVEN/Omit sentinel types (non-iterable placeholders).
    # See https://deepwiki.com/openai/openai-python/5-data-types-and-models#special-types-and-sentinels
    if not tools:
        return tools
    for tool in tools:
        if isinstance(tool, dict) and "function" in tool:
            tool.get("function", {}).pop("strict", None)
    return tools


def _strip_strict_from_kwargs(kwargs: dict) -> dict:
    """Strip 'strict' from top-level kwargs which causes issues for GPT models."""
    kwargs.pop("strict", None)  # Remove top-level strict if present
    return kwargs


def _should_strip_strict(model: str | None) -> bool:
    """Determine if strict should be stripped based on model name.

    GPT models (hosted via Databricks) support the strict field.
    Non-GPT models (Claude, Llama, etc.) do not.
    """
    if model is None:
        return True  # Default to stripping if model unknown
    return "gpt" not in model.lower()


def _is_claude_model(model: str | None) -> bool:
    """Returns True if the model is a Claude variant."""
    if not model:
        return False
    return "claude" in model.lower()


def _is_empty_content(content: Any) -> bool:
    """Check if message content is effectively empty."""
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        if not content:
            return True
        return all(
            isinstance(part, dict)
            and part.get("type") == "text"
            and not (part.get("text") or "").strip()
            for part in content
        )
    return False


def _fix_empty_assistant_content_in_messages(messages: Any) -> None:
    """Replace empty assistant content with a single space when tool_calls are present.

    Claude models on Databricks reject empty text content blocks in assistant messages.
    When tool_calls are present but content is empty, set content to " " to avoid errors.
    """
    if not messages:
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant" and message.get("tool_calls"):
            if _is_empty_content(message.get("content")):
                message["content"] = " "


def _discover_ai_gateway_host(
    http_client: Client,
    host: str,
) -> str | None:
    """Discover the AI Gateway host URL (scheme + netloc only).

    Calls GET /api/ai-gateway/v2/endpoints. If successful and endpoints exist,
    extracts the ai_gateway_url from the first endpoint response and returns
    just the scheme + netloc (no path). Returns None if gateway is not available.
    """
    try:
        response = http_client.get(f"{host}/api/ai-gateway/v2/endpoints")
        if response.status_code != 200:
            return None
        data = response.json()
        endpoints = data.get("endpoints", [])
        if not endpoints:
            return None
        gateway_url = endpoints[0].get("ai_gateway_url")
        if not gateway_url:
            return None
        parsed = urlparse(gateway_url)
        return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        return None


def _get_ai_gateway_base_url(
    http_client: Client,
    host: str,
) -> str | None:
    """Check if AI Gateway V2 is enabled and return its MLflow base URL.

    Returns the AI Gateway base URL with /mlflow/v1 path appended, or None if
    the gateway is not available.
    """
    gateway_host = _discover_ai_gateway_host(http_client, host)
    return f"{gateway_host}/mlflow/v1" if gateway_host else None


def _resolve_base_url(
    workspace_client: WorkspaceClient,
    base_url: str | None,
    use_ai_gateway: bool,
    http_client: Client,
    ai_gateway_native_api: bool = False,
) -> str:
    """Resolve the target base URL for the OpenAI client."""
    if base_url is not None:
        if _DATABRICKS_APPS_DOMAIN in base_url:
            _validate_oauth_for_apps(workspace_client)
        return base_url

    # Native provider API via AI Gateway (e.g. OpenAI-compatible /openai path)
    if ai_gateway_native_api:
        gateway_host = _discover_ai_gateway_host(http_client, workspace_client.config.host)
        if gateway_host:
            return f"{gateway_host}/openai"
        raise ValueError(
            "Please ensure AI Gateway V2 is enabled for the workspace "
            "when ai_gateway_native_api is set to True."
        )

    # MLflow-format AI Gateway endpoints
    if use_ai_gateway:
        gateway_url = _get_ai_gateway_base_url(http_client, workspace_client.config.host)
        if gateway_url:
            return gateway_url
        raise ValueError(
            "Please ensure AI Gateway V2 is enabled for the workspace "
            "when use_ai_gateway is set to True."
        )

    # Fallback to using serving endpoints
    return f"{workspace_client.config.host}/serving-endpoints"


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
        elif "DNS" in message or "resolution" in message.lower():
            hint = (
                f"Hint: App '{app_name}' may be stopped or unavailable. "
                f"Check the app status in the Databricks workspace."
            )
        elif status_code >= 500:
            hint = (
                f"Hint: App '{app_name}' encountered an internal error. "
                f"Check the app logs and status in the Databricks workspace."
            )
        elif status_code != 200:
            hint = (
                f"Hint: App '{app_name}' returned a non-200 status code: {status_code}. "
                f"Check the app logs and status in the Databricks workspace."
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
        if _is_claude_model(model):
            _fix_empty_assistant_content_in_messages(kwargs.get("messages"))
        kwargs = _strip_strict_from_kwargs(kwargs)
        return super().create(**kwargs)


class DatabricksChat(Chat):
    """Chat resource that uses Databricks completions with strict stripping."""

    completions: DatabricksCompletions


_FMAPI_MAX_ID_LENGTH = 64


def _truncate_response_ids(response: Any) -> None:
    """Truncate ids that exceed FMAPI's 64-char input limit.

    FMAPI returns response and output item ids longer than 64 chars, but rejects
    them on the next turn's input. We truncate to prevent multi-turn failures.
    """
    if hasattr(response, "id") and response.id and len(response.id) > _FMAPI_MAX_ID_LENGTH:
        response.id = response.id[:_FMAPI_MAX_ID_LENGTH]
    if not hasattr(response, "output"):
        return
    for item in response.output:
        item_id = getattr(item, "id", None)
        if item_id and len(item_id) > _FMAPI_MAX_ID_LENGTH:
            item.id = item_id[:_FMAPI_MAX_ID_LENGTH]


def _truncate_input_ids(input_items: Any) -> None:
    """Truncate ids in input items. Covers the streaming path where
    _truncate_response_ids can't intercept the assembled response.
    """
    if not input_items or not isinstance(input_items, list):
        return
    for item in input_items:
        if isinstance(item, dict):
            item_id = item.get("id")
            if isinstance(item_id, str) and len(item_id) > _FMAPI_MAX_ID_LENGTH:
                item["id"] = item_id[:_FMAPI_MAX_ID_LENGTH]
        else:
            item_id = getattr(item, "id", None)
            if isinstance(item_id, str) and len(item_id) > _FMAPI_MAX_ID_LENGTH:
                item.id = item_id[:_FMAPI_MAX_ID_LENGTH]


class DatabricksResponses(Responses):
    """Responses resource that handles apps/ prefix routing and id truncation."""

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
                api_key=_get_openai_api_key(),
                http_client=_get_authorized_http_client(self._workspace_client),
            )
        return self._app_clients_cache[app_name]

    def create(self, **kwargs):
        model = kwargs.get("model", "")
        _truncate_input_ids(kwargs.get("input"))

        if isinstance(model, str) and model.startswith(_APPS_ENDPOINT_PREFIX):
            app_name = model[len(_APPS_ENDPOINT_PREFIX) :]
            app_client = self._get_app_client(app_name)
            try:
                return app_client.responses.create(**kwargs)
            except (APIStatusError, APIConnectionError) as e:
                raise _wrap_app_error(e, app_name) from e

        response = super().create(**kwargs)
        _truncate_response_ids(response)
        return response


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
        ai_gateway_native_api: If True, auto-detect AI Gateway V2 and route requests through
            its native OpenAI-compatible API (``<ai_gateway_url>/openai``). This allows use of
            provider-native features not available through the MLflow API. Cannot be combined
            with ``base_url`` or ``use_ai_gateway``. Defaults to False.
        use_ai_gateway: If True, auto-detect AI Gateway V2 availability and route
            requests through it using the MLflow API. Defaults to False.

    Example - Query a serving or AI gateway endpoint:
        >>> client = DatabricksOpenAI()
        >>> response = client.chat.completions.create(
        ...     model="databricks-meta-llama-3-1-70b-instruct",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )

    Example - Query AI Gateway endpoints via the native OpenAI-compatible API:
        >>> client = DatabricksOpenAI(ai_gateway_native_api=True)
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
        ai_gateway_native_api: bool = False,
        use_ai_gateway: bool = False,
    ):
        if ai_gateway_native_api and base_url is not None:
            raise ValueError("Cannot specify both 'ai_gateway_native_api' and 'base_url'.")
        if ai_gateway_native_api and use_ai_gateway:
            raise ValueError(
                "Cannot specify both 'ai_gateway_native_api' and 'use_ai_gateway'."
            )

        if workspace_client is None:
            workspace_client = WorkspaceClient()

        self._workspace_client = workspace_client

        http_client = _get_authorized_http_client(workspace_client)
        target_base_url = _resolve_base_url(
            workspace_client, base_url, use_ai_gateway, http_client, ai_gateway_native_api
        )

        # Authentication is handled via http_client, not api_key
        super().__init__(
            base_url=target_base_url,
            api_key=_get_openai_api_key(),
            http_client=http_client,
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
        if _is_claude_model(model):
            _fix_empty_assistant_content_in_messages(kwargs.get("messages"))
        kwargs = _strip_strict_from_kwargs(kwargs)
        return await super().create(**kwargs)


class AsyncDatabricksChat(AsyncChat):
    """Async chat resource that uses Databricks completions with strict stripping."""

    completions: AsyncDatabricksCompletions


class AsyncDatabricksResponses(AsyncResponses):
    """Async Responses resource that handles apps/ prefix routing and id truncation."""

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
                api_key=_get_openai_api_key(),
                http_client=_get_authorized_async_http_client(self._workspace_client),
            )
        return self._app_clients_cache[app_name]

    async def create(self, **kwargs):
        model = kwargs.get("model", "")
        _truncate_input_ids(kwargs.get("input"))

        if isinstance(model, str) and model.startswith(_APPS_ENDPOINT_PREFIX):
            app_name = model[len(_APPS_ENDPOINT_PREFIX) :]
            app_client = self._get_app_client(app_name)
            try:
                return await app_client.responses.create(**kwargs)
            except (APIStatusError, APIConnectionError) as e:
                raise _wrap_app_error(e, app_name) from e

        response = await super().create(**kwargs)
        _truncate_response_ids(response)
        return response


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
        ai_gateway_native_api: If True, auto-detect AI Gateway V2 and route requests through
            its native OpenAI-compatible API (``<ai_gateway_url>/openai``). This allows use of
            provider-native features not available through the MLflow API. Cannot be combined
            with ``base_url`` or ``use_ai_gateway``. Defaults to False.
        use_ai_gateway: If True, auto-detect AI Gateway V2 availability and route
            requests through it using the MLflow API. Defaults to False.

    Example - Query a serving or AI gateway endpoint:
        >>> client = AsyncDatabricksOpenAI()
        >>> response = await client.chat.completions.create(
        ...     model="databricks-meta-llama-3-1-70b-instruct",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ... )

    Example - Query AI Gateway endpoints via the native OpenAI-compatible API:
        >>> client = AsyncDatabricksOpenAI(ai_gateway_native_api=True)
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
        ai_gateway_native_api: bool = False,
        use_ai_gateway: bool = False,
    ):
        if ai_gateway_native_api and base_url is not None:
            raise ValueError("Cannot specify both 'ai_gateway_native_api' and 'base_url'.")
        if ai_gateway_native_api and use_ai_gateway:
            raise ValueError(
                "Cannot specify both 'ai_gateway_native_api' and 'use_ai_gateway'."
            )

        if workspace_client is None:
            workspace_client = WorkspaceClient()

        self._workspace_client = workspace_client

        sync_http_client = _get_authorized_http_client(workspace_client)
        target_base_url = _resolve_base_url(
            workspace_client, base_url, use_ai_gateway, sync_http_client, ai_gateway_native_api
        )

        # Authentication is handled via http_client, not api_key
        super().__init__(
            base_url=target_base_url,
            api_key=_get_openai_api_key(),
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
