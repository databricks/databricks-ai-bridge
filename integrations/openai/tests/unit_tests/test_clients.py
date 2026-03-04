import os
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from databricks.sdk import WorkspaceClient
from httpx import Request
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI
from openai._types import NOT_GIVEN, Omit
from openai.resources.chat.completions import AsyncCompletions, Completions
from openai.resources.responses import AsyncResponses, Responses

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI
from databricks_openai.utils.clients import (
    _get_app_url,
    _get_authorized_async_http_client,
    _get_authorized_http_client,
    _get_openai_api_key,
    _should_strip_strict,
    _strip_strict_from_tools,
    _validate_oauth_for_apps,
    _wrap_app_error,
)


@pytest.fixture
def mock_workspace_client():
    """Create a mock WorkspaceClient for testing."""
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"

    # Mock the authenticate method to return headers
    mock_client.config.authenticate.return_value = {"Authorization": "Bearer test-token-123"}

    return mock_client


@pytest.fixture
def mock_workspace_client_with_oauth():
    """Create a mock WorkspaceClient with OAuth support for testing."""
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config.authenticate.return_value = {"Authorization": "Bearer oauth-token"}
    mock_client.config.oauth_token.return_value = "oauth-token"

    # Mock app lookup
    mock_app = MagicMock()
    mock_app.url = "https://my-app.aws.databricksapps.com"
    mock_client.apps.get.return_value = mock_app

    return mock_client


@pytest.fixture
def mock_workspace_client_no_oauth():
    """Create a mock WorkspaceClient without OAuth support for testing."""
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config.authenticate.return_value = {"Authorization": "Bearer pat-token"}
    mock_client.config.oauth_token.side_effect = Exception("No OAuth token available")

    return mock_client


class TestDatabricksOpenAI:
    """Tests for DatabricksOpenAI client."""

    def test_init_with_default_workspace_client(self):
        """Test initialization with default WorkspaceClient."""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws_client_class,
        ):
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://default.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer default-token"}
            mock_ws_client_class.return_value = mock_client

            client = DatabricksOpenAI()

            # Verify WorkspaceClient was created with no arguments
            mock_ws_client_class.assert_called_once_with()

            # Verify the client was initialized correctly
            assert isinstance(client, OpenAI)
            assert client.base_url.path == "/serving-endpoints/"
            assert "default.databricks.com" in str(client.base_url)
            assert client.api_key == "no-token"

    def test_init_uses_openai_api_key_env_var(self):
        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}),
            patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws_client_class,
        ):
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://default.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws_client_class.return_value = mock_client

            client = DatabricksOpenAI()
            assert client.api_key == "sk-from-env"

    def test_bearer_auth_flow(self, mock_workspace_client):
        """Test that BearerAuth correctly adds Authorization header."""

        http_client = _get_authorized_http_client(mock_workspace_client)

        # Create a test request
        request = Request("GET", "https://test.databricks.com/api/test")

        # Authenticate the request
        assert http_client.auth is not None
        auth_flow = http_client.auth.auth_flow(request)
        authenticated_request = next(auth_flow)

        # Verify Authorization header was added
        assert "Authorization" in authenticated_request.headers
        assert authenticated_request.headers["Authorization"] == "Bearer test-token-123"

        # Verify authenticate was called
        mock_workspace_client.config.authenticate.assert_called()


class TestAsyncDatabricksOpenAI:
    """Tests for AsyncDatabricksOpenAI client."""

    def test_init_with_default_workspace_client(self):
        """Test initialization with default WorkspaceClient."""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws_client_class,
        ):
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://default.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer default-token"}
            mock_ws_client_class.return_value = mock_client

            client = AsyncDatabricksOpenAI()

            # Verify the client was initialized correctly
            assert isinstance(client, AsyncOpenAI)
            assert client.base_url.path == "/serving-endpoints/"
            assert "default.databricks.com" in str(client.base_url)
            assert client.api_key == "no-token"

    def test_bearer_auth_flow(self, mock_workspace_client):
        """Test that BearerAuth correctly adds Authorization header for async client."""

        http_client = _get_authorized_async_http_client(mock_workspace_client)

        # Create a test request
        request = Request("GET", "https://test.databricks.com/api/test")

        # Authenticate the request
        assert http_client.auth is not None
        auth_flow = http_client.auth.auth_flow(request)
        authenticated_request = next(auth_flow)

        # Verify Authorization header was added
        assert "Authorization" in authenticated_request.headers
        assert authenticated_request.headers["Authorization"] == "Bearer test-token-123"

        # Verify authenticate was called
        mock_workspace_client.config.authenticate.assert_called()


class TestStrictFieldStripping:
    """Tests for strict field stripping helper functions."""

    def test_strip_strict_from_tools_removes_strict(self):
        tools = [
            {"type": "function", "function": {"name": "test", "strict": True, "parameters": {}}}
        ]
        _strip_strict_from_tools(tools)
        assert "strict" not in tools[0]["function"]

    def test_strip_strict_from_tools_handles_none(self):
        assert _strip_strict_from_tools(None) is None

    def test_strip_strict_from_tools_handles_openai_not_given_sentinel(self):
        """OpenAI Agents SDK may pass NOT_GIVEN instead of None or a list."""

        # Should not raise TypeError: 'NotGiven' object is not iterable
        result = _strip_strict_from_tools(NOT_GIVEN)
        assert result is NOT_GIVEN

    def test_strip_strict_from_tools_handles_openai_omit_sentinel(self):
        """OpenAI Agents SDK may pass Omit() instead of None or a list."""

        omit = Omit()
        # Should not raise TypeError: 'Omit' object is not iterable
        result = _strip_strict_from_tools(omit)
        assert result is omit

    def test_strip_strict_from_tools_handles_empty_list(self):
        tools = []
        _strip_strict_from_tools(tools)
        assert tools == []

    def test_strip_strict_from_tools_handles_tool_without_function(self):
        tools = [{"type": "other"}]
        _strip_strict_from_tools(tools)
        assert tools == [{"type": "other"}]

    def test_strip_strict_preserves_other_fields(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "desc",
                    "strict": True,
                    "parameters": {"type": "object"},
                },
            }
        ]
        _strip_strict_from_tools(tools)
        tool: dict[str, Any] = tools[0]
        function = cast(dict[str, Any], tool["function"])
        assert function["name"] == "test"
        assert function["description"] == "desc"
        assert function["parameters"] == {"type": "object"}
        assert "strict" not in tools[0]["function"]

    @pytest.mark.parametrize(
        "model,should_strip",
        [
            ("databricks-claude-3-7-sonnet", True),
            ("databricks-meta-llama-3-1-70b-instruct", True),
            ("databricks-mixtral-8x7b-instruct", True),
            ("databricks-gpt-4o", False),
            ("databricks-gpt-5-2", False),
            ("gpt-4", False),
            ("GPT-4-turbo", False),
            (None, True),
            ("", True),
        ],
    )
    def test_should_strip_strict_by_model(self, model, should_strip):
        assert _should_strip_strict(model) == should_strip


class TestDatabricksOpenAIStrictStripping:
    """Tests for strict stripping in DatabricksOpenAI."""

    def test_chat_completions_strips_strict_for_claude(self):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://test.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws.return_value = mock_client

            client = DatabricksOpenAI()

            with patch.object(Completions, "create") as mock_create:
                mock_create.return_value = MagicMock()
                tools: list[Any] = [
                    {"type": "function", "function": {"name": "test", "strict": True}}
                ]
                client.chat.completions.create(
                    model="databricks-claude-3-7-sonnet",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=cast(Any, tools),
                )

                call_kwargs = mock_create.call_args.kwargs
                assert "strict" not in call_kwargs["tools"][0]["function"]

    def test_chat_completions_preserves_strict_for_gpt(self):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://test.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws.return_value = mock_client

            client = DatabricksOpenAI()

            with patch.object(Completions, "create") as mock_create:
                mock_create.return_value = MagicMock()
                tools: list[Any] = [
                    {"type": "function", "function": {"name": "test", "strict": True}}
                ]
                client.chat.completions.create(
                    model="databricks-gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=cast(Any, tools),
                )

                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["tools"][0]["function"]["strict"] is True

    def test_chat_completions_works_without_tools(self):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://test.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws.return_value = mock_client

            client = DatabricksOpenAI()

            with patch.object(Completions, "create") as mock_create:
                mock_create.return_value = MagicMock()
                client.chat.completions.create(
                    model="databricks-claude-3-7-sonnet",
                    messages=[{"role": "user", "content": "hi"}],
                )
                mock_create.assert_called_once()


class TestAsyncDatabricksOpenAIStrictStripping:
    """Tests for strict stripping in AsyncDatabricksOpenAI."""

    @pytest.mark.asyncio
    async def test_chat_completions_strips_strict_for_claude(self):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://test.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws.return_value = mock_client

            client = AsyncDatabricksOpenAI()

            with patch.object(AsyncCompletions, "create", new_callable=AsyncMock) as mock_create:
                tools: list[Any] = [
                    {"type": "function", "function": {"name": "test", "strict": True}}
                ]
                await client.chat.completions.create(
                    model="databricks-claude-3-7-sonnet",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=cast(Any, tools),
                )

                call_kwargs = mock_create.call_args.kwargs
                assert "strict" not in call_kwargs["tools"][0]["function"]

    @pytest.mark.asyncio
    async def test_chat_completions_preserves_strict_for_gpt(self):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://test.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws.return_value = mock_client

            client = AsyncDatabricksOpenAI()

            with patch.object(AsyncCompletions, "create", new_callable=AsyncMock) as mock_create:
                tools: list[Any] = [
                    {"type": "function", "function": {"name": "test", "strict": True}}
                ]
                await client.chat.completions.create(
                    model="databricks-gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=cast(Any, tools),
                )

                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["tools"][0]["function"]["strict"] is True


class TestDatabricksAppsSupport:
    """Tests for Databricks Apps support."""

    def test_validate_oauth_for_apps_success(self, mock_workspace_client_with_oauth):
        _validate_oauth_for_apps(mock_workspace_client_with_oauth)
        mock_workspace_client_with_oauth.config.oauth_token.assert_called_once()

    def test_validate_oauth_for_apps_failure(self, mock_workspace_client_no_oauth):
        with pytest.raises(ValueError, match="OAuth authentication"):
            _validate_oauth_for_apps(mock_workspace_client_no_oauth)

    def test_get_app_url_success(self, mock_workspace_client_with_oauth):
        url = _get_app_url(mock_workspace_client_with_oauth, "my-app")
        assert url == "https://my-app.aws.databricksapps.com"
        mock_workspace_client_with_oauth.apps.get.assert_called_once_with(name="my-app")

    def test_get_app_url_app_not_found(self, mock_workspace_client_with_oauth):
        mock_workspace_client_with_oauth.apps.get.side_effect = Exception("App not found")
        with pytest.raises(ValueError, match="Failed to get Databricks App"):
            _get_app_url(mock_workspace_client_with_oauth, "nonexistent-app")

    def test_get_app_url_no_url(self, mock_workspace_client_with_oauth):
        mock_app = MagicMock()
        mock_app.url = None
        mock_workspace_client_with_oauth.apps.get.return_value = mock_app

        with pytest.raises(ValueError, match="has no URL"):
            _get_app_url(mock_workspace_client_with_oauth, "my-app")


class TestDatabricksClientWithBaseUrl:
    """Tests for DatabricksOpenAI and AsyncDatabricksOpenAI with base_url parameter."""

    @pytest.mark.parametrize("client_cls_name", ["DatabricksOpenAI", "AsyncDatabricksOpenAI"])
    def test_init_with_base_url_validates_oauth(
        self, client_cls_name, mock_workspace_client_with_oauth
    ):
        client_cls = (
            DatabricksOpenAI if client_cls_name == "DatabricksOpenAI" else AsyncDatabricksOpenAI
        )
        client = client_cls(
            workspace_client=mock_workspace_client_with_oauth,
            base_url="https://my-app.aws.databricksapps.com",
        )
        assert "my-app.aws.databricksapps.com" in str(client.base_url)
        mock_workspace_client_with_oauth.config.oauth_token.assert_called_once()

    @pytest.mark.parametrize("client_cls_name", ["DatabricksOpenAI", "AsyncDatabricksOpenAI"])
    def test_init_with_base_url_requires_oauth(
        self, client_cls_name, mock_workspace_client_no_oauth
    ):
        client_cls = (
            DatabricksOpenAI if client_cls_name == "DatabricksOpenAI" else AsyncDatabricksOpenAI
        )
        with pytest.raises(ValueError, match="OAuth authentication"):
            client_cls(
                workspace_client=mock_workspace_client_no_oauth,
                base_url="https://my-app.aws.databricksapps.com",
            )

    def test_init_without_base_url_uses_serving_endpoints(self, mock_workspace_client_with_oauth):
        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)
        assert "/serving-endpoints/" in str(client.base_url)
        mock_workspace_client_with_oauth.config.oauth_token.assert_not_called()

    @pytest.mark.parametrize("client_cls_name", ["DatabricksOpenAI", "AsyncDatabricksOpenAI"])
    def test_init_with_non_databricksapps_base_url_does_not_require_oauth(
        self, client_cls_name, mock_workspace_client_no_oauth
    ):
        client_cls = (
            DatabricksOpenAI if client_cls_name == "DatabricksOpenAI" else AsyncDatabricksOpenAI
        )
        # Non-databricksapps URLs should not require OAuth
        client = client_cls(
            workspace_client=mock_workspace_client_no_oauth,
            base_url="https://custom-endpoint.example.com/v1",
        )
        assert "custom-endpoint.example.com" in str(client.base_url)
        # OAuth should not be validated for non-databricksapps URLs
        mock_workspace_client_no_oauth.config.oauth_token.assert_not_called()


class TestAppsRouting:
    """Tests for apps/ prefix routing in DatabricksOpenAI and AsyncDatabricksOpenAI."""

    def test_sync_responses_create_routes_to_app(self, mock_workspace_client_with_oauth):
        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(Responses, "create") as mock_create:
            mock_create.return_value = MagicMock()
            client.responses.create(
                model="apps/my-agent",
                input=[{"role": "user", "content": "Hello"}],
            )
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "apps/my-agent"
            mock_workspace_client_with_oauth.apps.get.assert_called_once_with(name="my-agent")

    @pytest.mark.asyncio
    async def test_async_responses_create_routes_to_app(self, mock_workspace_client_with_oauth):
        client = AsyncDatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(AsyncResponses, "create", new_callable=AsyncMock) as mock_create:
            await client.responses.create(
                model="apps/my-agent",
                input=[{"role": "user", "content": "Hello"}],
            )
            mock_create.assert_called_once()
            mock_workspace_client_with_oauth.apps.get.assert_called_once_with(name="my-agent")

    def test_responses_caches_app_clients(self, mock_workspace_client_with_oauth):
        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(Responses, "create") as mock_create:
            mock_create.return_value = MagicMock()
            client.responses.create(model="apps/my-agent", input=[{"role": "user", "content": "1"}])
            client.responses.create(model="apps/my-agent", input=[{"role": "user", "content": "2"}])
            assert mock_workspace_client_with_oauth.apps.get.call_count == 1

    def test_sync_responses_validates_oauth_for_apps_prefix(self, mock_workspace_client_no_oauth):
        client = DatabricksOpenAI(workspace_client=mock_workspace_client_no_oauth)
        with pytest.raises(ValueError, match="OAuth authentication"):
            client.responses.create(
                model="apps/my-agent",
                input=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.asyncio
    async def test_async_responses_validates_oauth_for_apps_prefix(
        self, mock_workspace_client_no_oauth
    ):
        client = AsyncDatabricksOpenAI(workspace_client=mock_workspace_client_no_oauth)
        with pytest.raises(ValueError, match="OAuth authentication"):
            await client.responses.create(
                model="apps/my-agent",
                input=[{"role": "user", "content": "Hello"}],
            )


def _make_api_status_error(status_code: int, message: str) -> APIStatusError:
    """Helper to create an APIStatusError with a properly configured request/response."""
    request = httpx.Request("POST", "https://test.databricksapps.com/v1/responses")
    response = httpx.Response(status_code, json={"detail": message}, request=request)
    return APIStatusError(message=message, response=response, body=None)


class TestAppErrorWrapping:
    @pytest.mark.parametrize(
        "status_code,message,expected_hints",
        [
            (404, "Not Found", ["/responses endpoint"]),
            (405, "Method Not Allowed", ["/responses endpoint"]),
            (403, "Forbidden", ["CAN_USE"]),
            (500, "Internal Server Error", ["internal error", "Check the app logs"]),
            (502, "Bad Gateway", ["internal error", "Check the app logs"]),
            (503, "Service Unavailable", ["internal error", "Check the app logs"]),
            (429, "Too Many Requests", []),  # No specific hint for rate limiting
        ],
    )
    def test_wrap_app_error_status_errors(self, status_code, message, expected_hints):
        error = _make_api_status_error(status_code, message)
        wrapped = _wrap_app_error(error, "my-app")
        wrapped_str = str(wrapped)

        assert str(status_code) in wrapped_str
        assert message in wrapped_str
        for hint in expected_hints:
            assert "Hint:" in wrapped_str
            assert hint in wrapped_str

    @pytest.mark.parametrize(
        "message,expected_hint",
        [
            ("DNS resolution failure", "stopped or unavailable"),
            ("Connection refused", "starting up or unavailable"),
        ],
    )
    def test_wrap_app_error_connection_errors(self, message, expected_hint):
        request = httpx.Request("POST", "https://test.databricksapps.com/v1/responses")
        error = APIConnectionError(message=message, request=request)
        wrapped = _wrap_app_error(error, "my-app")
        wrapped_str = str(wrapped)

        assert message in wrapped_str
        assert "Hint:" in wrapped_str
        assert expected_hint in wrapped_str


class TestDatabricksOpenAIAppsErrorHandling:
    @pytest.mark.parametrize(
        "error,expected_match",
        [
            (_make_api_status_error(405, "Method Not Allowed"), r"(?s)405.*Hint:"),
            (
                APIConnectionError(
                    message="DNS resolution failure",
                    request=httpx.Request("POST", "https://test.databricksapps.com/v1/responses"),
                ),
                r"(?s)DNS resolution failure.*Hint:",
            ),
        ],
    )
    def test_responses_wraps_app_errors(
        self, mock_workspace_client_with_oauth, error, expected_match
    ):
        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(Responses, "create", side_effect=error):
            with pytest.raises(ValueError, match=expected_match):
                client.responses.create(
                    model="apps/my-agent",
                    input=[{"role": "user", "content": "Hello"}],
                )

    def test_responses_non_apps_model_does_not_wrap_errors(self, mock_workspace_client_with_oauth):
        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(
            Responses, "create", side_effect=_make_api_status_error(500, "Internal Server Error")
        ):
            with pytest.raises(APIStatusError):
                client.responses.create(
                    model="databricks-claude-3-7-sonnet",
                    input=[{"role": "user", "content": "Hello"}],
                )


class TestAsyncDatabricksOpenAIAppsErrorHandling:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error,expected_match",
        [
            (_make_api_status_error(405, "Method Not Allowed"), r"(?s)405.*Hint:"),
            (
                APIConnectionError(
                    message="DNS resolution failure",
                    request=httpx.Request("POST", "https://test.databricksapps.com/v1/responses"),
                ),
                r"(?s)DNS resolution failure.*Hint:",
            ),
        ],
    )
    async def test_responses_wraps_app_errors(
        self, mock_workspace_client_with_oauth, error, expected_match
    ):
        client = AsyncDatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(AsyncResponses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = error
            with pytest.raises(ValueError, match=expected_match):
                await client.responses.create(
                    model="apps/my-agent",
                    input=[{"role": "user", "content": "Hello"}],
                )


class TestChatCompletionsEmptyContentFix:
    @pytest.mark.parametrize(
        "model,expected_content",
        [
            ("databricks-claude-3-7-sonnet", " "),
            ("databricks-gpt-4o", ""),
        ],
    )
    def test_sync_empty_content_fix(self, model, expected_content):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_ws.return_value = _mock_workspace_client()
            client = DatabricksOpenAI()
            with patch.object(Completions, "create", return_value=MagicMock()) as mock_create:
                messages = _messages_with_empty_assistant_content()
                client.chat.completions.create(model=model, messages=cast(Any, messages))
                assert mock_create.call_args.kwargs["messages"][1]["content"] == expected_content

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "model,expected_content",
        [
            ("databricks-claude-3-7-sonnet", " "),
            ("databricks-gpt-4o", ""),
        ],
    )
    async def test_async_empty_content_fix(self, model, expected_content):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_ws.return_value = _mock_workspace_client()
            client = AsyncDatabricksOpenAI()
            with patch.object(AsyncCompletions, "create", new_callable=AsyncMock) as mock_create:
                messages = _messages_with_empty_assistant_content()
                await client.chat.completions.create(model=model, messages=cast(Any, messages))
                assert mock_create.call_args.kwargs["messages"][1]["content"] == expected_content


def _mock_workspace_client():
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"
    mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
    return mock_client


def _messages_with_empty_assistant_content() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "result", "tool_call_id": "1"},
    ]


class TestOpenAIApiKey:
    def test_uses_env_var_when_set(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"}):
            assert _get_openai_api_key() == "sk-test-key"

    def test_falls_back_to_no_token_when_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            assert _get_openai_api_key() == "no-token"

    def test_falls_back_to_no_token_when_empty_string(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            assert _get_openai_api_key() == "no-token"
