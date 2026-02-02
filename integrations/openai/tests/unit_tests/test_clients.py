from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from databricks.sdk import WorkspaceClient
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI


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
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws_client_class:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://default.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer default-token"}
            mock_ws_client_class.return_value = mock_client

            from databricks_openai.utils.clients import DatabricksOpenAI

            client = DatabricksOpenAI()

            # Verify WorkspaceClient was created with no arguments
            mock_ws_client_class.assert_called_once_with()

            # Verify the client was initialized correctly
            assert isinstance(client, OpenAI)
            assert client.base_url.path == "/serving-endpoints/"
            assert "default.databricks.com" in str(client.base_url)
            assert client.api_key == "no-token"

    def test_bearer_auth_flow(self, mock_workspace_client):
        """Test that BearerAuth correctly adds Authorization header."""
        from httpx import Request

        from databricks_openai.utils.clients import _get_authorized_http_client

        http_client = _get_authorized_http_client(mock_workspace_client)

        # Create a test request
        request = Request("GET", "https://test.databricks.com/api/test")

        # Authenticate the request
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
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws_client_class:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://default.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer default-token"}
            mock_ws_client_class.return_value = mock_client

            from databricks_openai.utils.clients import AsyncDatabricksOpenAI

            client = AsyncDatabricksOpenAI()

            # Verify the client was initialized correctly
            assert isinstance(client, AsyncOpenAI)
            assert client.base_url.path == "/serving-endpoints/"
            assert "default.databricks.com" in str(client.base_url)
            assert client.api_key == "no-token"

    def test_bearer_auth_flow(self, mock_workspace_client):
        """Test that BearerAuth correctly adds Authorization header for async client."""
        from httpx import Request

        from databricks_openai.utils.clients import _get_authorized_async_http_client

        http_client = _get_authorized_async_http_client(mock_workspace_client)

        # Create a test request
        request = Request("GET", "https://test.databricks.com/api/test")

        # Authenticate the request
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
        from databricks_openai.utils.clients import _strip_strict_from_tools

        tools = [
            {"type": "function", "function": {"name": "test", "strict": True, "parameters": {}}}
        ]
        _strip_strict_from_tools(tools)
        assert "strict" not in tools[0]["function"]

    def test_strip_strict_from_tools_handles_none(self):
        from databricks_openai.utils.clients import _strip_strict_from_tools

        assert _strip_strict_from_tools(None) is None

    def test_strip_strict_from_tools_handles_empty_list(self):
        from databricks_openai.utils.clients import _strip_strict_from_tools

        tools = []
        _strip_strict_from_tools(tools)
        assert tools == []

    def test_strip_strict_from_tools_handles_tool_without_function(self):
        from databricks_openai.utils.clients import _strip_strict_from_tools

        tools = [{"type": "other"}]
        _strip_strict_from_tools(tools)
        assert tools == [{"type": "other"}]

    def test_strip_strict_preserves_other_fields(self):
        from databricks_openai.utils.clients import _strip_strict_from_tools

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
        from databricks_openai.utils.clients import _should_strip_strict

        assert _should_strip_strict(model) == should_strip


class TestDatabricksOpenAIStrictStripping:
    """Tests for strict stripping in DatabricksOpenAI."""

    def test_chat_completions_strips_strict_for_claude(self):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://test.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws.return_value = mock_client

            from openai.resources.chat.completions import Completions

            from databricks_openai import DatabricksOpenAI

            client = DatabricksOpenAI()

            with patch.object(Completions, "create") as mock_create:
                mock_create.return_value = MagicMock()
                tools = [{"type": "function", "function": {"name": "test", "strict": True}}]
                client.chat.completions.create(
                    model="databricks-claude-3-7-sonnet",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=tools,
                )

                call_kwargs = mock_create.call_args.kwargs
                assert "strict" not in call_kwargs["tools"][0]["function"]

    def test_chat_completions_preserves_strict_for_gpt(self):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://test.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws.return_value = mock_client

            from openai.resources.chat.completions import Completions

            from databricks_openai import DatabricksOpenAI

            client = DatabricksOpenAI()

            with patch.object(Completions, "create") as mock_create:
                mock_create.return_value = MagicMock()
                tools = [{"type": "function", "function": {"name": "test", "strict": True}}]
                client.chat.completions.create(
                    model="databricks-gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=tools,
                )

                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["tools"][0]["function"]["strict"] is True

    def test_chat_completions_works_without_tools(self):
        with patch("databricks_openai.utils.clients.WorkspaceClient") as mock_ws:
            mock_client = MagicMock(spec=WorkspaceClient)
            mock_client.config.host = "https://test.databricks.com"
            mock_client.config.authenticate.return_value = {"Authorization": "Bearer token"}
            mock_ws.return_value = mock_client

            from openai.resources.chat.completions import Completions

            from databricks_openai import DatabricksOpenAI

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

            from openai.resources.chat.completions import AsyncCompletions

            from databricks_openai import AsyncDatabricksOpenAI

            client = AsyncDatabricksOpenAI()

            with patch.object(AsyncCompletions, "create", new_callable=AsyncMock) as mock_create:
                tools = [{"type": "function", "function": {"name": "test", "strict": True}}]
                await client.chat.completions.create(
                    model="databricks-claude-3-7-sonnet",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=tools,
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

            from openai.resources.chat.completions import AsyncCompletions

            from databricks_openai import AsyncDatabricksOpenAI

            client = AsyncDatabricksOpenAI()

            with patch.object(AsyncCompletions, "create", new_callable=AsyncMock) as mock_create:
                tools = [{"type": "function", "function": {"name": "test", "strict": True}}]
                await client.chat.completions.create(
                    model="databricks-gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=tools,
                )

                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["tools"][0]["function"]["strict"] is True


class TestDatabricksAppsSupport:
    """Tests for Databricks Apps support."""

    def test_validate_oauth_for_apps_success(self, mock_workspace_client_with_oauth):
        from databricks_openai.utils.clients import _validate_oauth_for_apps

        _validate_oauth_for_apps(mock_workspace_client_with_oauth)
        mock_workspace_client_with_oauth.config.oauth_token.assert_called_once()

    def test_validate_oauth_for_apps_failure(self, mock_workspace_client_no_oauth):
        from databricks_openai.utils.clients import _validate_oauth_for_apps

        with pytest.raises(ValueError, match="OAuth authentication"):
            _validate_oauth_for_apps(mock_workspace_client_no_oauth)

    def test_get_app_url_success(self, mock_workspace_client_with_oauth):
        from databricks_openai.utils.clients import _get_app_url

        url = _get_app_url(mock_workspace_client_with_oauth, "my-app")
        assert url == "https://my-app.aws.databricksapps.com"
        mock_workspace_client_with_oauth.apps.get.assert_called_once_with(name="my-app")

    def test_get_app_url_app_not_found(self, mock_workspace_client_with_oauth):
        from databricks_openai.utils.clients import _get_app_url

        mock_workspace_client_with_oauth.apps.get.side_effect = Exception("App not found")
        with pytest.raises(ValueError, match="Failed to get Databricks App"):
            _get_app_url(mock_workspace_client_with_oauth, "nonexistent-app")

    def test_get_app_url_no_url(self, mock_workspace_client_with_oauth):
        from databricks_openai.utils.clients import _get_app_url

        mock_app = MagicMock()
        mock_app.url = None
        mock_workspace_client_with_oauth.apps.get.return_value = mock_app

        with pytest.raises(ValueError, match="has no URL"):
            _get_app_url(mock_workspace_client_with_oauth, "my-app")


class TestDatabricksOpenAIWithBaseUrl:
    """Tests for DatabricksOpenAI with base_url parameter."""

    def test_init_with_base_url_validates_oauth(self, mock_workspace_client_with_oauth):
        from databricks_openai import DatabricksOpenAI

        client = DatabricksOpenAI(
            workspace_client=mock_workspace_client_with_oauth,
            base_url="https://my-app.aws.databricksapps.com",
        )
        assert "my-app.aws.databricksapps.com" in str(client.base_url)
        mock_workspace_client_with_oauth.config.oauth_token.assert_called_once()

    def test_init_with_base_url_requires_oauth(self, mock_workspace_client_no_oauth):
        from databricks_openai import DatabricksOpenAI

        with pytest.raises(ValueError, match="OAuth authentication"):
            DatabricksOpenAI(
                workspace_client=mock_workspace_client_no_oauth,
                base_url="https://my-app.aws.databricksapps.com",
            )

    def test_init_without_base_url_uses_serving_endpoints(self, mock_workspace_client_with_oauth):
        from databricks_openai import DatabricksOpenAI

        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)
        assert "/serving-endpoints/" in str(client.base_url)
        mock_workspace_client_with_oauth.config.oauth_token.assert_not_called()


class TestDatabricksOpenAIAppsRouting:
    """Tests for apps/ prefix routing in DatabricksOpenAI."""

    def test_responses_create_routes_to_app(self, mock_workspace_client_with_oauth):
        from openai.resources.responses import Responses

        from databricks_openai import DatabricksOpenAI

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

    def test_responses_create_non_apps_model_uses_default(self, mock_workspace_client_with_oauth):
        from openai.resources.responses import Responses

        from databricks_openai import DatabricksOpenAI

        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(Responses, "create") as mock_create:
            mock_create.return_value = MagicMock()
            client.responses.create(
                model="databricks-claude-3-7-sonnet",
                input=[{"role": "user", "content": "Hello"}],
            )
            mock_create.assert_called_once()
            mock_workspace_client_with_oauth.apps.get.assert_not_called()

    def test_responses_caches_app_clients(self, mock_workspace_client_with_oauth):
        from openai.resources.responses import Responses

        from databricks_openai import DatabricksOpenAI

        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(Responses, "create") as mock_create:
            mock_create.return_value = MagicMock()
            client.responses.create(model="apps/my-agent", input=[{"role": "user", "content": "1"}])
            client.responses.create(model="apps/my-agent", input=[{"role": "user", "content": "2"}])
            # App should only be looked up once due to caching
            assert mock_workspace_client_with_oauth.apps.get.call_count == 1

    def test_responses_validates_oauth_for_apps_prefix(self, mock_workspace_client_no_oauth):
        from databricks_openai import DatabricksOpenAI

        client = DatabricksOpenAI(workspace_client=mock_workspace_client_no_oauth)
        with pytest.raises(ValueError, match="OAuth authentication"):
            client.responses.create(
                model="apps/my-agent",
                input=[{"role": "user", "content": "Hello"}],
            )


class TestAsyncDatabricksOpenAIWithBaseUrl:
    """Tests for AsyncDatabricksOpenAI with base_url parameter."""

    def test_init_with_base_url_validates_oauth(self, mock_workspace_client_with_oauth):
        from databricks_openai import AsyncDatabricksOpenAI

        client = AsyncDatabricksOpenAI(
            workspace_client=mock_workspace_client_with_oauth,
            base_url="https://my-app.aws.databricksapps.com",
        )
        assert "my-app.aws.databricksapps.com" in str(client.base_url)
        mock_workspace_client_with_oauth.config.oauth_token.assert_called_once()

    def test_init_with_base_url_requires_oauth(self, mock_workspace_client_no_oauth):
        from databricks_openai import AsyncDatabricksOpenAI

        with pytest.raises(ValueError, match="OAuth authentication"):
            AsyncDatabricksOpenAI(
                workspace_client=mock_workspace_client_no_oauth,
                base_url="https://my-app.aws.databricksapps.com",
            )


class TestAsyncDatabricksOpenAIAppsRouting:
    """Tests for apps/ prefix routing in AsyncDatabricksOpenAI."""

    @pytest.mark.asyncio
    async def test_responses_create_routes_to_app(self, mock_workspace_client_with_oauth):
        from openai.resources.responses import AsyncResponses

        from databricks_openai import AsyncDatabricksOpenAI

        client = AsyncDatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(AsyncResponses, "create", new_callable=AsyncMock) as mock_create:
            await client.responses.create(
                model="apps/my-agent",
                input=[{"role": "user", "content": "Hello"}],
            )
            mock_create.assert_called_once()
            mock_workspace_client_with_oauth.apps.get.assert_called_once_with(name="my-agent")

    @pytest.mark.asyncio
    async def test_responses_create_non_apps_model_uses_default(
        self, mock_workspace_client_with_oauth
    ):
        from openai.resources.responses import AsyncResponses

        from databricks_openai import AsyncDatabricksOpenAI

        client = AsyncDatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        with patch.object(AsyncResponses, "create", new_callable=AsyncMock) as mock_create:
            await client.responses.create(
                model="databricks-claude-3-7-sonnet",
                input=[{"role": "user", "content": "Hello"}],
            )
            mock_create.assert_called_once()
            mock_workspace_client_with_oauth.apps.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_responses_validates_oauth_for_apps_prefix(self, mock_workspace_client_no_oauth):
        from databricks_openai import AsyncDatabricksOpenAI

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
    def test_wrap_app_error_404_not_found(self):
        from databricks_openai.utils.clients import _wrap_app_error

        error = _make_api_status_error(404, "Not Found")

        wrapped = _wrap_app_error(error, "my-app")
        assert "404" in str(wrapped)
        assert "Not Found" in str(wrapped)
        assert "Hint:" in str(wrapped)
        assert "/responses endpoint" in str(wrapped)

    def test_wrap_app_error_405_method_not_allowed(self):
        from databricks_openai.utils.clients import _wrap_app_error

        error = _make_api_status_error(405, "Method Not Allowed")

        wrapped = _wrap_app_error(error, "my-app")
        assert "405" in str(wrapped)
        assert "Method Not Allowed" in str(wrapped)
        assert "Hint:" in str(wrapped)
        assert "/responses endpoint" in str(wrapped)

    def test_wrap_app_error_403_permission_denied(self):
        from databricks_openai.utils.clients import _wrap_app_error

        error = _make_api_status_error(403, "Forbidden")

        wrapped = _wrap_app_error(error, "my-app")
        assert "403" in str(wrapped)
        assert "Forbidden" in str(wrapped)
        assert "Hint:" in str(wrapped)
        assert "CAN_QUERY" in str(wrapped)

    def test_wrap_app_error_dns_resolution_failure(self):
        from databricks_openai.utils.clients import _wrap_app_error

        request = httpx.Request("POST", "https://test.databricksapps.com/v1/responses")
        error = APIConnectionError(message="DNS resolution failure", request=request)

        wrapped = _wrap_app_error(error, "my-app")
        assert "DNS resolution failure" in str(wrapped)
        assert "Hint:" in str(wrapped)
        assert "stopped or unavailable" in str(wrapped)

    def test_wrap_app_error_connection_error(self):
        from databricks_openai.utils.clients import _wrap_app_error

        request = httpx.Request("POST", "https://test.databricksapps.com/v1/responses")
        error = APIConnectionError(message="Connection refused", request=request)

        wrapped = _wrap_app_error(error, "my-app")
        assert "Connection refused" in str(wrapped)
        assert "Hint:" in str(wrapped)

    def test_wrap_app_error_other_status_error(self):
        from databricks_openai.utils.clients import _wrap_app_error

        error = _make_api_status_error(500, "Internal Server Error")

        wrapped = _wrap_app_error(error, "my-app")
        assert "500" in str(wrapped)
        assert "Internal Server Error" in str(wrapped)

    def test_wrap_app_error_dns_in_status_error(self):
        from databricks_openai.utils.clients import _wrap_app_error

        error = _make_api_status_error(503, "DNS resolution failure")

        wrapped = _wrap_app_error(error, "my-app")
        assert "503" in str(wrapped)
        assert "DNS resolution failure" in str(wrapped)
        assert "Hint:" in str(wrapped)
        assert "stopped or unavailable" in str(wrapped)


class TestDatabricksOpenAIAppsErrorHandling:
    def test_responses_wraps_405_error(self, mock_workspace_client_with_oauth):
        from openai.resources.responses import Responses

        from databricks_openai import DatabricksOpenAI

        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        error = _make_api_status_error(405, "Method Not Allowed")

        with patch.object(Responses, "create", side_effect=error):
            with pytest.raises(ValueError, match=r"(?s)405.*Hint:"):
                client.responses.create(
                    model="apps/my-agent",
                    input=[{"role": "user", "content": "Hello"}],
                )

    def test_responses_wraps_dns_error(self, mock_workspace_client_with_oauth):
        from openai.resources.responses import Responses

        from databricks_openai import DatabricksOpenAI

        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        request = httpx.Request("POST", "https://test.databricksapps.com/v1/responses")
        error = APIConnectionError(message="DNS resolution failure", request=request)

        with patch.object(Responses, "create", side_effect=error):
            with pytest.raises(ValueError, match=r"(?s)DNS resolution failure.*Hint:"):
                client.responses.create(
                    model="apps/my-agent",
                    input=[{"role": "user", "content": "Hello"}],
                )

    def test_responses_non_apps_model_does_not_wrap_errors(
        self, mock_workspace_client_with_oauth
    ):
        from openai.resources.responses import Responses

        from databricks_openai import DatabricksOpenAI

        client = DatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        error = _make_api_status_error(500, "Internal Server Error")

        with patch.object(Responses, "create", side_effect=error):
            with pytest.raises(APIStatusError):
                client.responses.create(
                    model="databricks-claude-3-7-sonnet",
                    input=[{"role": "user", "content": "Hello"}],
                )


class TestAsyncDatabricksOpenAIAppsErrorHandling:
    @pytest.mark.asyncio
    async def test_responses_wraps_405_error(self, mock_workspace_client_with_oauth):
        from openai.resources.responses import AsyncResponses

        from databricks_openai import AsyncDatabricksOpenAI

        client = AsyncDatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        error = _make_api_status_error(405, "Method Not Allowed")

        with patch.object(AsyncResponses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = error
            with pytest.raises(ValueError, match=r"(?s)405.*Hint:"):
                await client.responses.create(
                    model="apps/my-agent",
                    input=[{"role": "user", "content": "Hello"}],
                )

    @pytest.mark.asyncio
    async def test_responses_wraps_dns_error(self, mock_workspace_client_with_oauth):
        from openai.resources.responses import AsyncResponses

        from databricks_openai import AsyncDatabricksOpenAI

        client = AsyncDatabricksOpenAI(workspace_client=mock_workspace_client_with_oauth)

        request = httpx.Request("POST", "https://test.databricksapps.com/v1/responses")
        error = APIConnectionError(message="DNS resolution failure", request=request)

        with patch.object(AsyncResponses, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = error
            with pytest.raises(ValueError, match=r"(?s)DNS resolution failure.*Hint:"):
                await client.responses.create(
                    model="apps/my-agent",
                    input=[{"role": "user", "content": "Hello"}],
                )
