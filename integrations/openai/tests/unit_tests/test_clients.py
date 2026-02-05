from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from databricks.sdk import WorkspaceClient
from openai import AsyncOpenAI, OpenAI


@pytest.fixture
def mock_workspace_client():
    """Create a mock WorkspaceClient for testing."""
    mock_client = MagicMock(spec=WorkspaceClient)
    mock_client.config.host = "https://test.databricks.com"

    # Mock the authenticate method to return headers
    mock_client.config.authenticate.return_value = {"Authorization": "Bearer test-token-123"}

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

            from openai.resources.chat.completions import Completions

            from databricks_openai import DatabricksOpenAI

            client = DatabricksOpenAI()

            with patch.object(Completions, "create") as mock_create:
                mock_create.return_value = MagicMock()
                tools = [{"type": "function", "function": {"name": "test", "strict": True}}]
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

            from openai.resources.chat.completions import AsyncCompletions

            from databricks_openai import AsyncDatabricksOpenAI

            client = AsyncDatabricksOpenAI()

            with patch.object(AsyncCompletions, "create", new_callable=AsyncMock) as mock_create:
                tools = [{"type": "function", "function": {"name": "test", "strict": True}}]
                await client.chat.completions.create(
                    model="databricks-gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    tools=cast(Any, tools),
                )

                call_kwargs = mock_create.call_args.kwargs
                assert call_kwargs["tools"][0]["function"]["strict"] is True
