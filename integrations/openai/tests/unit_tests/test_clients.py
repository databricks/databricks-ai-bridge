from unittest.mock import MagicMock, patch

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
