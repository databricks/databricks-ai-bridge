"""Test utilities module."""

from unittest.mock import Mock, patch

from databricks_langchain.utils import get_openai_client


def test_get_openai_client_with_timeout_and_max_retries() -> None:
    """Test that get_openai_client properly passes timeout and max_retries as kwargs to the SDK."""

    mock_openai_client = Mock()

    mock_workspace_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client

    # Test with workspace_client, timeout, and max_retries
    client = get_openai_client(workspace_client=mock_workspace_client, timeout=45.0, max_retries=3)

    # Verify the OpenAI client was obtained with the correct kwargs
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_called_once_with(
        timeout=45.0, max_retries=3
    )

    # Verify the client is returned
    assert client == mock_openai_client


def test_get_openai_client_with_default_workspace_client() -> None:
    """Test get_openai_client creates default WorkspaceClient when none provided."""

    mock_openai_client = Mock()

    mock_workspace_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client

    with patch("databricks.sdk.WorkspaceClient", return_value=mock_workspace_client):
        client = get_openai_client(timeout=30.0, max_retries=2)

    # Verify default WorkspaceClient was created and kwargs were passed
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_called_once_with(
        timeout=30.0, max_retries=2
    )

    # Verify the client is returned
    assert client == mock_openai_client


def test_get_openai_client_without_timeout_and_retries() -> None:
    """Test get_openai_client doesn't pass kwargs when not provided."""

    mock_openai_client = Mock()

    mock_workspace_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client

    client = get_openai_client(workspace_client=mock_workspace_client)

    # Verify the OpenAI client was obtained without kwargs
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_called_once_with()

    # Verify the client is returned
    assert client == mock_openai_client
