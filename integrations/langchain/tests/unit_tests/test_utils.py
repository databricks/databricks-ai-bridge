"""Test utilities module."""

from unittest.mock import Mock, patch

import pytest

from databricks_langchain.utils import get_openai_client


def test_get_openai_client_with_timeout_and_max_retries() -> None:
    """Test that get_openai_client properly sets timeout and max_retries on the OpenAI client."""
    
    mock_openai_client = Mock()
    mock_openai_client.timeout = None
    mock_openai_client.max_retries = None
    
    mock_workspace_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client
    
    # Test with workspace_client, timeout, and max_retries
    client = get_openai_client(
        workspace_client=mock_workspace_client,
        timeout=45.0,
        max_retries=3
    )
    
    # Verify the OpenAI client was obtained
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_called_once()
    
    # Verify timeout and max_retries were set
    assert client.timeout == 45.0
    assert client.max_retries == 3


def test_get_openai_client_with_default_workspace_client() -> None:
    """Test get_openai_client creates default WorkspaceClient when none provided."""
    
    mock_openai_client = Mock()
    mock_openai_client.timeout = None
    mock_openai_client.max_retries = None
    
    mock_workspace_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client
    
    with patch("databricks.sdk.WorkspaceClient", return_value=mock_workspace_client):
        client = get_openai_client(timeout=30.0, max_retries=2)
    
    # Verify default WorkspaceClient was created
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_called_once()
    
    # Verify timeout and max_retries were set
    assert client.timeout == 30.0
    assert client.max_retries == 2


def test_get_openai_client_without_timeout_and_retries() -> None:
    """Test get_openai_client doesn't set timeout/max_retries when not provided."""
    
    mock_openai_client = Mock()
    # Set initial values to check they're not changed
    mock_openai_client.timeout = "original_timeout"
    mock_openai_client.max_retries = "original_retries"
    
    mock_workspace_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client
    
    client = get_openai_client(workspace_client=mock_workspace_client)
    
    # Verify the OpenAI client was obtained
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_called_once()
    
    # Verify timeout and max_retries were NOT changed (None values don't override)
    assert client.timeout == "original_timeout"
    assert client.max_retries == "original_retries"