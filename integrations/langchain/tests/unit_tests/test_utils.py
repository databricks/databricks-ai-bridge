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


def test_get_openai_client_with_use_ai_gateway() -> None:
    """Test use_ai_gateway=True constructs DatabricksOpenAI instead of the SDK helper."""

    mock_workspace_client = Mock()
    mock_databricks_openai_client = Mock()

    with patch(
        "databricks_openai.DatabricksOpenAI", return_value=mock_databricks_openai_client
    ) as mock_databricks_openai:
        client = get_openai_client(
            workspace_client=mock_workspace_client, use_ai_gateway=True
        )

    mock_databricks_openai.assert_called_once_with(
        workspace_client=mock_workspace_client,
        use_ai_gateway=True,
        use_ai_gateway_native_api=False,
    )
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_not_called()
    assert client == mock_databricks_openai_client


def test_get_openai_client_with_use_ai_gateway_native_api() -> None:
    """Test use_ai_gateway_native_api=True constructs DatabricksOpenAI with that flag."""

    mock_workspace_client = Mock()
    mock_databricks_openai_client = Mock()

    with patch(
        "databricks_openai.DatabricksOpenAI", return_value=mock_databricks_openai_client
    ) as mock_databricks_openai:
        client = get_openai_client(
            workspace_client=mock_workspace_client, use_ai_gateway_native_api=True
        )

    mock_databricks_openai.assert_called_once_with(
        workspace_client=mock_workspace_client,
        use_ai_gateway=False,
        use_ai_gateway_native_api=True,
    )
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_not_called()
    assert client == mock_databricks_openai_client


def test_get_openai_client_without_gateway_uses_serving_endpoints() -> None:
    """Test that DatabricksOpenAI is NOT constructed when no gateway flags are set."""

    mock_workspace_client = Mock()
    mock_openai_client = Mock()
    mock_workspace_client.serving_endpoints.get_open_ai_client.return_value = mock_openai_client

    with patch("databricks_openai.DatabricksOpenAI") as mock_databricks_openai:
        client = get_openai_client(workspace_client=mock_workspace_client)

    mock_databricks_openai.assert_not_called()
    mock_workspace_client.serving_endpoints.get_open_ai_client.assert_called_once_with()
    assert client == mock_openai_client
