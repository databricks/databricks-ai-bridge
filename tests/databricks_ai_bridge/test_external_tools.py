from unittest.mock import MagicMock, patch

from databricks_ai_bridge.external_tools import http_request


@patch("databricks_ai_bridge.external_tools.WorkspaceClient")
@patch("databricks_ai_bridge.external_tools.requests.post")
def test_http_request_success(mock_post, mock_workspace_client):
    # Mock the WorkspaceClient config
    mock_workspace_config = MagicMock()
    mock_workspace_config.host = "https://mock-host"
    mock_workspace_config._header_factory.return_value = {"Authorization": "Bearer mock-token"}
    mock_workspace_client.return_value.config = mock_workspace_config

    # Mock the POST request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"success": True}
    mock_post.return_value = mock_response

    # Call the function
    response = http_request(
        conn="mock_connection",
        method="POST",
        path="/mock-path",
        json={"key": "value"},
        headers={"Custom-Header": "HeaderValue"},
        params={"query": "test"},
    )

    # Assertions
    assert response.status_code == 200
    assert response.json() == {"success": True}
    mock_post.assert_called_once_with(
        "https://mock-host/external-functions",
        headers={
            "Authorization": "Bearer mock-token",
        },
        json={
            "connection_name": "mock_connection",
            "method": "POST",
            "path": "/mock-path",
            "json": '{"key": "value"}',
            "headers": '{"Custom-Header": "HeaderValue"}',
            "params": '{"query": "test"}',
        },
    )


@patch("databricks_ai_bridge.external_tools.WorkspaceClient")
@patch("databricks_ai_bridge.external_tools.requests.post")
def test_http_request_error_response(mock_post, mock_workspace_client):
    # Mock the WorkspaceClient config
    mock_workspace_config = MagicMock()
    mock_workspace_config.host = "https://mock-host"
    mock_workspace_config._header_factory.return_value = {"Authorization": "Bearer mock-token"}
    mock_workspace_client.return_value.config = mock_workspace_config

    # Mock the POST request to return an error
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"error": "Bad Request"}
    mock_post.return_value = mock_response

    # Call the function
    response = http_request(
        conn="mock_connection",
        method="POST",
        path="/mock-path",
        json={"key": "value"},
    )

    # Assertions
    assert response.status_code == 400
    assert response.json() == {"error": "Bad Request"}
    mock_post.assert_called_once_with(
        "https://mock-host/external-functions",
        headers={"Authorization": "Bearer mock-token"},
        json={
            "connection_name": "mock_connection",
            "method": "POST",
            "path": "/mock-path",
            "json": '{"key": "value"}',
            "headers": "null",
            "params": "null",
        },
    )
