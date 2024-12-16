import json as js
from typing import Any, Dict, Optional

import requests
from databricks.sdk import WorkspaceClient

from databricks_ai_bridge.utils.annotations import experimental


@experimental
def http_request(
    conn: str,
    method: str,
    path: str,
    *,
    json: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> requests.Response:
    """
    Makes an HTTP request to a remote API using authentication from a Unity Catalog HTTP connection.

    Args:
        conn (str): The connection name to use. This is required to identify the external connection.
        method (str): The HTTP method to use (e.g., "GET", "POST"). This is required.
        path (str): The relative path for the API endpoint. This is required.
        json (Optional[Any]): JSON payload for the request.
        headers (Optional[Dict[str, str]]): Additional headers for the request.
                If not provided, only auth headers from connections would be passed.
        params (Optional[Dict[str, Any]]): Query parameters for the request.

    Returns:
        requests.Response: The HTTP response from the external function.

    Example Usage:
        response = http_request(
            conn="my_connection",
            method="POST",
            path="/api/v1/resource",
            json={"key": "value"},
            headers={"extra_header_key": "extra_header_value"},
            params={"query": "example"}
        )
    """
    workspaceConfig = WorkspaceClient().config
    url = f"{workspaceConfig.host}/external-functions"
    request_headers = workspaceConfig._header_factory()
    payload = {
        "connection_name": conn,
        "method": method,
        "path": path,
        "json": js.dumps(json),
        "header": headers,
        "params": params,
    }

    return requests.post(url, headers=request_headers, json=payload)
