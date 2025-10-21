import asyncio
import json
import logging
import re
from functools import wraps
from typing import Any, Callable, List, Optional
from urllib.parse import urlparse

import requests
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.utils.annotations import experimental
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, Tool
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksResource,
    DatabricksVectorSearchIndex,
)

from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider

logger = logging.getLogger(__name__)

# MCP URL types
UC_FUNCTIONS_MCP = "uc_functions_mcp"
VECTOR_SEARCH_MCP = "vector_search_mcp"
GENIE_MCP = "genie_mcp"

MCP_URL_PATTERNS = {
    UC_FUNCTIONS_MCP: r"^/api/2\.0/mcp/functions/[^/]+/[^/]+$",
    VECTOR_SEARCH_MCP: r"^/api/2\.0/mcp/vector-search/[^/]+/[^/]+$",
    GENIE_MCP: r"^/api/2\.0/mcp/genie/[^/]+$",
}


def _handle_mcp_errors(func: Callable) -> Callable:
    """Decorator to handle MCP connection errors for sync wrapper methods."""

    def _process_mcp_error(client_instance, error: Exception) -> None:
        """Process and enhance MCP connection errors with better context."""
        # For Databricks-managed MCP servers, no special error processing needed
        if client_instance._get_databricks_managed_mcp_url_type() is not None:
            raise error

        try:
            headers = client_instance.client.config.authenticate()
            authorization_header = headers["Authorization"]
            token = authorization_header.split("Bearer ")[1]

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {token}",
            }
            payload = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": "1av",
                    "params": {
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                    },
                }
            )
            response = requests.request(
                "POST",
                client_instance.server_url,
                headers=headers,
                data=payload,
                allow_redirects=False,
            )
        except Exception as e:
            # Error during processing the error, re-raise the original error
            raise error from None

        # Auth errors to Databricks Apps are redirected to a login page; a 302 often indicates an auth issue
        if response.status_code == 302:
            raise PermissionError(
                "Access denied to the MCP server. When accessing an MCP server hosted on a Databricks App please ensure you are using a valid OAuth token. "
                "If using a Service Principal, ensure that the service principal has query permissions on the Databricks App. "
                "For more information refer to the documentation here: "
                "https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp?language=Local+environment#connect-to-the-custom-mcp-server"
            ) from None
        # Not finding a `/initialize` endpoint means the MCP server is not running or the endpoint is not correct
        elif response.status_code == 404:
            raise ValueError(
                "MCP Server not found at the provided server url. Please ensure the server url specified hosts a MCP Server. For more information refer to the documentation here: "
                "https://docs.databricks.com/aws/en/generative-ai/mcp/custom-mcp"
            ) from None

        # If the error is not a 302 or 404, re-raise the original error
        raise error

    @wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as error:
            _process_mcp_error(self, error)

    return sync_wrapper


@experimental
class DatabricksMCPClient:
    """
    A client for interacting with a MCP(Model Context Protocol) servers on Databricks.

    This class provides a simplified interface to communicate with a specified MCP server URL with Databricks Authorization.
    Additionally this client provides helpers to retrieve the dependent resources for Databricks Managed MCP Resources to enable
    automatic authorization in Model Serving.

    Attributes:
        server_url (str): The base URL of the MCP server to which this client connects.
        client (databricks.sdk.WorkspaceClient): The Databricks workspace client used for authentication and requests.
    """

    def __init__(self, server_url: str, workspace_client: Optional[WorkspaceClient] = None):
        self.client = workspace_client or WorkspaceClient()
        self.server_url = server_url

    def _get_databricks_managed_mcp_url_type(self) -> str:
        """Determine the MCP URL type based on the path."""
        path = urlparse(self.server_url).path
        for mcp_type, pattern in MCP_URL_PATTERNS.items():
            if re.match(pattern, path):
                return mcp_type

        return None

    async def _get_tools_async(self) -> List[Tool]:
        """Fetch tools from the MCP endpoint asynchronously."""
        async with streamablehttp_client(
            url=self.server_url,
            auth=DatabricksOAuthClientProvider(self.client),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return (await session.list_tools()).tools

    async def _call_tools_async(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Call the tool with the given name and input."""
        async with streamablehttp_client(
            url=self.server_url,
            auth=DatabricksOAuthClientProvider(self.client),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.call_tool(tool_name, arguments)

    def _extract_genie_id(self) -> str:
        """Extract the Genie space ID from the URL."""
        path = urlparse(self.server_url).path
        if "/genie/" not in path:
            raise ValueError(f"Missing /genie/ segment in: {self.server_url}")
        genie_id = path.split("/genie/", 1)[1]
        if not genie_id:
            raise ValueError(f"Genie ID not found in: {self.server_url}")
        return genie_id

    def _normalize_tool_name(self, name: str) -> str:
        """Convert double underscores to dots for compatibility."""
        return name.replace("__", ".")

    @_handle_mcp_errors
    def list_tools(self) -> List[Tool]:
        """
        Lists the tools for the current MCP Server. This method uses the `streamablehttp_client` from mcp to fetch all the tools from the MCP server.

        Returns:
            List[mcp.types.Tool]: A list of tools for the current MCP Server.
        """
        return asyncio.run(self._get_tools_async())

    @_handle_mcp_errors
    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        """
        Calls the tool with the given name and input. This method uses the `streamablehttp_client` from mcp to call the tool.

        Args:
            tool_name (str): The name of the tool to call.
            arguments (dict[str, Any], optional): The arguments to pass to the tool. Defaults to None.

        Returns:
            mcp.types.CallToolResult: The result of the tool call.
        """
        return asyncio.run(self._call_tools_async(tool_name, arguments))

    def get_databricks_resources(self) -> List[DatabricksResource]:
        """
        Returns a list of dependent Databricks resources for the current MCP server URL.

        If authoring a custom code agent that runs tools from a Databricks Managed MCP server,
        call this method and pass the returned resources to `mlflow.pyfunc.log_model`
        when logging your agent, to enable your agent to authenticate to the MCP server and run tools when deployed.

        Note that this method only supports detecting resources for Databricks-managed MCP servers.
        For custom MCP servers or other MCP server URLs, this method returns an empty list
        """
        try:
            mcp_type = self._get_databricks_managed_mcp_url_type()
            if mcp_type is None:
                raise ValueError(
                    "Invalid Databricks MCP URL. Please ensure the url is of the form: <host>/api/2.0/mcp/functions/<catalog>/<schema>, "
                    "<host>/api/2.0/mcp/vector-search/<catalog>/<schema> "
                    "or <host>/api/2.0/mcp/genie/<genie-space-id>"
                )

            if mcp_type == GENIE_MCP:
                return [DatabricksGenieSpace(self._extract_genie_id())]

            tools = self.list_tools()
            normalized = [self._normalize_tool_name(tool.name) for tool in tools]

            if mcp_type == UC_FUNCTIONS_MCP:
                return [DatabricksFunction(name) for name in normalized]
            elif mcp_type == VECTOR_SEARCH_MCP:
                return [DatabricksVectorSearchIndex(name) for name in normalized]

            logger.warning(
                f"Unable to extract resources as the mcp type is not recognized: {mcp_type}"
            )
            return []

        except Exception as e:
            logger.error(f"Error retrieving Databricks resources: {e}")
            return []
