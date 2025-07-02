import asyncio
import logging
import re
from typing import Any, List, Optional
from urllib.parse import urlparse

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

    def list_tools(self) -> List[Tool]:
        """
        Lists the tools for the current MCP Server. This method uses the `streamablehttp_client` from mcp to fetch all the tools from the MCP server.

        Returns:
            List[mcp.types.Tool]: A list of tools for the current MCP Server.
        """
        return asyncio.run(self._get_tools_async())

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
