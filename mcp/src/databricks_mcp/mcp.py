import asyncio
import logging
import re
from typing import List, Any
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient
from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool, CallToolResult
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksResource,
    DatabricksVectorSearchIndex,
)

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


class McpServer:
    def __init__(self, client: WorkspaceClient, mcp_url: str):
        self.client = client
        self.mcp_url = mcp_url

    def _is_valid_databricks_managed_mcp_url(self) -> bool:
        """Validate the MCP URL against known patterns."""
        path = urlparse(self.mcp_url).path
        return any(re.match(pattern, path) for pattern in MCP_URL_PATTERNS.values())


    def _get_databricks_managed_mcp_url_type(self) -> str:
        """Determine the MCP URL type based on the path."""
        path = urlparse(self.mcp_url).path
        for mcp_type, pattern in MCP_URL_PATTERNS.items():
            if re.match(pattern, path):
                return mcp_type
        raise ValueError(f"Unrecognized MCP URL: {self.mcp_url}")


    async def _get_tools_async(self) -> List[Tool]:
        """Fetch tools from the MCP endpoint asynchronously."""
        async with streamablehttp_client(
            url=self.mcp_url,
            auth=DatabricksOAuthClientProvider(self.client),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return (await session.list_tools()).tools

    async def _call_tools_async(self, tool_name: str, arguments: dict[str, Any] | None = None,) -> CallToolResult:
        """Call the tool with the given name and input."""
        async with streamablehttp_client(
            url=self.mcp_url,
            auth=DatabricksOAuthClientProvider(self.client),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                return await session.call_tool(tool_name, arguments)

    def _extract_genie_id(self) -> str:
        """Extract the Genie space ID from the URL."""
        path = urlparse(self.mcp_url).path
        if "/genie/" not in path:
            raise ValueError(f"Missing /genie/ segment in: {self.mcp_url}")
        genie_id = path.split("/genie/", 1)[1]
        if not genie_id:
            raise ValueError(f"Genie ID not found in: {self.mcp_url}")
        return genie_id


    def _normalize_tool_name(self, name: str) -> str:
        """Convert double underscores to dots for compatibility."""
        return name.replace("__", ".")

    def list_tools(self) -> List[Tool]:
        """List the tools from the MCP URL."""
        return asyncio.run(self.get_tools_async())
    

    def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        """Call the tool with the given name and input."""
        return asyncio.run(self._call_tools_async(tool_name, arguments))

    def get_databricks_resources(self) -> List[DatabricksResource]:
        """
        This method is used to get the Databricks resources from the MCP URL.
        It will return a list of Databricks resources.

        Note that this method is only for databricks managed MCP URLs. It returns empty list for other MCP URLs
        """
        try:
            if not self.is_valid_databricks_managed_mcp_url():
                logger.warning(f"Ignoring invalid MCP URL: {self.mcp_url}")
                return []

            mcp_type = self._get_databricks_managed_mcp_url_type()

            if mcp_type == GENIE_MCP:
                return [DatabricksGenieSpace(self.extract_genie_id())]

            tools = self.list_tools()
            normalized = [self._normalize_tool_name(tool.name) for tool in tools]

            if mcp_type == UC_FUNCTIONS_MCP:
                return [DatabricksFunction(name) for name in normalized]
            elif mcp_type == VECTOR_SEARCH_MCP:
                return [DatabricksVectorSearchIndex(name) for name in normalized]

            logger.warning(f"Unknown MCP type: {mcp_type}")
            return []

        except ValueError as ve:
            logger.warning(f"Failed to parse MCP URL: {ve}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving Databricks resources: {e}")
            return []