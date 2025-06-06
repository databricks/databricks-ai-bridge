import asyncio
import logging
import re
from typing import List
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient
from databricks_mcp.oauth_provider import DatabricksOAuthClientProvider
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool
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


def is_valid_mcp_url(url: str) -> bool:
    """Validate the MCP URL against known patterns."""
    path = urlparse(url).path
    return any(re.match(pattern, path) for pattern in MCP_URL_PATTERNS.values())


def get_mcp_url_type(url: str) -> str:
    """Determine the MCP URL type based on the path."""
    path = urlparse(url).path
    for mcp_type, pattern in MCP_URL_PATTERNS.items():
        if re.match(pattern, path):
            return mcp_type
    raise ValueError(f"Unrecognized MCP URL: {url}")


async def get_tools_async(url: str, client: WorkspaceClient) -> List[Tool]:
    """Fetch tools from the MCP endpoint asynchronously."""
    async with streamablehttp_client(
        url=url,
        auth=DatabricksOAuthClientProvider(client),
    ) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return (await session.list_tools()).tools


def extract_genie_id(url: str) -> str:
    """Extract the Genie space ID from the URL."""
    path = urlparse(url).path
    if "/genie/" not in path:
        raise ValueError(f"Missing /genie/ segment in: {url}")
    genie_id = path.split("/genie/", 1)[1]
    if not genie_id:
        raise ValueError(f"Genie ID not found in: {url}")
    return genie_id


def normalize_tool_name(name: str) -> str:
    """Convert double underscores to dots for compatibility."""
    return name.replace("__", ".")


def get_databricks_resources(client: WorkspaceClient, url: str) -> List[DatabricksResource]:
    """Determine resource type from URL and return appropriate Databricks resource instances."""
    try:
        if not is_valid_mcp_url(url):
            logger.warning(f"Ignoring invalid MCP URL: {url}")
            return []

        mcp_type = get_mcp_url_type(url)

        if mcp_type == GENIE_MCP:
            return [DatabricksGenieSpace(extract_genie_id(url))]

        tools = asyncio.run(get_tools_async(url, client))
        normalized = [normalize_tool_name(tool.name) for tool in tools]

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