"""
Databricks AI support for Google Agent Development Kit (ADK).

This package provides tools and utilities for integrating Databricks AI features
with Google ADK agents.

Available classes and functions:

Tools:
- :class:`VectorSearchRetrieverTool` - Search Databricks Vector Search indexes
- :class:`GenieTool` - Query Databricks Genie AI/BI spaces
- :func:`create_genie_tool` - Factory function to create Genie tools

Toolsets:
- :class:`DatabricksToolset` - Bundle multiple Databricks tools together
- :class:`DatabricksMcpToolset` - Connect to Databricks MCP servers

Deployment:
- :class:`DatabricksAgentEngineApp` - Deploy agents to Vertex AI Agent Engine
- :func:`deploy_to_agent_engine` - One-step deployment helper
- :func:`create_agent_engine_config` - Create deployment configuration

Example:
    ```python
    from databricks_google_adk import VectorSearchRetrieverTool, GenieTool
    from google.adk.agents import Agent

    # Create tools
    vector_search = VectorSearchRetrieverTool(
        index_name="catalog.schema.my_index",
    )
    genie = GenieTool(space_id="your-genie-space-id")

    # Use with an ADK agent
    agent = Agent(
        name="data_assistant",
        model="gemini-2.0-flash",
        instruction="You help users find and analyze data.",
        tools=[vector_search.as_tool(), genie.as_tool()],
    )
    ```
"""

from databricks_google_adk.deployment import (
    DatabricksAgentEngineApp,
    create_agent_engine_config,
    deploy_to_agent_engine,
    get_databricks_requirements,
)
from databricks_google_adk.genie import GenieTool, create_genie_tool
from databricks_google_adk.mcp import (
    DatabricksMcpToolset,
    create_databricks_mcp_toolset,
)
from databricks_google_adk.toolset import DatabricksToolset
from databricks_google_adk.vector_search_retriever_tool import VectorSearchRetrieverTool

__all__ = [
    # Tools
    "VectorSearchRetrieverTool",
    "GenieTool",
    "create_genie_tool",
    # Toolsets
    "DatabricksToolset",
    "DatabricksMcpToolset",
    "create_databricks_mcp_toolset",
    # Deployment
    "DatabricksAgentEngineApp",
    "deploy_to_agent_engine",
    "create_agent_engine_config",
    "get_databricks_requirements",
]
