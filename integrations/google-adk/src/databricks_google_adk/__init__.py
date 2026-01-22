"""
Databricks AI support for Google Agent Development Kit (ADK).

This package provides tools and utilities for integrating Databricks AI features
with Google ADK agents.

Available classes and functions:

- :class:`VectorSearchRetrieverTool` - Search Databricks Vector Search indexes
- :class:`GenieTool` - Query Databricks Genie AI/BI spaces
- :func:`create_genie_tool` - Factory function to create Genie tools
- :class:`DatabricksToolset` - Bundle multiple Databricks tools together

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

from databricks_google_adk.genie import GenieTool, create_genie_tool
from databricks_google_adk.toolset import DatabricksToolset
from databricks_google_adk.vector_search_retriever_tool import VectorSearchRetrieverTool

__all__ = [
    "VectorSearchRetrieverTool",
    "GenieTool",
    "create_genie_tool",
    "DatabricksToolset",
]
