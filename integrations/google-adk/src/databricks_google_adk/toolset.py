from typing import Callable, Optional, Union

from databricks.sdk import WorkspaceClient
from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset, ToolPredicate

from databricks_google_adk.genie import GenieTool
from databricks_google_adk.vector_search_retriever_tool import VectorSearchRetrieverTool


class DatabricksToolset(BaseToolset):
    """
    A Google ADK toolset that bundles Databricks AI tools together.

    This toolset provides convenient access to Databricks Vector Search and Genie
    tools for use with Google ADK agents.

    Example:
        ```python
        from databricks_google_adk import DatabricksToolset
        from google.adk.agents import Agent

        # Create a toolset with Vector Search and Genie
        toolset = DatabricksToolset(
            vector_search_indexes=["catalog.schema.my_index"],
            genie_space_ids=["genie-space-123"],
        )

        # Use with an ADK agent
        agent = Agent(
            name="data_assistant",
            model="gemini-2.0-flash",
            instruction="You help users find and analyze data.",
            tools=[toolset],
        )
        ```
    """

    def __init__(
        self,
        *,
        vector_search_indexes: list[str] | None = None,
        genie_space_ids: list[str] | None = None,
        workspace_client: Optional[WorkspaceClient] = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
        tool_filter: Optional[Union[ToolPredicate, list[str]]] = None,
        tool_name_prefix: Optional[str] = None,
    ):
        """
        Initialize the DatabricksToolset.

        Args:
            vector_search_indexes: List of Vector Search index names to include.
                Each index name should be in the format "catalog.schema.index".
            genie_space_ids: List of Genie space IDs to include.
            workspace_client: Optional WorkspaceClient for authentication.
            embedding_fn: Optional embedding function for self-managed embeddings
                in Vector Search indexes.
            tool_filter: Optional filter to select specific tools by name or predicate.
            tool_name_prefix: Optional prefix to add to all tool names.
        """
        super().__init__(tool_filter=tool_filter, tool_name_prefix=tool_name_prefix)

        self._workspace_client = workspace_client
        self._embedding_fn = embedding_fn
        self._tools: list[BaseTool] = []

        # Create Vector Search tools
        for index_name in vector_search_indexes or []:
            vs_tool = VectorSearchRetrieverTool(
                index_name=index_name,
                workspace_client=workspace_client,
                embedding_fn=embedding_fn,
            )
            self._tools.append(vs_tool.as_tool())

        # Create Genie tools
        for space_id in genie_space_ids or []:
            genie_tool = GenieTool(
                space_id=space_id,
                tool_name=f"genie_{space_id.replace('-', '_')}",
                client=workspace_client,
            )
            self._tools.append(genie_tool.as_tool())

    async def get_tools(self, readonly_context=None) -> list[BaseTool]:
        """
        Return all tools in the toolset.

        Args:
            readonly_context: Optional context for filtering tools.

        Returns:
            List of BaseTool instances.
        """
        # Apply filtering if tool_filter is set
        if self.tool_filter is not None:
            return [
                tool for tool in self._tools
                if self._is_tool_selected(tool.name if hasattr(tool, 'name') else str(tool))
            ]
        return self._tools

    def add_vector_search_tool(
        self,
        index_name: str,
        tool_name: str | None = None,
        tool_description: str | None = None,
        num_results: int = 5,
        **kwargs,
    ) -> "DatabricksToolset":
        """
        Add a Vector Search tool to the toolset.

        Args:
            index_name: The name of the Vector Search index.
            tool_name: Optional custom name for the tool.
            tool_description: Optional custom description.
            num_results: Number of results to return (default: 5).
            **kwargs: Additional arguments passed to VectorSearchRetrieverTool.

        Returns:
            Self for method chaining.
        """
        vs_tool = VectorSearchRetrieverTool(
            index_name=index_name,
            tool_name=tool_name,
            tool_description=tool_description,
            num_results=num_results,
            workspace_client=self._workspace_client,
            embedding_fn=self._embedding_fn,
            **kwargs,
        )
        self._tools.append(vs_tool.as_tool())
        return self

    def add_genie_tool(
        self,
        space_id: str,
        tool_name: str | None = None,
        tool_description: str | None = None,
        **kwargs,
    ) -> "DatabricksToolset":
        """
        Add a Genie tool to the toolset.

        Args:
            space_id: The ID of the Genie space.
            tool_name: Optional custom name for the tool.
            tool_description: Optional custom description.
            **kwargs: Additional arguments passed to GenieTool.

        Returns:
            Self for method chaining.
        """
        genie = GenieTool(
            space_id=space_id,
            tool_name=tool_name or f"genie_{space_id.replace('-', '_')}",
            tool_description=tool_description,
            client=self._workspace_client,
            **kwargs,
        )
        self._tools.append(genie.as_tool())
        return self

    def add_custom_tool(self, tool: FunctionTool | BaseTool) -> "DatabricksToolset":
        """
        Add a custom tool to the toolset.

        Args:
            tool: A FunctionTool or BaseTool instance.

        Returns:
            Self for method chaining.
        """
        self._tools.append(tool)
        return self

    async def close(self) -> None:
        """Clean up resources."""
        # No persistent resources to clean up
        pass
