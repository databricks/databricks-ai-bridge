from typing import Any, Dict, List, Optional
from abc import abstractmethod
from pydantic import BaseModel, Field
from databricks_ai_bridge.utils.vector_search import IndexDetails
from databricks.vector_search.client import VectorSearchIndex

DEFAULT_TOOL_DESCRIPTION = "A vector search-based retrieval tool for querying indexed embeddings."

class VectorSearchRetrieverToolInput(BaseModel):
    query: str = Field(
        description="The string used to query the index with and identify the most similar "
                    "vectors and return the associated documents."
    )

class VectorSearchRetrieverToolMixin(BaseModel):
    """
    Mixin class for Databricks Vector Search retrieval tools.
    This class provides the common structure and interface that framework-specific
    implementations should follow.
    """

    index_name: str = Field(
        ..., description="The name of the index to use, format: 'catalog.schema.index'."
    )
    num_results: int = Field(10, description="The number of results to return.")
    columns: Optional[List[str]] = Field(
        None, description="Columns to return when doing the search."
    )
    filters: Optional[Dict[str, Any]] = Field(
        None, description="Filters to apply to the search."
    )
    query_type: str = Field(
        "ANN", description="The type of this query. Supported values are 'ANN' and 'HYBRID'."
    )
    tool_name: Optional[str] = Field(
        None, description="The name of the retrieval tool."
    )
    tool_description: Optional[str] = Field(
        None, description="A description of the tool."
    )

    def _get_default_tool_description(self, index_details: IndexDetails) -> str:
        if index_details.is_delta_sync_index():
            from databricks.sdk import WorkspaceClient

            source_table = index_details.index_spec.get("source_table", "")
            w = WorkspaceClient()
            source_table_comment = w.tables.get(full_name=source_table).comment
            if source_table_comment:
                return (
                        DEFAULT_TOOL_DESCRIPTION
                        + f" The queried index uses the source table {source_table} with the description: "
                        + source_table_comment
                )
            else:
                return (
                        DEFAULT_TOOL_DESCRIPTION
                        + f" The queried index uses the source table {source_table}"
                )
        return DEFAULT_TOOL_DESCRIPTION
