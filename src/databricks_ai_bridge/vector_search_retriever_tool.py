from functools import wraps
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    Resource,
)
from pydantic import BaseModel, Field

from databricks_ai_bridge.utils.vector_search import IndexDetails

DEFAULT_TOOL_DESCRIPTION = "A vector search-based retrieval tool for querying indexed embeddings."


def vector_search_retriever_tool_trace(func):
    """
    Decorator factory to trace VectorSearchRetrieverTool with the tool name
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create a new decorator with the instance's name
        traced_func = mlflow.trace(
            name=self.tool_name or self.index_name, span_type=SpanType.RETRIEVER
        )(func)
        # Call the traced function with self
        return traced_func(self, *args, **kwargs)

    return wrapper


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
    num_results: int = Field(5, description="The number of results to return.")
    columns: Optional[List[str]] = Field(
        None, description="Columns to return when doing the search."
    )
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply to the search.")
    query_type: str = Field(
        "ANN", description="The type of this query. Supported values are 'ANN' and 'HYBRID'."
    )
    tool_name: Optional[str] = Field(None, description="The name of the retrieval tool.")
    tool_description: Optional[str] = Field(None, description="A description of the tool.")
    resources: Optional[List[dict]] = Field(
        None, description="Resources required to log a model that uses this tool."
    )

    def _get_default_tool_description(self, index_details: IndexDetails) -> str:
        if index_details.is_delta_sync_index():
            source_table = index_details.index_spec.get("source_table", "")
            return (
                DEFAULT_TOOL_DESCRIPTION
                + f" The queried index uses the source table {source_table}"
            )
        return DEFAULT_TOOL_DESCRIPTION

    def _get_resources(self, index_name: str, embedding_endpoint: str) -> List[Resource]:
        return ([DatabricksVectorSearchIndex(index_name=index_name)] if index_name else []) + (
            [DatabricksServingEndpoint(endpoint_name=embedding_endpoint)]
            if embedding_endpoint
            else []
        )
