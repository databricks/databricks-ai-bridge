import logging
import re
from functools import wraps
from typing import Any, Dict, List, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    Resource,
)
from pydantic import BaseModel, ConfigDict, Field, validator

from databricks_ai_bridge.utils.vector_search import IndexDetails

_logger = logging.getLogger(__name__)
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

    model_config = ConfigDict(arbitrary_types_allowed=True)
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
    workspace_client: Optional[WorkspaceClient] = Field(
        default=None,
        description="When specified, will use workspace client credential strategy to instantiate VectorSearchClient",
    )

    @validator("tool_name")
    def validate_tool_name(cls, tool_name):
        if tool_name is not None:
            pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
            if not pattern.fullmatch(tool_name):
                raise ValueError("tool_name must match the pattern '^[a-zA-Z0-9_-]{1,64}$'")
        return tool_name

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

    def _get_tool_name(self) -> str:
        tool_name = self.tool_name or self.index_name.replace(".", "__")

        # Tool names must match the pattern '^[a-zA-Z0-9_-]+$'."
        # The '.' from the index name are not allowed
        if len(tool_name) > 64:
            _logger.warning(
                f"Tool name {tool_name} is too long, truncating to 64 characters {tool_name[-64:]}."
            )
            return tool_name[-64:]
        return tool_name
