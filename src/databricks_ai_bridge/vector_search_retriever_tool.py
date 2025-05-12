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
    model_config = ConfigDict(extra="allow")
    query: str = Field(
        description="The string used to query the index with and identify the most similar "
        "vectors and return the associated documents."
    )
    filters: Dict[str, Any] = Field(
        default=None,
        description=(
            "Optional filters to refine vector search results. Supports the following operators:\n\n"
            '- Inclusion: {"column": value} or {"column": [value1, value2]} (matches if the column equals any of the provided values)\n'
            '- Exclusion: {"column NOT": value}\n'
            '- Comparisons: {"column <": value}, {"column >=": value}, etc.\n'
            '- Pattern match: {"column LIKE": "word"} (matches full tokens separated by whitespace)\n'
            '- OR logic: {"column1 OR column2": [value1, value2]} '
            "(matches if column1 equals value1 or column2 equals value2; matches are position-specific)"
        ),
    )


class VectorSearchRetrieverToolMixin(BaseModel):
    """
    Mixin class for Databricks Vector Search retrieval tools.
    This class provides the common structure and interface that framework-specific
    implementations should follow.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
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
        None,
        description="When specified, will use workspace client credential strategy to instantiate VectorSearchClient",
    )
    doc_uri: Optional[str] = Field(
        None, description="The URI for the document, used for rendering a link in the UI."
    )
    primary_key: Optional[str] = Field(
        None,
        description="Identifies the chunk that the document is a part of. This is used by some evaluation metrics.",
    )

    @validator("tool_name")
    def validate_tool_name(cls, tool_name):
        if tool_name is not None:
            pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
            if not pattern.fullmatch(tool_name):
                raise ValueError("tool_name must match the pattern '^[a-zA-Z0-9_-]{1,64}$'")
        return tool_name

    def _describe_columns(self) -> str:
        try:
            from databricks.sdk import WorkspaceClient

            if self.workspace_client:
                table_info = self.workspace_client.tables.get(full_name=self.index_name)
            else:
                table_info = WorkspaceClient().tables.get(full_name=self.index_name)

            columns = []

            for column_info in table_info.columns:
                name = column_info.name
                comment = column_info.comment or "No description provided"
                col_type = column_info.type_name.name
                if not name.startswith("__"):
                    columns.append((name, col_type, comment))

            return "The vector search index includes the following columns:\n" + "\n".join(
                f"{name} ({col_type}): {comment}" for name, col_type, comment in columns
            )
        except Exception:
            _logger.warning(
                "Unable to retrieve column information automatically. Please manually specify column names, types, and descriptions in the tool description to help LLMs apply filters correctly."
            )

    def _get_default_tool_description(self, index_details: IndexDetails) -> str:
        if index_details.is_delta_sync_index():
            source_table = index_details.index_spec.get("source_table", "")
            description = (
                DEFAULT_TOOL_DESCRIPTION
                + f" The queried index uses the source table {source_table}."
            )
        else:
            description = DEFAULT_TOOL_DESCRIPTION

        column_description = self._describe_columns()
        if column_description:
            return f"{description}\n\n{column_description}"
        else:
            return description

    def _get_resources(
        self, index_name: str, embedding_endpoint: str, index_details: IndexDetails = None
    ) -> List[Resource]:
        resources = []
        if index_name:
            resources.append(DatabricksVectorSearchIndex(index_name=index_name))
        if embedding_endpoint:
            resources.append(DatabricksServingEndpoint(endpoint_name=embedding_endpoint))
        if (
            index_details
            and index_details.is_databricks_managed_embeddings
            and (
                managed_embedding := index_details.embedding_source_column.get(
                    "embedding_model_endpoint_name", None
                )
            )
        ):
            if managed_embedding != embedding_endpoint:
                resources.append(DatabricksServingEndpoint(endpoint_name=managed_embedding))
        return resources

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
