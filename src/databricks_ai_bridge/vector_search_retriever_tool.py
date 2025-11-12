import logging
import re
from functools import wraps
from typing import Any, Dict, List, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.vector_search.reranker import Reranker
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
    Resource,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator, validator

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


class FilterItem(BaseModel):
    key: str = Field(
        description="The filter key, which includes the column name and can include operators like 'NOT', '<', '>=', 'LIKE', 'OR'"
    )
    value: Any = Field(
        description="The filter value, which can be a single value or an array of values"
    )


class VectorSearchRetrieverToolInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str = Field(
        description="The string used to query the index with and identify the most similar "
        "vectors and return the associated documents."
    )
    filters: Optional[List[FilterItem]] = Field(
        default=None,
        description=(
            "Optional filters to refine vector search results as an array of key-value pairs. Supports the following operators:\n\n"
            '- Inclusion: [{"key": "column", "value": value}] or [{"key": "column", "value": [value1, value2]}] (matches if the column equals any of the provided values)\n'
            '- Exclusion: [{"key": "column NOT", "value": value}]\n'
            '- Comparisons: [{"key": "column <", "value": value}], [{"key": "column >=", "value": value}], etc.\n'
            '- Pattern match: [{"key": "column LIKE", "value": "word"}] (matches full tokens separated by whitespace)\n'
            '- OR logic: [{"key": "column1 OR column2", "value": [value1, value2]}] '
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
        description="WorkspaceClient instances with auth types PAT, OAuth-M2M (client ID and client secret) "
        "or model serving credential strategy will be used to instantiate the underlying VectorSearchClient.",
    )
    doc_uri: Optional[str] = Field(
        None, description="The URI for the document, used for rendering a link in the UI."
    )
    primary_key: Optional[str] = Field(
        None,
        description="Identifies the chunk that the document is a part of. This is used by some evaluation metrics.",
    )
    include_score: Optional[bool] = Field(
        False, description="When true, will return the similarity score with the metadata."
    )
    dynamic_filter: bool = Field(
        False,
        description="When true, enables LLM-generated filter parameters in the tool schema. "
        "This allows LLMs to dynamically generate filters based on natural language queries. "
        "Cannot be used together with predefined filters (filters parameter).",
    )
    reranker: Optional["Reranker"] = Field(
        None,
        description="When specified, will reranker the search results to improve relevance. "
        "\n\nRead more about reranking at: "
        "https://www.databricks.com/blog/reranking-mosaic-ai-vector-search-faster-smarter-retrieval-rag-agents",
    )

    @model_validator(mode="after")
    def validate_filter_configuration(self):
        """Validate that dynamic_filter and filters are not both enabled."""
        if self.dynamic_filter and self.filters:
            raise ValueError(
                "Cannot use both dynamic_filter=True and predefined filters. "
                "Please either enable dynamic_filter for LLM-generated filters, "
                "or provide predefined filters via the filters parameter, but not both."
            )
        return self

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

    def _get_filter_param_description(self) -> str:
        """Generate a comprehensive filter parameter description including available columns."""
        base_description = (
            "Optional filters to refine vector search results as an array of key-value pairs. "
            "IMPORTANT: If unsure about filter values, try searching WITHOUT filters first to get broad results, "
            "then optionally add filters to narrow down if needed. This ensures you don't miss relevant results due to incorrect filter values. "
        )

        # Get column information from Unity Catalog
        # This is required for dynamic filters to provide accurate column metadata to the LLM
        from databricks.sdk import WorkspaceClient

        column_info = []
        try:
            if self.workspace_client:
                table_info = self.workspace_client.tables.get(full_name=self.index_name)
            else:
                table_info = WorkspaceClient().tables.get(full_name=self.index_name)

            for column_info_item in table_info.columns:
                name = column_info_item.name
                col_type = column_info_item.type_name.name
                if not name.startswith("__"):
                    column_info.append((name, col_type))
        except Exception as e:
            raise ValueError(
                f"Failed to retrieve table metadata for index '{self.index_name}'. "
                f"Table metadata is required when dynamic_filter=True to provide accurate column information to the LLM. "
                f"Please ensure the table exists and you have permissions to access it. Error: {e}"
            ) from e

        # Validate that we got column information
        if not column_info:
            raise ValueError(
                f"No valid columns found in table metadata for index '{self.index_name}'. "
                f"Table metadata is required when dynamic_filter=True to provide accurate column information to the LLM. "
                f"Please ensure the table has columns defined."
            )

        base_description += f"Available columns for filtering: {', '.join([f'{name} ({col_type})' for name, col_type in column_info])}. "

        base_description += (
            "Supports the following operators:\n\n"
            '- Inclusion: [{"key": "column", "value": value}] or [{"key": "column", "value": [value1, value2]}] (matches if the column equals any of the provided values)\n'
            '- Exclusion: [{"key": "column NOT", "value": value}]\n'
            '- Comparisons: [{"key": "column <", "value": value}], [{"key": "column >=", "value": value}], etc.\n'
            '- Pattern match: [{"key": "column LIKE", "value": "word"}] (matches full tokens separated by whitespace)\n'
            '- OR logic: [{"key": "column1 OR column2", "value": [value1, value2]}] '
            "(matches if column1 equals value1 or column2 equals value2; matches are position-specific)\n\n"
            "Examples:\n"
            '- Filter by category: [{"key": "category", "value": "electronics"}]\n'
            '- Filter by price range: [{"key": "price >=", "value": 100}, {"key": "price <", "value": 500}]\n'
            '- Exclude specific status: [{"key": "status NOT", "value": "archived"}]\n'
            '- Pattern matching: [{"key": "description LIKE", "value": "wireless"}]'
        )

        return base_description

    def _create_enhanced_input_model(self):
        """Create an input model with filter parameters enabled."""
        filter_description = self._get_filter_param_description()

        class EnhancedVectorSearchRetrieverToolInput(BaseModel):
            model_config = ConfigDict(extra="allow")
            query: str = Field(
                description="The string used to query the index with and identify the most similar "
                "vectors and return the associated documents."
            )
            filters: Optional[List[FilterItem]] = Field(
                default=None,
                description=filter_description,
            )

        return EnhancedVectorSearchRetrieverToolInput

    def _create_basic_input_model(self):
        """Create an input model without filter parameters."""

        class BasicVectorSearchRetrieverToolInput(BaseModel):
            model_config = ConfigDict(extra="allow")
            query: str = Field(
                description="The string used to query the index with and identify the most similar "
                "vectors and return the associated documents."
            )

        return BasicVectorSearchRetrieverToolInput

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
