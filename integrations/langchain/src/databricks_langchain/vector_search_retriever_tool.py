from typing import List, Optional, Type, Dict, Any

from databricks_ai_bridge.utils.vector_search import IndexDetails
from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    VectorSearchRetrieverToolInput,
    VectorSearchRetrieverToolMixin,
    vector_search_retriever_tool_trace,
)
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from databricks_langchain import DatabricksEmbeddings
from databricks_langchain.vectorstores import DatabricksVectorSearch


class VectorSearchRetrieverTool(BaseTool, VectorSearchRetrieverToolMixin):
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with Databricks Vector Search and provides a convenient interface
    for building a retriever tool for agents.

    **Note**: Any additional keyword arguments passed to the constructor will be forwarded to
    `databricks.vector_search.index.VectorSearchIndex.similarity_search` when executing the tool.
    See documentation for the full set of supported keyword arguments (e.g., `score_threshold`).
    Also see the mixin docs for additional supported constructor arguments (e.g., `query_type`, `num_results`).

    **New**: `client_args` (optional) is forwarded to `VectorSearchClient` via `DatabricksVectorSearch`.
    Use this to pass service principal credentials (e.g., `service_principal_client_id`,
    `service_principal_client_secret`) or other client options such as `disable_notice`.
    """

    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
        "Required for direct-access index or delta-sync index with self-managed embeddings.",
    )
    embedding: Optional[Embeddings] = Field(
        None, description="Embedding model for self-managed embeddings."
    )

    # Optional pass-through for VectorSearchClient (SP/M2M auth, flags like disable_notice, etc.)
    client_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Additional args forwarded to VectorSearchClient via DatabricksVectorSearch "
            "(e.g., service_principal_client_id/service_principal_client_secret, disable_notice)."
        ),
    )

    # BaseTool requires these; populated in validate_tool_inputs()
    name: str = Field(default="", description="The name of the tool")
    description: str = Field(default="", description="The description of the tool")
    args_schema: Type[BaseModel] = VectorSearchRetrieverToolInput

    _vector_store: DatabricksVectorSearch = PrivateAttr()

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        kwargs: Dict[str, Any] = {
            "index_name": self.index_name,
            "embedding": self.embedding,
            "text_column": self.text_column,
            "doc_uri": self.doc_uri,
            "primary_key": self.primary_key,
            "columns": self.columns,
            "workspace_client": self.workspace_client,
            "include_score": self.include_score,
        }
        if self.client_args:
            kwargs["client_args"] = self.client_args  # <-- pass-through

        dbvs = DatabricksVectorSearch(**kwargs)
        self._vector_store = dbvs

        self.name = self._get_tool_name()
        self.description = self.tool_description or self._get_default_tool_description(
            IndexDetails(dbvs.index)
        )
        self.resources = self._get_resources(
            self.index_name,
            (self.embedding.endpoint if isinstance(self.embedding, DatabricksEmbeddings) else None),
            IndexDetails(dbvs.index),
        )
        return self

    @vector_search_retriever_tool_trace
    def _run(self, query: str, filters: Optional[List[FilterItem]] = None, **kwargs) -> str:
        kwargs = {**kwargs, **(self.model_extra or {})}
        # Since LLM can generate either a dict or FilterItem, convert to dict always
        filters_dict = {dict(item)["key"]: dict(item)["value"] for item in (filters or [])}
        combined_filters = {**filters_dict, **(self.filters or {})}

        # Allow kwargs to override the default values upon invocation
        num_results = kwargs.pop("k", self.num_results)
        query_type = kwargs.pop("query_type", self.query_type)

        # Ensure that we don't have duplicate keys
        kwargs.update(
            {
                "query": query,
                "k": num_results,
                "filter": combined_filters,
                "query_type": query_type,
            }
        )
        return self._vector_store.similarity_search(**kwargs)
