from typing import Any, Dict, Optional, Type

from databricks_ai_bridge.utils.vector_search import IndexDetails
from databricks_ai_bridge.vector_search_retriever_tool import (
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
    """

    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
        "Required for direct-access index or delta-sync index with "
        "self-managed embeddings.",
    )
    embedding: Optional[Embeddings] = Field(
        None, description="Embedding model for self-managed embeddings."
    )

    # The BaseTool class requires 'name' and 'description' fields which we will populate in validate_tool_inputs()
    name: str = Field(default="", description="The name of the tool")
    description: str = Field(default="", description="The description of the tool")
    args_schema: Type[BaseModel] = VectorSearchRetrieverToolInput

    _vector_store: DatabricksVectorSearch = PrivateAttr()

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        kwargs = {
            "index_name": self.index_name,
            "embedding": self.embedding,
            "text_column": self.text_column,
            "doc_uri": self.doc_uri,
            "primary_key": self.primary_key,
            "columns": self.columns,
            "workspace_client": self.workspace_client,
        }
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
    def _run(self, query: str, filters: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        kwargs = {**kwargs, **(self.model_extra or {})}
        combined_filters = {**(filters or {}), **(self.filters or {})}
        # Ensure that we don't have duplicate keys
        kwargs.update(
            {
                "query": query,
                "k": self.num_results,
                "filter": combined_filters,
                "query_type": self.query_type,
            }
        )
        return self._vector_store.similarity_search(**kwargs)
