from typing import Any, Dict, List, Optional, Type

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from databricks_langchain.vectorstores import DatabricksVectorSearch

from databricks_ai_bridge.vector_search_retriever_tool import VectorSearchRetrieverToolMixin, VectorSearchRetrieverToolInput
from databricks_ai_bridge.utils.vector_search import IndexDetails


class VectorSearchRetrieverTool(BaseTool, VectorSearchRetrieverToolMixin):
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with Databricks Vector Search and provides a convenient interface
    for building a retriever tool for agents.
    """

    # The BaseTool class requires 'name' and 'description' fields which we will populate in validate_tool_inputs()
    name: str = Field(default="", description="The name of the tool")
    description: str = Field(default="", description="The description of the tool")
    args_schema: Type[BaseModel] = VectorSearchRetrieverToolInput

    embedding: Optional[Embeddings] = Field(
        None, description="Embedding model for self-managed embeddings."
    )
    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
                    "Required for direct-access index or delta-sync index with "
                    "self-managed embeddings.",
    )

    _vector_store: DatabricksVectorSearch = PrivateAttr()

    @model_validator(mode="after")
    def _validate_tool_inputs(self):
        kwargs = {
            "index_name": self.index_name,
            "embedding": self.embedding,
            "text_column": self.text_column,
            "columns": self.columns,
        }
        dbvs = DatabricksVectorSearch(**kwargs)
        self._vector_store = dbvs

        self.name = self.tool_name or self.index_name
        self.description = self.tool_description or self._get_default_tool_description(IndexDetails(dbvs.index))

        return self

    def _run(self, query: str) -> str:
        return self._vector_store.similarity_search(
            query, k=self.num_results, filter=self.filters, query_type=self.query_type
        )