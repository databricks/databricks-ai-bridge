from typing import Any, Dict, List, Optional, Tuple

from databricks_ai_bridge.utils.vector_search import (
    IndexDetails,
    parse_vector_search_response,
    validate_and_get_return_columns,
    validate_and_get_text_column,
)
from databricks_ai_bridge.vector_search_retriever_tool import (
    VectorSearchRetrieverToolInput,
    VectorSearchRetrieverToolMixin,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.types import ToolMetadata
from pydantic import Field, PrivateAttr


class VectorSearchRetrieverTool(FunctionTool, VectorSearchRetrieverToolMixin):
    """Vector search retriever tool implementation."""

    text_column: Optional[str] = Field(
        None,
        description="The name of the text column to use for the embeddings. "
        "Required for direct-access index or delta-sync index with "
        "self-managed embeddings.",
    )
    embedding: Optional[BaseEmbedding] = Field(
        None, description="Embedding model for self-managed embeddings."
    )
    return_direct: bool = Field(
        default=False,
        description="Whether the tool should return the output directly",
    )

    _index = PrivateAttr()
    _index_details = PrivateAttr()

    def __init__(self, **data):
        # First initialize the VectorSearchRetrieverToolMixin
        VectorSearchRetrieverToolMixin.__init__(self, **data)

        # Initialize private attributes
        from databricks.vector_search.client import VectorSearchClient

        self._index = VectorSearchClient().get_index(index_name=self.index_name)
        self._index_details = IndexDetails(self._index)

        # Validate columns
        self.text_column = validate_and_get_text_column(self.text_column, self._index_details)
        self.columns = validate_and_get_return_columns(
            self.columns or [], self.text_column, self._index_details
        )

        # Define the similarity search function
        def similarity_search(query: str) -> List[Dict[str, Any]]:
            def get_query_text_vector(query: str) -> Tuple[Optional[str], Optional[List[float]]]:
                if self._index_details.is_databricks_managed_embeddings():
                    if self.embedding:
                        raise ValueError(
                            f"The index '{self._index_details.name}' uses Databricks-managed embeddings. "
                            "Do not pass the `embedding` parameter when executing retriever calls."
                        )
                    return query, None

                if not self.embedding:
                    raise ValueError(
                        "The embedding model name is required for non-Databricks-managed "
                        "embeddings Vector Search indexes in order to generate embeddings for retrieval queries."
                    )

                text = query if self.query_type and self.query_type.upper() == "HYBRID" else None
                vector = self.embedding.get_text_embedding(text=query)
                if (
                    index_embedding_dimension := self._index_details.embedding_vector_column.get(
                        "embedding_dimension"
                    )
                ) and len(vector) != index_embedding_dimension:
                    raise ValueError(
                        f"Expected embedding dimension {index_embedding_dimension} but got {len(vector)}"
                    )
                return text, vector

            query_text, query_vector = get_query_text_vector(query)
            search_resp = self._index.similarity_search(
                columns=self.columns,
                query_text=query_text,
                query_vector=query_vector,
                filters=self.filters,
                num_results=self.num_results,
                query_type=self.query_type,
            )
            return parse_vector_search_response(
                search_resp, self._index_details, self.text_column, document_class=dict
            )

        # Create tool metadata
        metadata = ToolMetadata(
            name=self.tool_name or self.index_name,
            description=self.tool_description
            or self._get_default_tool_description(self._index_details),
            fn_schema=VectorSearchRetrieverToolInput,
            return_direct=self.return_direct,
        )

        # Initialize FunctionTool with the similarity search function and metadata
        FunctionTool.__init__(self, fn=similarity_search, metadata=metadata)
