import inspect
from typing import Any, Dict, List, Optional, Tuple

from databricks_ai_bridge.utils.vector_search import (
    IndexDetails,
    RetrieverSchema,
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
from pydantic import Extra, Field, PrivateAttr


class VectorSearchRetrieverTool(FunctionTool, VectorSearchRetrieverToolMixin):
    """Vector search retriever tool implementation."""

    class Config:
        extra = Extra.allow  # allow FunctionTool to set unknown attributes

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
        from databricks.vector_search.utils import CredentialStrategy

        credential_strategy = None
        if (
            self.workspace_client is not None
            and self.workspace_client.config.auth_type == "model_serving_user_credentials"
        ):
            credential_strategy = CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS
        self._index = VectorSearchClient(
            disable_notice=True, credential_strategy=credential_strategy
        ).get_index(index_name=self.index_name)
        self._index_details = IndexDetails(self._index)

        # Validate columns
        self.text_column = validate_and_get_text_column(self.text_column, self._index_details)
        self.columns = validate_and_get_return_columns(
            self.columns or [],
            self.text_column,
            self._index_details,
            self.doc_uri,
            self.primary_key,
        )
        self._retriever_schema = RetrieverSchema(
            text_column=self.text_column,
            doc_uri=self.doc_uri,
            primary_key=self.primary_key,
            other_columns=self.columns,
        )

        # Define the similarity search function
        def similarity_search(
            query: str, filters: Optional[Dict[str, Any]] = None, **kwargs: Any
        ) -> List[Dict[str, Any]]:
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
            combined_filters = {**(filters or {}), **(self.filters or {})}

            signature = inspect.signature(self._index.similarity_search)
            kwargs = {**kwargs, **(self.model_extra or {})}
            kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

            # Ensure that we don't have duplicate keys
            kwargs.update(
                {
                    "query_text": query_text,
                    "query_vector": query_vector,
                    "columns": self.columns,
                    "filters": combined_filters,
                    "num_results": self.num_results,
                    "query_type": self.query_type,
                }
            )
            search_resp = self._index.similarity_search(**kwargs)
            return parse_vector_search_response(
                search_resp, retriever_schema=self._retriever_schema, document_class=dict
            )

        # Create tool metadata
        metadata = ToolMetadata(
            name=self._get_tool_name(),
            description=self.tool_description
            or self._get_default_tool_description(self._index_details),
            fn_schema=VectorSearchRetrieverToolInput,
            return_direct=self.return_direct,
        )

        # Initialize FunctionTool with the similarity search function and metadata
        FunctionTool.__init__(self, fn=similarity_search, metadata=metadata)
