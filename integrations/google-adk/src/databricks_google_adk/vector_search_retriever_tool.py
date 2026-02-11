import inspect
from typing import Any, Callable

from databricks_ai_bridge.utils.vector_search import (
    IndexDetails,
    RetrieverSchema,
    parse_vector_search_response,
    validate_and_get_return_columns,
    validate_and_get_text_column,
)
from databricks_ai_bridge.vector_search_retriever_tool import (
    FilterItem,
    VectorSearchRetrieverToolInput,
    VectorSearchRetrieverToolMixin,
    vector_search_retriever_tool_trace,
)
from google.adk.tools import FunctionTool
from pydantic import Field, PrivateAttr


class VectorSearchRetrieverTool(VectorSearchRetrieverToolMixin):
    """
    Databricks Vector Search retriever tool for Google ADK.

    This tool allows Google ADK agents to search and retrieve documents from
    Databricks Vector Search indexes.

    Example:
        ```python
        from databricks_google_adk import VectorSearchRetrieverTool
        from google.adk.agents import Agent

        # Create the tool
        vector_search_tool = VectorSearchRetrieverTool(
            index_name="catalog.schema.my_index",
            num_results=5,
        )

        # Use with an ADK agent
        agent = Agent(
            name="search_assistant",
            model="gemini-2.0-flash",
            instruction="You are a helpful assistant that searches documents.",
            tools=[vector_search_tool.as_tool()],
        )
        ```
    """

    text_column: str | None = Field(
        None,
        description="The name of the text column to use for the embeddings. "
        "Required for direct-access index or delta-sync index with "
        "self-managed embeddings.",
    )
    embedding_fn: Callable[[str], list[float]] | None = Field(
        None,
        description="Embedding function for self-managed embeddings. "
        "Should accept a string and return a list of floats.",
    )

    _index = PrivateAttr()
    _index_details = PrivateAttr()
    _retriever_schema = PrivateAttr()
    _adk_tool = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Initialize the vector search client and index after model creation."""
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

    def _get_query_text_vector(self, query: str) -> tuple[str | None, list[float] | None]:
        """Get the query text and vector based on the index configuration."""
        if self._index_details.is_databricks_managed_embeddings():
            if self.embedding_fn:
                raise ValueError(
                    f"The index '{self._index_details.name}' uses Databricks-managed embeddings. "
                    "Do not pass the `embedding_fn` parameter when executing retriever calls."
                )
            return query, None

        if not self.embedding_fn:
            raise ValueError(
                "The embedding_fn is required for non-Databricks-managed "
                "embeddings Vector Search indexes in order to generate embeddings for retrieval queries."
            )

        text = query if self.query_type and self.query_type.upper() == "HYBRID" else None
        vector = self.embedding_fn(query)
        if (
            index_embedding_dimension := self._index_details.embedding_vector_column.get(
                "embedding_dimension"
            )
        ) and len(vector) != index_embedding_dimension:
            raise ValueError(
                f"Expected embedding dimension {index_embedding_dimension} but got {len(vector)}"
            )
        return text, vector

    @vector_search_retriever_tool_trace
    def _search(
        self, query: str, filters: list[FilterItem] | None = None, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """
        Execute a similarity search against the vector index.

        Args:
            query: The search query string.
            filters: Optional list of filters to apply to the search.
            **kwargs: Additional keyword arguments passed to the search.

        Returns:
            A list of dictionaries containing the search results.
        """
        query_text, query_vector = self._get_query_text_vector(query)

        # Since LLM can generate either a dict or FilterItem, convert to dict always
        filters_dict = {dict(item)["key"]: dict(item)["value"] for item in (filters or [])}
        combined_filters = {**filters_dict, **(self.filters or {})}

        signature = inspect.signature(self._index.similarity_search)
        kwargs = {**kwargs, **(self.model_extra or {})}
        kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

        # Allow kwargs to override the default values upon invocation
        num_results = kwargs.pop("num_results", self.num_results)
        query_type = kwargs.pop("query_type", self.query_type)
        reranker = kwargs.pop("reranker", self.reranker)

        # Ensure that we don't have duplicate keys
        kwargs.update(
            {
                "query_text": query_text,
                "query_vector": query_vector,
                "columns": self.columns,
                "filters": combined_filters,
                "num_results": num_results,
                "query_type": query_type,
                "reranker": reranker,
            }
        )
        search_resp = self._index.similarity_search(**kwargs)
        return parse_vector_search_response(
            search_resp,
            retriever_schema=self._retriever_schema,
            include_score=self.include_score or False,
        )

    def as_tool(self) -> FunctionTool:
        """
        Convert this retriever to a Google ADK FunctionTool.

        Returns:
            A FunctionTool that can be used with Google ADK agents.
        """
        if self._adk_tool is not None:
            return self._adk_tool

        tool_name = self._get_tool_name()
        tool_description = self.tool_description or self._get_default_tool_description(
            self._index_details
        )

        if self.dynamic_filter:
            # Create a function with filter parameter for LLM-generated filters

            def search_with_filters(
                query: str, filters: list[dict[str, Any]] | None = None
            ) -> list[dict[str, Any]]:
                """Search the vector index with optional filters."""
                filter_items = None
                if filters:
                    filter_items = [FilterItem(**f) for f in filters]
                return self._search(query, filter_items)

            search_with_filters.__name__ = tool_name
            search_with_filters.__doc__ = tool_description
            self._adk_tool = FunctionTool(search_with_filters)
        else:
            # Create a simple function without filter parameter

            def search(query: str) -> list[dict[str, Any]]:
                """Search the vector index."""
                return self._search(query)

            search.__name__ = tool_name
            search.__doc__ = tool_description
            self._adk_tool = FunctionTool(search)

        return self._adk_tool
