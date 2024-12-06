from databricks_langchain import DatabricksVectorSearch
from typing import (
    List,
    Optional
)
from langchain_core.embeddings import Embeddings
from langchain.tools.retriever import create_retriever_tool

class VectorSearchRetrieverTool():
    """
    A utility class to create a vector search-based retrieval tool for querying indexed embeddings.
    This class integrates with a Databricks Vector Search and provides a convenient interface
    for building a retriever tool for agents.

    Parameters:
        tool_name (str):
            The name of the retrieval tool to be created. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        tool_description (str):
            A description of the tool's functionality. This will be passed to the language model,
            so should be descriptive.

        index_name (str):
            The name of the index to use. Format: “catalog.schema.index”. endpoint:
        endpoint (Optional[str]):
            The name of the Databricks Vector Search endpoint. If not specified, the endpoint name is
            automatically inferred based on the index name.
        embedding (Optional[Embeddings]):
            The embedding model. Required for direct-access index or delta-sync index with self-managed embeddings.
        text_column (Optional[str]):
            The name of the text column to use for the embeddings. Required for direct-access index or
            delta-sync index with self-managed embeddings. Make sure the text column specified is in the index.
        columns (Optional[List[str]]):
            The list of column names to get when doing the search. Defaults to [primary_key, text_column].

        search_type (Optional[str]): Defines the type of search that the Retriever should perform.
            Defaults to “similarity” (default).
        search_kwargs (Optional[Dict]): Keyword arguments to pass to the search function.
    """
    def __new__(
        cls,
        tool_name: str,
        tool_description: str,
        index_name: str,
        endpoint: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        text_column: Optional[str] = None,
        columns: Optional[List[str]] = None,
        search_type: Optional[str] = None,
        search_kwargs: Optional[dict] = None,
    ):
        vector_store_kwargs = {"index_name": index_name}
        if endpoint is not None:
            vector_store_kwargs["endpoint"] = endpoint
        if embedding is not None:
            vector_store_kwargs["embedding"] = embedding
        if text_column is not None:
            vector_store_kwargs["text_column"] = text_column
        if columns is not None:
            vector_store_kwargs["columns"] = columns
        vector_store = DatabricksVectorSearch(**vector_store_kwargs)

        retriever_kwargs = {}
        if search_type is not None:
            retriever_kwargs["search_type"] = search_type
        if search_kwargs is not None:
            retriever_kwargs["search_kwargs"] = search_kwargs

        retriever = vector_store.as_retriever(**retriever_kwargs)
        return create_retriever_tool(
            retriever,
            tool_name,
            tool_description,
        )