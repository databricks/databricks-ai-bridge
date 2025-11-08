from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.vector_search.reranker import DatabricksReranker, Reranker
from databricks_ai_bridge.utils.vector_search import (
    IndexDetails,
    RetrieverSchema,
    parse_vector_search_response,
    validate_and_get_return_columns,
    validate_and_get_text_column,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore

from databricks_langchain.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)


_DIRECT_ACCESS_ONLY_MSG = "`%s` is only supported for direct-access index."
_NON_MANAGED_EMB_ONLY_MSG = "`%s` is not supported for index with Databricks-managed embeddings."


class DatabricksVectorSearch(VectorStore):
    """Databricks vector store integration.

    Args:

        index_name: The name of the index to use. Format: "catalog.schema.index".

        endpoint: The name of the Databricks Vector Search ``endpoint``.
            If not specified, the endpoint name is automatically inferred based on the index name.

            .. note::

                If you are using `databricks-vectorsearch` version < 0.35, the `endpoint` parameter
                is required when initializing the vector store.

                .. code-block:: python

                    vector_store = DatabricksVectorSearch(
                        endpoint="<your-endpoint-name>",
                        index_name="<your-index-name>",
                        ...
                    )

        embedding: The embedding model.
                  Required for direct-access index or delta-sync index
                  with self-managed embeddings.
        text_column: The name of the text column to use for the embeddings.
                    Required for direct-access index or delta-sync index
                    with self-managed embeddings.
                    Make sure the text column specified is in the index.
        columns: The list of column names to get when doing the search.
                Defaults to ``[primary_key, text_column]``.
        client_args: Additional arguments to pass to the VectorSearchClient.
                    Allows you to pass in values like ``service_principal_client_id``
                    and ``service_principal_client_secret`` to allow for
                    service principal authentication instead of personal access token authentication.
        reranker: Optional reranker to apply on the top results. Pass an instance of
                    ``databricks.vector_search.reranker.DatabricksReranker`` with
                    ``columns_to_rerank=[...]``. The reranker reorders the initial results using
                    the specified text columns.

    **Instantiate**:

        `DatabricksVectorSearch` supports two types of indexes:

        * **Delta Sync Index** automatically syncs with a source Delta Table, automatically and incrementally updating the index as the underlying data in the Delta Table changes.

        * **Direct Vector Access Index** supports direct read and write of vectors and metadata. The user is responsible for updating this table using the REST API or the Python SDK.

        Also for delta-sync index, you can choose to use Databricks-managed embeddings or self-managed embeddings (via LangChain embeddings classes).

        If you are using a delta-sync index with Databricks-managed embeddings:

        .. code-block:: python

            from databricks_langchain.vectorstores import DatabricksVectorSearch

            vector_store = DatabricksVectorSearch(index_name="<your-index-name>")

        If you are using a direct-access index or a delta-sync index with self-managed embeddings,
        you also need to provide the embedding model and text column in your source table to
        use for the embeddings:

        .. code-block:: python

            from langchain_openai import OpenAIEmbeddings

            vector_store = DatabricksVectorSearch(
                index_name="<your-index-name>",
                embedding=OpenAIEmbeddings(),
                text_column="document_content",
            )

        If you want Databricks to rerank the results, you can provide a reranker when initializing the vector store:

        .. code-block:: python

            from databricks.vector_search.reranker import DatabricksReranker

            vector_store = DatabricksVectorSearch(
                index_name="<your-index-name>",
                reranker=DatabricksReranker(columns_to_rerank=["column1", "column2"]),
            )

    **Add Documents**:

        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")
            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    **Delete Documents**:

        .. code-block:: python

            vector_store.delete(ids=["3"])

        .. note::

            The `delete` method is only supported for direct-access index.

    **Search**:

        .. code-block:: python

            results = vector_store.similarity_search(query="thud", k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            *thud[{"id": "2"}]

        .. note:

            By default, similarity search only returns the primary key and text column.
            If you want to retrieve the custom metadata associated with the document,
            pass the additional columns in the `columns` parameter when initializing the vector store.

            .. code-block:: python

                vector_store = DatabricksVectorSearch(
                    endpoint="<your-endpoint-name>",
                    index_name="<your-index-name>",
                    columns=["baz", "bar"],
                )

                vector_store.similarity_search(query="thud", k=1)
                # Output: * thud [{'bar': 'baz', 'baz': None, 'id': '2'}]

    **Search with filter**:

        .. code-block:: python

            results = vector_store.similarity_search(query="thud", k=1, filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            *thud[{"id": "2"}]

    **Search with score**:

        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux", k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.748804] foo [{'id': '1'}]

    **Async**:

        .. code-block:: python

            # add documents
            await vector_store.aadd_documents(documents=documents, ids=ids)
            # delete documents
            await vector_store.adelete(ids=["3"])
            # search
            results = vector_store.asimilarity_search(query="thud", k=1)
            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux", k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.748807] foo [{'id': '1'}]

    **Use as Retriever**:

        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            [Document(metadata={"id": "2"}, page_content="thud")]
    """  # noqa: E501

    def __init__(
        self,
        index_name: str,
        endpoint: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        text_column: Optional[str] = None,
        doc_uri: Optional[str] = None,
        primary_key: Optional[str] = None,
        columns: Optional[List[str]] = None,
        workspace_client: Optional[WorkspaceClient] = None,
        client_args: Optional[Dict[str, Any]] = None,
        include_score: bool = False,
        reranker: Optional[DatabricksReranker] = None,
    ):
        if not isinstance(index_name, str):
            raise ValueError(
                f"The `index_name` parameter must be a string, but got {type(index_name).__name__}."
            )

        if index_name.count(".") != 2:
            raise ValueError(
                f"The `index_name` parameter must be in the format 'catalog.schema.name', but got {index_name!r}."
            )

        try:
            from databricks.vector_search.client import (  # type: ignore[import]
                VectorSearchClient,
            )
            from databricks.vector_search.utils import CredentialStrategy
        except ImportError as e:
            raise ImportError(
                "Could not import databricks-vectorsearch python package. "
                "Please install it with `pip install databricks-vectorsearch`."
            ) from e

        try:
            client_args = client_args or {}
            client_args.setdefault("disable_notice", True)
            if workspace_client is not None:
                config = workspace_client.config
                if config.auth_type == "model_serving_user_credentials":
                    client_args.setdefault(
                        "credential_strategy", CredentialStrategy.MODEL_SERVING_USER_CREDENTIALS
                    )
                elif config.auth_type == "pat":
                    client_args.setdefault("personal_access_token", config.token)
                elif config.auth_type == "oauth-m2m":
                    client_args.setdefault("workspace_url", config.host)
                    client_args.setdefault("service_principal_client_id", config.client_id)
                    client_args.setdefault("service_principal_client_secret", config.client_secret)
            self.index = VectorSearchClient(**client_args).get_index(
                endpoint_name=endpoint, index_name=index_name
            )
        except Exception as e:
            if endpoint is None and "Wrong vector search endpoint" in str(e):
                raise ValueError(
                    "The `endpoint` parameter is required for instantiating "
                    "DatabricksVectorSearch with the `databricks-vectorsearch` "
                    "version earlier than 0.35. Please provide the endpoint "
                    "name or upgrade to version 0.35 or later."
                ) from e
            else:
                raise

        self._index_details = IndexDetails(self.index)

        _validate_embedding(embedding, self._index_details)
        self._embeddings = embedding
        self._text_column = validate_and_get_text_column(text_column, self._index_details)
        self._columns = validate_and_get_return_columns(
            columns or [], self._text_column, self._index_details, doc_uri, primary_key
        )
        self._primary_key = self._index_details.primary_key
        self._retriever_schema = RetrieverSchema(
            text_column=self._text_column,
            doc_uri=doc_uri,
            primary_key=primary_key,
            other_columns=self._columns,
        )
        self._include_score = include_score
        self._reranker = reranker

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embeddings

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> VST:
        raise NotImplementedError(
            "`from_texts` is not supported. Use `add_texts` to add to existing direct-access index."
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the index.

        .. note::

            This method is only supported for a direct-access index.

        Args:
            texts: List of texts to add.
            metadatas: List of metadata for each text. Defaults to None.
            ids: List of ids for each text. Defaults to None.
                If not provided, a random uuid will be generated for each text.

        Returns:
            List of ids from adding the texts into the index.
        """
        if self._index_details.is_delta_sync_index():
            raise NotImplementedError(_DIRECT_ACCESS_ONLY_MSG % "add_texts")

        # Wrap to list if input texts is a single string
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        vectors = self._embeddings.embed_documents(texts)  # type: ignore[union-attr]
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]

        updates = [
            {
                self._primary_key: id_,
                self._text_column: text,
                self._index_details.embedding_vector_column["name"]: vector,
                **metadata,
            }
            for text, vector, id_, metadata in zip(texts, vectors, ids, metadatas)
        ]

        upsert_resp = self.index.upsert(updates)
        if upsert_resp.get("status") in ("PARTIAL_SUCCESS", "FAILURE"):
            failed_ids = upsert_resp.get("result", dict()).get("failed_primary_keys", [])
            if upsert_resp.get("status") == "FAILURE":
                logger.error("Failed to add texts to the index.")
            else:
                logger.warning("Some texts failed to be added to the index.")
            return [id_ for id_ in ids if id_ not in failed_ids]

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.add_texts, **kwargs), texts, metadatas
        )

    def delete(self, ids: Optional[List[Any]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents from the index.

        .. note::

            This method is only supported for a direct-access index.

        Args:
            ids: List of ids of documents to delete.

        Returns:
            True if successful.
        """
        if self._index_details.is_delta_sync_index():
            raise NotImplementedError(_DIRECT_ACCESS_ONLY_MSG % "delete")

        if ids is None:
            raise ValueError("ids must be provided.")
        self.index.delete(ids)
        return True

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_type: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".
            reranker: Allows reranking the results. Defaults to None.
            kwargs: Additional keyword arguments to pass to `databricks.vector_search.client.VectorSearchIndex.similarity_search`. `See
                    documentation <https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex.similarity_search>`_
                    to see the full set of supported keyword arguments

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_with_score = self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            query_type=query_type,
            reranker=reranker,
            **kwargs,
        )
        return [doc for doc, _ in docs_with_score]

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search, query, k=k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_type: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query, along with scores.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".
            reranker: Allows reranking the results. Defaults to None.
            kwargs: Additional keyword arguments to pass to `databricks.vector_search.client.VectorSearchIndex.similarity_search`. `See
                    documentation <https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex.similarity_search>`_
                    to see the full set of supported keyword arguments

        Returns:
            List of Documents most similar to the embedding and score for each.
        """
        if self._index_details.is_databricks_managed_embeddings():
            query_text = query
            query_vector = None
        else:
            # The value for `query_text` needs to be specified only for hybrid search.
            if query_type is not None and query_type.upper() == "HYBRID":
                query_text = query
            else:
                query_text = None
            query_vector = self._embeddings.embed_query(query)  # type: ignore[union-attr]

        signature = inspect.signature(self.index.similarity_search)
        kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
        kwargs.update(
            {
                "columns": self._columns,
                "query_text": query_text,
                "query_vector": query_vector,
                "filters": filter,
                "num_results": k,
                "query_type": query_type,
                "reranker": reranker or self._reranker,
            }
        )
        search_resp = self.index.similarity_search(**kwargs)
        return parse_vector_search_response(
            search_resp,
            retriever_schema=self._retriever_schema,
            document_class=Document,
            include_score=self._include_score,
        )

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        Databricks Vector search uses a normalized score 1/(1+d) where d
        is the L2 distance. Hence, we simply return the identity function.
        """
        return lambda score: score

    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search_with_score, *args, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Any] = None,
        *,
        query_type: Optional[str] = None,
        query: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".
            reranker: Allows reranking the results. Defaults to None.
            kwargs: Additional keyword arguments to pass to `databricks.vector_search.client.VectorSearchIndex.similarity_search`. `See
                    documentation <https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex.similarity_search>`_
                    to see the full set of supported keyword arguments

        Returns:
            List of Documents most similar to the embedding.
        """
        if self._index_details.is_databricks_managed_embeddings():
            raise NotImplementedError(_NON_MANAGED_EMB_ONLY_MSG % "similarity_search_by_vector")

        docs_with_score = self.similarity_search_by_vector_with_score(
            embedding=embedding,
            k=k,
            filter=filter,
            query_type=query_type,
            query=query,
            reranker=reranker,
            **kwargs,
        )
        return [doc for doc, _ in docs_with_score]

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(self.similarity_search_by_vector, embedding, k=k, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Any] = None,
        *,
        query_type: Optional[str] = None,
        query: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to embedding vector, along with scores.

        .. note::

            This method is not supported for index with Databricks-managed embeddings.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".
            reranker: Allows reranking the results. Defaults to None.
            kwargs: Additional keyword arguments to pass to `databricks.vector_search.client.VectorSearchIndex.similarity_search`. `See
                    documentation <https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex.similarity_search>`_
                    to see the full set of supported keyword arguments

        Returns:
            List of Documents most similar to the embedding and score for each.
        """
        if self._index_details.is_databricks_managed_embeddings():
            raise NotImplementedError(
                _NON_MANAGED_EMB_ONLY_MSG % "similarity_search_by_vector_with_score"
            )

        if query_type is not None and query_type.upper() == "HYBRID":
            if query is None:
                raise ValueError("A value for `query` must be specified for hybrid search.")
            query_text = query
        else:
            if query is not None:
                raise ValueError(
                    ('Cannot specify both `embedding` and `query` unless `query_type="HYBRID"')
                )
            query_text = None

        search_resp = self.index.similarity_search(
            columns=self._columns,
            query_vector=embedding,
            query_text=query_text,
            filters=filter,
            num_results=k,
            query_type=query_type,
            reranker=reranker or self._reranker,
            **kwargs,
        )
        return parse_vector_search_response(
            search_resp,
            retriever_schema=self._retriever_schema,
            document_class=Document,
            include_score=self._include_score,
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        *,
        query_type: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        .. note::

            This method is not supported for index with Databricks-managed embeddings.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".
            reranker: Allows reranking the results. Defaults to None.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._index_details.is_databricks_managed_embeddings():
            raise NotImplementedError(_NON_MANAGED_EMB_ONLY_MSG % "max_marginal_relevance_search")
        else:
            # The value for `query_text` needs to be specified only for hybrid search.
            if query_type is not None and query_type.upper() == "HYBRID":
                query_text = query
            else:
                query_text = None

        query_vector = self._embeddings.embed_query(query)  # type: ignore[union-attr]
        docs = self.max_marginal_relevance_search_by_vector(
            query,
            query_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            query_type=query_type,
            reranker=reranker,
        )
        return docs

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        # This is a temporary workaround to make the similarity search
        # asynchronous. The proper solution is to make the similarity search
        # asynchronous in the vector store implementations.
        func = partial(
            self.max_marginal_relevance_search,
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def max_marginal_relevance_search_by_vector(
        self,
        query: str,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Any] = None,
        *,
        query_type: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        .. note::

            This method is not supported for index with Databricks-managed embeddings.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filters to apply to the query. Defaults to None.
            query_type: The type of this query. Supported values are "ANN" and "HYBRID".
            reranker: Allows reranking the results. Defaults to None.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._index_details.is_databricks_managed_embeddings():
            raise NotImplementedError(
                _NON_MANAGED_EMB_ONLY_MSG % "max_marginal_relevance_search_by_vector"
            )

        embedding_column = self._index_details.embedding_vector_column["name"]
        search_resp = self.index.similarity_search(
            columns=list(set(self._columns + [embedding_column])),
            query_text=query,
            query_vector=embedding,
            filters=filter,
            num_results=fetch_k,
            query_type=query_type,
            reranker=reranker or self._reranker,
            **kwargs,
        )

        embeddings_result_index = (
            search_resp.get("manifest").get("columns").index({"name": embedding_column})
        )
        embeddings = [
            doc[embeddings_result_index] for doc in search_resp.get("result").get("data_array")
        ]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        ignore_cols: List = [embedding_column] if embedding_column not in self._columns else []
        candidates = parse_vector_search_response(
            search_resp,
            retriever_schema=self._retriever_schema,
            ignore_cols=ignore_cols,
            document_class=Document,
            include_score=self._include_score,
        )
        selected_results = [r[0] for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError


def _validate_embedding(embedding: Optional[Embeddings], index_details: IndexDetails) -> None:
    if index_details.is_databricks_managed_embeddings():
        if embedding is not None:
            raise ValueError(
                f"The index '{index_details.name}' uses Databricks-managed embeddings. "
                "Do not pass the `embedding` parameter when initializing vector store."
            )
    else:
        if not embedding:
            raise ValueError(
                "The `embedding` parameter is required for a direct-access index "
                "or delta-sync index with self-managed embedding."
            )
        _validate_embedding_dimension(embedding, index_details)


def _validate_embedding_dimension(embeddings: Embeddings, index_details: IndexDetails) -> None:
    """validate if the embedding dimension matches with the index's configuration."""
    if index_embedding_dimension := index_details.embedding_vector_column.get(
        "embedding_dimension"
    ):
        # Infer the embedding dimension from the embedding function."""
        actual_dimension = len(embeddings.embed_query("test"))
        if actual_dimension != index_embedding_dimension:
            raise ValueError(
                f"The specified embedding model's dimension '{actual_dimension}' does "
                f"not match with the index configuration '{index_embedding_dimension}'."
            )
