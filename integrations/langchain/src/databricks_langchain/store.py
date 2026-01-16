from __future__ import annotations

import warnings
from typing import Any, Iterable

from databricks.sdk import WorkspaceClient

try:
    from databricks_ai_bridge.lakebase import AsyncLakebasePool, LakebasePool
    from langgraph.store.base import BaseStore, Op, Result
    from langgraph.store.base.batch import AsyncBatchedBaseStore
    from langgraph.store.postgres import AsyncPostgresStore, PostgresStore
    from langgraph.store.postgres.base import PostgresIndexConfig

    from databricks_langchain.embeddings import DatabricksEmbeddings

    _store_imports_available = True
except ImportError:
    BaseStore = object  # type: ignore
    AsyncBatchedBaseStore = object  # type: ignore
    _store_imports_available = False


class DatabricksStore(BaseStore):
    """Provides APIs for working with long-term memory on Databricks using Lakebase.
    Extends LangGraph BaseStore interface using Databricks Lakebase for connection pooling,
    with semantic search capabilities via DatabricksEmbeddings.

    Operations borrow a connection from the pool, create a short-lived PostgresStore,
    execute the operation, and return the connection to the pool.
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        embedding_endpoint: str | None = None,
        embedding_dims: int | None = None,
        embedding_fields: list[str] | None = None,
        embeddings: DatabricksEmbeddings | None = None,
        **pool_kwargs: Any,
    ) -> None:
        """Initialize DatabricksStore with embedding support.

        Args:
            instance_name: The name of the Lakebase instance to connect to.
            workspace_client: Optional Databricks WorkspaceClient for authentication.
            embedding_endpoint: Name of the Databricks Model Serving endpoint for embeddings
                (e.g., "databricks-gte-large-en"). If provided, enables semantic search.
            embedding_dims: Dimension of the embedding vectors (e.g., 1024 for gte-large-en,
                1536 for OpenAI-compatible models). Required if embedding_endpoint is set.
            embedding_fields: List of JSON paths to vectorize. Defaults to ["$"] which
                vectorizes the entire JSON value.
            embeddings: Optional pre-configured DatabricksEmbeddings instance. If provided,
                takes precedence over embedding_endpoint.
            **pool_kwargs: Additional keyword arguments passed to LakebasePool.
        """
        if not _store_imports_available:
            raise ImportError(
                "DatabricksStore requires databricks-langchain[memory]. "
                "Install with: pip install 'databricks-langchain[memory]'"
            )

        self._lakebase: LakebasePool = LakebasePool(
            instance_name=instance_name,
            workspace_client=workspace_client,
            **pool_kwargs,
        )

        # Initialize embeddings and index configuration for semantic search
        self.embeddings: DatabricksEmbeddings | None = None
        self.index_config: PostgresIndexConfig | None = None

        if embeddings is not None:
            # Use pre-configured embeddings instance
            if embedding_endpoint is not None:
                warnings.warn(
                    "Both 'embeddings' and 'embedding_endpoint' were specified. "
                    "Using the provided 'embeddings' instance.",
                    UserWarning,
                    stacklevel=2,
                )
            self.embeddings = embeddings
            if embedding_dims is None:
                raise ValueError("embedding_dims is required when providing an embeddings instance")
            self.index_config = {
                "dims": embedding_dims,
                "embed": self.embeddings,
                "fields": embedding_fields or ["$"],
            }
        elif embedding_endpoint is not None:
            # Create embeddings from endpoint configuration
            if embedding_dims is None:
                raise ValueError("embedding_dims is required when embedding_endpoint is specified")
            self.embeddings = DatabricksEmbeddings(endpoint=embedding_endpoint)
            self.index_config = {
                "dims": embedding_dims,
                "embed": self.embeddings,
                "fields": embedding_fields or ["$"],
            }

    def _with_store(self, fn, *args, **kwargs):
        """
        Borrow a connection, create a short-lived PostgresStore with index config,
        call fn(store), then return the connection to the pool.
        """
        with self._lakebase.connection() as conn:
            if self.index_config is not None:
                store = PostgresStore(conn=conn, index=self.index_config)
            else:
                store = PostgresStore(conn=conn)
            return fn(store, *args, **kwargs)

    def setup(self) -> None:
        """Instantiate the store, setting up necessary persistent storage."""
        return self._with_store(lambda s: s.setup())

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations synchronously.

        This is the core method required by BaseStore. All other operations
        (get, put, search, delete, list_namespaces) are inherited from BaseStore
        and internally call this batch() method.
        """
        return self._with_store(lambda s: s.batch(ops))

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously.

        This is the second abstract method required by BaseStore.
        Currently delegates to sync batch() - for true async support,
        would need async-compatible connection pooling.
        """
        return self.batch(ops)


class AsyncDatabricksStore(AsyncBatchedBaseStore):
    """Async version of DatabricksStore for working with long-term memory on Databricks.

    Extends LangGraph AsyncBatchedBaseStore interface using Databricks Lakebase
    for async connection pooling, with semantic search capabilities via DatabricksEmbeddings.

    Operations borrow a connection from the async pool, create a short-lived AsyncPostgresStore,
    execute the operation, and return the connection to the pool.
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        embedding_endpoint: str | None = None,
        embedding_dims: int | None = None,
        embedding_fields: list[str] | None = None,
        embeddings: DatabricksEmbeddings | None = None,
        **pool_kwargs: Any,
    ) -> None:
        """Initialize AsyncDatabricksStore with embedding support.

        Args:
            instance_name: The name of the Lakebase instance to connect to.
            workspace_client: Optional Databricks WorkspaceClient for authentication.
            embedding_endpoint: Name of the Databricks Model Serving endpoint for embeddings
                (e.g., "databricks-gte-large-en"). If provided, enables semantic search.
            embedding_dims: Dimension of the embedding vectors (e.g., 1024 for gte-large-en,
                1536 for OpenAI-compatible models). Required if embedding_endpoint is set.
            embedding_fields: List of JSON paths to vectorize. Defaults to ["$"] which
                vectorizes the entire JSON value.
            embeddings: Optional pre-configured DatabricksEmbeddings instance. If provided,
                takes precedence over embedding_endpoint.
            **pool_kwargs: Additional keyword arguments passed to AsyncLakebasePool.
        """
        if not _store_imports_available:
            raise ImportError(
                "AsyncDatabricksStore requires databricks-langchain[memory]. "
                "Install with: pip install 'databricks-langchain[memory]'"
            )

        super().__init__()

        self._lakebase: AsyncLakebasePool = AsyncLakebasePool(
            instance_name=instance_name,
            workspace_client=workspace_client,
            **pool_kwargs,
        )

        # Initialize embeddings and index configuration for semantic search
        self.embeddings: DatabricksEmbeddings | None = None
        self.index_config: PostgresIndexConfig | None = None

        if embeddings is not None:
            # Use pre-configured embeddings instance
            if embedding_endpoint is not None:
                warnings.warn(
                    "Both 'embeddings' and 'embedding_endpoint' were specified. "
                    "Using the provided 'embeddings' instance.",
                    UserWarning,
                    stacklevel=2,
                )
            self.embeddings = embeddings
            if embedding_dims is None:
                raise ValueError("embedding_dims is required when providing an embeddings instance")
            self.index_config = {
                "dims": embedding_dims,
                "embed": self.embeddings,
                "fields": embedding_fields or ["$"],
            }
        elif embedding_endpoint is not None:
            # Create embeddings from endpoint configuration
            if embedding_dims is None:
                raise ValueError("embedding_dims is required when embedding_endpoint is specified")
            self.embeddings = DatabricksEmbeddings(endpoint=embedding_endpoint)
            self.index_config = {
                "dims": embedding_dims,
                "embed": self.embeddings,
                "fields": embedding_fields or ["$"],
            }

    async def _with_store(self, fn, *args, **kwargs):
        """
        Borrow an async connection, create a short-lived AsyncPostgresStore with index config,
        call fn(store), then return the connection to the pool.
        """
        async with self._lakebase.connection() as conn:
            if self.index_config is not None:
                store = AsyncPostgresStore(conn=conn, index=self.index_config)
            else:
                store = AsyncPostgresStore(conn=conn)
            return await fn(store, *args, **kwargs)

    async def setup(self) -> None:
        """Instantiate the store, setting up necessary persistent storage."""
        return await self._with_store(lambda s: s.setup())

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute a batch of operations asynchronously.

        This is the core method required by AsyncBatchedBaseStore. All other async operations
        (aget, aput, asearch, adelete, alist_namespaces) are inherited from AsyncBatchedBaseStore
        and internally call this abatch() method.
        """
        return await self._with_store(lambda s: s.abatch(ops))

    async def __aenter__(self):
        """Enter async context manager and open the connection pool."""
        await self._lakebase.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager and close the connection pool."""
        await self._lakebase.close()
        return False
