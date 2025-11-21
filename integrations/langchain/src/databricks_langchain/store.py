from __future__ import annotations

from typing import Any, Optional

from databricks.sdk import WorkspaceClient

try:
    from databricks_ai_bridge.lakebase import LakebasePool
    from langgraph.store.base import BaseStore, Item
    from langgraph.store.postgres import PostgresStore

    _store_imports_available = True
except ImportError:
    LakebasePool = object
    PostgresStore = object
    BaseStore = object
    Item = object
    _store_imports_available = False


class DatabricksStore(BaseStore):
    """Provides APIs for working with long-term memory on Databricks using Lakebase.
    Extends LangGraph BaseStore interface using Databricks Lakebase for connection pooling.

    Operations borrow a connection from the pool, create a short-lived PostgresStore,
    execute the operation, and return the connection to the pool.
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: Optional[WorkspaceClient] = None,
        **pool_kwargs: Any,
    ) -> None:
        if not _store_imports_available:
            raise ImportError(
                "DatabricksStore requires databricks-langchain[memory]. "
                "Install with: pip install 'databricks-langchain[memory]'"
            )

        # Store initialization parameters for lazy initialization, otherwise
        # if we directly initialize pool during deployment it will fail
        self._instance_name = instance_name
        self._workspace_client = workspace_client
        self._pool_kwargs = pool_kwargs
        self._lakebase: Optional[LakebasePool] = None
        self._pool = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of LakebasePool on first use after deployment is ready."""
        if self._lakebase is None:
            self._lakebase = LakebasePool(
                instance_name=self._instance_name,
                workspace_client=self._workspace_client,
                **self._pool_kwargs,
            )
            self._pool = self._lakebase.pool

    def _with_store(self, fn, *args, **kwargs):
        """
        Borrow a connection, create a short-lived PostgresStore, call fn(store),
        then return the connection to the pool.
        """
        self._ensure_initialized()
        with self._pool.connection() as conn:
            store = PostgresStore(conn=conn)
            return fn(store, *args, **kwargs)

    def setup(self) -> None:
        """Instantiate the store, setting up necessary persistent storage."""
        return self._with_store(lambda s: s.setup())

    def put(self, namespace: tuple[str, ...], key: str, value: Any) -> None:
        """Store a value in the store."""
        return self._with_store(lambda s: s.put(namespace, key, value))

    def search(
        self,
        namespace: tuple[str, ...],
        *,
        query: Optional[str] = None,
        limit: int = 20,
    ) -> list[Item]:
        """Search for items in the store."""
        return self._with_store(lambda s: s.search(namespace, query=query, limit=limit))
