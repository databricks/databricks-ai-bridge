from __future__ import annotations

from typing import Any, Iterable, Optional

from databricks.sdk import WorkspaceClient

try:
    from databricks_ai_bridge.lakebase import LakebasePool
    from langgraph.store.base import BaseStore, Item, Op, Result
    from langgraph.store.postgres import PostgresStore

    _store_imports_available = True
except ImportError:
    LakebasePool = object
    PostgresStore = object
    BaseStore = object
    Item = object
    Op = object
    Result = object
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

    # BaseStore abstract methods - REQUIRED to implement
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

    # def close(self) -> None:
    #     """Close the underlying Lakebase pool."""
    #     if self._lakebase is not None:
    #         self._lakebase.close()
