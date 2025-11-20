from __future__ import annotations

from typing import Any, Optional

from databricks.sdk import WorkspaceClient

try:
    from databricks_ai_bridge.lakebase import LakebasePool
    from langgraph.store.postgres import PostgresStore

    _store_imports_available = True
except ImportError:
    LakebasePool = object
    PostgresStore = object
    _store_imports_available = False


class DatabricksStore:
    """
    Wrapper around LangGraph's PostgresStore that uses a Lakebase
    connection pool and borrows a connection per call.
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

        self._lakebase: LakebasePool = LakebasePool(
            instance_name=instance_name,
            workspace_client=workspace_client,
            **pool_kwargs,
        )
        self._pool = self._lakebase.pool
        self.setup()

    def _with_store(self, fn, *args, **kwargs):
        """
        Borrow a connection, create a short-lived PostgresStore, call fn(store),
        then return the connection to the pool.
        """
        with self._pool.connection() as conn:
            store = PostgresStore(conn=conn)
            return fn(store, *args, **kwargs)

    def setup(self) -> None:
        """Set up the store database tables (first time setup)."""
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
    ) -> list[Any]:
        """Search for items in the store."""
        return self._with_store(lambda s: s.search(namespace, query=query, limit=limit))

    # def close(self) -> None:
    #     """Close the underlying Lakebase pool."""
    #     self._lakebase.close()

    # def __enter__(self) -> "DatabricksStore":
    #     """Enter context manager."""
    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb) -> None:
    #     """Exit context manager and close the connection pool."""
    #     self.close()
