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

        # Store initialization parameters for lazy initialization, otherwise
        # if we directly iniitalize pool during deployment it will fail
        self._instance_name = instance_name
        self._workspace_client = workspace_client
        self._pool_kwargs = pool_kwargs
        self._lakebase: Optional[LakebasePool] = None
        self._pool = None
        self._setup_called = False

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
        """Set up the store database tables."""
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
