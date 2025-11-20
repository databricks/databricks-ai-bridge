from __future__ import annotations

from databricks.sdk import WorkspaceClient

try:
    from databricks_ai_bridge.lakebase import LakebasePool
    from langgraph.store.postgres import PostgresStore

    _store_imports_available = True
except ImportError:
    PostgresStore = object
    _store_imports_available = False


class DatabricksStore(PostgresStore):
    """
    LangGraph PostgresStore using a Lakebase connection pool.
    instance_name: Name of Lakebase Instance
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        **pool_kwargs: object,
    ) -> None:
        # Lazy imports
        if not _store_imports_available:
            raise ImportError(
                "DatabricksStore requires databricks-langchain[memory]. "
                "Please install with: pip install databricks-langchain[memory]"
            )

        self._lakebase: LakebasePool = LakebasePool(
            instance_name=instance_name,
            workspace_client=workspace_client,
            **dict(pool_kwargs),
        )
        self._conn = self._lakebase.pool.getconn()
        super().__init__(conn=self._conn)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close the connection pool."""
        # Return connection to pool before closing
        self._lakebase.pool.putconn(self._conn)
        self._lakebase.close()
        # Return False to propagate errors to caller
        return False
