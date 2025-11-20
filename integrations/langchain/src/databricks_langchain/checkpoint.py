from __future__ import annotations

from databricks.sdk import WorkspaceClient

try:
    from databricks_ai_bridge.lakebase import LakebasePool
except ImportError:
    raise RuntimeError(
        "psycopg is needed to enable checkpoint feature. Please install with databricks-langchain[memory]"
    ) from None

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    raise RuntimeError(
        "langgraph-checkpoint-postgres is needed to enable checkpoint feature. Please install with databricks-langchain[memory]"
    ) from None


class CheckpointSaver(PostgresSaver):
    """
    LangGraph PostgresSaver using a Lakebase connection pool.

    instance_name: Name of Lakebase Instance
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        **pool_kwargs: object,
    ) -> None:
        self._lakebase = LakebasePool(
            instance_name=instance_name,
            workspace_client=workspace_client,
            **dict(pool_kwargs),
        )
        super().__init__(self._lakebase.pool)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close the connection pool."""
        self._lakebase.close()
        return False
