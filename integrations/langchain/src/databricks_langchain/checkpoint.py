from __future__ import annotations

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.lakebase import LakebasePool
from langgraph.checkpoint.postgres import PostgresSaver


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
