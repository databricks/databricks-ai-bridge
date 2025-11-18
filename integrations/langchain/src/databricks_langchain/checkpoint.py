from __future__ import annotations

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.lakebase import LakebasePool
from langgraph.checkpoint.postgres import PostgresSaver


class CheckpointSaver(PostgresSaver):
    """
    LangGraph PostgresSaver wired to a Lakebase connection pool.
    """

    def __init__(
        self,
        *,
        database_instance: str,
        workspace_client: WorkspaceClient | None = None,
        **pool_kwargs: object,
    ) -> None:
        self._lakebase = LakebasePool(
            instance_name=database_instance,
            workspace_client=workspace_client,
            **dict(pool_kwargs),
        )
        super().__init__(self._conn)
