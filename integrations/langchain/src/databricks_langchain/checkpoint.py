from __future__ import annotations

from typing import TYPE_CHECKING

from databricks.sdk import WorkspaceClient

if TYPE_CHECKING:
    from databricks_ai_bridge.lakebase import LakebasePool
    from langgraph.checkpoint.postgres import PostgresSaver

try:
    from langgraph.checkpoint.postgres import PostgresSaver as _PostgresSaverBase
except ImportError as e:
    raise ImportError(
        "CheckpointSaver requires databricks-langchain[memory]. "
        "Please install with: pip install databricks-langchain[memory]"
    ) from e


class CheckpointSaver(_PostgresSaverBase):
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
        # Lazy import LakebasePool
        try:
            from databricks_ai_bridge.lakebase import LakebasePool
        except ImportError as e:
            raise ImportError(
                "LakebasePool requires databricks-ai-bridge[memory]. "
                "Please install with: pip install databricks-langchain[memory]"
            ) from e

        self._lakebase: LakebasePool = LakebasePool(
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
