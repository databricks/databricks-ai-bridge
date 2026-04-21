from __future__ import annotations

from typing import Any

from databricks.sdk import WorkspaceClient

try:
    from databricks_ai_bridge.lakebase import AsyncLakebasePool, LakebaseClient, LakebasePool
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    _checkpoint_imports_available = True
except ImportError:
    PostgresSaver = object  # type: ignore
    AsyncPostgresSaver = object  # type: ignore

    _checkpoint_imports_available = False


class CheckpointSaver(PostgresSaver):
    """
    LangGraph PostgresSaver using a Lakebase connection pool.

    Supports two modes: Lakebase Provisioned VS Autoscaling
    https://docs.databricks.com/aws/en/oltp/#feature-comparison
    """

    def __init__(
        self,
        *,
        instance_name: str | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str | None = None,
        **pool_kwargs: Any,
    ) -> None:
        # Lazy imports
        if not _checkpoint_imports_available:
            raise ImportError(
                "CheckpointSaver requires databricks-langchain[memory]. "
                "Please install with: pip install databricks-langchain[memory]"
            )

        self._lakebase: LakebasePool = LakebasePool(
            instance_name=instance_name,
            autoscaling_endpoint=autoscaling_endpoint,
            project=project,
            branch=branch,
            workspace_client=workspace_client,
            schema=schema,
            **dict(pool_kwargs),
        )
        super().__init__(self._lakebase.pool)

    def setup(self) -> None:
        """Set up the checkpoint database, creating the schema if specified."""
        LakebaseClient.create_schema(self._lakebase)
        super().setup()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close the connection pool."""
        self._lakebase.close()
        return False


class AsyncCheckpointSaver(AsyncPostgresSaver):
    """
    Async LangGraph PostgresSaver using a Lakebase connection pool.

    Supports two modes: Lakebase Provisioned VS Autoscaling
    https://docs.databricks.com/aws/en/oltp/#feature-comparison

    Checkpoint tables are created automatically when entering the context manager.
    """

    def __init__(
        self,
        *,
        instance_name: str | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str | None = None,
        **pool_kwargs: Any,
    ) -> None:
        # Lazy imports
        if not _checkpoint_imports_available:
            raise ImportError(
                "AsyncCheckpointSaver requires databricks-langchain[memory]. "
                "Please install with: pip install databricks-langchain[memory]"
            )

        self._lakebase: AsyncLakebasePool = AsyncLakebasePool(
            instance_name=instance_name,
            autoscaling_endpoint=autoscaling_endpoint,
            project=project,
            branch=branch,
            workspace_client=workspace_client,
            schema=schema,
            **dict(pool_kwargs),
        )
        super().__init__(self._lakebase.pool)

    async def setup(self) -> None:
        """Set up the checkpoint database asynchronously, creating the schema if specified."""
        await LakebaseClient.acreate_schema(self._lakebase)
        await super().setup()

    async def __aenter__(self):
        """Enter async context manager and open the connection pool."""
        await self._lakebase.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager and close the connection pool."""
        await self._lakebase.close()
        return False
