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
        self._pool = self._lakebase.pool
        self._conn = self._pool.getconn()
        super().__init__(self._conn)
        self._close_pool = True

    def close(self) -> None:
        """Close the saver, release the connection, and optionally close the pool."""
        close_method = getattr(super(), "close", None)
        if callable(close_method):
            close_method()

        if getattr(self, "_conn", None) is not None:
            try:
                self._pool.putconn(self._conn)
            finally:
                self._conn = None

        if getattr(self, "_close_pool", False):
            self._lakebase.close()

    def __enter__(self):
        self._prev_close_pool = self._close_pool
        self._close_pool = False
        enter_method = getattr(super(), "__enter__", None)
        if callable(enter_method):
            return enter_method()
        return self

    def __exit__(self, exc_type, exc, tb):
        prev_close_pool = getattr(self, "_prev_close_pool", True)
        try:
            self._close_pool = False
            exit_method = getattr(super(), "__exit__", None)
            if callable(exit_method):
                return exit_method(exc_type, exc, tb)
            self.close()
            return False
        finally:
            self._close_pool = prev_close_pool
