from __future__ import annotations

from typing import Any, Optional

try:  # pragma: no cover - optional dependency guard
    from databricks.sdk import WorkspaceClient
    from databricks_ai_bridge.lakebase import (
        LakebasePool,
        PooledPostgresSaver,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    LakebasePool = None  # type: ignore[assignment]
    PooledPostgresSaver = None  # type: ignore[assignment]
    WorkspaceClient = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


if PooledPostgresSaver is None:
    class CheckpointSaver:  # type: ignore[override]
        """Runtime guard that informs the user about missing optional dependencies."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "CheckpointSaver requires databricks-ai-bridge[lakebase] and psycopg to be installed."
            ) from _IMPORT_ERROR
else:

    class CheckpointSaver(PooledPostgresSaver):

        def __init__(
            self,
            *,
            database_instance: str,
            workspace_client: WorkspaceClient | None = None,
            host: str | None = None,
            database: str | None = None,
            username: Optional[str] = None,
            port: Optional[int] = None,
            sslmode: Optional[str] = None,
            token_cache_minutes: Optional[int] = None,
            connection_kwargs: Optional[dict[str, object]] = None,
            probe: bool = True,
            **pool_kwargs: Any,
        ) -> None:
            self._lakebase = LakebasePool(
                instance_name=database_instance,
                workspace_client=workspace_client,
                host=host,
                database=database,
                username=username,
                port=port,
                sslmode=sslmode,
                token_cache_minutes=token_cache_minutes,
                connection_kwargs=connection_kwargs,
                probe=probe,
                **pool_kwargs,
            )
            self._close_pool = True
            super().__init__(self._lakebase.pool)

        def close(self) -> None:  # type: ignore[override]
            """Return the connection to the pool and close the underlying pool."""

            super().close()
            if self._close_pool:
                self._lakebase.close()

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            previous = self._close_pool
            self._close_pool = False
            try:
                super().__exit__(exc_type, exc, tb)
            finally:
                self._close_pool = previous
