from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk import WorkspaceClient
else:  # pragma: no cover - runtime fallback when dependency absent
    WorkspaceClient = Any  # type: ignore

_CHECKPOINT_SAVER_IMPL: type | None = None


def _load_lakebase_dependencies() -> tuple[type, type]:
    """Import Lakebase dependencies lazily."""

    try:
        from databricks_ai_bridge.lakebase import LakebasePool, PooledPostgresSaver
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced to caller
        raise ModuleNotFoundError(
            "CheckpointSaver requires databricks-ai-bridge[lakebase] (psycopg) to be installed."
        ) from exc

    return LakebasePool, PooledPostgresSaver


def _checkpoint_saver_impl() -> type:
    global _CHECKPOINT_SAVER_IMPL
    if _CHECKPOINT_SAVER_IMPL is not None:
        return _CHECKPOINT_SAVER_IMPL

    LakebasePool, PooledPostgresSaver = _load_lakebase_dependencies()

    class _CheckpointSaver(PooledPostgresSaver):  # type: ignore[misc]
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
            token_cache_seconds: Optional[int] = None,
            connection_kwargs: Optional[dict[str, object]] = None,
            **pool_kwargs: object,
        ) -> None:
            self._lakebase = LakebasePool(
                instance_name=database_instance,
                host=host,
                workspace_client=workspace_client,
                database=database,
                username=username,
                port=port,
                sslmode=sslmode,
                token_cache_seconds=token_cache_seconds,
                connection_kwargs=connection_kwargs,
                **pool_kwargs,
            )
            self._close_pool = True
            super().__init__(self._lakebase.pool)

        def close(self) -> None:  # type: ignore[override]
            super().close()
            if self._close_pool:
                self._lakebase.close()

        def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
            previous = self._close_pool
            self._close_pool = False
            try:
                return super().__exit__(exc_type, exc, tb)
            finally:
                self._close_pool = previous

    _CHECKPOINT_SAVER_IMPL = _CheckpointSaver
    return _CHECKPOINT_SAVER_IMPL


class CheckpointSaver:  # type: ignore[override]
    def __new__(cls, *args: Any, **kwargs: Any):
        impl_cls = _checkpoint_saver_impl()
        return impl_cls(*args, **kwargs)
