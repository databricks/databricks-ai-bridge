from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk import WorkspaceClient
else:  # pragma: no cover - runtime fallback when dependency absent
    WorkspaceClient = Any  # type: ignore

_IMPORT_ERROR: ModuleNotFoundError | None = None
_LAKEBASE_DEPS: dict[str, Any] | None = None
_CHECKPOINT_SAVER_CLASS: type | None = None


def _load_dependencies() -> dict[str, Any] | None:
    """Attempt to import Lakebase dependencies lazily."""

    global _IMPORT_ERROR, _LAKEBASE_DEPS

    if _LAKEBASE_DEPS is not None:
        return _LAKEBASE_DEPS

    try:  # pragma: no cover - import guarded at runtime
        from databricks_ai_bridge.lakebase import (
            LakebasePool,
            PooledPostgresSaver,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced to caller
        _IMPORT_ERROR = exc
        return None

    _LAKEBASE_DEPS = {
        "LakebasePool": LakebasePool,
        "PooledPostgresSaver": PooledPostgresSaver,
    }
    _IMPORT_ERROR = None
    return _LAKEBASE_DEPS


def _ensure_checkpoint_saver_class() -> type:
    """Materialise the concrete CheckpointSaver class when deps are present."""

    global _CHECKPOINT_SAVER_CLASS

    if _CHECKPOINT_SAVER_CLASS is not None:
        return _CHECKPOINT_SAVER_CLASS

    deps = _load_dependencies()
    if not deps:
        raise ModuleNotFoundError(
            "CheckpointSaver requires databricks-ai-bridge[lakebase] and psycopg to be installed."
        ) from _IMPORT_ERROR

    LakebasePool = deps["LakebasePool"]
    PooledPostgresSaver = deps["PooledPostgresSaver"]

    class _CheckpointSaverImpl(PooledPostgresSaver):
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

    _CHECKPOINT_SAVER_CLASS = _CheckpointSaverImpl
    globals()["CheckpointSaver"] = _CheckpointSaverImpl
    return _CHECKPOINT_SAVER_CLASS


class CheckpointSaver:  # type: ignore[override]
    def __new__(cls, *args: Any, **kwargs: Any):
        impl = _ensure_checkpoint_saver_class()
        instance = impl.__new__(impl)
        impl.__init__(instance, *args, **kwargs)
        return instance
