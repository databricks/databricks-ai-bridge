from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

try:  # pragma: no cover - optional dependency guard
    from langgraph.checkpoint.base import BaseCheckpointSaver
except ModuleNotFoundError:  # pragma: no cover - fallback when langgraph missing

    class BaseCheckpointSaver:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            raise ModuleNotFoundError("CheckpointSaver requires langgraph to be installed.")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk import WorkspaceClient
else:  # pragma: no cover - runtime fallback when dependency absent
    WorkspaceClient = Any  # type: ignore


def _load_checkpoint_saver_deps():
    try:
        from databricks_ai_bridge.lakebase import LakebasePool
        from langgraph.checkpoint.postgres import PostgresSaver
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "databricks-ai-bridge[lakebase] is required for Lakebase checkpointing.\n"
        ) from exc

    return LakebasePool, PostgresSaver


def _delegate(name: str) -> Callable[..., Any]:
    def _fn(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._inner, name)(*args, **kwargs)

    _fn.__name__ = name
    return _fn


class CheckpointSaver(BaseCheckpointSaver):
    """LangGraph checkpoint saver backed by a Lakebase connection pool."""

    def __init__(
        self,
        *,
        database_instance: str,
        workspace_client: WorkspaceClient | None = None,
        **pool_kwargs: object,
    ) -> None:
        LakebasePool, PostgresSaver = _load_checkpoint_saver_deps()

        self._lakebase = LakebasePool(
            instance_name=database_instance,
            workspace_client=workspace_client,
            **dict(pool_kwargs),
        )
        self._pool = self._lakebase.pool
        self._conn = self._pool.getconn()
        self._inner = PostgresSaver(self._conn)
        self._close_pool = True
        self.serde = getattr(self._inner, "serde", None)

    @property
    def config_specs(self):
        return self._inner.config_specs

    def setup(self) -> None:
        self._inner.setup()

    def close(self) -> None:
        inner_close = getattr(self._inner, "close", None)
        if inner_close is not None:
            inner_close()
        if self._conn is not None:
            try:
                self._pool.putconn(self._conn)
            finally:
                self._conn = None
        if self._close_pool:
            self._lakebase.close()

    def __enter__(self):
        self._prev_close_pool = self._close_pool
        self._close_pool = False
        enter = getattr(self._inner, "__enter__", None)
        if enter:
            enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            exit_method = getattr(self._inner, "__exit__", None)
            if exit_method:
                exit_method(exc_type, exc, tb)
        finally:
            try:
                self.close()
            finally:
                self._close_pool = getattr(self, "_prev_close_pool", True)

    get = _delegate("get")  # type: ignore[assignment]
    get_tuple = _delegate("get_tuple")  # type: ignore[assignment]
    list = _delegate("list")  # type: ignore[assignment]
    put = _delegate("put")  # type: ignore[assignment]
    put_writes = _delegate("put_writes")  # type: ignore[assignment]
    delete_thread = _delegate("delete_thread")  # type: ignore[assignment]
    aget = _delegate("aget")  # type: ignore[assignment]
    aget_tuple = _delegate("aget_tuple")  # type: ignore[assignment]
    alist = _delegate("alist")  # type: ignore[assignment]
    aput = _delegate("aput")  # type: ignore[assignment]
    aput_writes = _delegate("aput_writes")  # type: ignore[assignment]
    adelete_thread = _delegate("adelete_thread")  # type: ignore[assignment]
    get_next_version = _delegate("get_next_version")  # type: ignore[assignment]

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - defensive fallback
        return getattr(self._inner, name)
