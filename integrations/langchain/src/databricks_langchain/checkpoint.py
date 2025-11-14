from __future__ import annotations

from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.lakebase import LakebasePool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver


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

    get = _delegate("get")
    get_tuple = _delegate("get_tuple")
    list = _delegate("list")
    put = _delegate("put")
    put_writes = _delegate("put_writes")
    delete_thread = _delegate("delete_thread")
    aget = _delegate("aget")
    aget_tuple = _delegate("aget_tuple")
    alist = _delegate("alist")
    aput = _delegate("aput")
    aput_writes = _delegate("aput_writes")
    adelete_thread = _delegate("adelete_thread")
    get_next_version = _delegate("get_next_version")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
