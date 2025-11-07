from __future__ import annotations

import logging
import os
import time
import uuid
import weakref
from contextlib import contextmanager
from threading import Lock
from typing import Generator, Optional, Union

import psycopg
from databricks.sdk import WorkspaceClient
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

__all__ = [
    "LakebasePool",
    "RotatingCredentialConnection",
    "build_lakebase_pool",
    "pooled_connection",
    "make_checkpointer",
    "PooledPostgresSaver",
]

logger = logging.getLogger(__name__)

DEFAULT_CACHE_MINUTES = int(os.getenv("DB_TOKEN_CACHE_MINUTES", "50"))
DEFAULT_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "1"))
DEFAULT_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
DEFAULT_TIMEOUT = float(os.getenv("DB_POOL_TIMEOUT", "30.0"))
DEFAULT_SSLMODE = os.getenv("DB_SSL_MODE", "require")
DEFAULT_PORT = int(os.getenv("DB_PORT", "5432"))


class RotatingCredentialConnection(psycopg.Connection):
    """
    psycopg `Connection` that injects a Lakehouse (Postgres) OAuth token
    at connect-time. Tokens are minted with `WorkspaceClient` and cached
    for *N* minutes to avoid repeated API calls.
    """

    workspace_client: Optional[WorkspaceClient] = None
    instance_name: Optional[str] = None

    _cache_lock = Lock()
    _cached_token: Optional[str] = None
    _cache_ts: Optional[float] = None
    _cache_duration_sec: int = DEFAULT_CACHE_MINUTES * 60

    @classmethod
    def _mint_token(cls) -> str:
        if not cls.workspace_client or not cls.instance_name:
            raise RuntimeError("RotatingCredentialConnection not initialized.")

        cred = cls.workspace_client.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[cls.instance_name],
        )
        return cred.token

    @classmethod
    def _get_token(cls) -> str:
        now = time.time()
        with cls._cache_lock:
            if (
                cls._cached_token
                and cls._cache_ts
                and (now - cls._cache_ts) < cls._cache_duration_sec
            ):
                return cls._cached_token

            token = cls._mint_token()
            cls._cached_token = token
            cls._cache_ts = now
            return token

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs):  # type: ignore[override]
        kwargs = dict(kwargs)
        kwargs["password"] = cls._get_token()
        return super().connect(conninfo=conninfo, **kwargs)


def _infer_username(w: WorkspaceClient) -> str:
    """Resolve a default username preferring service-principal identity."""
    try:
        sp = w.current_service_principal.me()
        if sp and getattr(sp, "application_id", None):
            return sp.application_id
    except Exception:
        logger.debug("Falling back to current_user for Lakehouse credentials.")

    user = w.current_user.me()
    return user.user_name


def _reset_token_cache() -> None:
    """Reset the cached OAuth token (useful for tests)."""

    RotatingCredentialConnection._cached_token = None
    RotatingCredentialConnection._cache_ts = None


class LakebasePool:
    """Wrapper around a psycopg connection pool with rotating credentials."""

    def __init__(
        self,
        *,
        workspace_client: WorkspaceClient,
        instance_name: str,
        host: str,
        database: str,
        username: Optional[str] = None,
        port: int = DEFAULT_PORT,
        sslmode: str = DEFAULT_SSLMODE,
        min_size: int = DEFAULT_MIN_SIZE,
        max_size: int = DEFAULT_MAX_SIZE,
        timeout: float = DEFAULT_TIMEOUT,
        token_cache_minutes: int = DEFAULT_CACHE_MINUTES,
        open_pool: bool = True,
        connection_kwargs: Optional[dict[str, object]] = None,
        probe: bool = True,
    ) -> None:
        self.workspace_client = workspace_client
        self.instance_name = instance_name
        self.host = host
        self.database = database
        self.username = username or _infer_username(workspace_client)
        self.port = port
        self.sslmode = sslmode
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.token_cache_minutes = token_cache_minutes

        RotatingCredentialConnection.workspace_client = workspace_client
        RotatingCredentialConnection.instance_name = instance_name
        RotatingCredentialConnection._cache_duration_sec = token_cache_minutes * 60
        _reset_token_cache()

        conninfo = (
            f"dbname={database} user={self.username} host={host} port={port} sslmode={sslmode}"
        )

        default_kwargs: dict[str, object] = {
            "autocommit": True,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
        if connection_kwargs:
            default_kwargs.update(connection_kwargs)

        self._pool = ConnectionPool(
            conninfo=conninfo,
            connection_class=RotatingCredentialConnection,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            open=open_pool,
            kwargs=default_kwargs,
        )

        if probe:
            try:
                with self._pool.connection() as conn, conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except Exception:
                self._pool.close()
                raise

        logger.info(
            "lakebase pool ready: host=%s db=%s min=%s max=%s cache=%smin",
            host,
            database,
            min_size,
            max_size,
            token_cache_minutes,
        )

    @property
    def pool(self) -> ConnectionPool:
        return self._pool

    def connection(self):
        return self._pool.connection()

    def make_checkpointer(self) -> PooledPostgresSaver:
        return PooledPostgresSaver(self._pool)

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> LakebasePool:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()


@contextmanager
def pooled_connection(
    pool: Union[ConnectionPool, LakebasePool],
) -> Generator[psycopg.Connection, None, None]:
    """Context manager that yields a pooled psycopg connection."""

    connection_ctx = pool.connection()

    with connection_ctx as conn:
        yield conn


class PooledPostgresSaver(PostgresSaver):
    """PostgresSaver that automatically returns its connection to the pool."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool
        self._conn = pool.getconn()
        super().__init__(self._conn)
        self._finalizer = weakref.finalize(self, self._release)

    def _release(self) -> None:
        if self._conn is not None:
            try:
                self._pool.putconn(self._conn)
            finally:
                self._conn = None

    def close(self) -> None:
        self._release()
        self._finalizer.detach()

    def __enter__(self) -> PooledPostgresSaver:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()


def make_checkpointer(pool: Union[ConnectionPool, LakebasePool]) -> PooledPostgresSaver:
    """
    Create a LangGraph `PostgresSaver` backed by a pooled connection.

    The returned saver keeps a dedicated connection checked out until
    `close()` is invoked (or the object is GC'ed). Use it as a context
    manager to ensure timely release:

    >>> with make_checkpointer(pool) as saver:
    ...     graph = workflow.compile(checkpointer=saver)
    """
    if isinstance(pool, LakebasePool):
        return pool.make_checkpointer()

    return PooledPostgresSaver(pool)
