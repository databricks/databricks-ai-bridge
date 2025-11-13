from __future__ import annotations

import logging
import time
import uuid
import weakref
from contextlib import contextmanager
from threading import Lock
from typing import Any, Generator, Optional, Union

import psycopg
from databricks.sdk import WorkspaceClient
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolClosed, PoolTimeout

__all__ = [
    "LakebasePool",
    "RotatingCredentialConnection",
    "build_lakebase_pool",
    "pooled_connection",
    "make_checkpointer",
    "PooledPostgresSaver",
]

logger = logging.getLogger(__name__)

DEFAULT_CACHE_SECONDS = 50 * 60
DEFAULT_MIN_SIZE = 1
DEFAULT_MAX_SIZE = 10
DEFAULT_TIMEOUT = 30.0
DEFAULT_SSLMODE = "require"
DEFAULT_PORT = 5432
DEFAULT_HOST = None
LAKEBASE_NAME = None
DEFAULT_DATABASE = "databricks_postgres"
DEFAULT_USERNAME = None


class RotatingCredentialConnection(psycopg.Connection):
    """
    Base psycopg connection that injects a Lakebase (Postgres) OAuth token at
    connect-time. Concrete subclasses are generated per pool so that token
    caches don't leak across pools.
    """

    workspace_client: Optional[WorkspaceClient] = None
    instance_name: Optional[str] = None
    cache_duration_sec: int = DEFAULT_CACHE_SECONDS

    _cache_lock = Lock()
    _cached_token: Optional[str] = None
    _cache_ts: Optional[float] = None

    @classmethod
    def _mint_token(cls) -> str:
        if not cls.workspace_client or not cls.instance_name:
            raise RuntimeError("RotatingCredentialConnection not initialized.")

        try:
            cred = cls.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[cls.instance_name],
            )
        except Exception as exc:  # pragma: no cover - surfaced to user
            raise ConnectionError(
                f"Failed to obtain credential for Lakebase instance "
                f"'{cls.instance_name}'. Ensure the caller has access."
            ) from exc

        return cred.token

    @classmethod
    def _get_token(cls) -> str:
        with cls._cache_lock:
            now = time.time()
            if (
                cls._cached_token
                and cls._cache_ts
                and (now - cls._cache_ts) < cls.cache_duration_sec
            ):
                return cls._cached_token

            token = cls._mint_token()
            cls._cached_token = token
            cls._cache_ts = now
            return token

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs):
        kwargs = dict(kwargs)
        kwargs["password"] = cls._get_token()
        return super().connect(conninfo, **kwargs)


def _make_rotating_connection_class(
    workspace_client: WorkspaceClient, instance_name: str, cache_duration_sec: int
) -> type[RotatingCredentialConnection]:
    return type(
        f"LakebaseRotatingConnection_{instance_name}",
        (RotatingCredentialConnection,),
        {
            "workspace_client": workspace_client,
            "instance_name": instance_name,
            "cache_duration_sec": cache_duration_sec,
            "_cache_lock": Lock(),
            "_cached_token": None,
            "_cache_ts": None,
        },
    )


def _infer_username(w: WorkspaceClient) -> str:
    """Resolve a default username preferring service-principal identity."""
    try:
        sp = w.current_service_principal.me()
        if sp and getattr(sp, "application_id", None):
            return sp.application_id
    except Exception:
        logger.debug(
            "Could not get service principal, using current user for Lakebase credentials."
        )

    user = w.current_user.me()
    return user.user_name


class LakebasePool:
    """Wrapper around a psycopg connection pool with rotating Lakehouse credentials.

    instance_name:
        User-set name on Lakebase Instance
        (can retrieve from connection details page in Databricks workspace)
    workspace_client:
        Optional `WorkspaceClient` to use; default client is created otherwise.
    token_cache_seconds:
        Lifetime for cached OAuth tokens in seconds. Defaults to 50 minutes
        (3000 seconds).
    **pool_kwargs:
        Additional options passed to ``psycopg_pool.ConnectionPool`` (e.g.
        ``min_size``, ``max_size``).
    """

    def __init__(
        self,
        *,
        instance_name: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        token_cache_seconds: Optional[int] = None,
        **pool_kwargs: object,
    ) -> None:
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        resolved_instance = instance_name or LAKEBASE_NAME
        if resolved_instance is None:
            raise ValueError(
                "Lakebase instance name must be provided. Specify the instance_name argument or set the LAKEBASE_NAME environment variable."
            )

        resolved_host = DEFAULT_HOST
        if resolved_host is None:
            try:
                instance = workspace_client.database.get_database_instance(resolved_instance)
            except Exception as exc:  # pragma: no cover - propagated to caller
                raise ValueError(
                    "Lakebase host must be provided. Unable to resolve host from workspace metadata."
                ) from exc

            resolved_host = getattr(instance, "read_write_dns", None) or getattr(
                instance, "read_only_dns", None
            )

        if resolved_host is None:
            raise ValueError(
                "Lakebase host must be provided. Make sure your Lakebase instance name is correct, set DB_HOST, or ensure the workspace instance metadata exposes read_write_dns."
            )

        database = DEFAULT_DATABASE
        port = DEFAULT_PORT
        sslmode = DEFAULT_SSLMODE
        cache_seconds = (
            DEFAULT_CACHE_SECONDS if token_cache_seconds is None else int(token_cache_seconds)
        )

        self.workspace_client = workspace_client
        self.instance_name = resolved_instance
        self.host = resolved_host
        self.database = database
        self.username = DEFAULT_USERNAME or _infer_username(workspace_client)
        self.port = port
        self.sslmode = sslmode
        self.token_cache_seconds = cache_seconds
        typed_pool_kwargs = dict(pool_kwargs)
        self.pool_config = typed_pool_kwargs

        conninfo = (
            f"dbname={database} user={self.username} host={resolved_host} port={port} sslmode={sslmode}"
        )

        default_kwargs: dict[str, object] = {
            "autocommit": True,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }

        connection_class = _make_rotating_connection_class(
            workspace_client=workspace_client,
            instance_name=resolved_instance,
            cache_duration_sec=cache_seconds,
        )

        pool_params = dict(
            conninfo=conninfo,
            kwargs=default_kwargs,
            min_size=DEFAULT_MIN_SIZE,
            max_size=DEFAULT_MAX_SIZE,
            timeout=DEFAULT_TIMEOUT,
            open=True,
            connection_class=connection_class,
            **typed_pool_kwargs,
        )

        self._pool = ConnectionPool(**pool_params)
        self._connection_class = connection_class

        logger.info(
            "lakebase pool ready: host=%s db=%s min=%s max=%s cache=%ss",
            resolved_host,
            database,
            pool_params.get("min_size"),
            pool_params.get("max_size"),
            cache_seconds,
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


# Build lakebase pool instance with only instance name required
def build_lakebase_pool(
    *,
    instance_name: str,
    token_cache_seconds: Optional[int] = None,
    workspace_client: WorkspaceClient | None = None,
    **pool_kwargs: Any,
) -> ConnectionPool:
    lakebase = LakebasePool(
        instance_name=instance_name,
        workspace_client=workspace_client,
        token_cache_seconds=token_cache_seconds,
        **pool_kwargs,
    )
    return lakebase.pool


@contextmanager
def pooled_connection(
    pool: Union[ConnectionPool, LakebasePool],
) -> Generator[psycopg.Connection, None, None]:
    """Context manager that yields a pooled psycopg connection."""

    connection_ctx = pool.connection()

    with connection_ctx as conn:
        yield conn


class PooledPostgresSaver(PostgresSaver):
    """LangGraph PostgresSaver keeps one database connection checked out from a connection pool."""

    def __init__(self, pool: ConnectionPool):
        self._pool = pool
        self._conn = pool.getconn()
        super().__init__(self._conn)
        self._finalizer = weakref.finalize(self, self._release)

    def _release(self) -> None:
        if self._conn is not None:
            try:
                self._pool.putconn(self._conn)
            except (PoolClosed, PoolTimeout) as e:  # pragma: no cover - expected cleanup path
                logger.debug("Pool unavailable for connection return: %s", e)
            except Exception as e:  # pragma: no cover - unexpected
                logger.error("Unexpected error returning connection to pool: %s", e, exc_info=True)
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
