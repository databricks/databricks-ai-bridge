from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from threading import Lock
from typing import Optional, Type

import psycopg
from databricks.sdk import WorkspaceClient
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

__all__ = ["LakebasePool"]

logger = logging.getLogger(__name__)

DEFAULT_CACHE_SECONDS = 50 * 60  # Cache token for 50 minutes
DEFAULT_MIN_SIZE = 1
DEFAULT_MAX_SIZE = 10
DEFAULT_TIMEOUT = 30.0
DEFAULT_SSLMODE = "require"
DEFAULT_PORT = 5432
DEFAULT_DATABASE = "databricks_postgres"


class _RotatingCredentialConnection(psycopg.Connection):
    """
    Note: Don't use - use create_connection_class instead to avoid leaking tokens
    """

    workspace_client: Optional[WorkspaceClient] = None
    instance_name: Optional[str] = None
    token_cache_duration_seconds: int = DEFAULT_CACHE_SECONDS

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
        except Exception as exc:
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
                and (now - cls._cache_ts) < cls.token_cache_duration_seconds
            ):
                return cls._cached_token

            token = cls._mint_token()
            cls._cached_token = token
            cls._cache_ts = now
            return token

    @classmethod
    def connect(cls: Type["_RotatingCredentialConnection"], conninfo: str = "", **kwargs):
        kwargs = dict(kwargs)
        kwargs["password"] = cls._get_token()
        return super().connect(conninfo, **kwargs)


def create_connection_class(
    workspace_client: WorkspaceClient, instance_name: str, token_cache_duration_seconds: int
) -> type[_RotatingCredentialConnection]:
    """
    Create a psycopg `Connection` subclass that automatically injects a
    Lakebase OAuth token at connect-time with token refresh handled

    Parameters
    ----------
    workspace_client : WorkspaceClient
        The Databricks workspace client used to mint credentials.
    instance_name : str
        The Lakebase instance name
    token_cache_duration_seconds : int, optional
        Seconds to cache the minted token before refreshing. Defaults to 50 minutes/3000 sec

    Returns
    -------
    Type[psycopg.Connection]
        A subclass suitable for passing to psycopg / psycopg_pool as
        `connection_class`.

    Example
    -------
    >>> w = WorkspaceClient()
    >>> ConnectionClass = create_connection_class(w, "my-lakebase")
    >>> conn = ConnectionClass.connect("connection-string")
    """

    return type(
        f"LakebaseRotatingConnection_{instance_name}",
        (_RotatingCredentialConnection,),
        {
            "workspace_client": workspace_client,
            "instance_name": instance_name,
            "token_cache_duration_seconds": token_cache_duration_seconds,
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
    token_cache_duration_seconds:
        Lifetime for cached OAuth tokens in seconds. Defaults to 50 minutes
        (3000 seconds).
    **pool_kwargs:
        Additional options passed to ``psycopg_pool.ConnectionPool`` (e.g.
        ``min_size``, ``max_size``).
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        **pool_kwargs: object,
    ) -> None:
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        # Resolve host from the Lakebase name
        try:
            instance = workspace_client.database.get_database_instance(instance_name)
        except Exception as exc:
            raise ValueError(
                f"Unable to resolve Lakebase host for instance '{instance_name}'. "
                "Ensure the instance name is correct."
            ) from exc

        resolved_host = getattr(instance, "read_write_dns", None) or getattr(
            instance, "read_only_dns", None
        )

        if not resolved_host:
            raise ValueError(
                f"Lakebase host not found for instance '{instance_name}'. "
                "Ensure the instance exposes `read_write_dns` or `read_only_dns` in workspace metadata."
            )

        database = DEFAULT_DATABASE
        port = DEFAULT_PORT
        sslmode = DEFAULT_SSLMODE
        token_cache_duration_seconds = DEFAULT_CACHE_SECONDS

        self.workspace_client = workspace_client
        self.instance_name = instance_name
        self.host = resolved_host
        self.database = database
        self.port = port
        self.sslmode = sslmode
        self.token_cache_duration_seconds = token_cache_duration_seconds
        self.username = _infer_username(workspace_client)
        typed_pool_kwargs = dict(pool_kwargs)
        self.pool_config = dict(typed_pool_kwargs)

        conninfo = f"dbname={database} user={self.username} host={resolved_host} port={port} sslmode={sslmode}"

        default_kwargs: dict[str, object] = {
            "autocommit": True,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }

        self._connection_class = create_connection_class(
            workspace_client=workspace_client,
            instance_name=instance_name,
            token_cache_duration_seconds=token_cache_duration_seconds,
        )

        pool_params = dict(
            conninfo=conninfo,
            kwargs=default_kwargs,
            min_size=DEFAULT_MIN_SIZE,
            max_size=DEFAULT_MAX_SIZE,
            timeout=DEFAULT_TIMEOUT,
            open=True,
            connection_class=self._connection_class,
            **typed_pool_kwargs,
        )

        self._pool = ConnectionPool(**pool_params)

        logger.info(
            "lakebase pool ready: host=%s db=%s min=%s max=%s cache=%ss",
            resolved_host,
            database,
            pool_params.get("min_size"),
            pool_params.get("max_size"),
            token_cache_duration_seconds,
        )

    @property
    def pool(self) -> ConnectionPool:
        return self._pool

    def connection(self) -> contextmanager[psycopg.Connection]:
        return self._pool.connection()

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> LakebasePool:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()
