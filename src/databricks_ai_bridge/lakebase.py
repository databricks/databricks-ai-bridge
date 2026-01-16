from __future__ import annotations

import asyncio
import logging
import time
import uuid
from threading import Lock
from typing import Any

from databricks.sdk import WorkspaceClient
from psycopg.rows import DictRow

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool, ConnectionPool
except ImportError as e:
    raise ImportError(
        "LakebasePool requires databricks-ai-bridge[memory]. "
        "Please install with: pip install databricks-ai-bridge[memory]"
    ) from e

__all__ = ["AsyncLakebasePool", "LakebasePool"]

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_CACHE_DURATION_SECONDS = 50 * 60  # Cache token for 50 minutes
DEFAULT_MIN_SIZE = 1
DEFAULT_MAX_SIZE = 10
DEFAULT_TIMEOUT = 30.0
# Default values from https://docs.databricks.com/aws/en/oltp/projects/connect-overview#connection-string-components
DEFAULT_SSLMODE = "require"
DEFAULT_PORT = 5432
DEFAULT_DATABASE = "databricks_postgres"


class _LakebasePoolBase:
    """
    Base logic for Lakebase connection pools: resolve host, infer username,
    token cache + minting, and conninfo building.

    Subclasses implement pool-specific initialization and lifecycle methods.
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        token_cache_duration_seconds: int = DEFAULT_TOKEN_CACHE_DURATION_SECONDS,
    ) -> None:
        self.workspace_client: WorkspaceClient = workspace_client or WorkspaceClient()
        self.instance_name: str = instance_name
        self.token_cache_duration_seconds: int = token_cache_duration_seconds

        # Resolve host from the Lakebase name
        try:
            instance = self.workspace_client.database.get_database_instance(instance_name)
        except Exception as exc:
            raise ValueError(
                f"Unable to resolve Lakebase instance '{instance_name}'. "
                "Ensure the instance name is correct."
            ) from exc

        resolved_host = getattr(instance, "read_write_dns", None) or getattr(
            instance, "read_only_dns", None
        )

        if not resolved_host:
            raise ValueError(
                f"Lakebase host not found for instance '{instance_name}'. "
                "Ensure the instance is running and in AVAILABLE state."
            )

        self.host: str = resolved_host
        self.username: str = self._infer_username()

        self._cached_token: str | None = None
        self._cache_ts: float | None = None

    def _get_cached_token(self) -> str | None:
        """Check if the cached token is still valid."""
        if not self._cached_token or not self._cache_ts:
            return None
        if (time.time() - self._cache_ts) < self.token_cache_duration_seconds:
            return self._cached_token
        return None

    def _mint_token(self) -> str:
        try:
            cred = self.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[self.instance_name],
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to obtain credential for Lakebase instance "
                f"'{self.instance_name}'. Ensure the caller has access."
            ) from exc

        if not cred.token:
            raise RuntimeError("Failed to generate database credential: no token received")

        return cred.token

    def _conninfo(self) -> str:
        """Build the connection info string."""
        return (
            f"dbname={DEFAULT_DATABASE} user={self.username} "
            f"host={self.host} port={DEFAULT_PORT} sslmode={DEFAULT_SSLMODE}"
        )

    def _infer_username(self) -> str:
        """Get username for database connection."""
        try:
            user = self.workspace_client.current_user.me()
            if user and user.user_name:
                return user.user_name
        except Exception:
            logger.debug("Could not get username for Lakebase credentials.")
        raise ValueError("Unable to infer username for Lakebase connection.")


class LakebasePool(_LakebasePoolBase):
    """Sync Lakebase connection pool built on psycopg with rotating credentials.

    instance_name: Name of Lakebase Instance
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        token_cache_duration_seconds: int = DEFAULT_TOKEN_CACHE_DURATION_SECONDS,
        **pool_kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            instance_name=instance_name,
            workspace_client=workspace_client,
            token_cache_duration_seconds=token_cache_duration_seconds,
        )

        # Sync lock for thread-safe token caching
        self._cache_lock = Lock()

        # Create connection pool that fetches a rotating M2M OAuth token
        # https://docs.databricks.com/aws/en/oltp/instances/query/notebook#psycopg3
        pool = self

        class RotatingConnection(psycopg.Connection):
            @classmethod
            def connect(cls, conninfo: str = "", **kwargs):
                kwargs["password"] = pool._get_token()
                # Call the superclass's connect method with updated kwargs
                return super().connect(conninfo, **kwargs)

        default_kwargs: dict[str, object] = {
            "autocommit": True,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }

        # Get pool config values (overrides by user pool_kwargs)
        min_size = pool_kwargs.pop("min_size", DEFAULT_MIN_SIZE)
        max_size = pool_kwargs.pop("max_size", DEFAULT_MAX_SIZE)
        timeout = pool_kwargs.pop("timeout", DEFAULT_TIMEOUT)

        self._pool: ConnectionPool[psycopg.Connection[DictRow]] = ConnectionPool(
            conninfo=self._conninfo(),
            kwargs=default_kwargs,
            min_size=min_size,  # type: ignore[invalid-argument-type]
            max_size=max_size,  # type: ignore[invalid-argument-type]
            timeout=timeout,  # type: ignore[invalid-argument-type]
            open=True,
            connection_class=RotatingConnection,
            **pool_kwargs,  # type: ignore[invalid-argument-type]
        )

        logger.info(
            "lakebase pool ready: host=%s db=%s min=%s max=%s timeout=%s cache=%ss",
            self.host,
            DEFAULT_DATABASE,
            min_size,
            max_size,
            timeout,
            self.token_cache_duration_seconds,
        )

    def _get_token(self) -> str:
        """Get cached token or mint a new one if expired (thread-safe)."""
        with self._cache_lock:
            if cached_token := self._get_cached_token():
                return cached_token

            token = self._mint_token()
            self._cached_token = token
            self._cache_ts = time.time()
            return token

    @property
    def pool(self) -> ConnectionPool[psycopg.Connection[DictRow]]:
        """Access the underlying connection pool."""
        return self._pool

    def connection(self):
        """Get a connection from the pool."""
        return self._pool.connection()

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()


class AsyncLakebasePool(_LakebasePoolBase):
    """Async Lakebase connection pool built on psycopg with rotating credentials.

    instance_name: Name of Lakebase Instance
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        token_cache_duration_seconds: int = DEFAULT_TOKEN_CACHE_DURATION_SECONDS,
        **pool_kwargs: object,
    ) -> None:
        super().__init__(
            instance_name=instance_name,
            workspace_client=workspace_client,
            token_cache_duration_seconds=token_cache_duration_seconds,
        )

        # Async lock for coroutine-safe token caching
        self._cache_lock = asyncio.Lock()

        # Create async connection pool that fetches a rotating M2M OAuth token
        pool = self

        class AsyncRotatingConnection(psycopg.AsyncConnection):
            @classmethod
            async def connect(cls, conninfo: str = "", **kwargs):
                kwargs["password"] = await pool._get_token_async()
                # Call the superclass's connect method with updated kwargs
                return await super().connect(conninfo, **kwargs)

        default_kwargs: dict[str, object] = {
            "autocommit": True,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }

        # Get pool config values (overrides by user pool_kwargs)
        min_size = pool_kwargs.pop("min_size", DEFAULT_MIN_SIZE)
        max_size = pool_kwargs.pop("max_size", DEFAULT_MAX_SIZE)
        timeout = pool_kwargs.pop("timeout", DEFAULT_TIMEOUT)

        self._pool: AsyncConnectionPool[psycopg.AsyncConnection[DictRow]] = AsyncConnectionPool(
            conninfo=self._conninfo(),
            kwargs=default_kwargs,
            min_size=min_size,  # type: ignore[invalid-argument-type]
            max_size=max_size,  # type: ignore[invalid-argument-type]
            timeout=timeout,  # type: ignore[invalid-argument-type]
            open=False,  # Don't open yet, must be opened with await
            connection_class=AsyncRotatingConnection,
            **pool_kwargs,  # type: ignore[invalid-argument-type]
        )

        logger.info(
            "async lakebase pool created: host=%s db=%s min=%s max=%s timeout=%s cache=%ss",
            self.host,
            DEFAULT_DATABASE,
            min_size,
            max_size,
            timeout,
            self.token_cache_duration_seconds,
        )

    async def _get_token_async(self) -> str:
        """Get cached token or mint a new one if expired (async, non-blocking).

        Uses asyncio.Lock for coroutine coordination. Token minting (a sync SDK call)
        runs in an executor to avoid blocking the event loop.
        """
        async with self._cache_lock:
            if cached_token := self._get_cached_token():
                return cached_token

            # Run the sync SDK call in an executor to not block the event loop
            loop = asyncio.get_running_loop()
            token = await loop.run_in_executor(None, self._mint_token)
            self._cached_token = token
            self._cache_ts = time.time()
            return token

    @property
    def pool(self) -> AsyncConnectionPool[psycopg.AsyncConnection[DictRow]]:
        """Access the underlying async connection pool."""
        return self._pool

    def connection(self):
        """Get a connection from the async pool."""
        return self._pool.connection()

    async def open(self) -> None:
        """Open the connection pool."""
        await self._pool.open()

    async def close(self) -> None:
        """Close the connection pool."""
        await self._pool.close()

    async def __aenter__(self):
        """Enter async context manager."""
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager and close the connection pool."""
        await self.close()
        return False
