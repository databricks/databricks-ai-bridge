from __future__ import annotations

import logging
import time
import uuid
from threading import Lock
from typing import Optional

import psycopg
from databricks.sdk import WorkspaceClient
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

__all__ = ["LakebasePool"]

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_CACHE_DURATION_SECONDS = 50 * 60  # Cache token for 50 minutes
DEFAULT_MIN_SIZE = 1
DEFAULT_MAX_SIZE = 10
DEFAULT_TIMEOUT = 30.0
# Default values from https://docs.databricks.com/aws/en/oltp/projects/connect-overview#connection-string-components
DEFAULT_SSLMODE = "require"
DEFAULT_PORT = 5432
DEFAULT_DATABASE = "databricks_postgres"


def _infer_username(w: WorkspaceClient) -> str:
    """Get username for database connection with prioritizing service principal first, then user's username."""
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

    name: Name of Lakebase Instance
    """

    def __init__(
        self,
        *,
        instance_name: str,
        workspace_client: WorkspaceClient | None = None,
        token_cache_duration_seconds: int = DEFAULT_TOKEN_CACHE_DURATION_SECONDS,
        **pool_kwargs: object,
    ) -> None:
        if workspace_client is None:
            workspace_client = WorkspaceClient()

        # Resolve host from the Lakebase name
        try:
            instance = workspace_client.database.get_database_instance(instance_name)
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

        self.workspace_client = workspace_client
        self.name = instance_name
        self.host = resolved_host
        self.username = _infer_username(workspace_client)
        self.token_cache_duration_seconds = token_cache_duration_seconds

        # Token caching
        self._cache_lock = Lock()
        self._cached_token: Optional[str] = None
        self._cache_ts: Optional[float] = None

        # Create connection pool that fetches a rotating M2M OAuth token
        # https://docs.databricks.com/aws/en/oltp/instances/query/notebook#psycopg3
        class RotatingConnection(psycopg.Connection):
            @classmethod
            def connect(cls, conninfo: str = "", **kwargs):
                # Append new password to kwargs
                kwargs["password"] = self._get_token()

                # Call the superclass's connect method with updated kwargs
                return super().connect(conninfo, **kwargs)

        conninfo = f"dbname={DEFAULT_DATABASE} user={self.username} host={resolved_host} port={DEFAULT_PORT} sslmode={DEFAULT_SSLMODE}"

        default_kwargs: dict[str, object] = {
            "autocommit": True,
            "row_factory": dict_row,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }

        pool_params = dict(
            conninfo=conninfo,
            kwargs=default_kwargs,
            min_size=DEFAULT_MIN_SIZE,
            max_size=DEFAULT_MAX_SIZE,
            timeout=DEFAULT_TIMEOUT,
            open=True,
            connection_class=RotatingConnection,
            **pool_kwargs,
        )

        self._pool = ConnectionPool(**pool_params)

        logger.info(
            "lakebase pool ready: host=%s db=%s min=%s max=%s cache=%ss",
            resolved_host,
            DEFAULT_DATABASE,
            pool_params.get("min_size"),
            pool_params.get("max_size"),
            self.token_cache_duration_seconds,
        )

    def _get_token(self) -> str:
        """Get cached token or mint a new one if expired."""
        with self._cache_lock:
            now = time.time()
            if (
                self._cached_token
                and self._cache_ts
                and (now - self._cache_ts) < self.token_cache_duration_seconds
            ):
                return self._cached_token

            token = self._mint_token()
            self._cached_token = token
            self._cache_ts = now
            return token

    def _mint_token(self) -> str:
        try:
            cred = self.workspace_client.database.generate_database_credential(
                request_id=str(uuid.uuid4()),
                instance_names=[self.name],
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to obtain credential for Lakebase instance "
                f"'{self.name}'. Ensure the caller has access."
            ) from exc

        return cred.token

    @property
    def pool(self) -> ConnectionPool:
        """Access the underlying connection pool."""
        return self._pool

    def connection(self):
        """Get a connection from the pool."""
        return self._pool.connection()

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()
