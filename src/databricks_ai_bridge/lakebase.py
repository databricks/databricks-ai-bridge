from __future__ import annotations

import asyncio
import logging
import time
import uuid
from enum import Enum
from threading import Lock
from typing import Any, List, Literal, Optional, Sequence

from databricks.sdk import WorkspaceClient
from psycopg.rows import DictRow

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool, ConnectionPool
except ImportError as e:
    raise ImportError(
        "LakebasePool requires databricks-ai-bridge[memory]. "
        "Please install with: pip install databricks-ai-bridge[memory]"
    ) from e

__all__ = [
    "AsyncLakebasePool",
    "LakebasePool",
    "LakebaseClient",
    "TablePrivilege",
    "SchemaPrivilege",
]

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_CACHE_DURATION_SECONDS = 50 * 60  # Cache token for 50 minutes
DEFAULT_MIN_SIZE = 1
DEFAULT_MAX_SIZE = 10
DEFAULT_TIMEOUT = 30.0
# Default values from https://docs.databricks.com/aws/en/oltp/projects/connect-overview#connection-string-components
DEFAULT_SSLMODE = "require"
DEFAULT_PORT = 5432
DEFAULT_DATABASE = "databricks_postgres"

# Valid identity types for create_role
IdentityType = Literal["USER", "SERVICE_PRINCIPAL", "GROUP"]


class TablePrivilege(str, Enum):
    """PostgreSQL table privileges for GRANT statements."""

    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    TRUNCATE = "TRUNCATE"
    REFERENCES = "REFERENCES"
    TRIGGER = "TRIGGER"
    MAINTAIN = "MAINTAIN"
    ALL = "ALL"  # Renders as ALL PRIVILEGES


class SchemaPrivilege(str, Enum):
    """PostgreSQL schema privileges for GRANT statements."""

    USAGE = "USAGE"
    CREATE = "CREATE"
    ALL = "ALL"  # Renders as ALL PRIVILEGES


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


# =============================================================================
# LakebaseClient - SQL execution and operations
# =============================================================================


class LakebaseClient:
    """Client for executing SQL queries and managing Lakebase resources.

    Example (simple - creates pool internally):
        client = LakebaseClient(instance_name="my-lakebase")
        client.execute("SELECT * FROM users")
        client.create_role("user@example.com", PrincipalType.USER)
        client.close()

    Example (advanced - bring your own pool):
        pool = LakebasePool(instance_name="my-lakebase", max_size=20)
        client = LakebaseClient(pool=pool)
        client.execute("SELECT * FROM users")
        # Pool is managed externally, close it yourself
    """

    def __init__(
        self,
        *,
        pool: LakebasePool | None = None,
        instance_name: str | None = None,
        **pool_kwargs: object,
    ) -> None:
        """
        Initialize LakebaseClient.

        Provide EITHER:
        - pool: An existing LakebasePool instance (advanced usage where multiple clients can connect to same pool)
        - instance_name: Name of the Lakebase instance (creates pool internally)

        :param pool: Existing LakebasePool to use for connections.
        :param instance_name: Name of the Lakebase instance (used to create pool if pool not provided).
        :param workspace_client: Optional WorkspaceClient (only used when creating pool internally).
        :param pool_kwargs: Additional kwargs passed to LakebasePool (only used when creating pool internally).
        """
        if pool is not None and instance_name is not None:
            raise ValueError("Provide either 'pool' or 'instance_name', not both.")

        if pool is None and instance_name is None:
            raise ValueError("Must provide either 'pool' or 'instance_name'.")

        self._owns_pool = pool is None

        if pool is not None:
            self._pool = pool
        else:
            self._pool = LakebasePool(
                instance_name=instance_name,  # type: ignore[arg-type]
                **pool_kwargs,
            )

    @property
    def pool(self) -> LakebasePool:
        """Access the underlying LakebasePool."""
        return self._pool

    def close(self) -> None:
        """Close the client (and pool if it was created internally)."""
        if self._owns_pool:
            self._pool.close()

    # ---------------------------------------------------------
    # SQL Execution
    # ---------------------------------------------------------

    def execute(self, sql: str, params: Optional[tuple | dict] = None) -> List[Any] | None:
        """
        Execute a SQL query against the Lakebase instance.

        :param sql: The SQL query string.
        :param params: Optional parameters for query interpolation (prevents SQL injection).
        :return: List of rows (as dicts) if the query returns data, else None.

        Example:
            # DDL
            client.execute("CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT)")

            # Parameterized query (safe from SQL injection)
            client.execute("SELECT * FROM users WHERE name = %s", ("Alice",))

            # Named parameters
            client.execute("SELECT * FROM users WHERE id = %(id)s", {"id": 1})
        """
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                if cur.description:
                    return cur.fetchall()
                return None

    # ---------------------------------------------------------
    # Permission / Role Management
    # ---------------------------------------------------------

    def create_role(
        self,
        identity_name: str,
        identity_type: IdentityType,
        *,
        ensure_extension: bool = True,
    ) -> List[Any] | None:
        """
        Create a Databricks role for the given identity.
        https://docs.databricks.com/aws/en/oltp/instances/pg-roles?language=PostgreSQL#create-postgres-roles-and-grant-privileges-for-databricks-identities

        This enables role-based access control by registering a Databricks
        user, service principal, or group as a PostgreSQL role.

        If the role already exists, a warning is logged

        :param identity_name: The Databricks identity name
            (e.g., user email, service principal application ID, or group ID).
        :param identity_type: The type of identity - must be one of:
            "USER", "SERVICE_PRINCIPAL", or "GROUP".
        :param ensure_extension: If True (default), ensures the databricks_auth
            extension is created before creating the role.
        :return: Result from databricks_create_role, or None if role already exists.

        Example:
            # Create a role for a service principal
            client.create_role(
                "service-principal-uuid",
                "SERVICE_PRINCIPAL"
            )
        """
        # Create the databricks_auth extension. Each Postgres database must have its own extension.
        # https://docs.databricks.com/aws/en/oltp/instances/pg-roles#create-postgres-roles-and-grant-privileges-for-databricks-identities
        if ensure_extension:
            self.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth;")

        query = f"SELECT databricks_create_role(%s, '{identity_type}');"
        try:
            return self.execute(query, (identity_name,))
        except psycopg.errors.DuplicateObject:
            logger.info("Role '%s' already exists, skipping creation.", identity_name)
            return None

    def _format_privileges_str(
        self, privileges: Sequence[TablePrivilege] | Sequence[SchemaPrivilege]
    ) -> str:
        """Format privileges as a string for logging."""
        privilege_values = [p.value for p in privileges]
        if "ALL" in privilege_values:
            return "ALL PRIVILEGES"
        return ", ".join(privilege_values)

    def _format_privileges_sql(
        self, privileges: Sequence[TablePrivilege] | Sequence[SchemaPrivilege]
    ) -> sql.SQL:
        """Format privileges as a safe SQL fragment."""
        # Check if ALL is in the list - if so, use ALL PRIVILEGES
        privilege_values = [p.value for p in privileges]
        if "ALL" in privilege_values:
            return sql.SQL("ALL PRIVILEGES")
        # Privileges are SQL keywords, so use sql.SQL for each
        return sql.SQL(", ").join(sql.SQL(p) for p in privilege_values)

    def _execute_composed(self, query: sql.Composed) -> List[Any] | None:
        """Execute a composed SQL query safely."""
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                if cur.description:
                    return cur.fetchall()
                return None

    def grant_schema(
        self,
        grantee: str,
        privileges: Sequence[SchemaPrivilege],
        schemas: Sequence[str],
    ) -> None:
        """
        Grant schema-level privileges to a role.

        :param grantee: The role to grant privileges to (e.g., service principal UUID).
        :param privileges: List of SchemaPrivilege to grant.
        :param schemas: List of schema names to grant privileges on.

        Example:
            client.grant_schema(
                grantee="app-sp-uuid",
                privileges=[SchemaPrivilege.USAGE, SchemaPrivilege.CREATE],
                schemas=["drizzle", "ai_chatbot", "public"],
            )
        """
        privs = self._format_privileges_sql(privileges)
        privs_str = self._format_privileges_str(privileges)

        for schema in schemas:
            query = sql.SQL("GRANT {privs} ON SCHEMA {schema} TO {grantee}").format(
                privs=privs,
                schema=sql.Identifier(schema),
                grantee=sql.Identifier(grantee),
            )
            self._execute_composed(query)
            logger.info("Granted %s on schema '%s' to '%s'", privs_str, schema, grantee)

    def grant_all_tables_in_schema(
        self,
        grantee: str,
        privileges: Sequence[TablePrivilege],
        schemas: Sequence[str],
    ) -> None:
        """
        Grant table-level privileges on ALL tables in the specified schemas.

        :param grantee: The role to grant privileges to (e.g., service principal UUID).
        :param privileges: List of TablePrivilege to grant.
        :param schemas: List of schema names whose tables will receive the privileges.

        Example:
            client.grant_all_tables_in_schema(
                grantee="app-sp-uuid",
                privileges=[TablePrivilege.SELECT, TablePrivilege.INSERT, TablePrivilege.UPDATE],
                schemas=["drizzle", "ai_chatbot"],
            )
        """
        privs = self._format_privileges_sql(privileges)
        privs_str = self._format_privileges_str(privileges)

        for schema in schemas:
            query = sql.SQL("GRANT {privs} ON ALL TABLES IN SCHEMA {schema} TO {grantee}").format(
                privs=privs,
                schema=sql.Identifier(schema),
                grantee=sql.Identifier(grantee),
            )
            self._execute_composed(query)
            logger.info(
                "Granted %s on all tables in schema '%s' to '%s'",
                privs_str,
                schema,
                grantee,
            )

    def grant_table(
        self,
        grantee: str,
        privileges: Sequence[TablePrivilege],
        tables: Sequence[str],
    ) -> None:
        """
        Grant table-level privileges on specific tables.

        :param grantee: The role to grant privileges to (e.g., service principal UUID).
        :param privileges: List of TablePrivilege to grant.
        :param tables: List of table names (can be schema-qualified like "public.users").

        Example:
            client.grant_table(
                grantee="app-sp-uuid",
                privileges=[TablePrivilege.SELECT, TablePrivilege.INSERT, TablePrivilege.UPDATE],
                tables=[
                    "public.checkpoint_migrations",
                    "public.checkpoint_writes",
                    "public.checkpoints",
                    "public.checkpoint_blobs",
                ],
            )
        """
        privs = self._format_privileges_sql(privileges)
        privs_str = self._format_privileges_str(privileges)

        for table in tables:
            # Handle schema.table format
            if "." in table:
                schema_name, table_name = table.split(".", 1)
                table_ident = sql.Identifier(schema_name, table_name)
            else:
                table_ident = sql.Identifier(table)

            query = sql.SQL("GRANT {privs} ON {table} TO {grantee}").format(
                privs=privs,
                table=table_ident,
                grantee=sql.Identifier(grantee),
            )
            self._execute_composed(query)
            logger.info("Granted %s on table '%s' to '%s'", privs_str, table, grantee)
