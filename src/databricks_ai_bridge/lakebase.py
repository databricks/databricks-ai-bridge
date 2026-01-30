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
    """PostgreSQL table privileges for GRANT statements.

    See: https://www.postgresql.org/docs/16/sql-grant.html
    """

    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    TRUNCATE = "TRUNCATE"
    REFERENCES = "REFERENCES"
    TRIGGER = "TRIGGER"
    ALL = "ALL"  # Renders as ALL PRIVILEGES


class SchemaPrivilege(str, Enum):
    """PostgreSQL schema privileges for GRANT statements.

    See: https://www.postgresql.org/docs/current/sql-grant.html
    """

    USAGE = "USAGE"
    CREATE = "CREATE"
    ALL = "ALL"  # Renders as ALL PRIVILEGES


class SequencePrivilege(str, Enum):
    """PostgreSQL sequence privileges for GRANT statements.

    See: https://www.postgresql.org/docs/current/sql-grant.html
    """

    USAGE = "USAGE"
    SELECT = "SELECT"
    UPDATE = "UPDATE"
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
        self.token_cache_duration_seconds: int = token_cache_duration_seconds

        # If input is hostname (e.g., from Databricks Apps valueFrom resolution)
        # resolve to lakebase name
        if self._is_hostname(instance_name):
            # Input is a hostname - resolve to instance name
            self.instance_name, self.host = self._resolve_from_hostname(instance_name)
        else:
            # Input is an instance name
            self.instance_name = instance_name
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

            self.host = resolved_host

        self.username: str = self._infer_username()

        self._cached_token: str | None = None
        self._cache_ts: float | None = None

    @staticmethod
    def _is_hostname(value: str) -> bool:
        """Check if the value looks like a Lakebase hostname rather than an instance name."""
        # Hostname pattern: instance-{uuid}.database.{env}.cloud.databricks.com
        # or similar patterns containing ".database." and ending with a domain
        return ".database." in value and (value.endswith(".com") or value.endswith(".net"))

    def _resolve_from_hostname(self, hostname: str) -> tuple[str, str]:
        """
        Resolve instance name from a hostname by listing database instances.

        Args:
            hostname: The database hostname (e.g., from Databricks Apps valueFrom: "database")

        Returns:
            Tuple of (instance_name, host)

        Raises:
            ValueError: If no matching instance is found
        """
        try:
            instances = list(self.workspace_client.database.list_database_instances())
        except Exception as exc:
            raise ValueError(
                f"Unable to list database instances to resolve hostname '{hostname}'. "
                "Ensure you have access to database instances."
            ) from exc

        # Find the instance that matches this hostname
        for instance in instances:
            rw_dns = getattr(instance, "read_write_dns", None)
            ro_dns = getattr(instance, "read_only_dns", None)

            if hostname in (rw_dns, ro_dns):
                instance_name = getattr(instance, "name", None)
                if not instance_name:
                    raise ValueError(
                        f"Found matching instance for hostname '{hostname}' "
                        "but instance name is not available."
                    )
                return instance_name, hostname

        raise ValueError(
            f"Unable to find database instance matching hostname '{hostname}'. "
            "Ensure the hostname is correct and the instance exists."
        )

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

    Example (simple):
        client = LakebaseClient(instance_name="my-lakebase")
        client.execute("SELECT * FROM users")
        client.create_role("user@example.com", "USER")
        client.close()

    Example (end-to-end permission setup for an application):
        from databricks_ai_bridge.lakebase import (
            LakebaseClient,
            SchemaPrivilege,
            SequencePrivilege,
            TablePrivilege,
        )

        # Create client and set up permissions for a service principal
        with LakebaseClient(instance_name="my-lakebase") as client:
            # 1. Create a PostgreSQL role for the service principal
            client.create_role("my-app-service-principal-uuid", "SERVICE_PRINCIPAL")

            # 2. Grant schema access
            client.grant_schema(
                grantee="my-app-service-principal-uuid",
                privileges=[SchemaPrivilege.USAGE, SchemaPrivilege.CREATE],
                schemas=["public", "app_schema"],
            )

            # 3. Grant table privileges on all tables in the schema
            client.grant_all_tables_in_schema(
                grantee="my-app-service-principal-uuid",
                privileges=[TablePrivilege.SELECT, TablePrivilege.INSERT,
                            TablePrivilege.UPDATE, TablePrivilege.DELETE],
                schemas=["public", "app_schema"],
            )

            # 4. Grant sequence privileges (needed for INSERT with SERIAL columns)
            client.grant_all_sequences_in_schema(
                grantee="my-app-service-principal-uuid",
                privileges=[SequencePrivilege.USAGE, SequencePrivilege.SELECT],
                schemas=["public", "app_schema"],
            )

    Example (bring your own pool):
        pool = LakebasePool(instance_name="my-lakebase", max_size=20)
        client = LakebaseClient(pool=pool)
        client.execute("SELECT * FROM users")
        client.close()
        pool.close()  # Pool is managed externally
    """

    def __init__(
        self,
        *,
        pool: LakebasePool | None = None,
        instance_name: str | None = None,
        **pool_kwargs: Any,
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

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close the client."""
        self.close()
        return False

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
        :raises ValueError: If identity_name is empty.
        :raises PermissionError: If the caller lacks required permissions.

        Example:
            # Create a role for a service principal
            client.create_role(
                "service-principal-uuid",
                "SERVICE_PRINCIPAL"
            )
        """
        if not identity_name or not identity_name.strip():
            raise ValueError(
                "identity_name cannot be empty. Provide a valid Databricks identity "
                "(user email, service principal UUID, or group ID)."
            )

        # Create the databricks_auth extension. Each Postgres database must have its own extension.
        # https://docs.databricks.com/aws/en/oltp/instances/pg-roles#create-postgres-roles-and-grant-privileges-for-databricks-identities
        try:
            if ensure_extension:
                self.execute("CREATE EXTENSION IF NOT EXISTS databricks_auth;")

            query = f"SELECT databricks_create_role(%s, '{identity_type}');"
            return self.execute(query, (identity_name,))
        except psycopg.errors.DuplicateObject:
            logger.info("Role '%s' already exists, skipping creation.", identity_name)
            return None
        except psycopg.errors.InvalidParameterValue as e:
            raise ValueError(
                f"Identity '{identity_name}' not found in the Databricks workspace. "
                f"Ensure the {identity_type.lower().replace('_', ' ')} exists in your "
                f"Databricks workspace before creating a role. "
                f"Original error: {e}"
            ) from e
        except psycopg.errors.InsufficientPrivilege as e:
            raise PermissionError(
                f"Insufficient privileges to create role '{identity_name}'. "
                f"Ensure you have 'CAN MANAGE' permission on the Lakebase instance. "
                f"Original error: {e}"
            ) from e
        except psycopg.errors.UndefinedFunction as e:
            raise RuntimeError(
                f"The databricks_create_role function is not available. "
                f"Ensure the databricks_auth extension is properly installed. "
                f"See https://docs.databricks.com/aws/en/oltp/instances/pg-roles?language=PostgreSQL. "
                f"Original error: {e}"
            ) from e

    def _format_privileges_str(
        self,
        privileges: Sequence[TablePrivilege]
        | Sequence[SchemaPrivilege]
        | Sequence[SequencePrivilege],
    ) -> str:
        """Format privileges as a string for logging."""
        privilege_values = [p.value for p in privileges]
        if "ALL" in privilege_values:
            return "ALL PRIVILEGES"
        return ", ".join(privilege_values)

    def _format_privileges_sql(
        self,
        privileges: Sequence[TablePrivilege]
        | Sequence[SchemaPrivilege]
        | Sequence[SequencePrivilege],
    ) -> sql.SQL | sql.Composed:
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

    def _validate_non_empty(self, items: Sequence[Any], param_name: str) -> None:
        """Validate that a sequence is not empty."""
        if not items:
            raise ValueError(
                f"'{param_name}' cannot be empty. Provide at least one {param_name[:-1]}."
            )

    def _execute_grant(self, query: sql.Composed, operation_desc: str, grantee: str) -> None:
        """Execute a GRANT query with helpful error handling.

        :param query: The composed SQL query to execute.
        :param operation_desc: Description of the operation for error messages.
        :param grantee: The role being granted privileges (for error messages).
        :raises PermissionError: If the user lacks required permissions.
        :raises ValueError: If the target object does not exist.
        """
        try:
            self._execute_composed(query)
        except psycopg.errors.InsufficientPrivilege as e:
            raise PermissionError(
                f"Insufficient privileges to {operation_desc}. "
                f"Ensure you have 'CAN MANAGE' permission on the Lakebase instance "
                f"and appropriate ownership or GRANT OPTION on the target objects. "
                f"Original error: {e}"
            ) from e
        except psycopg.errors.UndefinedObject as e:
            raise ValueError(
                f"Failed to {operation_desc}: object does not exist. "
                f"Ensure the schema/table/sequence exists and the role '{grantee}' "
                f"has been created. Original error: {e}"
            ) from e
        except psycopg.errors.InvalidSchemaName as e:
            raise ValueError(
                f"Failed to {operation_desc}: schema does not exist. "
                f"Verify the schema name is correct. Original error: {e}"
            ) from e
        except psycopg.errors.UndefinedTable as e:
            raise ValueError(
                f"Failed to {operation_desc}: table does not exist. "
                f"Verify the table name is correct. Original error: {e}"
            ) from e
        except psycopg.errors.InvalidGrantor as e:
            raise PermissionError(
                f"Cannot {operation_desc}: you must be the owner or have "
                f"GRANT OPTION for these privileges. Original error: {e}"
            ) from e

    def _parse_table_identifier(self, table: str) -> sql.Identifier:
        """Parse a table name into a SQL identifier.

        :param table: Table name, optionally schema-qualified (e.g., "public.users").
        :return: A psycopg sql.Identifier for safe SQL composition.
        :raises ValueError: If the table name format is invalid.
        """
        if not table or not table.strip():
            raise ValueError(
                "Table name cannot be empty. Provide a valid table name "
                "(e.g., 'users' or 'public.users')."
            )

        # Handle schema.table format
        if "." in table:
            parts = table.split(".", 1)
            schema_name, table_name = parts[0].strip(), parts[1].strip()
            if not schema_name or not table_name:
                raise ValueError(
                    f"Invalid table format '{table}'. Expected 'schema.table' "
                    f"(e.g., 'public.users') or just 'table_name'."
                )
            return sql.Identifier(schema_name, table_name)
        return sql.Identifier(table.strip())

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
        :raises ValueError: If schemas or privileges is empty.
        :raises PermissionError: If the caller lacks required permissions.

        Example:
            client.grant_schema(
                grantee="app-sp-uuid",
                privileges=[SchemaPrivilege.USAGE, SchemaPrivilege.CREATE],
                schemas=["drizzle", "ai_chatbot", "public"],
            )
        """
        self._validate_non_empty(schemas, "schemas")
        self._validate_non_empty(privileges, "privileges")

        privs = self._format_privileges_sql(privileges)
        privs_str = self._format_privileges_str(privileges)

        for schema in schemas:
            query = sql.SQL("GRANT {privs} ON SCHEMA {schema} TO {grantee}").format(
                privs=privs,
                schema=sql.Identifier(schema),
                grantee=sql.Identifier(grantee),
            )
            self._execute_grant(query, f"grant {privs_str} on schema '{schema}'", grantee)
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
        :raises ValueError: If schemas or privileges is empty.
        :raises PermissionError: If the caller lacks required permissions.

        Example:
            client.grant_all_tables_in_schema(
                grantee="app-sp-uuid",
                privileges=[TablePrivilege.SELECT, TablePrivilege.INSERT, TablePrivilege.UPDATE],
                schemas=["drizzle", "ai_chatbot"],
            )
        """
        self._validate_non_empty(schemas, "schemas")
        self._validate_non_empty(privileges, "privileges")

        privs = self._format_privileges_sql(privileges)
        privs_str = self._format_privileges_str(privileges)

        for schema in schemas:
            query = sql.SQL("GRANT {privs} ON ALL TABLES IN SCHEMA {schema} TO {grantee}").format(
                privs=privs,
                schema=sql.Identifier(schema),
                grantee=sql.Identifier(grantee),
            )
            self._execute_grant(
                query, f"grant {privs_str} on all tables in schema '{schema}'", grantee
            )
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
        :raises ValueError: If tables or privileges is empty, or table name format is invalid.
        :raises PermissionError: If the caller lacks required permissions.

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
        self._validate_non_empty(tables, "tables")
        self._validate_non_empty(privileges, "privileges")

        privs = self._format_privileges_sql(privileges)
        privs_str = self._format_privileges_str(privileges)

        for table in tables:
            table_ident = self._parse_table_identifier(table)

            query = sql.SQL("GRANT {privs} ON {table} TO {grantee}").format(
                privs=privs,
                table=table_ident,
                grantee=sql.Identifier(grantee),
            )
            self._execute_grant(query, f"grant {privs_str} on table '{table}'", grantee)
            logger.info("Granted %s on table '%s' to '%s'", privs_str, table, grantee)

    def grant_all_sequences_in_schema(
        self,
        grantee: str,
        privileges: Sequence[SequencePrivilege],
        schemas: Sequence[str],
    ) -> None:
        """
        Grant sequence-level privileges on ALL sequences in the specified schemas.

        :param grantee: The role to grant privileges to (e.g., service principal UUID).
        :param privileges: List of SequencePrivilege to grant (USAGE, SELECT, UPDATE, ALL).
        :param schemas: List of schema names whose sequences will receive the privileges.
        :raises ValueError: If schemas or privileges is empty.
        :raises PermissionError: If the caller lacks required permissions.

        Example:
            client.grant_all_sequences_in_schema(
                grantee="app-sp-uuid",
                privileges=[SequencePrivilege.USAGE, SequencePrivilege.SELECT, SequencePrivilege.UPDATE],
                schemas=["public", "app_schema"],
            )
        """
        self._validate_non_empty(schemas, "schemas")
        self._validate_non_empty(privileges, "privileges")

        privs = self._format_privileges_sql(privileges)
        privs_str = self._format_privileges_str(privileges)

        for schema in schemas:
            query = sql.SQL(
                "GRANT {privs} ON ALL SEQUENCES IN SCHEMA {schema} TO {grantee}"
            ).format(
                privs=privs,
                schema=sql.Identifier(schema),
                grantee=sql.Identifier(grantee),
            )
            self._execute_grant(
                query,
                f"grant {privs_str} on all sequences in schema '{schema}'",
                grantee,
            )
            logger.info(
                "Granted %s on all sequences in schema '%s' to '%s'",
                privs_str,
                schema,
                grantee,
            )
