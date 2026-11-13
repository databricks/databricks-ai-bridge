"""Lakebase registry with a Postgres advisory lock shared by App replicas."""

import hashlib
import os
import re
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from databricks_agentkit.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_USERNAME_ENV,
)


class SessionBusy(RuntimeError):
    pass


class ExecutionIdentityConflict(ValueError):
    pass


class Registry:
    def __init__(
        self,
        workspace,
        namespace: str,
        *,
        endpoint=None,
        host=None,
        database=None,
        username=None,
        schema=None,
    ):
        self.workspace = workspace
        self.namespace = namespace
        self.schema = schema or os.getenv("SANDBOX_SCHEMA", "sandbox_runtime_prototype")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", self.schema):
            raise ValueError("Invalid sandbox runtime schema")
        managed = {
            key: os.getenv(key)
            for key in (
                RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
                RUNTIME_STORE_DATABASE_ENV,
                RUNTIME_STORE_USERNAME_ENV,
            )
        }
        if endpoint is None and any(managed.values()):
            missing = [key for key, value in managed.items() if not value]
            if missing:
                raise ValueError(
                    "Managed Runtime Store configuration is missing: " + ", ".join(missing)
                )
            branch = managed[RUNTIME_STORE_LAKEBASE_BRANCH_ENV]
            endpoints = workspace.postgres.list_endpoints(parent=branch)
            writable = next(
                (
                    item
                    for item in endpoints
                    if item.status and "READ_WRITE" in str(item.status.endpoint_type)
                ),
                None,
            )
            if not writable or not writable.status.hosts or not writable.status.hosts.host:
                raise ValueError("Managed Runtime Store has no ready READ_WRITE endpoint")
            self.endpoint, self.host = writable.name, writable.status.hosts.host
            # Never mix inherited PG* coordinates from a legacy App resource with
            # the App-owned backend returned by the Runtime Store API.
            self.database = managed[RUNTIME_STORE_DATABASE_ENV]
            self.username = managed[RUNTIME_STORE_USERNAME_ENV]
        else:
            self.endpoint = endpoint or os.environ["LAKEBASE_ENDPOINT"]
            self.host = (
                host
                or os.getenv("PGHOST")
                or workspace.postgres.get_endpoint(self.endpoint).status.hosts.host
            )
            self.database = database or os.getenv("PGDATABASE", "databricks_postgres")
            self.username = username or os.getenv("PGUSER") or workspace.current_user.me().user_name

    def connect(self):
        credential = self.workspace.postgres.generate_database_credential(endpoint=self.endpoint)
        return psycopg.Connection[dict].connect(
            host=self.host,
            dbname=self.database,
            user=self.username,
            password=credential.token,
            sslmode="require",
            connect_timeout=20,
            autocommit=True,
            row_factory=dict_row,
            options=f"-csearch_path={self.schema}",
        )

    def initialize(self):
        with self.connect() as db:
            db.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema))
            )
            db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    namespace TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    sandbox_id TEXT,
                    source_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    bootstrap_metrics JSONB,
                    source_revision TEXT, execution_principal TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY(namespace, session_key)
                )
            """)

    def runtime(self, session_id):
        """Read runtime state without reserving a sandbox or acquiring ownership."""
        key = self.session_key(session_id)
        with self.connect() as db:
            row = Session(db, self.namespace, key).get()
        return {
            "execution_principal": row.get("execution_principal") if row else None,
            "sandbox": row,
        }

    def runtimes(self):
        """List this environment's session mappings without touching compute."""
        with self.connect() as db:
            return db.execute(
                "SELECT * FROM sessions WHERE namespace=%s ORDER BY updated_at DESC LIMIT 100",
                (self.namespace,),
            ).fetchall()

    def execution_principals(self):
        with self.connect() as db:
            return [
                row["execution_principal"]
                for row in db.execute(
                    "SELECT DISTINCT execution_principal FROM sessions WHERE namespace=%s "
                    "AND execution_principal IS NOT NULL",
                    (self.namespace,),
                ).fetchall()
            ]

    def grant_app_access(self, principal):
        """Reconcile caller-owned prototype objects after local-first setup.

        App-owned objects already have access.
        """
        with self.connect() as db:
            schema = db.execute(
                "SELECT pg_get_userbyid(nspowner) = current_user AS owned "
                "FROM pg_namespace WHERE nspname=%s",
                (self.schema,),
            ).fetchone()
            if schema and schema["owned"]:
                db.execute(
                    sql.SQL("GRANT USAGE, CREATE ON SCHEMA {} TO {}").format(
                        sql.Identifier(self.schema), sql.Identifier(principal)
                    )
                )
            tables = db.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname=%s AND tableowner=current_user",
                (self.schema,),
            ).fetchall()
            for table in tables:
                db.execute(
                    sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON {} TO {}").format(
                        sql.Identifier(table["tablename"]), sql.Identifier(principal)
                    )
                )

    @contextmanager
    def session(self, session_id: str):
        key = self.session_key(session_id)
        lock_id = int.from_bytes(bytes.fromhex(key[:16]), "big", signed=True)
        with self.connect() as db:
            if not db.execute("SELECT pg_try_advisory_lock(%s) AS acquired", (lock_id,)).fetchone()[
                "acquired"
            ]:
                raise SessionBusy("This session already has an invocation in progress; retry later")
            try:
                yield Session(db, self.namespace, key)
            finally:
                db.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))

    def session_key(self, session_id):
        return hashlib.sha256(f"{self.namespace}\0{session_id}".encode()).hexdigest()


class Session:
    def __init__(self, db, namespace: str, key: str):
        self.db, self.namespace, self.key = db, namespace, key

    def get(self):
        return self.db.execute(
            "SELECT * FROM sessions WHERE namespace = %s AND session_key = %s",
            (self.namespace, self.key),
        ).fetchone()

    def bind_owner(self, principal):
        # Reserve ownership in the mapping itself, even before create returns an ID.
        self.db.execute(
            "INSERT INTO sessions(namespace,session_key,source_hash,status,execution_principal) "
            "VALUES (%s,%s,'','pending',%s) ON CONFLICT(namespace,session_key) DO UPDATE SET "
            "execution_principal=COALESCE(sessions.execution_principal,EXCLUDED.execution_principal)",
            (self.namespace, self.key, principal),
        )
        owner = self.get()["execution_principal"]
        if owner != principal:
            raise ExecutionIdentityConflict(
                f"This session sandbox belongs to {owner}; current identity is {principal}"
            )

    def save(self, source_hash, sandbox_id, status, metrics=None, *, source_revision=None):
        from psycopg.types.json import Jsonb

        self.db.execute(
            """
            INSERT INTO sessions
                (namespace, session_key, source_hash, sandbox_id, status, bootstrap_metrics, source_revision)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (namespace, session_key) DO UPDATE SET
                source_hash = EXCLUDED.source_hash, sandbox_id = EXCLUDED.sandbox_id,
                source_revision = COALESCE(EXCLUDED.source_revision, sessions.source_revision),
                status = EXCLUDED.status,
                bootstrap_metrics = COALESCE(sessions.bootstrap_metrics, '{}'::jsonb) || COALESCE(EXCLUDED.bootstrap_metrics, '{}'::jsonb),
                updated_at = now()
            """,
            (
                self.namespace,
                self.key,
                source_hash,
                sandbox_id,
                status,
                Jsonb(metrics) if metrics is not None else None,
                source_revision,
            ),
        )
