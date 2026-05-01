"""Async database connection pool for Lakebase persistence."""

import logging
import os
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy
from databricks_ai_bridge.long_running.models import AGENT_DB_SCHEMA, Base

logger = logging.getLogger(__name__)

_session_factory: async_sessionmaker[AsyncSession] | None = None
_engine = None
_lakebase: AsyncLakebaseSQLAlchemy | None = None
_initialized = False


def is_db_configured() -> bool:
    """Check if database is configured via provisioned instance or autoscaling."""
    return bool(
        os.getenv("LAKEBASE_AUTOSCALING_ENDPOINT")
        or os.getenv("LAKEBASE_INSTANCE_NAME")
        or (os.getenv("LAKEBASE_AUTOSCALING_PROJECT") and os.getenv("LAKEBASE_AUTOSCALING_BRANCH"))
    )


async def init_db(
    *,
    instance_name: str | None = None,
    autoscaling_endpoint: str | None = None,
    project: str | None = None,
    branch: str | None = None,
    pool_size: int = 10,
    max_overflow: int = 0,
    db_statement_timeout_ms: int = 5000,
) -> None:
    """Create engine, schema, and tables. Call once on app startup."""
    global _session_factory, _engine, _lakebase, _initialized

    if _initialized:
        logger.debug("[DB] Already initialized, skipping")
        return

    lakebase_kwargs: dict = {
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_pre_ping": True,
    }
    if instance_name:
        lakebase_kwargs["instance_name"] = instance_name
    if autoscaling_endpoint:
        lakebase_kwargs["autoscaling_endpoint"] = autoscaling_endpoint
    if project:
        lakebase_kwargs["project"] = project
    if branch:
        lakebase_kwargs["branch"] = branch

    _lakebase = AsyncLakebaseSQLAlchemy(**lakebase_kwargs)
    _engine = _lakebase.engine

    @event.listens_for(_engine.sync_engine, "checkout")
    def _set_statement_timeout(dbapi_conn, connection_record, connection_proxy):
        cursor = dbapi_conn.cursor()
        cursor.execute(f"SET statement_timeout = {int(db_statement_timeout_ms)}")
        cursor.close()

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )

    # AGENT_DB_SCHEMA is a trusted constant ("agent_server"), not user input.
    async with _engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {AGENT_DB_SCHEMA}"))
        await conn.run_sync(Base.metadata.create_all)

    # Idempotent migration for tables created by earlier versions: add any
    # columns introduced for durable-resume support. Each statement runs in
    # its own transaction so an InsufficientPrivilege on one ALTER (another
    # pod's SP owns the table but the schema is already migrated) doesn't
    # poison the rest. A single mega-transaction would abort entirely on the
    # first owner-check failure even with IF NOT EXISTS.
    migration_stmts = (
        f"ALTER TABLE {AGENT_DB_SCHEMA}.responses "
        "ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ",
        f"ALTER TABLE {AGENT_DB_SCHEMA}.responses "
        "ADD COLUMN IF NOT EXISTS attempt_number INTEGER NOT NULL DEFAULT 1",
        f"ALTER TABLE {AGENT_DB_SCHEMA}.responses ADD COLUMN IF NOT EXISTS original_request TEXT",
        f"ALTER TABLE {AGENT_DB_SCHEMA}.messages "
        "ADD COLUMN IF NOT EXISTS attempt_number INTEGER NOT NULL DEFAULT 1",
        f"CREATE INDEX IF NOT EXISTS idx_responses_stale "
        f"ON {AGENT_DB_SCHEMA}.responses (status, heartbeat_at) "
        "WHERE status = 'in_progress'",
    )
    skipped_migrations: list[str] = []
    for stmt in migration_stmts:
        try:
            async with _engine.begin() as conn:
                await conn.execute(text(stmt))
        except Exception as exc:
            msg = str(exc).lower()
            if "insufficientprivilege" in msg or "must be owner" in msg:
                skipped_migrations.append(stmt.split("\n")[0])
                continue
            raise

    _initialized = True
    if skipped_migrations:
        # WARN-level summary: if the DB was previously migrated by another SP
        # this is fine, but if it's genuinely a new table and our SP lacks
        # ALTER, claim/heartbeat queries will fail later with a confusing
        # "column does not exist" — surface it clearly at startup.
        logger.warning(
            "[DB] Skipped %d durability migration(s) due to insufficient "
            "privilege — assuming table was already migrated by another "
            "service principal. Crash-resume will fail with 'column does "
            "not exist' if this assumption is wrong. Skipped: %s",
            len(skipped_migrations),
            ", ".join(skipped_migrations),
        )
    logger.info("[DB] Engine and schema ready")


async def dispose_db() -> None:
    """Dispose engine and clear registration. Call on app shutdown."""
    global _session_factory, _engine, _lakebase, _initialized

    if _engine is not None:
        await _engine.dispose()
        logger.info("[DB] Engine disposed")
    _session_factory = None
    _engine = None
    _lakebase = None
    _initialized = False


def session_scope():
    """Return an async context manager yielding a session from the pool."""

    @asynccontextmanager
    async def _session_cm():
        if _session_factory is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        async with _session_factory() as session:
            yield session

    return _session_cm()
