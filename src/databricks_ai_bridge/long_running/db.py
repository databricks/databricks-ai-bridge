"""Async database connection pool for Lakebase persistence."""

import logging
import os
from contextlib import asynccontextmanager

try:
    from sqlalchemy import event, text
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
except ImportError as e:
    raise ImportError(
        "Long-running server requires databricks-ai-bridge[server]. "
        "Please install with: pip install databricks-ai-bridge[server]"
    ) from e

from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy
from databricks_ai_bridge.long_running.models import AGENT_DB_SCHEMA, Base

logger = logging.getLogger(__name__)

_session_factory: async_sessionmaker[AsyncSession] | None = None
_engine = None
_lakebase: AsyncLakebaseSQLAlchemy | None = None


def is_db_configured() -> bool:
    """Check if database is configured via provisioned instance or autoscaling."""
    return bool(
        os.getenv("LAKEBASE_INSTANCE_NAME")
        or (os.getenv("LAKEBASE_AUTOSCALING_PROJECT") and os.getenv("LAKEBASE_AUTOSCALING_BRANCH"))
    )


async def init_db(
    *,
    instance_name: str | None = None,
    project: str | None = None,
    branch: str | None = None,
    pool_size: int = 10,
    max_overflow: int = 0,
    db_statement_timeout_ms: int = 5000,
) -> None:
    """Create engine, schema, and tables. Call on app startup."""
    global _session_factory, _engine, _lakebase

    lakebase_kwargs: dict = {
        "pool_size": pool_size,
        "max_overflow": max_overflow,
        "pool_pre_ping": True,
    }
    if instance_name:
        lakebase_kwargs["instance_name"] = instance_name
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

    logger.info("[DB] Engine and schema ready")


async def dispose_db() -> None:
    """Dispose engine and clear registration. Call on app shutdown."""
    global _session_factory, _engine, _lakebase

    if _engine is not None:
        await _engine.dispose()
        logger.info("[DB] Engine disposed")
    _session_factory = None
    _engine = None
    _lakebase = None


def get_async_session():
    """Return an async context manager yielding a session from the pool."""

    @asynccontextmanager
    async def _session_cm():
        if _session_factory is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        async with _session_factory() as session:
            yield session

    return _session_cm()
