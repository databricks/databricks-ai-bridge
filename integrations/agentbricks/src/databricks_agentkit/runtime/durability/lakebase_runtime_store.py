"""Lakebase-backed Runtime Store with heartbeat-based recovery."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from sqlalchemy import URL, event, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from databricks_agentkit.runtime.durability.store import DurableRuntimeStore
from databricks_agentkit.runtime.store import DEFAULT_RUNTIME_STORE_SCHEMA
from databricks_agentkit.runtime.types import (
    Invocation,
    InvocationConflictError,
    InvocationEvent,
    InvocationStatus,
    JsonObject,
    JsonValue,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


class _AsyncLakebase(Protocol):
    engine: AsyncEngine

    async def create_schema(self) -> None: ...


_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN_CACHE_SECONDS = 15 * 60
_POOL_RECYCLE_SECONDS = 14 * 60


def _serialize_json_value(value: JsonValue) -> str:
    return json.dumps(value, allow_nan=False)


def _validate_invocation_id(invocation_id: str) -> None:
    if not invocation_id:
        raise ValueError("invocation_id must not be empty")


def _validate_session_id(session_id: str | None) -> None:
    if session_id is not None and not session_id:
        raise ValueError("session_id must not be empty")


def _validate_read_scope(invocation_id: str | None, session_id: str | None) -> None:
    if (invocation_id is None) == (session_id is None):
        raise ValueError("exactly one of invocation_id or session_id must be provided")
    if invocation_id is not None:
        _validate_invocation_id(invocation_id)
    if session_id is not None:
        _validate_session_id(session_id)


def _runnable_queued_predicate(table: str, alias: str) -> str:
    """Return the queue-head predicate used after taking the session advisory lock."""
    return f"""
        (
            {alias}.session_id IS NULL
            OR (
                NOT EXISTS (
                    SELECT 1
                    FROM {table} AS active
                    WHERE active.session_id={alias}.session_id
                      AND active.status='ACTIVE'
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM {table} AS earlier
                    WHERE earlier.session_id={alias}.session_id
                      AND earlier.status='QUEUED'
                      AND (earlier.session_sequence_number, earlier.invocation_id)
                          < ({alias}.session_sequence_number, {alias}.invocation_id)
                )
            )
        )
    """


def _same_affinity_predicate(alias: str) -> str:
    return f"{alias}.recovery_affinity IS NOT DISTINCT FROM CAST(:recovery_affinity AS TEXT)"


def _claim_owner_predicate(alias: str) -> str:
    return f"({alias}.worker_id IS NULL OR {alias}.worker_id = CAST(:worker_id AS UUID))"


def _stale_heartbeat_predicate(alias: str) -> str:
    return (
        f"({alias}.heartbeat_at IS NULL OR {alias}.heartbeat_at < "
        "clock_timestamp() - (:stale * INTERVAL '1 second'))"
    )


class _AppsPostgresLakebase:
    """SQLAlchemy connection for a Databricks Apps Postgres resource.

    Apps injects the selected resource's connection coordinates through the standard ``PG*``
    variables. The endpoint resource path is kept separately because OAuth credentials must be
    refreshed through the Databricks Postgres API.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        host: str,
        port: int,
        database: str,
        username: str,
        sslmode: str,
        workspace_client: WorkspaceClient | None,
        schema: str,
    ) -> None:
        if not endpoint or not host or not database or not username:
            raise ValueError("endpoint, host, database, and username must not be empty")
        if port <= 0:
            raise ValueError("port must be positive")

        if workspace_client is None:
            from databricks.sdk import WorkspaceClient

            workspace_client = WorkspaceClient()

        self._endpoint = endpoint
        self._workspace_client = workspace_client
        self._schema = schema
        self._token: str | None = None
        self._token_time = 0.0
        self._token_lock = Lock()

        url = URL.create(
            drivername="postgresql+psycopg",
            username=username,
            host=host,
            port=port,
            database=database,
        )
        self.engine: AsyncEngine = create_async_engine(
            url,
            pool_recycle=_POOL_RECYCLE_SECONDS,
            pool_pre_ping=True,
            connect_args={"sslmode": sslmode},
        )

        @event.listens_for(self.engine.sync_engine, "do_connect")
        def inject_token(dialect, connection_record, args, params) -> None:
            params["password"] = self._get_token()

    async def create_schema(self) -> None:
        async with self.engine.begin() as connection:
            await connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self._schema}"))

    def _get_token(self) -> str:
        with self._token_lock:
            if self._token and time.monotonic() - self._token_time < _TOKEN_CACHE_SECONDS:
                return self._token
            credential = self._workspace_client.postgres.generate_database_credential(
                endpoint=self._endpoint
            )
            token = getattr(credential, "token", None)
            if not token:
                raise RuntimeError(
                    f"failed to generate a database credential for endpoint {self._endpoint!r}"
                )
            self._token = token
            self._token_time = time.monotonic()
            return token


class LakebaseDurableRuntimeStore(DurableRuntimeStore):
    """Persist invocation state, attempt leases, and ordered events in Lakebase.

    The ``invocations`` table is the source of truth for idempotency and lifecycle state. Conditional
    SQL updates claim an attempt and fence completion, failure, heartbeat, and event writes by its
    attempt number. The ``invocation_events`` table provides an ordered replay cursor. Because every
    app replica connects to the same schema, another replica can detect a stale heartbeat, claim the
    next attempt, and continue after process or pod loss.

    Each store reserves newly accepted queued work for its generated ``worker_id``. The identity is
    unique to the store process. ``recovery_affinity`` optionally groups workers that may take over
    one another's stale work; ``None`` is the default group.

    This store requires a Lakebase Postgres database. ``agentbricks deploy`` reuses or provisions a
    dedicated app-owned database, then assigns the app its own schema. ``agentbricks dev`` uses
    ``InMemoryRuntimeStore`` instead.
    """

    def __init__(
        self,
        *,
        recovery_affinity: str | None = None,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        database: str | None = None,
        username: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = DEFAULT_RUNTIME_STORE_SCHEMA,
        lakebase: _AsyncLakebase | None = None,
    ) -> None:
        """Create a store connection and establish this worker's ownership identity.

        A fresh UUID identifies this store instance for queued-work ownership. A non-empty
        ``recovery_affinity`` permits stale work transfer within that affinity group.
        """
        if not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError(f"invalid Runtime Store schema name: {schema!r}")
        if recovery_affinity is not None and not recovery_affinity:
            raise ValueError("recovery_affinity must not be empty")

        self._worker_id = str(uuid4())
        self._recovery_affinity = recovery_affinity

        if lakebase is None:
            from databricks_ai_bridge.lakebase import DEFAULT_DATABASE, AsyncLakebaseSQLAlchemy

            autoscaling_endpoint = autoscaling_endpoint or os.getenv(
                "LAKEBASE_AUTOSCALING_ENDPOINT"
            )
            if autoscaling_endpoint is None:
                project = project or os.getenv("LAKEBASE_AUTOSCALING_PROJECT")
                branch = branch or os.getenv("LAKEBASE_AUTOSCALING_BRANCH")
            lakebase = AsyncLakebaseSQLAlchemy(
                autoscaling_endpoint=autoscaling_endpoint,
                project=project,
                branch=branch,
                database=database or DEFAULT_DATABASE,
                username=username,
                workspace_client=workspace_client,
                schema=schema,
                pool_pre_ping=True,
            )

        self._lakebase = lakebase
        self._engine = lakebase.engine
        self._table = f"{schema}.invocations"
        self._events_table = f"{schema}.invocation_events"

    @property
    def worker_id(self) -> str:
        """Identity used to reserve and claim this store's queued invocations."""
        return self._worker_id

    @classmethod
    def from_app_resource(
        cls,
        *,
        endpoint: str,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        username: str | None = None,
        sslmode: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = DEFAULT_RUNTIME_STORE_SCHEMA,
    ) -> "LakebaseDurableRuntimeStore":
        """Use connection coordinates injected for a Databricks Apps Postgres resource."""
        if not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError(f"invalid Runtime Store schema name: {schema!r}")
        host = host or os.getenv("PGHOST")
        database = database or os.getenv("PGDATABASE")
        username = username or os.getenv("PGUSER")
        if port is None:
            raw_port = os.getenv("PGPORT")
            try:
                port = int(raw_port or "")
            except ValueError as exc:
                raise RuntimeError("PGPORT must be an integer") from exc
        missing = [
            name
            for name, value in {
                "PGHOST": host,
                "PGPORT": port,
                "PGDATABASE": database,
                "PGUSER": username,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Databricks Apps Postgres resource is missing: " + ", ".join(missing)
            )
        assert host is not None
        assert port is not None
        assert database is not None
        assert username is not None
        lakebase = _AppsPostgresLakebase(
            endpoint=endpoint,
            host=host,
            port=port,
            database=database,
            username=username,
            sslmode=sslmode or os.getenv("PGSSLMODE", "require"),
            workspace_client=workspace_client,
            schema=schema,
        )
        return cls(schema=schema, lakebase=lakebase)

    @classmethod
    def from_managed_runtime_store(
        cls,
        *,
        branch: str,
        database: str,
        username: str,
        workspace_client: WorkspaceClient | None = None,
        schema: str = DEFAULT_RUNTIME_STORE_SCHEMA,
    ) -> "LakebaseDurableRuntimeStore":
        """Connect using the backend coordinates returned by the Runtime Store API."""
        return cls(
            branch=branch,
            database=database,
            username=username,
            workspace_client=workspace_client,
            schema=schema,
        )

    async def initialize(self) -> None:
        await self._lakebase.create_schema()
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        invocation_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        session_sequence_number BIGINT,
                        status TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        worker_id UUID,
                        heartbeat_at TIMESTAMPTZ,
                        recovery_affinity TEXT,
                        request JSONB NOT NULL,
                        response JSONB,
                        CHECK (status IN ('QUEUED', 'ACTIVE', 'COMPLETED', 'FAILED'))
                    )
                    """
                )
            )
            await connection.execute(
                text(
                    f"""
                    ALTER TABLE {self._table}
                    ADD COLUMN IF NOT EXISTS session_id TEXT,
                    ADD COLUMN IF NOT EXISTS session_sequence_number BIGINT,
                    ADD COLUMN IF NOT EXISTS worker_id UUID,
                    ADD COLUMN IF NOT EXISTS recovery_affinity TEXT
                    """
                )
            )
            await connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._events_table} (
                        sequence_number BIGSERIAL PRIMARY KEY,
                        invocation_id TEXT NOT NULL
                            REFERENCES {self._table}(invocation_id) ON DELETE CASCADE,
                        attempt INTEGER NOT NULL,
                        event JSONB NOT NULL,
                        CHECK (jsonb_typeof(event) = 'object')
                    )
                    """
                )
            )
            await connection.execute(
                text(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS invocations_session_sequence_number_idx
                    ON {self._table} (session_id, session_sequence_number)
                    WHERE session_id IS NOT NULL
                    """
                )
            )
            await connection.execute(
                text(
                    f"""
                    CREATE UNIQUE INDEX IF NOT EXISTS invocations_active_session_idx
                    ON {self._table} (session_id)
                    WHERE session_id IS NOT NULL AND status='ACTIVE'
                    """
                )
            )
            await connection.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS invocation_events_replay_idx
                    ON {self._events_table} (invocation_id, sequence_number)
                    """
                )
            )
            await connection.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS invocations_recovery_idx
                    ON {self._table} (status, heartbeat_at)
                    WHERE status IN ('QUEUED', 'ACTIVE')
                    """
                )
            )

    async def close(self) -> None:
        await self._engine.dispose()

    async def accept(
        self,
        invocation_id: str,
        request: JsonValue,
        session_id: str | None = None,
    ) -> Invocation:
        _validate_invocation_id(invocation_id)
        _validate_session_id(session_id)
        serialized_request = _serialize_json_value(request)
        async with self._engine.begin() as connection:
            session_sequence_number = None
            if session_id is not None:
                await self._lock_session(connection, session_id)
                session_sequence_number = int(
                    (
                        await connection.execute(
                            text(
                                f"""
                                SELECT COALESCE(MAX(session_sequence_number), 0) + 1
                                FROM {self._table}
                                WHERE session_id=:session_id
                                """
                            ),
                            {"session_id": session_id},
                        )
                    ).scalar_one()
                )
            await connection.execute(
                text(
                    f"""
                    INSERT INTO {self._table}
                        (invocation_id, session_id, session_sequence_number, status,
                         worker_id, heartbeat_at, recovery_affinity, request)
                    VALUES
                        (:invocation_id, :session_id, :session_sequence_number, 'QUEUED',
                         CAST(:worker_id AS UUID), clock_timestamp(), :recovery_affinity,
                         CAST(:request AS JSONB))
                    ON CONFLICT (invocation_id) DO NOTHING
                    """
                ),
                {
                    "invocation_id": invocation_id,
                    "session_id": session_id,
                    "session_sequence_number": session_sequence_number,
                    "worker_id": self._worker_id,
                    "recovery_affinity": self._recovery_affinity,
                    "request": serialized_request,
                },
            )
            row = (
                (
                    await connection.execute(
                        text(
                            f"""
                        SELECT invocation_id, session_id, session_sequence_number, status, attempt,
                               worker_id, recovery_affinity,
                               request::TEXT AS request_json,
                               response::TEXT AS response_json
                        FROM {self._table}
                        WHERE invocation_id=:invocation_id
                        """
                        ),
                        {"invocation_id": invocation_id},
                    )
                )
                .mappings()
                .one()
            )

        state = self._to_invocation(row)
        if state.request != request or state.session_id != session_id:
            raise InvocationConflictError(
                f"invocation {invocation_id!r} was already accepted with a different session "
                "or request"
            )
        return state

    async def get(
        self,
        invocation_id: str | None = None,
        session_id: str | None = None,
    ) -> Invocation | None:
        _validate_read_scope(invocation_id, session_id)
        if invocation_id is not None:
            predicate = "invocation_id=:invocation_id"
            parameters = {"invocation_id": invocation_id}
            ordering = ""
        else:
            predicate = "session_id=:session_id AND status IN ('ACTIVE', 'QUEUED')"
            parameters = {"session_id": session_id}
            ordering = """
                ORDER BY CASE WHEN status='ACTIVE' THEN 0 ELSE 1 END,
                         session_sequence_number
                LIMIT 1
            """
        async with self._engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        text(
                            f"""
                        SELECT invocation_id, session_id, session_sequence_number, status, attempt,
                               worker_id, recovery_affinity,
                               request::TEXT AS request_json,
                               response::TEXT AS response_json
                        FROM {self._table}
                        WHERE {predicate}
                        {ordering}
                        """
                        ),
                        parameters,
                    )
                )
                .mappings()
                .one_or_none()
            )
        return self._to_invocation(row) if row is not None else None

    async def queued_invocation_ids(self) -> list[str]:
        async with self._engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        text(
                            f"""
                        SELECT invocation_id
                        FROM {self._table} AS candidate
                        WHERE candidate.status='QUEUED'
                          AND {_claim_owner_predicate("candidate")}
                          AND {_same_affinity_predicate("candidate")}
                          AND {_runnable_queued_predicate(self._table, "candidate")}
                        ORDER BY candidate.session_id NULLS FIRST,
                                 candidate.session_sequence_number NULLS FIRST,
                                 candidate.invocation_id
                        """
                        ),
                        {
                            "worker_id": self._worker_id,
                            "recovery_affinity": self._recovery_affinity,
                        },
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def stale_invocation_ids(self, stale_seconds: float) -> list[str]:
        async with self._engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        text(
                            f"""
                        SELECT invocation_id
                        FROM {self._table} AS candidate
                        WHERE {_same_affinity_predicate("candidate")}
                          AND {_stale_heartbeat_predicate("candidate")}
                          AND (
                              candidate.status='ACTIVE'
                              OR (
                                  candidate.status='QUEUED'
                                  AND candidate.worker_id IS NOT NULL
                                  AND {_runnable_queued_predicate(self._table, "candidate")}
                              )
                          )
                        ORDER BY heartbeat_at NULLS FIRST
                        """
                        ),
                        {
                            "stale": stale_seconds,
                            "recovery_affinity": self._recovery_affinity,
                        },
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def claim(self, invocation_id: str) -> Invocation | None:
        """Claim a newly queued invocation for its first attempt."""
        return await self._claim(invocation_id, stale_seconds=None)

    async def claim_recoverable(
        self,
        invocation_id: str,
        stale_seconds: float,
    ) -> Invocation | None:
        """Replace stale active work or take over abandoned queued work."""
        return await self._claim(invocation_id, stale_seconds=stale_seconds)

    async def _claim(
        self,
        invocation_id: str,
        *,
        stale_seconds: float | None,
    ) -> Invocation | None:
        _validate_invocation_id(invocation_id)
        parameters: dict[str, str | float | None] = {
            "invocation_id": invocation_id,
            "worker_id": self._worker_id,
            "recovery_affinity": self._recovery_affinity,
        }
        if stale_seconds is not None:
            # Ownership transfer is recovery even if the first attempt never started.
            attempt_expression = "GREATEST(target.attempt + 1, 2)"
            eligibility_filter_str = f"""
                {_same_affinity_predicate("target")}
                AND target.status IN ('ACTIVE', 'QUEUED')
                AND (
                    target.status='ACTIVE'
                    OR (
                        target.worker_id IS NOT NULL
                        AND {_runnable_queued_predicate(self._table, "target")}
                    )
                )
                AND {_stale_heartbeat_predicate("target")}
            """
            parameters["stale"] = stale_seconds
        else:
            attempt_expression = "target.attempt + 1"
            eligibility_filter_str = f"""
                target.status='QUEUED'
                AND {_claim_owner_predicate("target")}
                AND {_same_affinity_predicate("target")}
                AND {_runnable_queued_predicate(self._table, "target")}
            """
        async with self._engine.begin() as connection:
            await self._lock_invocation_session(connection, invocation_id)
            row = (
                (
                    await connection.execute(
                        text(
                            f"""
                        UPDATE {self._table} AS target
                        SET status='ACTIVE',
                            attempt={attempt_expression},
                            heartbeat_at=clock_timestamp(),
                            worker_id=CAST(:worker_id AS UUID)
                        WHERE target.invocation_id=:invocation_id
                          AND ({eligibility_filter_str})
                        RETURNING target.invocation_id, target.session_id, target.session_sequence_number,
                                  target.status, target.attempt, target.worker_id,
                                  target.recovery_affinity,
                                  request::TEXT AS request_json,
                                  response::TEXT AS response_json
                        """
                        ),
                        parameters,
                    )
                )
                .mappings()
                .one_or_none()
            )
            if row is not None:
                await connection.execute(
                    text(
                        f"""
                        INSERT INTO {self._events_table} (invocation_id, attempt, event)
                        VALUES (:invocation_id, :attempt, CAST(:event AS JSONB))
                        """
                    ),
                    {
                        "invocation_id": invocation_id,
                        "attempt": int(row["attempt"]),
                        "event": _serialize_json_value({"type": "run.started"}),
                    },
                )
        return self._to_invocation(row) if row is not None else None

    async def heartbeat(self, invocation_id: str, attempt: int) -> bool:
        _validate_invocation_id(invocation_id)
        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    f"""
                    UPDATE {self._table}
                    SET heartbeat_at=clock_timestamp()
                    WHERE invocation_id=:invocation_id
                      AND attempt=:attempt
                      AND status='ACTIVE'
                      AND (worker_id IS NULL OR worker_id=CAST(:worker_id AS UUID))
                    """
                ),
                {
                    "invocation_id": invocation_id,
                    "attempt": attempt,
                    "worker_id": self._worker_id,
                },
            )
        return result.rowcount == 1

    async def complete(
        self,
        invocation_id: str,
        attempt: int,
        response: JsonValue,
    ) -> bool:
        _validate_invocation_id(invocation_id)
        serialized_response = _serialize_json_value(response)
        async with self._engine.begin() as connection:
            session_id = await self._lock_invocation_session(connection, invocation_id)
            result = await connection.execute(
                text(
                    f"""
                    UPDATE {self._table}
                    SET status='COMPLETED', response=CAST(:response AS JSONB)
                    WHERE invocation_id=:invocation_id
                      AND attempt=:attempt
                      AND status='ACTIVE'
                      AND (worker_id IS NULL OR worker_id=CAST(:worker_id AS UUID))
                    """
                ),
                {
                    "invocation_id": invocation_id,
                    "attempt": attempt,
                    "worker_id": self._worker_id,
                    "response": serialized_response,
                },
            )
            if result.rowcount == 1:
                await connection.execute(
                    text(
                        f"""
                        INSERT INTO {self._events_table} (invocation_id, attempt, event)
                        VALUES (:invocation_id, :attempt, CAST(:event AS JSONB))
                        """
                    ),
                    {
                        "invocation_id": invocation_id,
                        "attempt": attempt,
                        "event": _serialize_json_value({"type": "run.completed"}),
                    },
                )
                if session_id is not None:
                    await self._refresh_next_owned_queued_head(connection, session_id)
        return result.rowcount == 1

    async def fail(
        self,
        invocation_id: str,
        attempt: int,
        response: JsonValue = None,
    ) -> bool:
        _validate_invocation_id(invocation_id)
        serialized_response = _serialize_json_value(response)
        async with self._engine.begin() as connection:
            session_id = await self._lock_invocation_session(connection, invocation_id)
            result = await connection.execute(
                text(
                    f"""
                    UPDATE {self._table}
                    SET status='FAILED', response=CAST(:response AS JSONB)
                    WHERE invocation_id=:invocation_id
                      AND attempt=:attempt
                      AND status='ACTIVE'
                      AND (worker_id IS NULL OR worker_id=CAST(:worker_id AS UUID))
                    """
                ),
                {
                    "invocation_id": invocation_id,
                    "attempt": attempt,
                    "worker_id": self._worker_id,
                    "response": serialized_response,
                },
            )
            if result.rowcount == 1:
                await connection.execute(
                    text(
                        f"""
                        INSERT INTO {self._events_table} (invocation_id, attempt, event)
                        VALUES (:invocation_id, :attempt, CAST(:event AS JSONB))
                        """
                    ),
                    {
                        "invocation_id": invocation_id,
                        "attempt": attempt,
                        "event": _serialize_json_value({"type": "run.failed"}),
                    },
                )
                if session_id is not None:
                    await self._refresh_next_owned_queued_head(connection, session_id)
        return result.rowcount == 1

    async def append_event(
        self,
        invocation_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        """Append an event only while the caller owns the active attempt."""
        _validate_invocation_id(invocation_id)
        serialized_event = _serialize_json_value(event)
        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    f"""
                    WITH owned_invocation AS (
                        SELECT 1
                        FROM {self._table}
                        WHERE invocation_id=:invocation_id
                          AND attempt=:attempt
                          AND status='ACTIVE'
                          AND (worker_id IS NULL OR worker_id=CAST(:worker_id AS UUID))
                        FOR UPDATE
                    )
                    INSERT INTO {self._events_table} (invocation_id, attempt, event)
                    SELECT :invocation_id, :attempt, CAST(:event AS JSONB)
                    FROM owned_invocation
                    RETURNING sequence_number
                    """
                ),
                {
                    "invocation_id": invocation_id,
                    "attempt": attempt,
                    "worker_id": self._worker_id,
                    "event": serialized_event,
                },
            )
            sequence_number = result.scalar_one_or_none()
        return int(sequence_number) if sequence_number is not None else None

    async def events(
        self,
        invocation_id: str | None = None,
        after_sequence: int | None = None,
        session_id: str | None = None,
    ) -> list[InvocationEvent]:
        """Return ordered events for one invocation or session after an optional cursor."""
        _validate_read_scope(invocation_id, session_id)
        if invocation_id is not None:
            predicate = "event.invocation_id=:invocation_id"
            parameters: dict[str, str | int | None] = {
                "invocation_id": invocation_id,
                "after_sequence": after_sequence,
            }
        else:
            predicate = "invocation.session_id=:session_id"
            parameters = {"session_id": session_id, "after_sequence": after_sequence}
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    f"""
                    SELECT event.sequence_number, event.invocation_id, event.attempt,
                           event.event::TEXT AS event_json
                    FROM {self._events_table} AS event
                    JOIN {self._table} AS invocation
                      ON invocation.invocation_id=event.invocation_id
                    WHERE {predicate}
                      AND (
                          CAST(:after_sequence AS BIGINT) IS NULL
                          OR event.sequence_number > :after_sequence
                      )
                    ORDER BY event.sequence_number
                    """
                ),
                parameters,
            )
            rows = result.mappings().all()
        return [
            InvocationEvent(
                sequence_number=int(row["sequence_number"]),
                invocation_id=str(row["invocation_id"]),
                attempt=int(row["attempt"]),
                event=json.loads(row["event_json"]),
            )
            for row in rows
        ]

    @staticmethod
    def _to_invocation(row: Mapping[str, Any] | RowMapping) -> Invocation:
        return Invocation(
            invocation_id=str(row["invocation_id"]),
            status=InvocationStatus(str(row["status"])),
            attempt=int(row["attempt"]),
            request=json.loads(row["request_json"]),
            response=json.loads(row["response_json"]) if row["response_json"] else None,
            session_id=str(row["session_id"]) if row["session_id"] is not None else None,
            session_sequence_number=int(row["session_sequence_number"])
            if row["session_sequence_number"] is not None
            else None,
            worker_id=str(row["worker_id"]) if row.get("worker_id") is not None else None,
            recovery_affinity=(
                str(row["recovery_affinity"]) if row.get("recovery_affinity") is not None else None
            ),
        )

    async def _lock_invocation_session(self, connection: Any, invocation_id: str) -> str | None:
        session_id = (
            await connection.execute(
                text(
                    f"""
                    SELECT session_id
                    FROM {self._table}
                    WHERE invocation_id=:invocation_id
                    """
                ),
                {"invocation_id": invocation_id},
            )
        ).scalar_one_or_none()
        if session_id is not None:
            session_id = str(session_id)
            await self._lock_session(connection, session_id)
        return session_id

    async def _refresh_next_owned_queued_head(self, connection: Any, session_id: str) -> None:
        """Extend the next owned queue head while the session lock is still held."""
        await connection.execute(
            text(
                f"""
                UPDATE {self._table} AS next_invocation
                SET heartbeat_at=clock_timestamp()
                WHERE next_invocation.invocation_id = (
                    SELECT head.invocation_id
                    FROM {self._table} AS head
                    WHERE head.session_id=:session_id
                      AND head.status='QUEUED'
                      AND head.worker_id IS NOT NULL
                      AND {_runnable_queued_predicate(self._table, "head")}
                    ORDER BY head.session_sequence_number, head.invocation_id
                    LIMIT 1
                )
                """
            ),
            {"session_id": session_id},
        )

    async def _lock_session(self, connection: Any, session_id: str) -> None:
        await connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:session_lock_key, 0))"),
            {"session_lock_key": f"{self._table}:{session_id}"},
        )
