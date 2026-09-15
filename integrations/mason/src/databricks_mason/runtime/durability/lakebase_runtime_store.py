"""Lakebase-backed durability for Mason Runtime."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import URL, event, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from databricks_mason.runtime.durability.store import DurableRuntimeStore
from databricks_mason.runtime.store import DEFAULT_RUNTIME_SCHEMA
from databricks_mason.runtime.types import (
    DurableEvent,
    DurableExecution,
    DurableExecutionStatus,
    DurableRequestConflictError,
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


def _validate_execution_id(execution_id: str) -> None:
    if not execution_id:
        raise ValueError("execution_id must not be empty")


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
    """Persist shared execution state, attempt leases, and ordered events in Lakebase.

    The ``executions`` table is the source of truth for idempotency and lifecycle state. Conditional
    SQL updates claim an attempt and fence completion, failure, heartbeat, and event writes by its
    attempt number. The ``execution_events`` table provides an ordered replay cursor. Because every
    app replica connects to the same schema, another replica can detect a stale heartbeat, claim the
    next attempt, and continue after process or pod loss.

    This store requires a Lakebase Postgres database. ``mason deploy`` reuses or provisions a
    dedicated app-owned database, then assigns the app its own schema. ``mason dev`` uses
    ``InMemoryRuntimeStore`` instead.
    """

    def __init__(
        self,
        *,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = DEFAULT_RUNTIME_SCHEMA,
        lakebase: _AsyncLakebase | None = None,
    ) -> None:
        if not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError(f"invalid Runtime Store schema name: {schema!r}")

        if lakebase is None:
            from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

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
                workspace_client=workspace_client,
                schema=schema,
                pool_pre_ping=True,
            )

        self._lakebase = lakebase
        self._engine = lakebase.engine
        self._table = f"{schema}.executions"
        self._events_table = f"{schema}.execution_events"

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
        schema: str = DEFAULT_RUNTIME_SCHEMA,
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

    async def initialize(self) -> None:
        await self._lakebase.create_schema()
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        execution_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 0,
                        heartbeat_at TIMESTAMPTZ,
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
                    CREATE TABLE IF NOT EXISTS {self._events_table} (
                        sequence_number BIGSERIAL PRIMARY KEY,
                        execution_id TEXT NOT NULL
                            REFERENCES {self._table}(execution_id) ON DELETE CASCADE,
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
                    CREATE INDEX IF NOT EXISTS execution_events_replay_idx
                    ON {self._events_table} (execution_id, sequence_number)
                    """
                )
            )
            await connection.execute(
                text(
                    f"""
                    CREATE INDEX IF NOT EXISTS executions_recovery_idx
                    ON {self._table} (status, heartbeat_at)
                    WHERE status IN ('QUEUED', 'ACTIVE')
                    """
                )
            )

    async def close(self) -> None:
        await self._engine.dispose()

    async def accept(self, execution_id: str, request: JsonValue) -> DurableExecution:
        _validate_execution_id(execution_id)
        serialized_request = _serialize_json_value(request)
        async with self._engine.begin() as connection:
            await connection.execute(
                text(
                    f"""
                    INSERT INTO {self._table} (execution_id, status, request)
                    VALUES (:execution_id, 'QUEUED', CAST(:request AS JSONB))
                    ON CONFLICT (execution_id) DO NOTHING
                    """
                ),
                {"execution_id": execution_id, "request": serialized_request},
            )
            row = (
                (
                    await connection.execute(
                        text(
                            f"""
                        SELECT execution_id, status, attempt, heartbeat_at,
                               request::TEXT AS request_json,
                               response::TEXT AS response_json
                        FROM {self._table}
                        WHERE execution_id=:execution_id
                        """
                        ),
                        {"execution_id": execution_id},
                    )
                )
                .mappings()
                .one()
            )

        state = self._to_execution(row)
        if state.request != request:
            raise DurableRequestConflictError(
                f"execution {execution_id!r} was already accepted with a different request"
            )
        return state

    async def get(self, execution_id: str) -> DurableExecution | None:
        _validate_execution_id(execution_id)
        async with self._engine.connect() as connection:
            row = (
                (
                    await connection.execute(
                        text(
                            f"""
                        SELECT execution_id, status, attempt, heartbeat_at,
                               request::TEXT AS request_json,
                               response::TEXT AS response_json
                        FROM {self._table}
                        WHERE execution_id=:execution_id
                        """
                        ),
                        {"execution_id": execution_id},
                    )
                )
                .mappings()
                .one_or_none()
            )
        return self._to_execution(row) if row is not None else None

    async def recoverable_execution_ids(self, stale_seconds: float) -> list[str]:
        async with self._engine.connect() as connection:
            rows = (
                (
                    await connection.execute(
                        text(
                            f"""
                        SELECT execution_id
                        FROM {self._table}
                        WHERE status='QUEUED'
                           OR (status='ACTIVE' AND (
                               heartbeat_at IS NULL
                               OR heartbeat_at < NOW() - (:stale * INTERVAL '1 second')
                           ))
                        ORDER BY heartbeat_at NULLS FIRST
                        """
                        ),
                        {"stale": stale_seconds},
                    )
                )
                .scalars()
                .all()
            )
        return list(rows)

    async def claim(self, execution_id: str) -> DurableExecution | None:
        """Claim a newly queued invocation for its first attempt."""
        return await self._claim(execution_id, stale_seconds=None)

    async def claim_recoverable(
        self,
        execution_id: str,
        stale_seconds: float,
    ) -> DurableExecution | None:
        """Claim a queued or stale invocation for a replacement attempt."""
        return await self._claim(execution_id, stale_seconds=stale_seconds)

    async def _claim(
        self,
        execution_id: str,
        *,
        stale_seconds: float | None,
    ) -> DurableExecution | None:
        _validate_execution_id(execution_id)
        eligibility = "status='QUEUED'"
        parameters: dict[str, str | float] = {"execution_id": execution_id}
        if stale_seconds is not None:
            eligibility = """
                status='QUEUED'
                OR (status='ACTIVE' AND (
                    heartbeat_at IS NULL
                    OR heartbeat_at < NOW() - (:stale * INTERVAL '1 second')
                ))
            """
            parameters["stale"] = stale_seconds
        async with self._engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        text(
                            f"""
                        UPDATE {self._table}
                        SET status='ACTIVE', attempt=attempt+1, heartbeat_at=NOW()
                        WHERE execution_id=:execution_id
                          AND ({eligibility})
                        RETURNING execution_id, status, attempt, heartbeat_at,
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
                        INSERT INTO {self._events_table} (execution_id, attempt, event)
                        VALUES (:execution_id, :attempt, CAST(:event AS JSONB))
                        """
                    ),
                    {
                        "execution_id": execution_id,
                        "attempt": int(row["attempt"]),
                        "event": _serialize_json_value({"type": "run.started"}),
                    },
                )
        return self._to_execution(row) if row is not None else None

    async def heartbeat(self, execution_id: str, attempt: int) -> bool:
        _validate_execution_id(execution_id)
        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    f"""
                    UPDATE {self._table}
                    SET heartbeat_at=NOW()
                    WHERE execution_id=:execution_id
                      AND attempt=:attempt
                      AND status='ACTIVE'
                    """
                ),
                {"execution_id": execution_id, "attempt": attempt},
            )
        return result.rowcount == 1

    async def complete(
        self,
        execution_id: str,
        attempt: int,
        response: JsonValue,
    ) -> bool:
        _validate_execution_id(execution_id)
        serialized_response = _serialize_json_value(response)
        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    f"""
                    UPDATE {self._table}
                    SET status='COMPLETED', response=CAST(:response AS JSONB)
                    WHERE execution_id=:execution_id
                      AND attempt=:attempt
                      AND status='ACTIVE'
                    """
                ),
                {
                    "execution_id": execution_id,
                    "attempt": attempt,
                    "response": serialized_response,
                },
            )
            if result.rowcount == 1:
                await connection.execute(
                    text(
                        f"""
                        INSERT INTO {self._events_table} (execution_id, attempt, event)
                        VALUES (:execution_id, :attempt, CAST(:event AS JSONB))
                        """
                    ),
                    {
                        "execution_id": execution_id,
                        "attempt": attempt,
                        "event": _serialize_json_value({"type": "run.completed"}),
                    },
                )
        return result.rowcount == 1

    async def fail(self, execution_id: str, attempt: int) -> bool:
        _validate_execution_id(execution_id)
        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    f"""
                    UPDATE {self._table}
                    SET status='FAILED'
                    WHERE execution_id=:execution_id
                      AND attempt=:attempt
                      AND status='ACTIVE'
                    """
                ),
                {"execution_id": execution_id, "attempt": attempt},
            )
            if result.rowcount == 1:
                await connection.execute(
                    text(
                        f"""
                        INSERT INTO {self._events_table} (execution_id, attempt, event)
                        VALUES (:execution_id, :attempt, CAST(:event AS JSONB))
                        """
                    ),
                    {
                        "execution_id": execution_id,
                        "attempt": attempt,
                        "event": _serialize_json_value({"type": "run.failed"}),
                    },
                )
        return result.rowcount == 1

    async def append_event(
        self,
        execution_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        """Append an event only while the caller owns the active attempt."""
        _validate_execution_id(execution_id)
        serialized_event = _serialize_json_value(event)
        async with self._engine.begin() as connection:
            result = await connection.execute(
                text(
                    f"""
                    INSERT INTO {self._events_table} (execution_id, attempt, event)
                    SELECT :execution_id, :attempt, CAST(:event AS JSONB)
                    WHERE EXISTS (
                        SELECT 1
                        FROM {self._table}
                        WHERE execution_id=:execution_id
                          AND attempt=:attempt
                          AND status='ACTIVE'
                    )
                    RETURNING sequence_number
                    """
                ),
                {
                    "execution_id": execution_id,
                    "attempt": attempt,
                    "event": serialized_event,
                },
            )
            sequence_number = result.scalar_one_or_none()
        return int(sequence_number) if sequence_number is not None else None

    async def events(
        self,
        execution_id: str,
        after_sequence: int | None = None,
    ) -> list[DurableEvent]:
        """Return ordered events for one execution after an optional cursor."""
        _validate_execution_id(execution_id)
        async with self._engine.connect() as connection:
            result = await connection.execute(
                text(
                    f"""
                    SELECT sequence_number, execution_id, attempt,
                           event::TEXT AS event_json
                    FROM {self._events_table}
                    WHERE execution_id=:execution_id
                      AND (:after_sequence IS NULL OR sequence_number > :after_sequence)
                    ORDER BY sequence_number
                    """
                ),
                {
                    "execution_id": execution_id,
                    "after_sequence": after_sequence,
                },
            )
            rows = result.mappings().all()
        return [
            DurableEvent(
                sequence_number=int(row["sequence_number"]),
                execution_id=str(row["execution_id"]),
                attempt=int(row["attempt"]),
                event=json.loads(row["event_json"]),
            )
            for row in rows
        ]

    @staticmethod
    def _to_execution(row: Mapping[str, Any] | RowMapping) -> DurableExecution:
        return DurableExecution(
            execution_id=str(row["execution_id"]),
            status=DurableExecutionStatus(str(row["status"])),
            attempt=int(row["attempt"]),
            heartbeat_at=row["heartbeat_at"],
            request=json.loads(row["request_json"]),
            response=json.loads(row["response_json"]) if row["response_json"] else None,
        )
