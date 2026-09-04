"""Lakebase persistence for durable request execution."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import URL, event, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from databricks_mason.runtime.types import (
    DurabilityStore,
    DurableEvent,
    DurableExecution,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


class _AsyncLakebase(Protocol):
    engine: AsyncEngine

    async def create_schema(self) -> None: ...


DEFAULT_DURABILITY_SCHEMA = "databricks_mason_runtime"
RUNTIME_ENDPOINT_ENV = "DATABRICKS_MASON_RUNTIME_ENDPOINT"
_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TOKEN_CACHE_SECONDS = 15 * 60
_POOL_RECYCLE_SECONDS = 14 * 60


def _serialize_json_object(value: JsonObject) -> str:
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
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


class LakebaseDurabilityStore:
    """Store durable execution state and events in one Lakebase schema."""

    def __init__(
        self,
        *,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = DEFAULT_DURABILITY_SCHEMA,
        lakebase: _AsyncLakebase | None = None,
    ) -> None:
        if not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError(f"invalid durability schema name: {schema!r}")

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
        schema: str = DEFAULT_DURABILITY_SCHEMA,
    ) -> "LakebaseDurabilityStore":
        """Use connection coordinates injected for a Databricks Apps Postgres resource."""
        if not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError(f"invalid durability schema name: {schema!r}")
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
                        CHECK (status IN ('QUEUED', 'ACTIVE', 'COMPLETED', 'FAILED')),
                        CHECK (jsonb_typeof(request) = 'object'),
                        CHECK (response IS NULL OR jsonb_typeof(response) = 'object')
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

    async def accept(self, execution_id: str, request: JsonObject) -> DurableExecution:
        _validate_execution_id(execution_id)
        serialized_request = _serialize_json_object(request)
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

    async def claim(
        self,
        execution_id: str,
        stale_seconds: float,
    ) -> DurableExecution | None:
        _validate_execution_id(execution_id)
        async with self._engine.begin() as connection:
            row = (
                (
                    await connection.execute(
                        text(
                            f"""
                        UPDATE {self._table}
                        SET status='ACTIVE', attempt=attempt+1, heartbeat_at=NOW()
                        WHERE execution_id=:execution_id
                          AND (
                              status='QUEUED'
                              OR (status='ACTIVE' AND (
                                  heartbeat_at IS NULL
                                  OR heartbeat_at < NOW() - (:stale * INTERVAL '1 second')
                              ))
                          )
                        RETURNING execution_id, status, attempt, heartbeat_at,
                                  request::TEXT AS request_json,
                                  response::TEXT AS response_json
                        """
                        ),
                        {"execution_id": execution_id, "stale": stale_seconds},
                    )
                )
                .mappings()
                .one_or_none()
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
        response: JsonObject,
    ) -> bool:
        _validate_execution_id(execution_id)
        serialized_response = _serialize_json_object(response)
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
        return result.rowcount == 1

    async def append_event(
        self,
        execution_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        """Append an event only while the caller owns the active attempt."""
        _validate_execution_id(execution_id)
        serialized_event = _serialize_json_object(event)
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


class InMemoryDurabilityStore:
    """Process-local durability store for development and tests."""

    def __init__(self) -> None:
        self.states: dict[str, DurableExecution] = {}
        self.persisted_events: list[DurableEvent] = []
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def accept(self, execution_id: str, request: JsonObject) -> DurableExecution:
        _validate_execution_id(execution_id)
        async with self._lock:
            existing = self.states.get(execution_id)
            if existing is not None:
                if existing.request != request:
                    raise DurableRequestConflictError(execution_id)
                return copy.deepcopy(existing)
            state = DurableExecution(
                execution_id=execution_id,
                status=DurableExecutionStatus.QUEUED,
                attempt=0,
                heartbeat_at=None,
                request=copy.deepcopy(request),
                response=None,
            )
            self.states[execution_id] = state
            return copy.deepcopy(state)

    async def get(self, execution_id: str) -> DurableExecution | None:
        _validate_execution_id(execution_id)
        async with self._lock:
            state = self.states.get(execution_id)
            return copy.deepcopy(state) if state is not None else None

    async def recoverable_execution_ids(self, stale_seconds: float) -> list[str]:
        async with self._lock:
            return [
                execution_id
                for execution_id, state in self.states.items()
                if self._is_recoverable(state, stale_seconds)
            ]

    async def claim(
        self,
        execution_id: str,
        stale_seconds: float,
    ) -> DurableExecution | None:
        _validate_execution_id(execution_id)
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or not self._is_recoverable(state, stale_seconds):
                return None
            claimed = DurableExecution(
                execution_id=execution_id,
                status=DurableExecutionStatus.ACTIVE,
                attempt=state.attempt + 1,
                heartbeat_at=datetime.now(timezone.utc),
                request=copy.deepcopy(state.request),
                response=None,
            )
            self.states[execution_id] = claimed
            return copy.deepcopy(claimed)

    async def heartbeat(self, execution_id: str, attempt: int) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or not self._owns_attempt(state, attempt):
                return False
            self.states[execution_id] = DurableExecution(
                execution_id=state.execution_id,
                status=state.status,
                attempt=state.attempt,
                heartbeat_at=datetime.now(timezone.utc),
                request=state.request,
                response=state.response,
            )
            return True

    async def complete(self, execution_id: str, attempt: int, response: JsonObject) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or not self._owns_attempt(state, attempt):
                return False
            self.states[execution_id] = DurableExecution(
                execution_id=state.execution_id,
                status=DurableExecutionStatus.COMPLETED,
                attempt=state.attempt,
                heartbeat_at=state.heartbeat_at,
                request=state.request,
                response=copy.deepcopy(response),
            )
            return True

    async def fail(self, execution_id: str, attempt: int) -> bool:
        async with self._lock:
            state = self.states.get(execution_id)
            if state is None or not self._owns_attempt(state, attempt):
                return False
            self.states[execution_id] = DurableExecution(
                execution_id=state.execution_id,
                status=DurableExecutionStatus.FAILED,
                attempt=state.attempt,
                heartbeat_at=state.heartbeat_at,
                request=state.request,
                response=None,
            )
            return True

    async def append_event(
        self,
        execution_id: str,
        attempt: int,
        event: JsonObject,
    ) -> int | None:
        async with self._lock:
            state = self.states.get(execution_id)
            if not self._owns_attempt(state, attempt):
                return None
            persisted = DurableEvent(
                sequence_number=len(self.persisted_events) + 1,
                execution_id=execution_id,
                attempt=attempt,
                event=copy.deepcopy(event),
            )
            self.persisted_events.append(persisted)
            return persisted.sequence_number

    async def events(
        self,
        execution_id: str,
        after_sequence: int | None = None,
    ) -> list[DurableEvent]:
        _validate_execution_id(execution_id)
        async with self._lock:
            return [
                copy.deepcopy(event)
                for event in self.persisted_events
                if event.execution_id == execution_id
                and (after_sequence is None or event.sequence_number > after_sequence)
            ]

    @staticmethod
    def _owns_attempt(state: DurableExecution | None, attempt: int) -> bool:
        return bool(
            state and state.status == DurableExecutionStatus.ACTIVE and state.attempt == attempt
        )

    @staticmethod
    def _is_recoverable(state: DurableExecution, stale_seconds: float) -> bool:
        if state.status == DurableExecutionStatus.QUEUED:
            return True
        if state.status != DurableExecutionStatus.ACTIVE or state.heartbeat_at is None:
            return state.status == DurableExecutionStatus.ACTIVE
        heartbeat_at = state.heartbeat_at
        if heartbeat_at.tzinfo is None:
            heartbeat_at = heartbeat_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - heartbeat_at >= timedelta(seconds=stale_seconds)


def default_durability_store() -> DurabilityStore:
    """Use the attached Lakebase resource when deployed, otherwise process-local state."""
    if endpoint := os.getenv(RUNTIME_ENDPOINT_ENV):
        return LakebaseDurabilityStore.from_app_resource(endpoint=endpoint)
    return InMemoryDurabilityStore()
