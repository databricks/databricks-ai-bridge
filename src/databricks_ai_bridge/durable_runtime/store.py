"""Lakebase persistence for durable request execution."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from databricks_ai_bridge.durable_runtime.types import (
    DurableExecution,
    DurableExecutionStatus,
    DurableRequestConflictError,
    JsonObject,
)
from databricks_ai_bridge.lakebase import AsyncLakebaseSQLAlchemy

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

DEFAULT_DURABILITY_SCHEMA = "databricks_durable_runtime"
_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _serialize_json_object(value: JsonObject) -> str:
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object, got {type(value).__name__}")
    return json.dumps(value, allow_nan=False)


def _validate_execution_id(execution_id: str) -> None:
    if not execution_id:
        raise ValueError("execution_id must not be empty")


class LakebaseDurabilityStore:
    """Store durable execution state in one Lakebase table."""

    def __init__(
        self,
        *,
        autoscaling_endpoint: str | None = None,
        project: str | None = None,
        branch: str | None = None,
        workspace_client: WorkspaceClient | None = None,
        schema: str = DEFAULT_DURABILITY_SCHEMA,
        lakebase: AsyncLakebaseSQLAlchemy | None = None,
    ) -> None:
        if not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError(f"invalid durability schema name: {schema!r}")

        if lakebase is None:
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

    @staticmethod
    def _to_execution(row: Mapping[str, Any]) -> DurableExecution:
        return DurableExecution(
            execution_id=str(row["execution_id"]),
            status=DurableExecutionStatus(str(row["status"])),
            attempt=int(row["attempt"]),
            heartbeat_at=row["heartbeat_at"],
            request=json.loads(row["request_json"]),
            response=json.loads(row["response_json"]) if row["response_json"] else None,
        )
