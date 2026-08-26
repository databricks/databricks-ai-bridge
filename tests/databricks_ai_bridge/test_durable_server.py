"""Tests for the fixed-protocol durable JSON server."""

from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from databricks_ai_bridge.durable_runtime import (
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionStatus,
)
from databricks_ai_bridge.durable_server import DatabricksDurableServer


@pytest.mark.asyncio
async def test_handler_receives_session_and_attempt_context():
    emitted: list[dict] = []

    async def handler(payload, context):
        cursor = await context.emit({"type": "progress"})
        return {
            "payload": payload,
            "run_id": context.run_id,
            "session_id": context.session_id,
            "attempt": context.attempt,
            "is_recovery": context.is_recovery,
            "cursor": cursor,
        }

    async def emit(event: dict) -> int:
        emitted.append(event)
        return 9

    server = DatabricksDurableServer(handler, durability_store=AsyncMock())
    result = await server._execute(
        {"session_id": "session-1", "input": {"message": "hello"}},
        DurableExecutionContext("run-1", 2, _emit=emit),
    )

    assert result == {
        "payload": {"message": "hello"},
        "run_id": "run-1",
        "session_id": "session-1",
        "attempt": 2,
        "is_recovery": True,
        "cursor": 9,
    }
    assert emitted == [{"type": "progress"}]


def test_background_invocation_uses_standard_protocol():
    async def handler(payload, context):
        return payload

    server = DatabricksDurableServer(handler, durability_store=AsyncMock())
    submit = AsyncMock(
        return_value=DurableExecution(
            execution_id="run-1",
            status=DurableExecutionStatus.QUEUED,
            attempt=0,
            heartbeat_at=None,
            request={"session_id": "session-1", "input": {"message": "hello"}},
            response=None,
        )
    )

    with (
        patch.object(server.runtime, "start", new_callable=AsyncMock),
        patch.object(server.runtime, "stop", new_callable=AsyncMock),
        patch.object(server.runtime, "submit", submit),
        TestClient(server.app) as client,
    ):
        response = client.post(
            "/invocations",
            json={
                "id": "run-1",
                "session_id": "session-1",
                "input": {"message": "hello"},
                "background": True,
            },
        )

    assert response.status_code == 202
    assert response.json() == {
        "id": "run-1",
        "status": "queued",
        "attempt": 0,
        "output": None,
    }
    submit.assert_awaited_once_with(
        "run-1",
        {"session_id": "session-1", "input": {"message": "hello"}},
    )
