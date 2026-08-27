"""Tests for the generic durable entrypoint application."""

from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from databricks_ai_bridge.durable_app import DatabricksDurableApp
from databricks_ai_bridge.durable_runtime import (
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionStatus,
)


@pytest.mark.asyncio
async def test_entrypoint_receives_stable_session_and_recovery_context():
    durable_app = DatabricksDurableApp(durability_store=AsyncMock())
    emitted: list[dict] = []

    async def emit(event: dict) -> int:
        emitted.append(event)
        return 4

    @durable_app.entrypoint
    async def agent(payload, context):
        cursor = await context.emit({"type": "progress"})
        return {
            "payload": payload,
            "run_id": context.run_id,
            "session_id": context.session_id,
            "attempt": context.attempt,
            "is_recovery": context.is_recovery,
            "cursor": cursor,
        }

    result = await durable_app._execute(
        {"session_id": "session-1", "payload": {"message": "hello"}},
        DurableExecutionContext("run-1", 2, _emit=emit),
    )

    assert result == {
        "payload": {"message": "hello"},
        "run_id": "run-1",
        "session_id": "session-1",
        "attempt": 2,
        "is_recovery": True,
        "cursor": 4,
    }
    assert emitted == [{"type": "progress"}]


@pytest.mark.asyncio
async def test_resume_entrypoint_handles_recovered_attempt():
    durable_app = DatabricksDurableApp(durability_store=AsyncMock())

    @durable_app.entrypoint
    async def agent(payload, context):
        return {"handler": "entrypoint"}

    @durable_app.on_resume
    async def resume_agent(payload, context):
        return {
            "handler": "resume",
            "payload": payload,
            "session_id": context.session_id,
        }

    result = await durable_app._execute(
        {"session_id": "session-1", "payload": {"message": "hello"}},
        DurableExecutionContext("run-1", 2, _emit=AsyncMock()),
    )

    assert result == {
        "handler": "resume",
        "payload": {"message": "hello"},
        "session_id": "session-1",
    }


def test_builtin_submission_route_maps_generic_payload():
    durable_app = DatabricksDurableApp(durability_store=AsyncMock())

    @durable_app.entrypoint
    async def agent(payload, context):
        return payload

    submit = AsyncMock(
        return_value=DurableExecution(
            execution_id="run-1",
            status=DurableExecutionStatus.QUEUED,
            attempt=0,
            heartbeat_at=None,
            request={"session_id": "session-1", "payload": {"message": "hello"}},
            response=None,
        )
    )

    with (
        patch.object(durable_app.runtime, "start", new_callable=AsyncMock),
        patch.object(durable_app.runtime, "stop", new_callable=AsyncMock),
        patch.object(durable_app.runtime, "submit", submit),
        TestClient(durable_app) as client,
    ):
        response = client.post(
            "/runs",
            json={
                "run_id": "run-1",
                "session_id": "session-1",
                "payload": {"message": "hello"},
                "background": True,
                "stream": False,
            },
        )

    assert response.status_code == 202
    assert response.json() == {
        "run_id": "run-1",
        "status": "QUEUED",
        "attempt": 0,
        "result": None,
    }
    submit.assert_awaited_once_with(
        "run-1",
        {"session_id": "session-1", "payload": {"message": "hello"}},
    )


def test_only_one_entrypoint_can_be_registered():
    durable_app = DatabricksDurableApp(durability_store=AsyncMock())

    @durable_app.entrypoint
    async def first(payload, context):
        return payload

    with pytest.raises(RuntimeError, match="one entrypoint"):

        @durable_app.entrypoint
        async def second(payload, context):
            return payload
