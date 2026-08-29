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


def test_header_submission_preserves_application_payload():
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
            "/invocations",
            headers={
                "Idempotency-Key": "run-1",
                "Databricks-Agent-Session-Id": "session-1",
                "Databricks-Background": "true",
            },
            json={"message": "hello"},
        )

    assert response.status_code == 202
    assert response.json() == {
        "run_id": "run-1",
        "status": "QUEUED",
        "attempt": 0,
        "result": None,
    }
    assert response.headers["Databricks-Run-Id"] == "run-1"
    assert response.headers["Databricks-Agent-Session-Id"] == "session-1"
    submit.assert_awaited_once_with(
        "run-1",
        {"session_id": "session-1", "payload": {"message": "hello"}},
    )


def test_foreground_submission_preserves_agent_response_body():
    durable_app = DatabricksDurableApp(durability_store=AsyncMock())

    @durable_app.entrypoint
    async def agent(payload, context):
        return payload

    with (
        patch.object(durable_app.runtime, "start", new_callable=AsyncMock),
        patch.object(durable_app.runtime, "stop", new_callable=AsyncMock),
        patch.object(
            durable_app.runtime,
            "invoke",
            new=AsyncMock(return_value={"answer": "unchanged"}),
        ),
        TestClient(durable_app) as client,
    ):
        response = client.post(
            "/invocations",
            headers={"Databricks-Agent-Session-Id": "session-1"},
            json={"message": "hello"},
        )

    assert response.status_code == 200
    assert response.json() == {"answer": "unchanged"}
    assert response.headers["Databricks-Agent-Session-Id"] == "session-1"


def test_header_background_stream_preserves_body_and_returns_run_id():
    durable_app = DatabricksDurableApp(durability_store=AsyncMock())

    @durable_app.entrypoint
    async def agent(payload, context):
        return payload

    completed = DurableExecution(
        execution_id="run-1",
        status=DurableExecutionStatus.COMPLETED,
        attempt=1,
        heartbeat_at=None,
        request={"session_id": "session-1", "payload": {"message": "hello"}},
        response={"message": "hello"},
    )
    submit = AsyncMock(return_value=completed)

    with (
        patch.object(durable_app.runtime, "start", new_callable=AsyncMock),
        patch.object(durable_app.runtime, "stop", new_callable=AsyncMock),
        patch.object(durable_app.runtime, "submit", submit),
        patch.object(durable_app.runtime, "events", new=AsyncMock(return_value=[])),
        patch.object(durable_app.runtime, "get", new=AsyncMock(return_value=completed)),
        TestClient(durable_app) as client,
    ):
        response = client.post(
            "/invocations",
            headers={
                "Idempotency-Key": "run-1",
                "Databricks-Agent-Session-Id": "session-1",
                "Databricks-Background": "true",
                "Databricks-Stream": "true",
            },
            json={"message": "hello"},
        )

    assert response.status_code == 200
    assert response.headers["Databricks-Run-Id"] == "run-1"
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
