"""Tests for the SDK-provided durable agent application."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from uuid import UUID

import httpx
import pytest

from databricks_mason.runtime import DurableAgentApp
from databricks_mason.runtime.app import RUN_ID_HEADER, SESSION_ID_HEADER
from databricks_mason.runtime.memory_store import InMemoryDurabilityStore
from databricks_mason.runtime.types import (
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionStatus,
)


def make_app() -> DurableAgentApp:
    return DurableAgentApp(
        durability_store=InMemoryDurabilityStore(),
        heartbeat_seconds=0.01,
        stale_seconds=0.05,
        scan_seconds=0.01,
        poll_seconds=0.005,
    )


@asynccontextmanager
async def running_client(app: DurableAgentApp):
    await app._runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            yield client
    finally:
        await app._runtime.stop()


@pytest.mark.asyncio
async def test_body_runtime_parameters_are_removed_and_session_id_is_ignored() -> None:
    durable_app = make_app()

    @durable_app.invoke
    async def invoke(payload, context):
        return {
            "received": payload,
            "attempt": context.attempt,
            "is_recovery": context.is_recovery,
        }

    async with running_client(durable_app) as client:
        response = await client.post(
            "/invocations",
            json={
                "run_id": "run-1",
                "session_id": "session-1",
                "background": False,
                "stream": False,
                "input": "hello",
            },
        )

    assert response.status_code == 200
    assert response.headers[RUN_ID_HEADER] == "run-1"
    assert response.headers[SESSION_ID_HEADER] != "session-1"
    assert UUID(response.headers[SESSION_ID_HEADER])
    assert response.json() == {
        "received": {"input": "hello"},
        "attempt": 1,
        "is_recovery": False,
    }


@pytest.mark.asyncio
async def test_recovery_attempt_uses_registered_recovery_hook() -> None:
    durable_app = make_app()
    calls = []

    @durable_app.invoke
    async def invoke(payload, context):
        calls.append("invoke")
        return payload

    @durable_app.recover
    async def recover(payload, context):
        calls.append("recover")
        return {"attempt": context.attempt, "session_id": context.session_id}

    result = await durable_app._execute(
        {
            "payload": {"input": "hello"},
            "session_id": "session-1",
            "actor": "alice@example.com",
        },
        DurableExecutionContext("run-1", 2),
    )

    assert result == {"attempt": 2, "session_id": "session-1"}
    assert calls == ["recover"]


@pytest.mark.asyncio
async def test_background_invocation_can_be_polled() -> None:
    durable_app = make_app()

    @durable_app.invoke
    async def invoke(payload, context):
        return {"output": payload["input"]}

    async with running_client(durable_app) as client:
        submitted = await client.post(
            "/invocations",
            json={
                "run_id": "run-bg",
                "session_id": "session-bg",
                "input": "hello",
                "background": True,
            },
        )
        session_id = submitted.json()["session_id"]
        assert submitted.status_code == 202
        assert submitted.json() == {
            "id": "run-bg",
            "session_id": session_id,
            "status": "in_progress",
        }

        for _ in range(100):
            polled = await client.get("/invocations/run-bg")
            if polled.json()["status"] != "in_progress":
                break
            await asyncio.sleep(0.005)

    assert polled.json() == {
        "id": "run-bg",
        "session_id": session_id,
        "status": "completed",
        "result": {"output": "hello"},
    }


@pytest.mark.asyncio
async def test_background_stream_returns_events_url() -> None:
    durable_app = make_app()

    @durable_app.invoke
    async def invoke(payload, context):
        return {"output": payload}

    async with running_client(durable_app) as client:
        response = await client.post(
            "/api/invocations",
            json={"run_id": "run-bg", "input": "hello", "background": True, "stream": True},
        )

    assert response.status_code == 202
    assert response.json()["events_url"] == "/api/invocations/run-bg/events"
    assert UUID(response.json()["session_id"])


@pytest.mark.asyncio
async def test_retry_without_session_id_reuses_the_generated_session() -> None:
    durable_app = make_app()

    @durable_app.invoke
    async def invoke(payload, context):
        return {"output": payload}

    async with running_client(durable_app) as client:
        first = await client.post(
            "/invocations",
            json={"run_id": "run-retry", "input": "hello", "background": True},
        )
        retry = await client.post(
            "/invocations",
            json={"run_id": "run-retry", "input": "hello", "background": True},
        )

    assert first.status_code in {200, 202}
    assert retry.status_code in {200, 202}
    assert retry.json()["session_id"] == first.json()["session_id"]
    assert UUID(first.json()["session_id"])


@pytest.mark.asyncio
async def test_stream_replays_persisted_events_and_ends_with_done() -> None:
    durable_app = make_app()

    @durable_app.invoke
    async def invoke(payload, context):
        await context.emit({"type": "delta", "content": "hel"})
        await context.emit({"type": "delta", "content": "lo"})
        return {"output": "hello"}

    async with running_client(durable_app) as client:
        async with client.stream(
            "POST",
            "/invocations",
            json={"run_id": "run-stream", "session_id": "session-stream", "stream": True},
        ) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert 'data: {"type": "delta", "content": "hel"}' in body
    assert 'data: {"type": "delta", "content": "lo"}' in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_reusing_run_id_with_different_payload_returns_conflict() -> None:
    durable_app = make_app()

    @durable_app.invoke
    async def invoke(payload, context):
        return {"output": payload}

    async with running_client(durable_app) as client:
        first = await client.post(
            "/invocations", json={"run_id": "run-1", "session_id": "session-1", "input": "one"}
        )
        conflict = await client.post(
            "/invocations", json={"run_id": "run-1", "session_id": "session-1", "input": "two"}
        )

    assert first.status_code == 200
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_transport_mode_does_not_change_idempotent_request() -> None:
    durable_app = make_app()

    @durable_app.invoke
    async def invoke(payload, context):
        return {"output": payload}

    async with running_client(durable_app) as client:
        first = await client.post(
            "/invocations", json={"run_id": "run-1", "session_id": "session-1", "input": "one"}
        )
        second = await client.post(
            "/invocations",
            json={
                "run_id": "run-1",
                "session_id": "session-1",
                "input": "one",
                "background": True,
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["result"] == {"output": {"input": "one"}}


def test_local_default_uses_in_memory_store(monkeypatch) -> None:
    for name in (
        "AGENT_SESSION_STORE",
        "DATABRICKS_MASON_RUNTIME_ENDPOINT",
        "PGHOST",
        "PGPORT",
        "PGDATABASE",
        "PGUSER",
        "LAKEBASE_AUTOSCALING_ENDPOINT",
        "LAKEBASE_AUTOSCALING_PROJECT",
        "LAKEBASE_AUTOSCALING_BRANCH",
    ):
        monkeypatch.delenv(name, raising=False)

    durable_app = DurableAgentApp()

    assert isinstance(durable_app._runtime.durability_store, InMemoryDurabilityStore)


def test_managed_session_store_alone_keeps_runtime_in_memory(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")

    durable_app = DurableAgentApp()

    assert isinstance(durable_app._runtime.durability_store, InMemoryDurabilityStore)


def test_apps_postgres_resource_selects_lakebase(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABRICKS_MASON_RUNTIME_ENDPOINT",
        "projects/session-store/branches/production/endpoints/primary",
    )
    monkeypatch.setenv("PGHOST", "session-store.example.com")
    monkeypatch.setenv("PGPORT", "5432")
    monkeypatch.setenv("PGDATABASE", "sessions")
    monkeypatch.setenv("PGUSER", "app-service-principal")
    monkeypatch.setenv("PGSSLMODE", "verify-full")
    store = AsyncMock()

    with patch(
        "databricks_mason.runtime.app.LakebaseDurabilityStore.from_app_resource",
        return_value=store,
    ) as lakebase:
        durable_app = DurableAgentApp()

    assert durable_app._runtime.durability_store is store
    lakebase.assert_called_once_with(
        endpoint="projects/session-store/branches/production/endpoints/primary",
        host="session-store.example.com",
        port=5432,
        database="sessions",
        username="app-service-principal",
        sslmode="verify-full",
        workspace_client=None,
        schema="databricks_mason_runtime",
    )


def test_partial_apps_postgres_resource_fails_fast(monkeypatch) -> None:
    monkeypatch.setenv(
        "DATABRICKS_MASON_RUNTIME_ENDPOINT",
        "projects/session-store/branches/production/endpoints/primary",
    )
    monkeypatch.setenv("PGHOST", "session-store.example.com")
    monkeypatch.delenv("PGPORT", raising=False)
    monkeypatch.delenv("PGDATABASE", raising=False)
    monkeypatch.delenv("PGUSER", raising=False)

    with pytest.raises(RuntimeError, match="PGPORT, PGDATABASE, PGUSER"):
        DurableAgentApp()


def test_hooks_can_only_be_registered_once() -> None:
    durable_app = make_app()

    @durable_app.invoke
    async def first_invoke(payload, context):
        return payload

    with pytest.raises(RuntimeError, match="one invoke hook"):

        @durable_app.invoke
        async def second_invoke(payload, context):
            return payload

    @durable_app.recover
    async def first_recover(payload, context):
        return payload

    with pytest.raises(RuntimeError, match="one recovery hook"):

        @durable_app.recover
        async def second_recover(payload, context):
            return payload


def test_state_payload_keeps_application_result_opaque() -> None:
    state = DurableExecution(
        execution_id="run-1",
        status=DurableExecutionStatus.COMPLETED,
        attempt=1,
        heartbeat_at=None,
        request={"payload": {}, "session_id": "session-1"},
        response={"status": "interrupted", "output": []},
    )

    assert DurableAgentApp._state_payload(state, session_id="session-1") == {
        "id": "run-1",
        "session_id": "session-1",
        "status": "completed",
        "result": {"status": "interrupted", "output": []},
    }
