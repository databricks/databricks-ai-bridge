"""Tests for the SDK-provided durable agent application."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import httpx
import pytest

from databricks_mason import DurableAgentApp
from databricks_mason.durable_server.app import RUN_ID_HEADER, SESSION_ID_HEADER
from databricks_mason.runtime.store import InMemoryDurabilityStore
from databricks_mason.runtime.types import (
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionStatus,
)


async def echo(payload, context):
    return {"output": payload}


def make_app(invoke=echo, *, on_resume=None) -> DurableAgentApp:
    return DurableAgentApp(
        invoke,
        on_resume=on_resume,
        durability_store=InMemoryDurabilityStore(),
        heartbeat_seconds=0.01,
        stale_seconds=0.05,
        scan_seconds=0.01,
        poll_seconds=0.005,
    )


@asynccontextmanager
async def running_client(server: DurableAgentApp) -> AsyncIterator[httpx.AsyncClient]:
    await server._runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://testserver",
        ) as client:
            yield client
    finally:
        await server._runtime.stop()


@pytest.mark.asyncio
async def test_body_transport_parameters_are_removed_and_session_id_is_ignored() -> None:
    async def invoke(payload, context):
        return {
            "received": payload,
            "attempt": context.attempt,
            "is_recovery": context.is_recovery,
        }

    server = make_app(invoke)
    async with running_client(server) as client:
        response = await client.post(
            "/invocations",
            json={
                "run_id": "run-1",
                "session_id": "session-1",
                "actor": "body-actor",
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
async def test_recovery_attempt_uses_on_resume_hook() -> None:
    calls = []

    async def invoke(payload, context):
        calls.append("invoke")
        return payload

    async def recover(payload, context):
        calls.append("recover")
        return {"attempt": context.attempt, "session_id": context.session_id}

    server = make_app(invoke, on_resume=recover)
    result = await server._execute(
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
async def test_recovery_without_on_resume_reuses_invoke_hook() -> None:
    attempts = []

    async def invoke(payload, context):
        attempts.append(context.attempt)
        return payload

    server = make_app(invoke)
    await server._execute(
        {"payload": {}, "session_id": "session-1", "actor": "agent"},
        DurableExecutionContext("run-1", 2),
    )

    assert attempts == [2]


@pytest.mark.asyncio
async def test_background_invocation_can_be_polled() -> None:
    async def invoke(payload, context):
        return {"output": payload["input"]}

    server = make_app(invoke)
    async with running_client(server) as client:
        submitted = await client.post(
            "/invocations",
            json={"run_id": "run-bg", "input": "hello", "background": True},
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
        "output": "hello",
    }


@pytest.mark.asyncio
async def test_background_stream_returns_api_events_url() -> None:
    server = make_app()
    async with running_client(server) as client:
        response = await client.post(
            "/api/invocations",
            json={"run_id": "run-bg", "input": "hello", "background": True, "stream": True},
        )

    assert response.status_code == 202
    assert response.json()["events_url"] == "/api/invocations/run-bg/events"
    assert UUID(response.json()["session_id"])


@pytest.mark.asyncio
async def test_retry_reuses_cookie_derived_session() -> None:
    server = make_app()
    async with running_client(server) as client:
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
    async def invoke(payload, context):
        await context.emit({"type": "delta", "content": "hel"})
        await context.emit({"type": "delta", "content": "lo"})
        return {"output": "hello"}

    server = make_app(invoke)
    async with running_client(server) as client:
        async with client.stream(
            "POST",
            "/invocations",
            json={"run_id": "run-stream", "stream": True},
        ) as response:
            body = "".join([chunk async for chunk in response.aiter_text()])

    assert response.status_code == 200
    assert 'data: {"type": "delta", "content": "hel"}' in body
    assert 'data: {"type": "delta", "content": "lo"}' in body
    assert body.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_reusing_run_id_with_different_payload_returns_conflict() -> None:
    server = make_app()
    async with running_client(server) as client:
        first = await client.post("/invocations", json={"run_id": "run-1", "input": "one"})
        conflict = await client.post("/invocations", json={"run_id": "run-1", "input": "two"})

    assert first.status_code == 200
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_transport_mode_does_not_change_idempotent_request() -> None:
    server = make_app()
    async with running_client(server) as client:
        first = await client.post("/invocations", json={"run_id": "run-1", "input": "one"})
        second = await client.post(
            "/invocations",
            json={"run_id": "run-1", "input": "one", "background": True},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["output"] == {"input": "one"}


@pytest.mark.asyncio
async def test_forwarded_actor_is_passed_to_agent_context() -> None:
    async def invoke(payload, context):
        return {"actor": context.actor}

    server = make_app(invoke)
    async with running_client(server) as client:
        response = await client.post(
            "/invocations",
            json={},
            headers={"X-Forwarded-Email": "alice@example.com"},
        )

    assert response.json() == {"actor": "alice@example.com"}


@pytest.mark.asyncio
async def test_run_id_is_generated_with_invocation_prefix() -> None:
    server = make_app()
    async with running_client(server) as client:
        response = await client.post("/invocations", json={})

    assert response.headers[RUN_ID_HEADER].startswith("inv_")


@pytest.mark.asyncio
async def test_new_session_rotates_local_cookie() -> None:
    server = make_app()
    async with running_client(server) as client:
        health = await client.get("/health")
        previous_session_id = health.cookies["mason-local-session"]
        response = await client.post("/api/session/new")

    assert response.json()["previous_session_id"] == previous_session_id
    assert response.json()["session_id"] != previous_session_id
    assert response.cookies["mason-local-session"] == response.json()["session_id"]


def test_app_exposes_only_fastapi_surface() -> None:
    server = make_app()
    paths = server.app.openapi()["paths"]

    assert "/invocations" in paths
    assert "/api/invocations" in paths
    assert "/api/session/new" in paths
    assert not hasattr(server, "asgi_app")
    assert not hasattr(server, "run")


def test_durability_store_is_required() -> None:
    with pytest.raises(TypeError, match="durability_store"):
        DurableAgentApp(echo)  # type: ignore[call-arg]


def test_state_payload_flattens_completed_application_response() -> None:
    state = DurableExecution(
        execution_id="run-1",
        status=DurableExecutionStatus.COMPLETED,
        attempt=1,
        heartbeat_at=None,
        request={"payload": {}, "session_id": "session-1"},
        response={"output": [], "custom": "value"},
    )

    assert DurableAgentApp._state_payload(state, session_id="session-1") == {
        "id": "run-1",
        "session_id": "session-1",
        "status": "completed",
        "output": [],
        "custom": "value",
    }
