"""Tests for the SDK-provided durable agent application."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import httpx
import pytest

from databricks_mason import DurableAgentApp
from databricks_mason.runtime.store import RUNTIME_ENDPOINT_ENV, InMemoryDurabilityStore
from databricks_mason.runtime.types import (
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionStatus,
)

_ROUTING_COOKIE = "__Host-databricks-app-router"


async def echo(payload, context):
    return {"output": payload}


def make_app(invoke=echo, *, on_resume=None) -> DurableAgentApp:
    return DurableAgentApp(
        invoke,
        on_resume=on_resume,
        durability_store=InMemoryDurabilityStore(),
    )


@asynccontextmanager
async def running_client(server: DurableAgentApp) -> AsyncIterator[httpx.AsyncClient]:
    await server._runtime.start(recover=server._on_resume is not None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="https://testserver",
        ) as client:
            yield client
    finally:
        await server._runtime.stop()


@pytest.mark.asyncio
async def test_routing_cookie_is_the_only_session_source() -> None:
    async def invoke(payload, context):
        return {
            "received": payload,
            "run_id": context.run_id,
            "session_id": context.session_id,
            "attempt": context.attempt,
            "is_recovery": context.is_recovery,
        }

    server = make_app(invoke)
    async with running_client(server) as client:
        client.cookies.set(_ROUTING_COOKIE, "session-1")
        response = await client.post(
            "/invocations",
            json={"id": "run-1", "input": "hello"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "received": {"input": "hello"},
        "run_id": "run-1",
        "session_id": "session-1",
        "attempt": 1,
        "is_recovery": False,
    }
    assert "x-databricks-run-id" not in response.headers
    assert "x-databricks-session-id" not in response.headers


@pytest.mark.asyncio
async def test_missing_routing_cookie_is_initialized_once() -> None:
    seen_sessions = []

    async def invoke(payload, context):
        seen_sessions.append(context.session_id)
        return {"output": payload}

    server = make_app(invoke)
    async with running_client(server) as client:
        response = await client.post("/invocations", json={"id": "run-1"})
        session_id = response.cookies[_ROUTING_COOKIE]

    assert UUID(session_id)
    assert seen_sessions == [session_id]
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_body_session_metadata_is_rejected() -> None:
    server = make_app()
    async with running_client(server) as client:
        response = await client.post(
            "/invocations",
            json={"id": "run-1", "session_id": "body-session"},
        )

    assert response.status_code == 422


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
        {"input": {"input": "hello"}, "session_id": "session-1"},
        DurableExecutionContext("run-1", 2),
    )

    assert result == {"attempt": 2, "session_id": "session-1"}
    assert calls == ["recover"]


@pytest.mark.asyncio
async def test_recovery_without_on_resume_is_disabled() -> None:
    server = make_app()
    await server._runtime.start(recover=False)
    try:
        assert server._runtime._scanner is None
        with pytest.raises(RuntimeError, match="on_resume"):
            await server._execute(
                {"input": {}, "session_id": "session-1"},
                DurableExecutionContext("run-1", 2),
            )
    finally:
        await server._runtime.stop()


@pytest.mark.asyncio
async def test_background_invocation_can_be_polled() -> None:
    async def invoke(payload, context):
        return {"output": payload["input"]}

    server = make_app(invoke)
    async with running_client(server) as client:
        submitted = await client.post(
            "/invocations",
            json={"id": "run-bg", "input": "hello", "background": True},
        )
        assert submitted.status_code in {200, 202}
        assert submitted.json()["id"] == "run-bg"

        for _ in range(100):
            polled = await client.get("/invocations/run-bg")
            if polled.json()["status"] not in {"queued", "active"}:
                break
            await asyncio.sleep(0.005)

    assert polled.json() == {
        "id": "run-bg",
        "status": "completed",
        "attempt": 1,
        "output": "hello",
    }


@pytest.mark.asyncio
async def test_stream_replays_events_without_context_headers() -> None:
    async def invoke(payload, context):
        await context.emit({"type": "delta", "content": "hello"})
        return {"output": []}

    server = make_app(invoke)
    async with running_client(server) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": "run-stream", "stream": True},
        )

    assert response.status_code == 200
    assert 'event: delta\ndata: {"type": "delta", "content": "hello"}' in response.text
    assert "[DONE]" not in response.text
    assert "x-databricks-run-id" not in response.headers
    assert "x-databricks-session-id" not in response.headers


@pytest.mark.asyncio
async def test_reusing_id_with_different_input_returns_conflict() -> None:
    server = make_app()
    async with running_client(server) as client:
        first = await client.post("/invocations", json={"id": "run-1", "input": "one"})
        conflict = await client.post("/invocations", json={"id": "run-1", "input": "two"})

    assert first.status_code == 200
    assert conflict.status_code == 409


@pytest.mark.asyncio
async def test_agent_failure_returns_500() -> None:
    async def fail(payload, context):
        raise RuntimeError("boom")

    server = make_app(fail)
    async with running_client(server) as client:
        response = await client.post("/invocations", json={"id": "run-1"})

    assert response.status_code == 500
    assert response.json() == {"detail": "agent execution failed"}


def test_app_exposes_local_and_deployed_invocation_routes() -> None:
    server = make_app()
    paths = server.app.openapi()["paths"]

    assert "/invocations" in paths
    assert "/api/invocations" in paths
    assert "/invocations/{run_id}" in paths
    assert "/api/invocations/{run_id}" in paths
    assert "/invocations/{run_id}/events" in paths
    assert "/api/invocations/{run_id}/events" in paths
    assert "/api/session/new" not in paths
    assert not hasattr(server, "asgi_app")
    assert not hasattr(server, "run")


def test_durability_store_defaults_to_in_memory(monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_ENDPOINT_ENV, raising=False)

    server = DurableAgentApp(echo)

    assert isinstance(server._runtime.durability_store, InMemoryDurabilityStore)


def test_state_payload_flattens_completed_application_response() -> None:
    state = DurableExecution(
        execution_id="run-1",
        status=DurableExecutionStatus.COMPLETED,
        attempt=1,
        heartbeat_at=None,
        request={"input": {}, "session_id": "session-1"},
        response={"output": [], "custom": "value"},
    )

    assert DurableAgentApp._state_payload(state) == {
        "id": "run-1",
        "status": "completed",
        "attempt": 1,
        "output": [],
        "custom": "value",
    }
