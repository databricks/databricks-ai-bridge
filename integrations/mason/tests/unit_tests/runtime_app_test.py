"""Tests for the SDK-provided agent application."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI

from databricks_mason import AgentApp
from databricks_mason.runtime.durability.store import (
    RUNTIME_ENDPOINT_ENV,
    RUNTIME_LOCAL_ENV,
    RUNTIME_SCHEMA_ENV,
    InMemoryDurabilityStore,
)
from databricks_mason.runtime.durability.types import (
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionStatus,
)

_ROUTING_COOKIE = "__Host-databricks-app-router"
_RUN_1 = "11111111-1111-4111-8111-111111111111"
_RUN_2 = "22222222-2222-4222-8222-222222222222"


async def echo(input, context):
    return input


def make_app(invoke=echo, *, on_recovery=None) -> AgentApp:
    app = AgentApp(durable_runtime=True, durability_store=InMemoryDurabilityStore())
    app.invoke(invoke)
    if on_recovery is not None:
        app.on_recovery(on_recovery)
    return app


@asynccontextmanager
async def running_client(app: AgentApp) -> AsyncIterator[httpx.AsyncClient]:
    await app._runtime.start(recover=app.durable_runtime and app._on_recovery_hook is not None)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://testserver",
        ) as client:
            yield client
    finally:
        await app._runtime.stop()


async def poll(client: httpx.AsyncClient, invocation_id: str) -> dict:
    for _ in range(100):
        response = await client.get(f"/api/invocations/{invocation_id}")
        if response.json()["status"] not in {"queued", "active"}:
            return response.json()
        await asyncio.sleep(0.005)
    raise AssertionError("run did not finish")


@pytest.mark.asyncio
async def test_routing_cookie_is_the_only_session_source() -> None:
    async def invoke(input, context):
        return {
            "received": input,
            "invocation_id": context.invocation_id,
            "session_id": context.session_id,
        }

    app = make_app(invoke)
    async with running_client(app) as client:
        client.cookies.set(_ROUTING_COOKIE, "session-1")
        response = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "hello"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": _RUN_1,
        "status": "completed",
        "output": {
            "received": "hello",
            "invocation_id": _RUN_1,
            "session_id": "session-1",
        },
    }


@pytest.mark.asyncio
async def test_missing_forwarded_routing_cookie_uses_invocation_id() -> None:
    seen_sessions = []

    async def invoke(input, context):
        seen_sessions.append(context.session_id)
        return input

    app = make_app(invoke)
    async with running_client(app) as client:
        response = await client.post("/api/invocations", json={"id": _RUN_1})

    assert seen_sessions == [_RUN_1]
    assert response.status_code == 200
    assert _ROUTING_COOKIE not in response.cookies


@pytest.mark.asyncio
async def test_body_session_and_resume_metadata_are_rejected() -> None:
    app = make_app()
    async with running_client(app) as client:
        session = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "session_id": "body-session"},
        )
        resume = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "resume": {"answer": "yes"}},
        )

    assert session.status_code == 422
    assert resume.status_code == 422


@pytest.mark.asyncio
async def test_recovery_attempt_uses_on_recovery_hook() -> None:
    calls = []

    async def invoke(input, context):
        calls.append("invoke")
        return input

    async def recover(input, context):
        calls.append("recover")
        return {"input": input, "session_id": context.session_id}

    app = make_app(invoke, on_recovery=recover)
    result = await app._execute(
        {"input": "hello", "session_id": "session-1"},
        DurableExecutionContext(_RUN_1, 2),
    )

    assert result == {"input": "hello", "session_id": "session-1"}
    assert calls == ["recover"]


@pytest.mark.asyncio
async def test_omitting_on_recovery_warns_and_disables_recovery(caplog) -> None:
    app = make_app()
    with caplog.at_level(logging.WARNING):
        async with app.router.lifespan_context(app):
            assert app._runtime._recovery_scheduler._scanner is None

    assert "crash recovery is disabled" in caplog.text
    with pytest.raises(RuntimeError, match="@app.on_recovery"):
        await app._execute(
            {"input": {}, "session_id": "session-1"},
            DurableExecutionContext(_RUN_1, 2),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [None, True, 7, "text", [1, "two"], {"nested": [None]}],
)
async def test_foreground_sync_accepts_any_json_input_and_output(value) -> None:
    app = make_app()
    async with running_client(app) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": value},
        )

    assert response.status_code == 200
    assert response.json() == {"id": _RUN_1, "status": "completed", "output": value}


@pytest.mark.asyncio
async def test_background_sync_returns_202_and_can_be_polled() -> None:
    app = make_app()
    async with running_client(app) as client:
        submitted = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "hello", "background": True},
        )
        completed = await poll(client, _RUN_1)

    assert submitted.status_code == 202
    assert submitted.json() == {
        "id": _RUN_1,
        "status": "queued",
        "status_url": f"/api/invocations/{_RUN_1}",
    }
    assert completed == {"id": _RUN_1, "status": "completed", "output": "hello"}


@pytest.mark.asyncio
async def test_foreground_stream_returns_sse() -> None:
    async def invoke(input, context):
        await context.emit({"type": "delta", "content": input})
        return [input]

    app = make_app(invoke)
    async with running_client(app) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "hello", "stream": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: run.started" in response.text
    assert 'event: delta\ndata: {"type": "delta", "content": "hello"}' in response.text
    assert "event: run.completed" in response.text


@pytest.mark.asyncio
async def test_background_stream_returns_202_with_polling_urls() -> None:
    app = make_app()
    async with running_client(app) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "hello", "background": True, "stream": True},
        )

    assert response.status_code == 202
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "id": _RUN_1,
        "status": "queued",
        "status_url": f"/api/invocations/{_RUN_1}",
        "events_url": f"/api/invocations/{_RUN_1}/events",
    }


@pytest.mark.asyncio
async def test_invocation_id_is_idempotency_key_for_every_mode() -> None:
    calls = 0

    async def invoke(input, context):
        nonlocal calls
        calls += 1
        return input

    app = make_app(invoke)
    async with running_client(app) as client:
        first = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "one"},
        )
        replay = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "one", "background": True, "stream": True},
        )
        conflict = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "two"},
        )

    assert first.status_code == 200
    assert replay.status_code == 202
    assert conflict.status_code == 409
    assert calls == 1


@pytest.mark.asyncio
async def test_retry_remains_idempotent_when_proxy_consumes_routing_cookie() -> None:
    calls = 0

    async def invoke(input, context):
        nonlocal calls
        calls += 1
        return input

    app = make_app(invoke)
    async with running_client(app) as client:
        first = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "one"},
        )
        client.cookies.clear()
        replay = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": "one", "background": True},
        )

    assert first.status_code == 200
    assert replay.status_code == 202
    assert calls == 1


@pytest.mark.asyncio
async def test_invocation_id_must_be_uuid() -> None:
    app = make_app()
    async with running_client(app) as client:
        submitted = await client.post("/api/invocations", json={"id": "not-a-uuid"})
        polled = await client.get("/api/invocations/not-a-uuid")

    assert submitted.status_code == 422
    assert polled.status_code == 422


@pytest.mark.asyncio
async def test_application_output_cannot_overwrite_protocol_metadata() -> None:
    async def invoke(input, context):
        return {"id": "application-id", "status": "application-status", "attempt": 99}

    app = make_app(invoke)
    async with running_client(app) as client:
        response = await client.post("/api/invocations", json={"id": _RUN_1})

    assert response.json() == {
        "id": _RUN_1,
        "status": "completed",
        "output": {"id": "application-id", "status": "application-status", "attempt": 99},
    }


@pytest.mark.asyncio
async def test_agent_failure_returns_500_and_failed_event() -> None:
    async def fail(input, context):
        raise RuntimeError("boom")

    app = make_app(fail)
    async with running_client(app) as client:
        response = await client.post("/api/invocations", json={"id": _RUN_1})
        events = await app._runtime.get_events(_RUN_1)

    assert response.status_code == 500
    assert response.json() == {"detail": "agent execution failed"}
    assert [event.event for event in events] == [
        {"type": "run.started"},
        {"type": "run.failed"},
    ]


def test_app_is_asgi_app_with_instance_scoped_decorators() -> None:
    app = AgentApp(durability_store=InMemoryDurabilityStore())

    @app.invoke
    async def invoke(input, context):
        return input

    @app.on_recovery
    async def recover(input, context):
        return input

    assert isinstance(app, FastAPI)
    assert app._invoke_hook is invoke
    assert app._on_recovery_hook is recover
    with pytest.raises(ValueError, match="already registered"):
        app.invoke(echo)


def test_app_exposes_only_api_invocation_routes() -> None:
    app = make_app()
    paths = app.openapi()["paths"]

    assert set(paths) == {
        "/api/invocations",
        "/api/invocations/{invocation_id}",
        "/api/invocations/{invocation_id}/events",
    }
    assert {getattr(route, "path", None) for route in app.routes} == set(paths)


def test_agent_app_defaults_to_process_local_state_even_inside_apps(monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_LOCAL_ENV, raising=False)
    monkeypatch.setenv("DATABRICKS_APP_NAME", "mason-agent")
    monkeypatch.delenv(RUNTIME_ENDPOINT_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_SCHEMA_ENV, raising=False)

    app = AgentApp()

    assert app.durable_runtime is False
    assert isinstance(app._runtime.durability_store, InMemoryDurabilityStore)


def test_deployed_app_without_durability_resource_fails_startup(monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_LOCAL_ENV, raising=False)
    monkeypatch.setenv("DATABRICKS_APP_NAME", "mason-agent")
    monkeypatch.delenv(RUNTIME_ENDPOINT_ENV, raising=False)

    with pytest.raises(RuntimeError, match=RUNTIME_ENDPOINT_ENV):
        AgentApp(durable_runtime=True)


def test_state_payload_nests_completed_application_response() -> None:
    state = DurableExecution(
        execution_id=_RUN_2,
        status=DurableExecutionStatus.COMPLETED,
        attempt=1,
        heartbeat_at=None,
        request={"input": {}, "session_id": "session-1"},
        response={"id": "application-id", "status": "application-status"},
    )

    assert AgentApp._state_payload(state) == {
        "id": _RUN_2,
        "status": "completed",
        "output": {"id": "application-id", "status": "application-status"},
    }
