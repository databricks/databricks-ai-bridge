"""Tests for the SDK-provided agent application."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI

from databricks_agentbricks.agent_project import AgentProject, ToolSpec
from databricks_agentkit import DurableAgentServer
from databricks_agentkit.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
    RUNTIME_STORE_LOCAL_ENV,
    RUNTIME_STORE_SCHEMA_ENV,
    RUNTIME_STORE_USERNAME_ENV,
    InMemoryRuntimeStore,
)
from databricks_agentkit.runtime.types import (
    Invocation,
    InvocationAttemptContext,
    InvocationStatus,
)

_ROUTING_COOKIE = "__Host-databricks-app-router"
_RUN_1 = "11111111-1111-4111-8111-111111111111"
_RUN_2 = "22222222-2222-4222-8222-222222222222"


async def echo(input, context):
    return input


def make_app(invoke=echo, *, recover=None) -> DurableAgentServer:
    app = DurableAgentServer(runtime_store=InMemoryRuntimeStore())
    app.invoke(invoke)
    if recover is not None:
        app.recover(recover)
    return app


@asynccontextmanager
async def running_client(app: DurableAgentServer) -> AsyncIterator[httpx.AsyncClient]:
    runtime = app._runtime
    assert runtime is not None
    await runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://testserver",
        ) as client:
            yield client
    finally:
        await runtime.stop()


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
async def test_recovery_attempt_uses_recovery_hook() -> None:
    calls = []

    async def invoke(input, context):
        calls.append("invoke")
        return input

    async def recover(input, context):
        calls.append("recover")
        return {"input": input, "session_id": context.session_id}

    app = make_app(invoke, recover=recover)
    result = await app._execute(
        {"input": "hello", "session_id": "session-1"},
        InvocationAttemptContext(_RUN_1, 2),
    )

    assert result == {"input": "hello", "session_id": "session-1"}
    assert calls == ["recover"]


@pytest.mark.asyncio
async def test_recovery_attempt_requires_a_recovery_hook() -> None:
    app = make_app()

    with pytest.raises(RuntimeError, match="@app.recover"):
        await app._execute(
            {"input": {}, "session_id": "session-1"},
            InvocationAttemptContext(_RUN_1, 2),
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
        assert app._runtime is not None
        events = await app._runtime.get_events(_RUN_1)

    assert response.status_code == 500
    assert response.json() == {"detail": "agent invocation failed"}
    assert [event.event for event in events] == [
        {"type": "run.started"},
        {"type": "run.failed"},
    ]


def test_app_is_asgi_app_with_instance_scoped_decorators() -> None:
    app = DurableAgentServer(runtime_store=InMemoryRuntimeStore())

    @app.invoke
    async def invoke(input, context):
        return input

    @app.recover
    async def recover(input, context):
        return input

    assert isinstance(app, FastAPI)
    assert app._invoke_hook is invoke
    assert app._recovery_hook is recover
    with pytest.raises(ValueError, match="already registered"):
        app.invoke(echo)


def test_app_allows_custom_routes_alongside_invocation_routes() -> None:
    app = make_app()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    paths = app.openapi()["paths"]

    assert set(paths) == {
        "/api/invocations",
        "/api/invocations/{invocation_id}",
        "/api/invocations/{invocation_id}/events",
        "/health",
    }
    assert "/health" in {getattr(route, "path", None) for route in app.routes}


def test_durable_agent_server_defaults_to_process_local_state_outside_apps(monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_STORE_LOCAL_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_LAKEBASE_BRANCH_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_DATABASE_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_USERNAME_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_STORE_SCHEMA_ENV, raising=False)

    app = DurableAgentServer()

    assert app._runtime is not None
    assert app._runtime.is_durable is False
    assert isinstance(app._runtime.runtime_store, InMemoryRuntimeStore)


def test_durable_agent_server_infers_request_user_policy_from_manifest(
    tmp_path, monkeypatch
) -> None:
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.add_tool(ToolSpec.mcp("search", service="system.ai.web_search", auth="user"))
    project.add_tool(ToolSpec.mcp("docs", service="system.ai.docs", auth="app"))
    project.write()
    monkeypatch.chdir(tmp_path)

    app = DurableAgentServer(runtime_store=InMemoryRuntimeStore())

    assert app.auth_policy.user_tools == ("search",)


def test_state_payload_nests_completed_application_response() -> None:
    state = Invocation(
        invocation_id=_RUN_2,
        status=InvocationStatus.COMPLETED,
        attempt=1,
        request={"input": {}, "session_id": "session-1"},
        response={"id": "application-id", "status": "application-status"},
    )

    assert DurableAgentServer._state_payload(state) == {
        "id": _RUN_2,
        "status": "completed",
        "output": {"id": "application-id", "status": "application-status"},
    }
