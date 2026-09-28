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
    InvocationEvent,
    InvocationStatus,
)

_ROUTING_COOKIE = "__Host-databricks-app-router"
_SESSION_HEADER = "X-Databricks-Session-Id"
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
async def running_client(
    app: DurableAgentServer, session_id: str | None = "test-session"
) -> AsyncIterator[httpx.AsyncClient]:
    runtime = app._runtime
    assert runtime is not None
    await runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://testserver",
            headers={_SESSION_HEADER: session_id} if session_id is not None else {},
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
async def test_session_header_sets_context_independently_of_routing_cookie() -> None:
    async def invoke(input, context):
        return {
            "received": input,
            "invocation_id": context.invocation_id,
            "session_id": context.session_id,
        }

    app = make_app(invoke)
    async with running_client(app, session_id="session-1") as client:
        client.cookies.set(_ROUTING_COOKIE, "routing-only")
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
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("legacy_source", ["none", "cookie", "input"])
async def test_missing_session_header_is_rejected_without_fallback(
    background, stream, legacy_source
) -> None:
    seen_sessions = []

    async def invoke(input, context):
        seen_sessions.append(context.session_id)
        return input

    app = make_app(invoke)
    async with running_client(app, session_id=None) as client:
        if legacy_source == "cookie":
            client.cookies.set(_ROUTING_COOKIE, "cookie-session")
        body = {"id": _RUN_1, "background": background, "stream": stream}
        if legacy_source == "input":
            body["input"] = {"session_id": "payload-session"}
        response = await client.post("/api/invocations", json=body)
        state = await app._runtime.get_invocation(_RUN_1)

    assert seen_sessions == []
    assert state is None
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["header", _SESSION_HEADER]
    assert _ROUTING_COOKIE not in response.cookies


@pytest.mark.asyncio
async def test_header_session_is_saved_once_and_input_does_not_override_context() -> None:
    async def invoke(input, context):
        return {"input": input, "session_id": context.session_id}

    app = make_app(invoke)
    payload = {"session_id": "application-input-only", "messages": ["hello"]}
    async with running_client(app, session_id="header-session") as client:
        client.cookies.set(_ROUTING_COOKIE, "routing-only")
        response = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "input": payload},
        )
        state = await app._runtime.get_invocation(_RUN_1)

    assert response.status_code == 200
    assert response.json()["output"] == {"input": payload, "session_id": "header-session"}
    assert state is not None
    assert state.session_id == "header-session"
    assert state.session_sequence_number == 1
    assert state.request == {"input": payload}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session_id",
    ["", "   ", " leading", "trailing ", "a\nb", "a\rb", "a\tb", "a\x7fb", "one,two", "one, two"],
)
async def test_invalid_session_header_is_rejected(session_id) -> None:
    app = make_app()
    async with running_client(app, session_id=None) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": _RUN_1},
            headers={_SESSION_HEADER: session_id},
        )
        state = await app._runtime.get_invocation(_RUN_1)

    assert response.status_code == 422
    assert state is None


@pytest.mark.asyncio
@pytest.mark.parametrize("second_session", ["session-1", "session-2"])
async def test_duplicate_session_headers_are_rejected(second_session):
    app = make_app()
    async with running_client(app, session_id=None) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": _RUN_1},
            headers=[(_SESSION_HEADER, "session-1"), (_SESSION_HEADER.lower(), second_session)],
        )
        state = await app._runtime.get_invocation(_RUN_1)

    assert response.status_code == 422
    assert state is None


@pytest.mark.asyncio
@pytest.mark.parametrize("body_session", ["test-session", "another-session", None, 7, [], {}])
async def test_top_level_body_session_is_rejected_even_with_valid_header(body_session):
    app = make_app()
    async with running_client(app) as client:
        response = await client.post(
            "/api/invocations", json={"id": _RUN_1, "session_id": body_session}
        )
        state = await app._runtime.get_invocation(_RUN_1)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "session_id"]
    assert state is None


@pytest.mark.asyncio
async def test_transport_resume_metadata_is_rejected() -> None:
    app = make_app()
    async with running_client(app) as client:
        resume = await client.post(
            "/api/invocations",
            json={"id": _RUN_1, "resume": {"answer": "yes"}},
        )

    assert resume.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fail_first", [False, True])
async def test_header_session_queues_every_transport_mode(background, stream, fail_first):
    entered = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def invoke(input, context):
        seen.append((input, context.session_id))
        if input == "first":
            entered.set()
            await release.wait()
            if fail_first:
                raise RuntimeError("first turn failed")
        await context.emit({"type": "message", "content": input})
        return input

    app = make_app(invoke)
    async with running_client(app, session_id="conversation") as client:
        first = await client.post(
            "/api/invocations",
            json={
                "id": _RUN_1,
                "input": "first",
                "background": True,
            },
        )
        assert first.status_code == 202
        await asyncio.wait_for(entered.wait(), 2)
        first_state = await app._runtime.get_invocation(_RUN_1)
        assert first_state is not None
        assert first_state.session_id == "conversation"
        assert first_state.session_sequence_number == 1
        assert first_state.request == {"input": "first"}
        second_request = asyncio.create_task(
            client.post(
                "/api/invocations",
                json={
                    "id": _RUN_2,
                    "input": "second",
                    "background": background,
                    "stream": stream,
                },
            )
        )
        try:
            for _ in range(100):
                second_state = await app._runtime.get_invocation(_RUN_2)
                if second_state is not None:
                    break
                await asyncio.sleep(0.005)
            assert second_state is not None
            assert second_state.status == InvocationStatus.QUEUED
            assert second_state.session_id == "conversation"
            assert second_state.session_sequence_number == 2
            assert second_state.request == {"input": "second"}
            assert seen == [("first", "conversation")]
        finally:
            release.set()

        response = await asyncio.wait_for(second_request, 2)
        completed = await poll(client, _RUN_2)
        first_state = await poll(client, _RUN_1)

    assert response.status_code == (202 if background else 200)
    if stream and not background:
        assert "event: run.completed" in response.text
    assert completed["output"] == "second"
    assert first_state["status"] == ("failed" if fail_first else "completed")
    assert seen == [("first", "conversation"), ("second", "conversation")]


@pytest.mark.asyncio
@pytest.mark.parametrize("nested_session", [False, True])
async def test_different_header_sessions_execute_concurrently(nested_session):
    both_entered = asyncio.Event()
    seen = []

    async def invoke(input, context):
        seen.append(context.session_id)
        if len(seen) == 2:
            both_entered.set()
        await both_entered.wait()
        return input

    app = make_app(invoke)
    async with running_client(app) as client:
        client.cookies.set(_ROUTING_COOKIE, "same-legacy-cookie")
        payload = {"session_id": "same-application-session"} if nested_session else {}
        responses = await asyncio.wait_for(
            asyncio.gather(
                *(
                    client.post(
                        "/api/invocations",
                        json={"id": invocation_id, "input": payload},
                        headers={_SESSION_HEADER: f"session-{i}"},
                    )
                    for i, invocation_id in enumerate((_RUN_1, _RUN_2))
                )
            ),
            2,
        )
        states = [await app._runtime.get_invocation(id) for id in (_RUN_1, _RUN_2)]

    assert all(response.status_code == 200 for response in responses)
    assert set(seen) == {"session-0", "session-1"}
    for i, state in enumerate(states):
        assert state is not None
        assert state.session_id == f"session-{i}"
        assert state.session_sequence_number == 1


@pytest.mark.asyncio
async def test_header_session_is_part_of_invocation_idempotency():
    calls = []

    async def invoke(input, context):
        calls.append(context.session_id)
        return input

    app = make_app(invoke)
    body = {"id": _RUN_1, "input": "hello"}
    async with running_client(app, session_id="conversation") as client:
        first = await client.post("/api/invocations", json=body)
        replay = await client.post("/api/invocations", json={**body, "background": True})
        conflict = await client.post(
            "/api/invocations", json=body, headers={_SESSION_HEADER: "another-conversation"}
        )
        del client.headers[_SESSION_HEADER]
        missing_header_retry = await client.post("/api/invocations", json=body)

    assert first.status_code == 200
    assert replay.status_code == 202
    assert conflict.status_code == 409
    assert missing_header_retry.status_code == 422
    assert calls == ["conversation"]


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
        {"input": "hello"},
        InvocationAttemptContext(_RUN_1, 2, session_id="session-1"),
    )

    assert result == {"input": "hello", "session_id": "session-1"}
    assert calls == ["recover"]


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", [1, 2])
async def test_attempt_context_is_the_only_source_of_session_identity(attempt) -> None:
    async def invoke(input, context):
        return context.session_id

    app = make_app(invoke, recover=invoke)

    result = await app._execute(
        {"input": "hello", "session_id": "legacy-session"},
        InvocationAttemptContext(_RUN_1, attempt, session_id="stored-session"),
    )

    assert result == "stored-session"


@pytest.mark.asyncio
@pytest.mark.parametrize("attempt", [1, 2])
async def test_attempt_without_saved_session_does_not_use_payload_fallback(attempt):
    app = make_app()

    with pytest.raises(TypeError, match="invocation attempt must contain session_id"):
        await app._execute(
            {"input": "hello", "session_id": "legacy-session"},
            InvocationAttemptContext(_RUN_1, attempt),
        )


@pytest.mark.asyncio
async def test_recovery_attempt_requires_a_recovery_hook() -> None:
    app = make_app()

    with pytest.raises(RuntimeError, match="@app.recover"):
        await app._execute(
            {"input": {}},
            InvocationAttemptContext(_RUN_1, 2, session_id="session-1"),
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
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("fail", [False, True])
async def test_stream_includes_events_committed_after_event_read(background, fail) -> None:
    snapshot_taken = asyncio.Event()
    release_snapshot = asyncio.Event()

    class PausingEventReadStore(InMemoryRuntimeStore):
        async def events(
            self,
            invocation_id: str | None = None,
            after_sequence: int | None = None,
            session_id: str | None = None,
        ) -> list[InvocationEvent]:
            events = await super().events(invocation_id, after_sequence, session_id)
            if not snapshot_taken.is_set():
                # Keep a real event snapshot while the handler commits its terminal state.
                snapshot_taken.set()
                await release_snapshot.wait()
            return events

    store = PausingEventReadStore()
    app = DurableAgentServer(runtime_store=store)

    @app.invoke
    async def invoke(input, context):
        await snapshot_taken.wait()
        await context.emit({"type": "delta", "content": input})
        if fail:
            raise RuntimeError("intentional execution failure")
        return input

    async with running_client(app) as client:
        if background:
            accepted = await client.post(
                "/api/invocations",
                json={"id": _RUN_1, "input": "hello", "background": True},
            )
            assert accepted.status_code == 202
            request = asyncio.create_task(client.get(f"/api/invocations/{_RUN_1}/events"))
        else:
            request = asyncio.create_task(
                client.post(
                    "/api/invocations",
                    json={"id": _RUN_1, "input": "hello", "stream": True},
                )
            )
        try:
            await asyncio.wait_for(snapshot_taken.wait(), 2)
            completed = await poll(client, _RUN_1)
            release_snapshot.set()
            response = await asyncio.wait_for(request, 2)
        finally:
            release_snapshot.set()
            if not request.done():
                request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        events = await store.events(_RUN_1)
        replay = await client.get(
            f"/api/invocations/{_RUN_1}/events?after={events[-2].sequence_number}"
        )
        past_terminal = await client.get(
            f"/api/invocations/{_RUN_1}/events?after={events[-1].sequence_number}"
        )

    terminal_type = "run.failed" if fail else "run.completed"
    assert completed["status"] == ("failed" if fail else "completed")
    if not fail:
        assert completed["output"] == "hello"
    assert response.status_code == 200
    assert [line[7:] for line in response.text.splitlines() if line.startswith("event: ")] == [
        "run.started",
        "delta",
        terminal_type,
    ]
    assert [int(line[4:]) for line in response.text.splitlines() if line.startswith("id: ")] == [
        event.sequence_number for event in events
    ]
    assert replay.text == (
        f"id: {events[-1].sequence_number}\nevent: {terminal_type}\n"
        f'data: {{"type": "{terminal_type}"}}\n\n'
    )
    assert past_terminal.text == ""


@pytest.mark.asyncio
async def test_stream_closes_when_invocation_disappears() -> None:
    store = InMemoryRuntimeStore()
    app = DurableAgentServer(runtime_store=store)

    @app.invoke
    async def invoke(input, context):
        await context.emit({"type": "delta", "content": input})
        store.states.pop(context.invocation_id)
        return input

    async with running_client(app) as client:
        response = await asyncio.wait_for(
            client.post(
                "/api/invocations",
                json={"id": _RUN_1, "input": "hello", "stream": True},
            ),
            2,
        )

    assert response.status_code == 200
    assert [line[7:] for line in response.text.splitlines() if line.startswith("event: ")] == [
        "run.started",
        "delta",
    ]


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
        client.cookies.set(_ROUTING_COOKIE, "routing-cookie")
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
async def test_invocation_status_and_event_reads_do_not_require_session_header():
    app = make_app()
    async with running_client(app) as client:
        submitted = await client.post("/api/invocations", json={"id": _RUN_1, "input": "hello"})
        del client.headers[_SESSION_HEADER]
        status = await client.get(f"/api/invocations/{_RUN_1}")
        events = await client.get(f"/api/invocations/{_RUN_1}/events")

    assert submitted.status_code == status.status_code == events.status_code == 200
    assert status.json()["output"] == "hello"
    assert "event: run.completed" in events.text


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


def test_openapi_declares_required_session_header_only_for_submission():
    paths = make_app().openapi()["paths"]
    submission_headers = [
        parameter
        for parameter in paths["/api/invocations"]["post"]["parameters"]
        if parameter["in"] == "header"
    ]

    assert len(submission_headers) == 1
    assert submission_headers[0]["name"] == _SESSION_HEADER
    assert submission_headers[0]["required"] is True
    for path in ("/api/invocations/{invocation_id}", "/api/invocations/{invocation_id}/events"):
        assert not any(
            parameter["in"] == "header" for parameter in paths[path]["get"]["parameters"]
        )


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
        request={"input": {}},
        session_id="session-1",
        response={"id": "application-id", "status": "application-status"},
    )

    assert DurableAgentServer._state_payload(state) == {
        "id": _RUN_2,
        "status": "completed",
        "output": {"id": "application-id", "status": "application-status"},
    }
