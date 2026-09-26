import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
import pytest

from databricks_agentkit import DurableAgentServer
from databricks_agentkit.runtime.auth import AuthError, InvocationAuthPolicy, RequestAuthContext
from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from databricks_agentkit.runtime.types import InvocationAttemptContext


@pytest.fixture
def deployed(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "auth-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")


def headers(subject="user-a", token="token-sentinel"):
    return {"x-forwarded-user": subject, "x-forwarded-access-token": token}


def make_app(handler):
    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(),
        auth_policy=InvocationAuthPolicy(user_tools=("sandbox",)),
    )
    app.invoke(handler)
    return app


def assert_auth_not_persisted(store: InMemoryRuntimeStore, token="token-sentinel"):
    persisted = {
        "states": store.states,
        "events": store.persisted_events,
    }
    assert token not in repr(persisted)


@asynccontextmanager
async def running_client(app: DurableAgentServer) -> AsyncIterator[httpx.AsyncClient]:
    await app._runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="https://test"
        ) as client:
            yield client
    finally:
        await app._runtime.stop()


@pytest.mark.asyncio
async def test_request_user_sync_uses_runtime_without_persisting_auth(deployed):
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        sequences = [
            await context.emit({"type": "delta", "content": "one"}),
            await context.emit({"type": "delta", "content": "two"}),
        ]
        return {"value": value, "sequences": sequences}

    app = make_app(handler)
    invocation_id = str(uuid4())
    body = {"id": invocation_id, "input": "hello"}
    async with running_client(app) as client:
        first = await client.post("/api/invocations", json=body, headers=headers())
        status = await client.get(f"/api/invocations/{invocation_id}", headers=headers())
        events = await client.get(f"/api/invocations/{invocation_id}/events", headers=headers())
        second = await client.post(
            "/api/invocations", json=body, headers=headers(token="refreshed")
        )

    assert first.status_code == second.status_code == 200
    assert (
        first.json()
        == second.json()
        == {
            "id": invocation_id,
            "status": "completed",
            "output": {"value": "hello", "sequences": [2, 3]},
        }
    )
    assert status.status_code == events.status_code == 200
    assert status.json() == first.json()
    assert 'event: delta\ndata: {"type": "delta", "content": "one"}' in events.text
    assert 'event: delta\ndata: {"type": "delta", "content": "two"}' in events.text
    assert len(contexts) == 1
    store = app._runtime.runtime_store
    assert isinstance(store, InMemoryRuntimeStore)
    assert_auth_not_persisted(store)
    with pytest.raises(AuthError):
        contexts[0].request_auth.client_for("user")


@pytest.mark.asyncio
async def test_missing_request_user_auth_never_executes(deployed):
    seen = []

    async def handler(value, context):
        seen.append(value)

    app = make_app(handler)
    async with running_client(app) as client:
        response = await client.post("/api/invocations", json={"id": str(uuid4())})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MCP_USER_AUTH_REQUIRED"
    assert not seen


@pytest.mark.asyncio
async def test_request_user_auth_composes_with_existing_runtime_background_mode(deployed):
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        await context.emit({"type": "delta", "content": value})
        return {"value": value}

    store = InMemoryRuntimeStore()
    app = DurableAgentServer(
        runtime_store=store,
        auth_policy=InvocationAuthPolicy(user_tools=("sandbox",)),
    )
    app.invoke(handler)
    invocation_id = str(uuid4())
    async with running_client(app) as client:
        submitted = await client.post(
            "/api/invocations",
            json={"id": invocation_id, "input": "hello", "background": True},
            headers=headers(),
        )
        for _ in range(100):
            completed = await client.get(f"/api/invocations/{invocation_id}", headers=headers())
            if completed.json().get("status") == "completed":
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("request-user invocation did not finish")

    assert submitted.status_code == 202
    assert submitted.json() == {
        "id": invocation_id,
        "status": "queued",
        "status_url": f"/api/invocations/{invocation_id}",
    }
    assert completed.status_code == 200
    assert completed.json() == {
        "id": invocation_id,
        "status": "completed",
        "output": {"value": "hello"},
    }
    assert_auth_not_persisted(store)
    assert len(contexts) == 1
    with pytest.raises(AuthError):
        contexts[0].request_auth.client_for("user")


@pytest.mark.asyncio
async def test_request_user_streams_ordered_persisted_events(deployed):
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        assert await context.emit({"type": "delta", "content": value}) == 2
        return {"ignored": "stream output"}

    app = make_app(handler)
    invocation_id = str(uuid4())
    async with running_client(app) as client:
        client.cookies.set("__Host-databricks-app-router", "routing-session")
        response = await client.post(
            "/api/invocations",
            json={"id": invocation_id, "input": "hello", "stream": True},
            headers=headers(),
        )
        status = await client.get(f"/api/invocations/{invocation_id}", headers=headers())
        events = await client.get(f"/api/invocations/{invocation_id}/events", headers=headers())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'id: 1\nevent: run.started\ndata: {"type": "run.started"}\n\n'
        'id: 2\nevent: delta\ndata: {"type": "delta", "content": "hello"}\n\n'
        'id: 3\nevent: run.completed\ndata: {"type": "run.completed"}\n\n'
    )
    assert status.status_code == events.status_code == 200
    assert status.json() == {
        "id": invocation_id,
        "status": "completed",
        "output": {"ignored": "stream output"},
    }
    assert events.text == response.text
    assert contexts[0].session_id == contexts[0].request_auth.namespace(
        "session", "routing-session"
    )
    with pytest.raises(AuthError):
        contexts[0].request_auth.client_for("user")


@pytest.mark.asyncio
async def test_request_user_stream_does_not_treat_handler_event_as_terminal(deployed):
    async def handler(value, context):
        assert await context.emit({"type": "run.completed", "source": "handler"}) == 2
        assert await context.emit({"type": "delta", "content": "after-terminal-looking-event"}) == 3

    app = make_app(handler)
    async with running_client(app) as client:
        response = await asyncio.wait_for(
            client.post(
                "/api/invocations",
                json={"id": str(uuid4()), "stream": True},
                headers=headers(),
            ),
            timeout=2,
        )

    assert response.text == (
        'id: 1\nevent: run.started\ndata: {"type": "run.started"}\n\n'
        'id: 2\nevent: run.completed\ndata: {"type": "run.completed", "source": "handler"}\n\n'
        'id: 3\nevent: delta\ndata: {"type": "delta", '
        '"content": "after-terminal-looking-event"}\n\n'
        'id: 4\nevent: run.completed\ndata: {"type": "run.completed"}\n\n'
    )


@pytest.mark.asyncio
async def test_missing_request_user_auth_fails_before_stream_starts(deployed):
    app = make_app(lambda value, context: value)
    async with running_client(app) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": str(uuid4()), "stream": True},
        )

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "MCP_USER_AUTH_REQUIRED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        AuthError("MCP_PERMISSION_DENIED", "Permission denied", 403, "sandbox"),
        RuntimeError("token-sentinel"),
    ],
)
async def test_request_user_stream_failure_is_credential_free(deployed, failure):
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        raise failure

    app = make_app(handler)
    async with running_client(app) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": str(uuid4()), "stream": True},
            headers=headers(),
        )

    frames = [frame for frame in response.text.split("\n\n") if frame]
    assert response.status_code == 200
    assert len(frames) == 2
    assert frames[0].endswith('data: {"type": "run.started"}')
    assert frames[1].endswith('data: {"type": "run.failed"}')
    assert "token-sentinel" not in response.text
    with pytest.raises(AuthError):
        contexts[0].request_auth.client_for("user")


@pytest.mark.asyncio
async def test_runtime_stop_cancels_request_user_attempt_before_auth_closes(deployed):
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            assert repr(context.request_auth) == "RequestAuthContext(closed=False)"
            cleaned.set()

    app = make_app(handler)
    await app._runtime.start()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        submitted = await client.post(
            "/api/invocations",
            json={"id": str(uuid4()), "background": True},
            headers=headers(),
        )
        assert submitted.status_code == 202
        await asyncio.wait_for(entered.wait(), 2)
        assert repr(contexts[0].request_auth) == "RequestAuthContext(closed=False)"
    await app._runtime.stop()

    assert cleaned.is_set()
    with pytest.raises(AuthError):
        contexts[0].request_auth.client_for("user")


@pytest.mark.asyncio
async def test_request_user_auth_error_is_safe_and_closes_credentials(deployed):
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        raise AuthError("MCP_PERMISSION_DENIED", "Permission denied", 403, "sandbox")

    app = make_app(handler)
    async with running_client(app) as client:
        response = await client.post(
            "/api/invocations", json={"id": str(uuid4())}, headers=headers()
        )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "MCP_PERMISSION_DENIED",
            "message": "Permission denied",
            "integration_id": "sandbox",
        }
    }
    assert "token-sentinel" not in response.text
    with pytest.raises(AuthError):
        contexts[0].request_auth.client_for("user")


@pytest.mark.asyncio
async def test_cancelled_request_closes_auth_and_handler(deployed):
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    app = make_app(handler)
    async with running_client(app) as client:
        task = asyncio.create_task(
            client.post("/api/invocations", json={"id": str(uuid4())}, headers=headers())
        )
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert cleaned.is_set()
    with pytest.raises(AuthError):
        contexts[0].request_auth.client_for("user")


@pytest.mark.asyncio
async def test_concurrent_request_users_have_isolated_auth_and_sessions(deployed):
    contexts = []
    both_entered = asyncio.Event()

    async def handler(value, context):
        contexts.append(context)
        if len(contexts) == 2:
            both_entered.set()
        await both_entered.wait()
        return context.session_id

    app = make_app(handler)
    body = {"id": str(uuid4())}
    async with running_client(app) as client:
        first, second = await asyncio.wait_for(
            asyncio.gather(
                client.post("/api/invocations", json=body, headers=headers("user-a")),
                client.post("/api/invocations", json=body, headers=headers("user-b")),
            ),
            2,
        )

    assert first.status_code == second.status_code == 200
    assert first.json()["output"] != second.json()["output"]
    assert contexts[0].request_auth is not contexts[1].request_auth


@pytest.mark.asyncio
async def test_request_user_recovery_fails_before_handlers(deployed):
    calls = []

    async def invoke(value, context):
        calls.append("invoke")

    async def recover(value, context):
        calls.append("recover")

    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(),
        auth_policy=InvocationAuthPolicy(("sandbox",)),
    )
    app.invoke(invoke)
    app.recover(recover)
    invocation_id = str(uuid4())
    request_auth = RequestAuthContext.from_headers(headers())
    runtime_invocation_id = request_auth.namespace("invocation", invocation_id)
    app._request_auth[runtime_invocation_id] = request_auth

    with pytest.raises(AuthError) as caught:
        await app._execute(
            {"input": "hello", "session_id": "session-1", "invocation_id": invocation_id},
            InvocationAttemptContext(runtime_invocation_id, 2),
        )

    assert caught.value.code == "MCP_USER_AUTH_RECOVERY_UNSUPPORTED"
    assert str(caught.value) == "Request-user execution does not survive failure recovery yet"
    assert not calls
    assert not app._request_auth
    with pytest.raises(AuthError, match="no longer active"):
        request_auth.client_for("user")
