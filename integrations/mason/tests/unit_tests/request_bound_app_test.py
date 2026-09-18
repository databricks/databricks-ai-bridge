import asyncio
from uuid import uuid4

import httpx
import pytest
from fastapi import Request

from databricks_mason import AgentApp
from databricks_mason.runtime.app import _InvocationRequest
from databricks_mason.runtime.auth import AuthError, InvocationAuthPolicy


@pytest.fixture
def deployed(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "auth-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")


def headers(subject="user-a", token="token-sentinel"):
    return {"x-forwarded-user": subject, "x-forwarded-access-token": token}


def make_app(handler):
    app = AgentApp(auth_policy=InvocationAuthPolicy(user_tools=("sandbox",)))
    app.invoke(handler)
    return app


@pytest.mark.asyncio
async def test_request_user_sync_executes_directly_without_retaining_state(deployed):
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
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
            "output": {"value": "hello", "sequences": [1, 2]},
        }
    )
    assert status.status_code == events.status_code == 404
    assert len(contexts) == 2
    assert contexts[0].request_auth is not contexts[1].request_auth
    assert not hasattr(app, "_request_execution")
    assert app._runtime is None
    for context in contexts:
        with pytest.raises(AuthError):
            context.request_auth.client_for("user")


@pytest.mark.asyncio
async def test_missing_request_user_auth_never_executes(deployed):
    seen = []

    async def handler(value, context):
        seen.append(value)

    app = make_app(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post("/api/invocations", json={"id": str(uuid4())})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MCP_USER_AUTH_REQUIRED"
    assert not seen


@pytest.mark.asyncio
async def test_request_user_rejects_background_execution(deployed):
    seen = []

    async def handler(value, context):
        seen.append(value)

    app = make_app(handler)
    body = {"id": str(uuid4()), "background": True}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post("/api/invocations", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED"
    assert response.json()["error"]["message"] == "Request-user tools require foreground execution"
    assert not seen


@pytest.mark.asyncio
async def test_request_user_streams_ordered_events_without_retaining_state(deployed):
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        assert await context.emit({"type": "delta", "content": value}) == 2
        return {"ignored": "stream output"}

    app = make_app(handler)
    invocation_id = str(uuid4())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
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
    assert status.status_code == events.status_code == 404
    assert app._runtime is None
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": str(uuid4()), "stream": True},
        )

    assert response.status_code == 401
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "MCP_USER_AUTH_REQUIRED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            AuthError("MCP_PERMISSION_DENIED", "Permission denied", 403, "sandbox"),
            {
                "type": "run.failed",
                "error": "Permission denied",
                "code": "MCP_PERMISSION_DENIED",
                "integration_id": "sandbox",
            },
        ),
        (
            RuntimeError("token-sentinel"),
            {"type": "run.failed", "error": "agent invocation failed"},
        ),
    ],
)
async def test_request_user_stream_failure_is_credential_free(deployed, failure, expected):
    contexts = []

    async def handler(value, context):
        contexts.append(context)
        raise failure

    app = make_app(handler)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post(
            "/api/invocations",
            json={"id": str(uuid4()), "stream": True},
            headers=headers(),
        )

    frames = [frame for frame in response.text.split("\n\n") if frame]
    assert response.status_code == 200
    assert len(frames) == 2
    assert frames[0].endswith('data: {"type": "run.started"}')
    assert frames[1].endswith(f"data: {__import__('json').dumps(expected)}")
    assert "token-sentinel" not in response.text
    with pytest.raises(AuthError):
        contexts[0].request_auth.client_for("user")


@pytest.mark.asyncio
async def test_closing_request_user_stream_cancels_handler_before_auth_closes(deployed):
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
    invocation_id = uuid4()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/invocations",
        "headers": [(name.encode(), value.encode()) for name, value in headers().items()],
    }
    request = Request(scope)
    request.state.session_id = None
    response = await app._invoke_request_user(
        request,
        _InvocationRequest(id=invocation_id, stream=True),
        str(invocation_id),
    )

    first = await anext(response.body_iterator)
    await asyncio.wait_for(entered.wait(), 2)
    await response.body_iterator.aclose()

    assert first == 'id: 1\nevent: run.started\ndata: {"type": "run.started"}\n\n'
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
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


def test_user_policy_rejects_recovery_and_explicit_store(deployed):
    from databricks_mason.runtime.store import InMemoryRuntimeStore

    with pytest.raises(ValueError, match="Runtime Store"):
        AgentApp(
            auth_policy=InvocationAuthPolicy(("sandbox",)), runtime_store=InMemoryRuntimeStore()
        )
    app = make_app(None)
    with pytest.raises(ValueError, match="recovery"):
        app.recover(lambda value, context: value)
