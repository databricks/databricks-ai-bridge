import asyncio
from uuid import uuid4

import httpx
import pytest

from databricks_mason import AgentApp
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
    assert not app._runtime.runtime_store.states
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
@pytest.mark.parametrize(
    ("option", "code"),
    [
        ("background", "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED"),
        ("stream", "MCP_USER_AUTH_STREAMING_UNSUPPORTED"),
    ],
)
async def test_request_user_rejects_non_sync_modes(deployed, option, code):
    seen = []

    async def handler(value, context):
        seen.append(value)

    app = make_app(handler)
    body = {"id": str(uuid4()), option: True}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="https://test"
    ) as client:
        response = await client.post("/api/invocations", json=body, headers=headers())

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    assert not seen


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
