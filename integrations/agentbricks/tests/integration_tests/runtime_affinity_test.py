"""HTTP/runtime affinity tests against a real shared PostgreSQL store.

These tests are opt-in through ``AGENTKIT_TEST_POSTGRES_URL``.  The fixture in
``integration_tests/conftest.py`` gives every test a disposable schema and a
separate store connection for each worker.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
import pytest

from databricks_agentkit import DurableAgentServer
from databricks_agentkit.runtime.auth import (
    AuthError,
    InvocationAuthPolicy,
    RequestAuthContext,
)
from databricks_agentkit.runtime.durability.execution import DurableInvocationExecutor
from databricks_agentkit.runtime.types import InvocationStatus

pytestmark = pytest.mark.asyncio


def _headers(*, token: str, user: str = "user-a") -> dict[str, str]:
    return {
        "x-forwarded-access-token": token,
        "x-forwarded-user": user,
    }


def _runtime_invocation_id(invocation_id: str, headers: dict[str, str]) -> str:
    auth = RequestAuthContext.from_headers(headers)
    try:
        return auth.namespace("invocation", invocation_id)
    finally:
        auth.close()


def _runtime_id_is_clean(app: DurableAgentServer, invocation_id: str) -> bool:
    return invocation_id not in app._request_auth


def _make_app(
    store: Any,
    invoke,
    *,
    recover=None,
    user_auth: bool = True,
) -> DurableAgentServer:
    app = DurableAgentServer(
        runtime_store=store,
        auth_policy=InvocationAuthPolicy(user_tools=("sandbox",))
        if user_auth
        else InvocationAuthPolicy(),
    )
    app.invoke(invoke)
    if recover is not None:
        app.recover(recover)

    # Keep failure/recovery tests fast while preserving a real heartbeat window.
    executor = app._runtime.executor
    assert isinstance(executor, DurableInvocationExecutor)
    executor._heartbeat_seconds = 1.0
    executor._stale_seconds = 30.0
    executor._scan_seconds = 0.01
    app._runtime.poll_seconds = 0.005
    return app


@asynccontextmanager
async def _running_apps(*apps: DurableAgentServer) -> AsyncIterator[None]:
    async with AsyncExitStack() as stack:
        for app in apps:
            await stack.enter_async_context(app.router.lifespan_context(app))
        yield


@asynccontextmanager
async def _client(app: DurableAgentServer) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://testserver",
    ) as client:
        yield client


async def _wait_for_state(
    store: Any,
    invocation_id: str,
    *,
    status: InvocationStatus | None = None,
    attempt_at_least: int | None = None,
    timeout: float = 5,
):
    deadline = asyncio.get_running_loop().time() + timeout
    latest = None
    while asyncio.get_running_loop().time() < deadline:
        latest = await store.get(invocation_id)
        if (
            latest is not None
            and (status is None or latest.status == status)
            and (attempt_at_least is None or latest.attempt >= attempt_at_least)
        ):
            return latest
        await asyncio.sleep(0.01)
    raise AssertionError(f"invocation {invocation_id} did not reach {status}: {latest!r}")


async def _wait_for_map_entry(app: DurableAgentServer, invocation_id: str):
    deadline = asyncio.get_running_loop().time() + 2
    while asyncio.get_running_loop().time() < deadline:
        auth = app._request_auth.get(invocation_id)
        if auth is not None:
            return auth
        await asyncio.sleep(0.005)
    raise AssertionError(f"request auth was not retained for {invocation_id}")


def _assert_closed(auth: RequestAuthContext) -> None:
    assert repr(auth) == "RequestAuthContext(closed=True)"
    with pytest.raises(AuthError, match="no longer active"):
        auth.client_for("user")


@pytest.fixture
def deployed(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "runtime-affinity-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setenv("DATABRICKS_WORKSPACE_ID", "workspace-affinity-test")


@pytest.mark.asyncio
async def test_cross_replica_retry_keeps_original_obo_context_and_cleans_duplicate(
    deployed, postgres_runtime_store
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    seen_a = []
    seen_b = []

    async def invoke_a(value, context):
        auth = context.request_auth
        assert auth is not None
        seen_a.append((value, auth, auth._token))
        if value == "first":
            first_started.set()
            await release_first.wait()
        return value

    async def invoke_b(value, context):
        seen_b.append(value)
        return value

    store_a = await postgres_runtime_store.create("shared")
    store_b = await postgres_runtime_store.create("shared")
    app_a = _make_app(store_a, invoke_a)
    app_b = _make_app(store_b, invoke_b)
    public_id = str(uuid4())
    headers_a = _headers(token="obo-token-a")
    headers_b = _headers(token="obo-token-b")
    runtime_id = _runtime_invocation_id(public_id, headers_a)

    async with _running_apps(app_a, app_b), _client(app_a) as client_a, _client(app_b) as client_b:
        accepted = await client_a.post(
            "/api/invocations",
            json={"id": public_id, "input": "first", "background": True},
            headers=headers_a,
        )
        assert accepted.status_code == 202
        await asyncio.wait_for(first_started.wait(), 2)

        retry = await client_b.post(
            "/api/invocations",
            json={"id": public_id, "input": "first", "background": True},
            headers=headers_b,
        )
        assert retry.status_code == 202
        state = await _wait_for_state(store_a, runtime_id, status=InvocationStatus.ACTIVE)
        assert state.worker_id == store_a.worker_id
        assert state.recovery_affinity == "shared"
        assert _runtime_id_is_clean(app_b, runtime_id)
        assert seen_b == []

        release_first.set()
        completed = await _wait_for_state(store_a, runtime_id, status=InvocationStatus.COMPLETED)

    assert completed.attempt == 1
    assert len(seen_a) == 1
    assert seen_a[0][0] == "first"
    assert seen_a[0][2] == "obo-token-a"
    _assert_closed(seen_a[0][1])
    assert "obo-token-a" not in repr(completed)
    assert "obo-token-b" not in repr(completed)
    assert all("obo-token" not in repr(event) for event in await store_a.events(runtime_id))


@pytest.mark.asyncio
async def test_fifo_queue_retains_obo_context_and_rejects_wrong_affinity_retry(
    deployed, postgres_runtime_store
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    seen_a = []
    seen_b = []
    seen_c = []

    async def invoke_a(value, context):
        auth = context.request_auth
        assert auth is not None
        seen_a.append((value, auth, auth._token))
        assert value == "first"
        first_started.set()
        await release_first.wait()
        return value

    async def invoke_b(value, context):
        auth = context.request_auth
        assert auth is not None
        seen_b.append((value, auth, auth._token))
        assert value == "second"
        second_started.set()
        await release_second.wait()
        return value

    async def invoke_c(value, context):
        seen_c.append(value)
        return value

    store_a = await postgres_runtime_store.create("shared")
    store_b = await postgres_runtime_store.create("shared")
    store_c = await postgres_runtime_store.create("different")
    app_a = _make_app(store_a, invoke_a)
    app_b = _make_app(store_b, invoke_b)
    app_c = _make_app(store_c, invoke_c)
    first_id = str(uuid4())
    second_id = str(uuid4())
    headers_a = _headers(token="obo-first")
    headers_b = _headers(token="obo-second")
    headers_c = _headers(token="obo-retry")
    runtime_second = _runtime_invocation_id(second_id, headers_b)

    async with (
        _running_apps(app_a, app_b, app_c),
        _client(app_a) as client_a,
        _client(app_b) as client_b,
        _client(app_c) as client_c,
    ):
        first = await client_a.post(
            "/api/invocations",
            json={"id": first_id, "session_id": "session-1", "input": "first", "background": True},
            headers=headers_a,
        )
        assert first.status_code == 202
        await asyncio.wait_for(first_started.wait(), 2)

        second = await client_b.post(
            "/api/invocations",
            json={
                "id": second_id,
                "session_id": "session-1",
                "input": "second",
                "background": True,
            },
            headers=headers_b,
        )
        assert second.status_code == 202
        queued = await _wait_for_state(store_b, runtime_second, status=InvocationStatus.QUEUED)
        queued_auth = await _wait_for_map_entry(app_b, runtime_second)
        assert queued.worker_id == store_b.worker_id
        assert app_b._request_auth[runtime_second] is queued_auth
        assert queued_auth._token == "obo-second"

        wrong_affinity_retry = await client_c.post(
            "/api/invocations",
            json={
                "id": second_id,
                "session_id": "session-1",
                "input": "second",
                "background": True,
            },
            headers=headers_c,
        )
        assert wrong_affinity_retry.status_code == 202
        assert _runtime_id_is_clean(app_c, runtime_second)
        assert seen_c == []

        release_first.set()
        await asyncio.wait_for(second_started.wait(), 2)
        assert seen_b[0][0] == "second"
        assert seen_b[0][1] is queued_auth
        assert seen_b[0][2] == "obo-second"
        release_second.set()
        await _wait_for_state(store_b, runtime_second, status=InvocationStatus.COMPLETED)

    assert [item[0] for item in seen_a] == ["first"]
    assert [item[0] for item in seen_b] == ["second"]
    _assert_closed(seen_a[0][1])
    _assert_closed(seen_b[0][1])


@pytest.mark.asyncio
async def test_cancelled_foreground_request_retains_queued_obo_context(
    deployed, postgres_runtime_store
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    seen = []

    async def invoke(value, context):
        auth = context.request_auth
        assert auth is not None
        seen.append((value, auth, auth._token))
        if value == "first":
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()
            await release_second.wait()
        return value

    store = await postgres_runtime_store.create("shared")
    app = _make_app(store, invoke)
    first_id = str(uuid4())
    second_id = str(uuid4())
    headers_first = _headers(token="foreground-first")
    headers_second = _headers(token="foreground-second")
    runtime_second = _runtime_invocation_id(second_id, headers_second)

    async with _running_apps(app), _client(app) as client:
        first = await client.post(
            "/api/invocations",
            json={"id": first_id, "session_id": "session-1", "input": "first", "background": True},
            headers=headers_first,
        )
        assert first.status_code == 202
        await asyncio.wait_for(first_started.wait(), 2)

        foreground = asyncio.create_task(
            client.post(
                "/api/invocations",
                json={
                    "id": second_id,
                    "session_id": "session-1",
                    "input": "second",
                },
                headers=headers_second,
            )
        )
        queued = await _wait_for_state(store, runtime_second, status=InvocationStatus.QUEUED)
        queued_auth = await _wait_for_map_entry(app, runtime_second)
        foreground.cancel()
        with pytest.raises(asyncio.CancelledError):
            await foreground

        assert queued.status == InvocationStatus.QUEUED
        assert app._request_auth[runtime_second] is queued_auth
        assert queued_auth._token == "foreground-second"

        release_first.set()
        await asyncio.wait_for(second_started.wait(), 2)
        assert seen[1][1] is queued_auth
        release_second.set()
        completed = await _wait_for_state(store, runtime_second, status=InvocationStatus.COMPLETED)

    assert completed.attempt == 1
    assert seen[1][2] == "foreground-second"
    _assert_closed(seen[0][1])
    _assert_closed(seen[1][1])


@pytest.mark.asyncio
async def test_stale_queued_takeover_rejects_fresh_obo_before_recovery_hook(
    deployed, postgres_runtime_store
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    recovery_calls = []

    async def invoke_a(value, context):
        if value == "first":
            first_started.set()
            await release_first.wait()
        return value

    async def invoke_b(value, context):
        raise AssertionError("request-user recovery must not call @app.invoke")

    async def recover_b(value, context):
        recovery_calls.append((value, context.attempt))
        return value

    store_a = await postgres_runtime_store.create("shared")
    store_b = await postgres_runtime_store.create("shared")
    app_a = _make_app(store_a, invoke_a)
    app_b = _make_app(store_b, invoke_b, recover=recover_b)
    first_id = str(uuid4())
    second_id = str(uuid4())
    headers_a = _headers(token="abandoned-first")
    headers_second = _headers(token="abandoned-second")
    headers_b = _headers(token="fresh-retry")
    runtime_first = _runtime_invocation_id(first_id, headers_a)
    runtime_second = _runtime_invocation_id(second_id, headers_second)

    async with _running_apps(app_a, app_b), _client(app_a) as client_a, _client(app_b) as client_b:
        first = await client_a.post(
            "/api/invocations",
            json={"id": first_id, "session_id": "session-1", "input": "first", "background": True},
            headers=headers_a,
        )
        assert first.status_code == 202
        await asyncio.wait_for(first_started.wait(), 2)
        second = await client_a.post(
            "/api/invocations",
            json={
                "id": second_id,
                "session_id": "session-1",
                "input": "second",
                "background": True,
            },
            headers=headers_second,
        )
        assert second.status_code == 202
        await _wait_for_state(store_a, runtime_second, status=InvocationStatus.QUEUED)
        queued_auth = await _wait_for_map_entry(app_a, runtime_second)

        # Simulate worker A disappearing before the queued attempt starts.  The
        # queued context must remain associated with A until recovery claims it.
        await app_a._runtime.stop()
        assert app_a._request_auth[runtime_second] is queued_auth

        retry = await client_b.post(
            "/api/invocations",
            json={
                "id": second_id,
                "session_id": "session-1",
                "input": "second",
                "background": True,
            },
            headers=headers_b,
        )
        assert retry.status_code == 202
        assert _runtime_id_is_clean(app_b, runtime_second)

        await postgres_runtime_store.age(runtime_first)
        await _wait_for_state(
            store_b,
            runtime_first,
            status=InvocationStatus.FAILED,
            attempt_at_least=2,
        )
        # Recovery of the predecessor promotes this queued head.  Age it only
        # after promotion so recovery does not depend on a short wall-clock lease.
        await postgres_runtime_store.age(runtime_second)
        failed = await _wait_for_state(
            store_b,
            runtime_second,
            status=InvocationStatus.FAILED,
            attempt_at_least=2,
        )
        assert failed.response == {
            "error": {
                "code": "MCP_USER_AUTH_RECOVERY_UNSUPPORTED",
                "message": "Request-user execution does not survive failure recovery yet",
            },
            "status_code": 500,
        }
        assert recovery_calls == []

        # The original worker's lifespan is still active even though its runtime
        # stopped. Its cleanup loop must release credentials after the replacement
        # worker reaches the terminal recovery failure, before lifespan shutdown.
        deadline = asyncio.get_running_loop().time() + 2
        while (
            runtime_second in app_a._request_auth and asyncio.get_running_loop().time() < deadline
        ):
            await asyncio.sleep(0.01)
        assert runtime_second not in app_a._request_auth
        _assert_closed(queued_auth)

    _assert_closed(queued_auth)
    assert "fresh-retry" not in repr(failed)


@pytest.mark.asyncio
async def test_app_auth_active_attempt_recovers_on_second_runtime(
    postgres_runtime_store,
):
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    recovery_calls = []

    async def invoke_a(value, context):
        first_started.set()
        await release_first.wait()
        return value

    async def invoke_b(value, context):
        raise AssertionError("the replacement attempt must use @app.recover")

    async def recover_b(value, context):
        recovery_calls.append((value, context.attempt, context.is_recovery))
        return {"recovered": value}

    store_a = await postgres_runtime_store.create("shared")
    store_b = await postgres_runtime_store.create("shared")
    app_a = _make_app(store_a, invoke_a, user_auth=False)
    app_b = _make_app(store_b, invoke_b, recover=recover_b, user_auth=False)
    invocation_id = str(uuid4())

    async with _running_apps(app_a, app_b), _client(app_a) as client:
        accepted = await client.post(
            "/api/invocations",
            json={"id": invocation_id, "input": "app-auth", "background": True},
        )
        assert accepted.status_code == 202
        await asyncio.wait_for(first_started.wait(), 2)
        await app_a._runtime.stop()
        await postgres_runtime_store.age(invocation_id)

        recovered = await _wait_for_state(
            store_b,
            invocation_id,
            status=InvocationStatus.COMPLETED,
            attempt_at_least=2,
        )

    assert recovered.response == {"recovered": "app-auth"}
    assert recovery_calls == [("app-auth", 2, True)]
