"""Tests for the generic durable entrypoint application."""

import time
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("psycopg")
pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from databricks_ai_bridge.durable_app import (
    RUN_ID_HEADER,
    SESSION_ID_HEADER,
    DatabricksDurableApp,
)
from databricks_ai_bridge.durable_runtime import (
    DurableExecutionContext,
    InMemoryDurabilityStore,
)


def make_app():
    return DatabricksDurableApp(
        durability_store=InMemoryDurabilityStore(),
        heartbeat_seconds=0.01,
        stale_seconds=0.05,
        scan_seconds=0.01,
        poll_seconds=0.005,
    )


@pytest.mark.asyncio
async def test_entrypoint_receives_stable_session_and_recovery_context():
    durable_app = make_app()
    emitted: list[dict] = []

    async def emit(event: dict) -> int:
        emitted.append(event)
        return 4

    @durable_app.entrypoint
    async def agent(payload, context):
        cursor = await context.emit({"type": "progress"})
        return {
            "payload": payload,
            "run_id": context.run_id,
            "session_id": context.session_id,
            "attempt": context.attempt,
            "is_recovery": context.is_recovery,
            "cursor": cursor,
        }

    result = await durable_app._execute(
        {"session_id": "session-1", "payload": {"message": "hello"}},
        DurableExecutionContext("run-1", 2, _emit=emit),
    )

    assert result == {
        "payload": {"message": "hello"},
        "run_id": "run-1",
        "session_id": "session-1",
        "attempt": 2,
        "is_recovery": True,
        "cursor": 4,
    }
    assert emitted == [{"type": "progress"}]


def test_sync_invocation_keeps_payload_in_body_and_context_in_headers():
    durable_app = make_app()

    @durable_app.entrypoint
    async def agent(payload, context):
        return {
            "received": payload,
            "run_id": context.run_id,
            "session_id": context.session_id,
        }

    with TestClient(durable_app) as client:
        response = client.post(
            "/invocations",
            headers={RUN_ID_HEADER: "run-1", SESSION_ID_HEADER: "session-1"},
            json={"input": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert response.headers[RUN_ID_HEADER] == "run-1"
    assert response.headers[SESSION_ID_HEADER] == "session-1"
    assert response.json() == {
        "received": {"input": [{"role": "user", "content": "hello"}]},
        "run_id": "run-1",
        "session_id": "session-1",
    }


def test_generated_context_is_returned_in_response_headers():
    durable_app = make_app()

    @durable_app.entrypoint
    async def agent(payload, context):
        return {"output": payload}

    with TestClient(durable_app) as client:
        response = client.post("/invocations", json={"input": "hello"})

    assert response.status_code == 200
    assert response.headers[RUN_ID_HEADER].startswith("inv_")
    assert response.headers[SESSION_ID_HEADER]


def test_background_invocation_can_be_polled():
    durable_app = make_app()

    @durable_app.entrypoint
    async def agent(payload, context):
        return {"output": payload["input"], "session_id": context.session_id}

    with TestClient(durable_app) as client:
        submitted = client.post(
            "/invocations",
            headers={RUN_ID_HEADER: "run-bg", SESSION_ID_HEADER: "session-bg"},
            json={"input": "hello", "background": True},
        )
        assert submitted.status_code == 202
        assert submitted.json() == {"id": "run-bg", "status": "in_progress"}

        for _ in range(100):
            polled = client.get("/invocations/run-bg")
            if polled.json()["status"] != "in_progress":
                break
            time.sleep(0.005)

    assert polled.json() == {
        "id": "run-bg",
        "status": "completed",
        "output": "hello",
        "session_id": "session-bg",
    }


def test_stream_replays_persisted_events_and_ends_with_done():
    durable_app = make_app()

    @durable_app.entrypoint
    async def agent(payload, context):
        await context.emit({"type": "delta", "content": "hel"})
        await context.emit({"type": "delta", "content": "lo"})
        return {"output": "hello"}

    with TestClient(durable_app) as client:
        with client.stream(
            "POST",
            "/invocations",
            headers={RUN_ID_HEADER: "run-stream", SESSION_ID_HEADER: "session-stream"},
            json={"input": "hello", "stream": True},
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert 'data: {"type": "delta", "content": "hel"}' in body
    assert 'data: {"type": "delta", "content": "lo"}' in body
    assert body.endswith("data: [DONE]\n\n")


def test_reusing_idempotency_key_with_different_body_returns_conflict():
    durable_app = make_app()

    @durable_app.entrypoint
    async def agent(payload, context):
        return {"output": payload}

    headers = {RUN_ID_HEADER: "run-1", SESSION_ID_HEADER: "session-1"}
    with TestClient(durable_app) as client:
        assert (
            client.post("/invocations", headers=headers, json={"input": "one"}).status_code == 200
        )
        conflict = client.post("/invocations", headers=headers, json={"input": "two"})

    assert conflict.status_code == 409


def test_local_default_uses_in_memory_store(monkeypatch):
    for name in (
        "AGENT_SESSION_STORE",
        "LAKEBASE_AUTOSCALING_ENDPOINT",
        "LAKEBASE_AUTOSCALING_PROJECT",
        "LAKEBASE_AUTOSCALING_BRANCH",
    ):
        monkeypatch.delenv(name, raising=False)

    durable_app = DatabricksDurableApp()

    assert isinstance(durable_app.runtime.durability_store, InMemoryDurabilityStore)


def test_managed_session_store_selects_lakebase(monkeypatch):
    monkeypatch.setenv("AGENT_SESSION_STORE", "sessions")
    store = AsyncMock()

    with patch(
        "databricks_ai_bridge.durable_app.app.LakebaseDurabilityStore", return_value=store
    ) as lakebase:
        durable_app = DatabricksDurableApp()

    assert durable_app.runtime.durability_store is store
    lakebase.assert_called_once_with(
        autoscaling_endpoint=None,
        project="databricks-internal-lakebase-agent-session-store",
        branch="production",
        workspace_client=None,
        schema="databricks_durable_app",
    )


def test_only_one_entrypoint_can_be_registered():
    durable_app = make_app()

    @durable_app.entrypoint
    async def first(payload, context):
        return payload

    with pytest.raises(RuntimeError, match="one entrypoint"):

        @durable_app.entrypoint
        async def second(payload, context):
            return payload
