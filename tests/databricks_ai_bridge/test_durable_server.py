"""Tests for the standalone durable FastAPI server."""

from datetime import datetime, timezone
from typing import cast

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("psycopg")
pytest.importorskip("sqlalchemy")

from databricks_ai_bridge.durable_runtime import (
    DurabilityStore,
    DurableExecution,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
)
from databricks_ai_bridge.durable_server import (
    DatabricksDurableServer,
    PreparedDurableRequest,
)


def execution(
    status: DurableExecutionStatus,
    *,
    response: dict | None = None,
) -> DurableExecution:
    return DurableExecution(
        execution_id="execution-1",
        status=status,
        attempt=1,
        heartbeat_at=datetime.now(timezone.utc),
        request={"input": "hello"},
        response=response,
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.invoke_calls: list[tuple[str, dict]] = []
        self.submit_calls: list[tuple[str, dict]] = []
        self.invoke_error: Exception | None = None
        self.submit_state = execution(DurableExecutionStatus.QUEUED)
        self.get_state: DurableExecution | None = self.submit_state

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def invoke(self, execution_id: str, request: dict) -> dict:
        if self.invoke_error is not None:
            raise self.invoke_error
        self.invoke_calls.append((execution_id, request))
        return {"id": execution_id, "status": "completed", "output": []}

    async def submit(self, execution_id: str, request: dict) -> DurableExecution:
        self.submit_calls.append((execution_id, request))
        return self.submit_state

    async def get(self, execution_id: str) -> DurableExecution | None:
        return self.get_state


def make_server(**kwargs) -> tuple[DatabricksDurableServer, FakeRuntime, list[dict]]:
    prepared_requests: list[dict] = []

    def prepare_request(request: dict) -> PreparedDurableRequest:
        prepared_requests.append(request)
        return PreparedDurableRequest("execution-1", request)

    async def unused_executor(request, context):
        raise AssertionError("fake runtime should handle execution")

    server = DatabricksDurableServer(
        unused_executor,
        prepare_request=prepare_request,
        durability_store=cast(DurabilityStore, object()),
        **kwargs,
    )
    runtime = FakeRuntime()
    server.runtime = runtime
    return server, runtime, prepared_requests


def test_blocking_invocation_uses_prepared_request_and_runtime_lifecycle():
    startup_calls = []
    shutdown_calls = []

    async def startup():
        startup_calls.append(True)

    async def shutdown():
        shutdown_calls.append(True)

    server, runtime, prepared_requests = make_server(
        on_startup=startup,
        on_shutdown=shutdown,
    )
    with TestClient(server.app) as client:
        response = client.post(
            "/responses",
            json={"input": "hello", "background": False, "stream": False},
        )
        assert runtime.started is True

    assert response.status_code == 200
    assert response.json() == {"id": "execution-1", "status": "completed", "output": []}
    assert prepared_requests == [{"input": "hello"}]
    assert runtime.invoke_calls == [("execution-1", {"input": "hello"})]
    assert runtime.stopped is True
    assert startup_calls == [True]
    assert shutdown_calls == [True]


def test_invocations_alias_supports_background_and_retrieval():
    server, runtime, _ = make_server()
    with TestClient(server.app) as client:
        submitted = client.post(
            "/invocations",
            json={"input": "hello", "background": True},
        )
        retrieved = client.get("/responses/execution-1")

    assert submitted.status_code == 202
    assert submitted.json()["status"] == "QUEUED"
    assert runtime.submit_calls == [("execution-1", {"input": "hello"})]
    assert retrieved.status_code == 200
    assert retrieved.json()["execution_id"] == "execution-1"


def test_background_cache_hit_returns_completed_response():
    server, runtime, _ = make_server()
    runtime.submit_state = execution(
        DurableExecutionStatus.COMPLETED,
        response={"id": "execution-1", "status": "completed", "output": []},
    )
    with TestClient(server.app) as client:
        response = client.post(
            "/responses",
            json={"input": "hello", "background": True},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"


@pytest.mark.parametrize(
    ("error", "status_code"),
    [
        (DurableRequestConflictError("conflict"), 409),
        (DurableExecutionFailedError("failed"), 500),
    ],
)
def test_runtime_errors_are_mapped_to_http(error, status_code):
    server, runtime, _ = make_server()
    runtime.invoke_error = error
    with TestClient(server.app) as client:
        response = client.post("/responses", json={"input": "hello"})

    assert response.status_code == status_code


def test_rejects_streaming_and_invalid_transport_flags():
    server, _, _ = make_server()
    with TestClient(server.app) as client:
        streaming = client.post("/responses", json={"input": "hello", "stream": True})
        invalid = client.post("/responses", json={"input": "hello", "background": "yes"})

    assert streaming.status_code == 400
    assert invalid.status_code == 400


def test_missing_execution_returns_not_found():
    server, runtime, _ = make_server()
    runtime.get_state = None
    with TestClient(server.app) as client:
        response = client.get("/responses/missing")

    assert response.status_code == 404


def test_health_routes():
    server, _, _ = make_server()
    with TestClient(server.app) as client:
        assert client.get("/health").json() == {"status": "healthy"}
        assert client.get("/api/healthz").json() == {"status": "healthy"}
