"""Tests for the durable MLflow AgentServer subclass."""

from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from mlflow.genai.agent_server import AgentServer
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

pytest.importorskip("psycopg")
pytest.importorskip("sqlalchemy")

from databricks_ai_bridge.durable_agent_server import (
    DatabricksDurableAgentServer,
    PreparedDurableRequest,
    get_durable_execution_context,
)
from databricks_ai_bridge.durable_runtime import (
    DurabilityStore,
    DurableExecution,
    DurableExecutionContext,
    DurableExecutionFailedError,
    DurableExecutionStatus,
    DurableRequestConflictError,
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
        request={"input": "hello", "custom_inputs": {"session_id": "session-1"}},
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


def make_server(
    **kwargs,
) -> tuple[DatabricksDurableAgentServer, FakeRuntime, list[ResponsesAgentRequest]]:
    prepared_requests: list[ResponsesAgentRequest] = []

    def prepare_request(request: ResponsesAgentRequest) -> PreparedDurableRequest:
        prepared_requests.append(request)
        payload = request.model_dump(mode="json", exclude_none=True)
        custom_inputs = dict(payload.get("custom_inputs") or {})
        custom_inputs["session_id"] = "session-1"
        payload["custom_inputs"] = custom_inputs
        return PreparedDurableRequest("execution-1", payload)

    server = DatabricksDurableAgentServer(
        prepare_request=prepare_request,
        durability_store=cast(DurabilityStore, object()),
        **kwargs,
    )
    runtime = FakeRuntime()
    server.runtime = runtime
    return server, runtime, prepared_requests


def test_extends_agent_server():
    server, _, _ = make_server()
    assert isinstance(server, AgentServer)


def test_blocking_invocation_uses_agent_validation_and_runtime_lifecycle():
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
            json={
                "input": [{"role": "user", "content": "hello"}],
                "background": False,
                "stream": False,
            },
        )
        assert runtime.started is True

    assert response.status_code == 200
    assert response.json() == {"id": "execution-1", "status": "completed", "output": []}
    assert prepared_requests[0].stream is None
    prepared_json = prepared_requests[0].model_dump(mode="json", exclude_none=True)
    assert prepared_json["input"][0]["role"] == "user"
    assert runtime.invoke_calls[0][0] == "execution-1"
    assert runtime.stopped is True
    assert startup_calls == [True]
    assert shutdown_calls == [True]


def test_invocations_alias_supports_background_and_retrieval():
    server, runtime, _ = make_server()
    with TestClient(server.app) as client:
        submitted = client.post(
            "/invocations",
            json={
                "input": [{"role": "user", "content": "hello"}],
                "background": True,
            },
        )
        retrieved = client.get("/responses/execution-1")

    assert submitted.status_code == 202
    assert submitted.json()["status"] == "in_progress"
    assert submitted.json()["custom_outputs"]["session_id"] == "session-1"
    assert runtime.submit_calls[0][0] == "execution-1"
    assert retrieved.status_code == 200
    assert retrieved.json()["custom_outputs"]["attempt"] == 1


def test_background_cache_hit_returns_completed_response():
    server, runtime, _ = make_server()
    runtime.submit_state = execution(
        DurableExecutionStatus.COMPLETED,
        response={"id": "execution-1", "status": "completed", "output": []},
    )
    with TestClient(server.app) as client:
        response = client.post(
            "/responses",
            json={
                "input": [{"role": "user", "content": "hello"}],
                "background": True,
            },
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
        response = client.post(
            "/responses",
            json={"input": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == status_code


def test_rejects_invalid_agent_request_streaming_and_transport_flags():
    server, _, _ = make_server()
    with TestClient(server.app) as client:
        invalid_request = client.post("/responses", json={})
        streaming = client.post(
            "/responses",
            json={"input": "hello", "stream": True},
        )
        invalid_flag = client.post(
            "/responses",
            json={"input": "hello", "background": "yes"},
        )

    assert invalid_request.status_code == 400
    assert streaming.status_code == 400
    assert invalid_flag.status_code == 400


@pytest.mark.asyncio
async def test_registered_invoke_handler_receives_attempt_context():
    server, _, _ = make_server()
    observed = []

    async def registered_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        observed.append((request, get_durable_execution_context()))
        return ResponsesAgentResponse(id="execution-1", status="completed", output=[])

    span = MagicMock()
    span.__enter__.return_value = span
    span.__exit__.return_value = False
    context = DurableExecutionContext(execution_id="execution-1", attempt=2)
    with (
        patch("mlflow.genai.agent_server.server._invoke_function", registered_handler),
        patch("mlflow.genai.agent_server.server.mlflow.start_span", return_value=span),
    ):
        response = await server._execute_durable(
            {"input": [{"role": "user", "content": "hello"}]},
            context,
        )

    assert response["status"] == "completed"
    assert isinstance(observed[0][0], ResponsesAgentRequest)
    assert observed[0][1] == context
    with pytest.raises(RuntimeError, match="only available inside"):
        get_durable_execution_context()


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
        assert client.get("/agent/info").json()["agent_api"] == "responses"
