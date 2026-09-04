from types import SimpleNamespace

import pytest
from agent.agent import run_agent
from fastapi.testclient import TestClient
from runtime.main import app


async def test_langgraph_agent_emits_progress_and_marks_recovery() -> None:
    events = []

    async def emit(event):
        events.append(event)
        return len(events)

    context = SimpleNamespace(attempt=2, is_recovery=True, emit=emit)
    result = await run_agent(
        {"input": {"message": "hello"}},
        context,
    )

    assert result == {"result": "Processed: hello", "attempt": 2, "recovered": True}
    assert events == [
        {"type": "progress", "stage": "recovered", "attempt": 2},
        {"type": "progress", "stage": "completed", "attempt": 2},
    ]


async def test_langgraph_agent_requires_message() -> None:
    async def emit(event):
        return 1

    context = SimpleNamespace(attempt=1, is_recovery=False, emit=emit)
    with pytest.raises(ValueError, match="message must be a string"):
        await run_agent({"input": {}}, context)


def test_app_exposes_only_durable_invocation_routes() -> None:
    with TestClient(app, base_url="https://testserver") as client:
        response = client.post(
            "/invocations",
            json={"id": "run-1", "input": {"message": "hello"}},
        )

    assert response.status_code == 200
    assert response.json() == {
        "result": "Processed: hello",
        "attempt": 1,
        "recovered": False,
    }
    paths = app.openapi()["paths"]
    assert "/invocations" in paths
    assert "/invocations/{run_id}" in paths
    assert "/invocations/{run_id}/events" in paths
