from fastapi.testclient import TestClient

from databricks_agentkit import DurableAgentServer
from databricks_agentkit.runtime.store import InMemoryRuntimeStore
from runtime.adapter import _local_sessions, invoke, recover


def test_managed_http_invocation_and_replay(monkeypatch):
    monkeypatch.delenv("AGENT_SESSION_STORE", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_STORE", raising=False)
    _local_sessions.clear()
    app = DurableAgentServer(runtime_store=InMemoryRuntimeStore())
    app.invoke(invoke)
    app.recover(recover)
    body = {
        "id": "11111111-1111-4111-8111-111111111111",
        "session_id": "demo",
        "input": {"messages": [{"role": "user", "content": "Hello"}]},
    }
    with TestClient(app) as client:
        response = client.post("/api/invocations", json=body)
        assert response.status_code == 200
        assert response.json()["output"]["output"][0]["content"] == "Echo: Hello"
        assert client.post("/api/invocations", json=body).json() == response.json()
    assert len(_local_sessions["demo"][1]) == 2
