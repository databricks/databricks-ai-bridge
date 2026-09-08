from types import SimpleNamespace

import agent.agent as agent_module
import pytest
import runtime.main as main_module
from fastapi.testclient import TestClient
from langchain.messages import AIMessage, AIMessageChunk


class _FakeAgent:
    async def astream(self, *, input, stream_mode):
        assert input == {"messages": [{"role": "user", "content": "hello"}]}
        assert stream_mode == ["updates", "messages"]
        yield "messages", (AIMessageChunk(content="hello", id="chunk-1"), {})
        yield "updates", {
            "model": {"messages": [AIMessage(content="hello", id="message-1")]}
        }


async def test_langgraph_agent_persists_streaming_events(monkeypatch) -> None:
    events = []

    async def emit(event):
        events.append(event)
        return len(events)

    async def create_agent_graph():
        return _FakeAgent()

    monkeypatch.setattr(agent_module, "create_agent_graph", create_agent_graph)
    context = SimpleNamespace(
        invocation_id="invocation-1",
        session_id="session-1",
        is_recovery=False,
        emit=emit,
    )

    result = await agent_module.run_agent(
        [{"role": "user", "content": "hello"}],
        context,
    )

    assert events[0] == {"type": "delta", "content": "hello", "id": "chunk-1"}
    assert events[1]["type"] == "message"
    assert events[1]["message"]["content"] == "hello"
    assert result == {
        "output": [events[1]["message"]],
        "session_id": "session-1",
        "recovered": False,
    }


async def test_langgraph_agent_requires_message_list() -> None:
    async def emit(event):
        return 1

    context = SimpleNamespace(
        invocation_id="invocation-1",
        session_id="session-1",
        is_recovery=False,
        emit=emit,
    )
    with pytest.raises(ValueError, match="input must be a list"):
        await agent_module.run_agent({}, context)


@pytest.mark.asyncio
async def test_recovery_handler_explains_the_recovery_attempt(monkeypatch) -> None:
    received_input = None

    async def run_agent(input, context):
        nonlocal received_input
        received_input = input
        return {"recovered": context.is_recovery}

    monkeypatch.setattr(main_module, "run_agent", run_agent)
    context = SimpleNamespace(is_recovery=True)
    original_input = [{"role": "user", "content": "hello"}]

    result = await main_module.recover(original_input, context)

    assert result == {"recovered": True}
    assert received_input[1:] == original_input
    assert received_input[0]["role"] == "system"
    assert "previous pod crashed" in received_input[0]["content"]


def test_app_exposes_only_durable_invocation_routes(monkeypatch) -> None:
    async def run_agent(input, context):
        return {
            "output": [{"role": "assistant", "content": "hello"}],
            "session_id": context.session_id,
            "recovered": context.is_recovery,
        }

    monkeypatch.setattr(main_module, "run_agent", run_agent)
    with TestClient(main_module.app, base_url="https://testserver") as client:
        response = client.post(
            "/api/invocations",
            json={
                "id": "11111111-1111-4111-8111-111111111111",
                "input": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": "11111111-1111-4111-8111-111111111111",
        "status": "completed",
        "output": {
            "output": [{"role": "assistant", "content": "hello"}],
            "session_id": "11111111-1111-4111-8111-111111111111",
            "recovered": False,
        },
    }
    paths = main_module.app.openapi()["paths"]
    assert set(paths) == {
        "/api/invocations",
        "/api/invocations/{invocation_id}",
        "/api/invocations/{invocation_id}/events",
    }
