"""A tool's error result reaches the model's next turn through the generated LangGraph agent."""

import json
from contextlib import nullcontext
from uuid import uuid4

import pytest
from template_fixtures import (
    TEMPLATES,
    durable_app,
    forget_template,
    import_template,
    runtime_client,
    scripted_model,
)

MARKER = "TOOL_ERROR_MARKER"


@pytest.fixture
def agent_app(monkeypatch):
    """Real HTTP runtime, adapter and graph; only the model and the tool are local."""
    for dependency in ("databricks_langchain", "langgraph", "langchain", "langchain_mcp_adapters"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[langgraph]")
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    from databricks_agentkit.runtime.store import InMemoryRuntimeStore

    adapter, agent = import_template(monkeypatch, TEMPLATES / "agent-langgraph")
    monkeypatch.setenv("DATABRICKS_APP_NAME", "tool-error-test")
    monkeypatch.delenv("DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL", raising=False)
    call = {
        "name": "run_code",
        "args": {"language": "python", "code": f"raise ValueError({MARKER!r})"},
        "id": "run-code-call",
        "type": "tool_call",
    }
    model = scripted_model(
        AIMessage(content="", tool_calls=[call]),
        AIMessage(content=f"The code failed with ValueError: {MARKER}"),
    )

    # The shape the sandbox MCP tool returns for code that raises: text blocks plus the
    # structured outcome as the artifact, with no exception at the tool boundary.
    @tool(response_format="content_and_artifact")
    async def run_code(language: str, code: str):
        """Run code in the sandbox."""
        error = f"ValueError: {MARKER}"
        return (
            [{"type": "text", "text": error, "id": "block-1"}],
            {"structured_content": {"outcome": "execution_error", "output": error}},
        )

    async def create_graph(actor, model_name=None, *, workspace_client_for=None):
        return create_agent(model, [run_code], checkpointer=InMemorySaver())

    monkeypatch.setattr(agent, "create_agent_graph", create_graph)
    monkeypatch.setattr(agent, "start_trace", lambda **kwargs: nullcontext())
    yield durable_app(adapter, runtime_store=InMemoryRuntimeStore()), model
    forget_template()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_tool_error_result_reaches_next_model_turn(agent_app, stream):
    app, model = agent_app
    invocation_id = str(uuid4())
    request = {
        "id": invocation_id,
        "session_id": invocation_id,
        "stream": stream,
        "input": {"messages": [{"role": "user", "content": "Run the failing code."}]},
    }
    async with runtime_client(app) as client:
        response = await client.post("/api/invocations", json=request)
        assert response.status_code == 200, response.text
        if stream:
            assert "event: run.completed" in response.text, response.text
            assert "event: run.failed" not in response.text, response.text
            response = await client.get(f"/api/invocations/{invocation_id}")
            assert response.status_code == 200, response.text

    body = response.json()
    assert body["status"] == "completed", body
    messages = body["output"]["output"]
    (result,) = [message for message in messages if message.get("type") == "tool"]
    assert result["name"] == "run_code", result
    assert result["tool_call_id"] == "run-code-call", result
    assert result["artifact"]["structured_content"]["outcome"] == "execution_error", result
    assert len(model.turns) == 2
    assert f"ValueError: {MARKER}" in json.dumps(
        [message.content for message in model.turns[1] if message.type == "tool"]
    )
    assert messages[-1]["type"] == "ai", messages
