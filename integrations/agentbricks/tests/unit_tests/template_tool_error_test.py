"""A tool's error result reaches the model's next turn through the generated LangGraph agent."""

import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import pytest

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"
MARKER = "TOOL_ERROR_MARKER"


@pytest.fixture
def agent_app(monkeypatch):
    """Real HTTP runtime, adapter and graph; only the model and the tool are local."""
    for dependency in ("databricks_langchain", "langgraph", "langchain", "langchain_mcp_adapters"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[langgraph]")
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver

    from databricks_agentkit import DurableAgentServer
    from databricks_agentkit.runtime.store import InMemoryRuntimeStore

    monkeypatch.syspath_prepend(str(TEMPLATES / "agent-langgraph"))
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, name)
    adapter = importlib.import_module("runtime.adapter")
    agent = importlib.import_module("agent.agent")
    monkeypatch.setenv("DATABRICKS_APP_NAME", "tool-error-test")
    monkeypatch.delenv("DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL", raising=False)
    model_turns = []

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            model_turns.append(messages)
            tool_results = [message for message in messages if message.type == "tool"]
            if not tool_results:
                return self._generate(messages, stop, run_manager, **kwargs)
            self.responses = [AIMessage(content=f"The code failed: {tool_results[-1].text}")]
            self.i = 0
            return self._generate(messages, stop, run_manager, **kwargs)

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

    async def create_graph(actor, model=None, *, workspace_client_for=None):
        call = {
            "name": "run_code",
            "args": {"language": "python", "code": f"raise ValueError({MARKER!r})"},
            "id": "run-code-call",
            "type": "tool_call",
        }
        return create_agent(
            Model(responses=[AIMessage(content="", tool_calls=[call])]),
            [run_code],
            checkpointer=InMemorySaver(),
        )

    monkeypatch.setattr(agent, "create_agent_graph", create_graph)
    monkeypatch.setattr(agent, "start_trace", lambda **kwargs: nullcontext())
    app = DurableAgentServer(runtime_store=InMemoryRuntimeStore())
    app.invoke(adapter.invoke)
    app.recover(adapter.recover)
    yield app, model_turns
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            sys.modules.pop(name, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
async def test_tool_error_result_reaches_next_model_turn(agent_app, stream):
    import httpx

    app, model_turns = agent_app
    invocation_id = str(uuid4())
    request = {
        "id": invocation_id,
        "session_id": invocation_id,
        "stream": stream,
        "input": {"messages": [{"role": "user", "content": "Run the failing code."}]},
    }
    await app._runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="https://test"
        ) as client:
            response = await client.post("/api/invocations", json=request)
            assert response.status_code == 200, response.text
            if stream:
                assert "event: run.completed" in response.text, response.text
                assert "event: run.failed" not in response.text, response.text
                response = await client.get(f"/api/invocations/{invocation_id}")
                assert response.status_code == 200, response.text
    finally:
        await app._runtime.stop()

    body = response.json()
    assert body["status"] == "completed", body
    messages = body["output"]["output"]
    (result,) = [message for message in messages if message.get("type") == "tool"]
    assert result["name"] == "run_code", result
    assert result["tool_call_id"] == "run-code-call", result
    assert result["artifact"]["structured_content"]["outcome"] == "execution_error", result
    assert len(model_turns) == 2
    assert f"ValueError: {MARKER}" in json.dumps(
        [message.content for message in model_turns[1] if message.type == "tool"]
    )
    assert messages[-1]["type"] == "ai", messages
    assert MARKER in json.dumps(messages[-1]["content"]), messages[-1]
