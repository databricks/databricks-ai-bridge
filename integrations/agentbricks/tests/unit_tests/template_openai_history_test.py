"""Restore real Agents SDK session items using invocation's request-user namespace."""

import importlib
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from template_history_fixtures import SessionStoreTransport

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"


@pytest.fixture(params=["memory", "managed"])
def template(request, monkeypatch):
    for dependency in ("agents", "databricks_openai"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[openai]")
    from databricks_agentkit.openai import sessions

    monkeypatch.syspath_prepend(str(TEMPLATES / "agent-openai"))
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, name)
    agent = importlib.import_module("agent.agent")
    adapter = importlib.import_module("runtime.adapter")
    spec = importlib.util.spec_from_file_location(
        "openai_history_ui", TEMPLATES / "ui/agent-openai/runtime/ui.py"
    )
    assert spec and spec.loader
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    store = "test" if request.param == "managed" else None
    monkeypatch.setattr(
        "databricks_agentkit.runtime.tool_manifest.resolve_session_store", lambda *_args: store
    )
    monkeypatch.setattr(sessions, "_local_sessions", {})
    transport = SessionStoreTransport()
    monkeypatch.setattr(sessions, "SessionStoreClient", lambda *_args: transport)
    monkeypatch.setattr(
        ui,
        "_state_client",
        lambda: SimpleNamespace(
            list_session_items=lambda session_id: {
                "session_id": session_id,
                "session_items": [
                    {"item_id": item.item_id, "data": item.data}
                    for item in transport.list_items(
                        transport.get_session(session_id=session_id), order_by="create_time asc"
                    )
                ],
            }
        ),
    )
    monkeypatch.setattr(agent, "start_trace", lambda **_kwargs: nullcontext())
    yield ui, agent, adapter
    for session in sessions._local_sessions.values():
        session.close()
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            sys.modules.pop(name, None)


@pytest.mark.asyncio
async def test_obo_history_reads_native_session_without_live_tools(template, monkeypatch):
    from agents import Agent, Model, ModelResponse
    from agents.usage import Usage
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseOutputMessage,
        ResponseOutputText,
    )

    from databricks_agentkit import DurableAgentServer
    from databricks_agentkit.runtime.auth import AuthError, InvocationAuthPolicy
    from databricks_agentkit.runtime.store import InMemoryRuntimeStore

    class ReplyModel(Model):
        async def get_response(self, *_args, **_kwargs):
            return ModelResponse(output=[message], usage=Usage(), response_id="reply")

        async def stream_response(self, *_args, **_kwargs):
            yield ResponseCompletedEvent(
                type="response.completed",
                sequence_number=0,
                response=Response(
                    id="reply",
                    created_at=0,
                    model="test",
                    object="response",
                    output=[message],
                    parallel_tool_calls=False,
                    tool_choice="none",
                    tools=[],
                ),
            )

    message = ResponseOutputMessage(
        id="answer",
        role="assistant",
        status="completed",
        type="message",
        content=[ResponseOutputText(text="saved answer", type="output_text", annotations=[])],
    )
    ui, agent, adapter = template
    monkeypatch.setenv("DATABRICKS_APP_NAME", "history-test")
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example")
    monkeypatch.setattr(
        agent, "create_agent", lambda *_args, **_kwargs: Agent(name="history", model=ReplyModel())
    )

    async def no_servers(*_args, **_kwargs):
        return []

    monkeypatch.setattr(agent, "mcp_servers", no_servers)
    app = DurableAgentServer(
        runtime_store=InMemoryRuntimeStore(),
        auth_policy=InvocationAuthPolicy(user_tools=("genie",)),
    )
    app.invoke(adapter.invoke)
    ui.install_ui(app)
    headers = {
        "x-forwarded-user": "alice-id",
        "x-forwarded-email": "alice@example.com",
        "x-forwarded-access-token": "test-token",
    }
    await app._runtime.start()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="https://test"
        ) as client:
            invocation = await client.post(
                "/api/invocations",
                headers=headers,
                json={
                    "id": str(uuid4()),
                    "session_id": "public",
                    "input": {
                        "actor": "alice@example.com",
                        "messages": [{"role": "user", "content": "saved prompt"}],
                    },
                },
            )
            assert invocation.status_code == 200, invocation.text
            assert invocation.json()["output"]["output"] == [
                {"role": "assistant", "content": "saved answer"}
            ]

            def unavailable(*_args, **_kwargs):
                raise AuthError(
                    "MCP_USER_AUTHORIZATION_MISSING", "History must not initialize live tools"
                )

            monkeypatch.setattr(agent, "create_agent", unavailable)
            for _attempt in range(2):
                restored = await client.get(
                    "/api/demo/session/items?session_id=public", headers=headers
                )
                assert restored.status_code == 200, restored.text
                assert restored.json()["session_id"] == "public"
                messages = [item["data"] for item in restored.json()["session_items"]]
                assert messages == [
                    {"role": "user", "content": "saved prompt"},
                    message.model_dump(exclude_unset=True),
                ]
                assert restored.json()["interrupts"] == []
            other = await client.get(
                "/api/demo/session/items?session_id=public",
                headers={**headers, "x-forwarded-user": "bob-id"},
            )
            assert other.status_code == 200, other.text
            assert other.json()["session_items"] == []
            missing = await client.get("/api/demo/session/items?session_id=public")
            assert missing.status_code == 401
    finally:
        await app._runtime.stop()
