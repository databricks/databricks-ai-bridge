"""The shipped MAF templates own their request-scoped model HTTP clients."""

import asyncio
import importlib
import json
import sys
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

import httpx
import pytest

pytest.importorskip("agent_framework")
pytest.importorskip("databricks_openai")

from agent_framework import InMemoryHistoryProvider, Message, tool
from openai import AsyncOpenAI

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"


@pytest.fixture(params=["agent-maf", "custom-agent-maf"])
def template(request, monkeypatch):
    monkeypatch.syspath_prepend(str(TEMPLATES / request.param))
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("agent.agent")
    monkeypatch.setattr(module, "workspace_client", lambda: None)
    monkeypatch.setattr(module, "workspace_headers", lambda: {})
    if request.param == "agent-maf":
        monkeypatch.setattr(module, "all_tools", lambda: [])
        monkeypatch.setattr(module, "memory_tools", lambda _: [])
        monkeypatch.setattr(module, "genie_tools", lambda: [])
        monkeypatch.setattr(module, "mcp_tools", lambda: [])
        monkeypatch.setattr(module, "session_store", lambda _: InMemoryHistoryProvider())
        monkeypatch.setattr(module, "start_trace", lambda *_args, **_kwargs: nullcontext())
    yield request.param, module
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            sys.modules.pop(name, None)


def _response(stream: bool) -> httpx.Response:
    text = {"type": "output_text", "text": "hello", "annotations": []}
    item = {
        "id": "message",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [text],
    }
    response = {
        "id": "response",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "test-model",
        "output": [item],
    }
    if not stream:
        return httpx.Response(200, json=response)
    part = {"item_id": "message", "output_index": 0, "content_index": 0}
    events = [
        {"type": "response.created", "response": {**response, "output": []}},
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.output_text.delta", **part, "delta": "hello"},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
        {"type": "response.completed", "response": response},
    ]
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        content="".join(
            f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': index})}\n\n"
            for index, event in enumerate(events)
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
async def test_model_client_closes_after_request(template, monkeypatch, outcome):
    kind, module = template
    started = asyncio.Event()
    clients = []
    requests = []
    mcp_exits = []

    if kind == "agent-maf":

        @tool
        def sample() -> str:
            """An unused tool from the request's MCP connection."""
            return "sample"

        @asynccontextmanager
        async def connection():
            try:
                yield sample
            finally:
                mcp_exits.append(clients[0].is_closed())

        monkeypatch.setattr(module, "mcp_tools", lambda: [connection()])

    async def model(request):
        requests.append(request)
        started.set()
        if outcome == "cancelled":
            await asyncio.Event().wait()
        if outcome == "failure":
            return httpx.Response(
                400, json={"error": {"message": "model failed", "type": "invalid_request_error"}}
            )
        return _response(json.loads(request.content).get("stream", False))

    def client(**_kwargs):
        raw = AsyncOpenAI(
            api_key="local-test-key",
            base_url="https://model.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(model)),
            max_retries=0,
        )
        clients.append(raw)
        return raw

    monkeypatch.setattr(module, "AsyncDatabricksOpenAI", client)

    async def invoke():
        if kind == "custom-agent-maf":
            response = await module.invoke([{"role": "user", "content": "hello"}])
            return response["output"][0]["content"]
        async with module.run_agent(
            [Message("user", ["hello"])], session_id="conversation", actor="alice"
        ) as stream:
            async for _ in stream:
                pass
            result = await stream.get_final_response()
            assert not clients[0].is_closed(), "keep the client open until the native run finishes"
            return result.text

    try:
        if outcome == "cancelled":
            task = asyncio.create_task(invoke())
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        elif outcome == "failure":
            with pytest.raises(Exception, match="model failed"):
                await invoke()
        else:
            assert await invoke() == "hello"
        assert len(requests) == 1
        assert len(clients) == 1
        assert clients[0].is_closed(), "the template must close its model HTTP client"
        if kind == "agent-maf":
            assert mcp_exits == [True], "close the native stream and model before leaving MCP scope"
    finally:
        # Also clean up when the regression fails against the original template.
        for raw in clients:
            await raw.close()
