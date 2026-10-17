"""Exercise signature compatibility through real OpenAI JSON and SSE decoding."""

import json
from copy import deepcopy
from unittest.mock import Mock, patch

import httpx
import pytest
from databricks.sdk import WorkspaceClient
from openai import AsyncStream, Stream

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI

MODEL = "catalog.schema.neutral-alias"


def _gateway_call():
    return {
        "id": "echo",
        "type": "function",
        "function": {"name": "echo", "arguments": '{"text":"one"}'},
        "thoughtSignature": "opaque-signature",
    }


def _completion(streaming):
    call = _gateway_call()
    if streaming:
        call["index"] = 0
    message = {"role": "assistant", "tool_calls": [call]}
    return {
        "id": "completion-1",
        "created": 1,
        "object": "chat.completion.chunk" if streaming else "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "delta" if streaming else "message": message,
            }
        ],
    }


def _history():
    call = _gateway_call()
    del call["thoughtSignature"]
    call["extra_content"] = {"google": {"thought_signature": "opaque-signature"}}
    return [
        {"role": "user", "content": "Echo one."},
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "echo", "content": "one"},
    ]


def _transport(streaming, captured):
    def handle(request):
        captured.append(json.loads(request.content))
        response = _completion(streaming)
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(
                    f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n".encode()
                ),
            )
        return httpx.Response(200, json=response)

    return httpx.MockTransport(handle)


def _workspace():
    workspace = Mock(spec=WorkspaceClient)
    workspace.config.host = "https://example.databricks.com"
    return workspace


def _assert_signature_and_request(result, captured, history, original, streaming):
    message = result.choices[0].delta if streaming else result.choices[0].message
    call = message.tool_calls[0].model_dump()
    assert call["thoughtSignature"] == "opaque-signature"
    assert call["extra_content"] == {"google": {"thought_signature": "opaque-signature"}}
    assert call["id"] == "echo"
    assert captured[0]["messages"][1]["tool_calls"][0]["thoughtSignature"] == "opaque-signature"
    assert captured[0]["messages"][1]["tool_calls"][0]["extra_content"] == call["extra_content"]
    assert history == original


@pytest.mark.parametrize("streaming", [False, True])
def test_sync_signature_round_trip_and_stream_lifecycle(streaming):
    captured = []
    history = _history()
    original = deepcopy(history)
    with httpx.Client(transport=_transport(streaming, captured)) as http_client:
        with patch(
            "databricks_openai.utils.clients._get_authorized_http_client", return_value=http_client
        ):
            with DatabricksOpenAI(workspace_client=_workspace()) as client:
                result = client.chat.completions.create(
                    model=MODEL, messages=history, stream=streaming
                )
                if streaming:
                    assert isinstance(result, Stream)
                    assert not result.response.is_closed
                    with result:
                        chunk = next(result)
                    assert result.response.is_closed
                    result = chunk
                _assert_signature_and_request(result, captured, history, original, streaming)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_async_signature_round_trip_and_stream_lifecycle(streaming):
    captured = []
    history = _history()
    original = deepcopy(history)
    async with httpx.AsyncClient(transport=_transport(streaming, captured)) as http_client:
        with patch(
            "databricks_openai.utils.clients._get_authorized_async_http_client",
            return_value=http_client,
        ):
            async with AsyncDatabricksOpenAI(workspace_client=_workspace()) as client:
                result = await client.chat.completions.create(
                    model=MODEL, messages=history, stream=streaming
                )
                if streaming:
                    assert isinstance(result, AsyncStream)
                    assert not result.response.is_closed
                    async with result:
                        chunk = await anext(result)
                    assert result.response.is_closed
                    result = chunk
                _assert_signature_and_request(result, captured, history, original, streaming)


def test_signature_replay_preserves_existing_gateway_value():
    history = _history()
    history[1]["tool_calls"][0]["thoughtSignature"] = "existing-gateway-value"
    original = deepcopy(history)
    captured = []
    with httpx.Client(transport=_transport(False, captured)) as http_client:
        with patch(
            "databricks_openai.utils.clients._get_authorized_http_client", return_value=http_client
        ):
            with DatabricksOpenAI(workspace_client=_workspace()) as client:
                client.chat.completions.create(model=MODEL, messages=history)
    assert (
        captured[0]["messages"][1]["tool_calls"][0]["thoughtSignature"] == "existing-gateway-value"
    )
    assert history == original


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_native_agents_sdk_preserves_signatures_across_two_tool_turns(streaming):
    from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool

    executed = []
    captured = []

    @function_tool
    def echo(text: str) -> str:
        """Echo text."""
        executed.append(text)
        return text

    def handle(request):
        body = json.loads(request.content)
        captured.append(body)
        turn = len(captured) - 1
        for message in body["messages"]:
            for call in message.get("tool_calls", []):
                if call.get("thoughtSignature") != f"signature-{call['id']}":
                    return httpx.Response(400, json={"error": {"message": "Missing signature"}})
        response = _completion(streaming)
        message = response["choices"][0]["delta" if streaming else "message"]
        if turn < 2:
            call = message["tool_calls"][0]
            call["id"] = f"call-{turn}"
            call["function"]["arguments"] = json.dumps({"text": ["one", "two"][turn]})
            call["thoughtSignature"] = f"signature-call-{turn}"
        else:
            message.pop("tool_calls")
            message["content"] = "DONE"
            response["choices"][0]["finish_reason"] = "stop"
        if streaming:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(
                    f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n".encode()
                ),
            )
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        with patch(
            "databricks_openai.utils.clients._get_authorized_async_http_client",
            return_value=http_client,
        ):
            async with AsyncDatabricksOpenAI(workspace_client=_workspace()) as client:
                agent = Agent(
                    name="Test",
                    model=OpenAIChatCompletionsModel(
                        model="system.ai.gemini-3-8-flash", openai_client=client
                    ),
                    tools=[echo],
                )
                config = RunConfig(tracing_disabled=True)
                if streaming:
                    result = Runner.run_streamed(
                        agent, "Echo twice.", run_config=config, max_turns=3
                    )
                    async for _ in result.stream_events():
                        pass
                else:
                    result = await Runner.run(agent, "Echo twice.", run_config=config, max_turns=3)

    assert result.final_output == "DONE"
    assert executed == ["one", "two"]
    assert len(captured) == 3
    assert [
        call["id"] for message in captured[-1]["messages"] for call in message.get("tool_calls", [])
    ] == ["call-0", "call-1"]
