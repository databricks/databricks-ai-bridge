"""Usage regression tests through real OpenAI JSON and SSE deserialization."""

import json
from functools import reduce
from typing import Any, Iterator, cast
from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from openai import AsyncOpenAI, OpenAI

from databricks_langchain import ChatDatabricks


@pytest_asyncio.fixture
async def usage_client():
    requests = []
    payload = {}

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=payload["stream"],
            )
        return httpx.Response(200, json=payload["response"])

    transport = httpx.MockTransport(respond)
    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=transport)
    ) as async_client:
        with (
            OpenAI(api_key="test", http_client=httpx.Client(transport=transport)) as client,
            patch("databricks_langchain.chat_models.get_openai_client", return_value=client),
            patch(
                "databricks_langchain.chat_models.get_async_openai_client",
                return_value=async_client,
            ),
        ):
            yield payload, requests


def _sse(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


def _chat_payload(usage, *, usage_only=False):
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "usage": usage,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world!"},
                "finish_reason": "stop",
            }
        ],
    }
    events = [
        {
            **response,
            "object": "chat.completion.chunk",
            "usage": None if usage_only else usage,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}
            ],
        }
        for text in ("Hello", " world", "!")
    ]
    if usage_only:
        events.append({**response, "object": "chat.completion.chunk", "choices": []})
    return {"response": response, "stream": _sse(events)}


_CHAT_USAGE = [
    pytest.param(
        {"prompt_tokens": 2697, "completion_tokens": 8, "total_tokens": 2705},
        {},
        id="reported-claude-stream-count",
    ),
    pytest.param(
        {
            "prompt_tokens": 6676,
            "completion_tokens": 8,
            "total_tokens": 6684,
            "prompt_tokens_details": {"cached_tokens": 6000},
        },
        {"cache_read": 6000},
        id="reported-gemini-stream-count",
    ),
    pytest.param(
        {
            "prompt_tokens": 100,
            "completion_tokens": 8,
            "total_tokens": 108,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 20,
        },
        {"cache_read": 30, "cache_creation": 20},
        id="claude-inclusive-cache",
    ),
    pytest.param(
        {
            "prompt_tokens": 100,
            "completion_tokens": 8,
            "total_tokens": 108,
            "cache_creation_input_tokens": 80,
        },
        {"cache_creation": 80},
        id="claude-cache-write-only",
    ),
    pytest.param(
        {
            "prompt_tokens": 100,
            "completion_tokens": 8,
            "total_tokens": 108,
            "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 20},
        },
        {"cache_read": 30, "cache_creation": 20},
        id="gpt-cache-write",
    ),
]


@pytest.mark.parametrize(("usage", "details"), _CHAT_USAGE)
@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_chat_usage_matches_gateway_payload(usage_client, usage, details, mode):
    payload, requests = usage_client
    payload.update(_chat_payload(usage))
    llm = ChatDatabricks(model="test-model")
    messages = [HumanMessage(content="Hello")]
    if mode == "invoke":
        result = llm.invoke(messages)
    elif mode == "ainvoke":
        result = await llm.ainvoke(messages)
    else:
        chunks = (
            list(llm.stream(messages))
            if mode == "stream"
            else [chunk async for chunk in llm.astream(messages)]
        )
        result = reduce(lambda left, right: left + right, chunks)
        assert sum(chunk.usage_metadata is not None for chunk in chunks) == 1
    assert result.content == "Hello world!"
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] == usage["prompt_tokens"]
    assert result.usage_metadata["output_tokens"] == usage["completion_tokens"]
    assert result.usage_metadata["total_tokens"] == usage["total_tokens"]
    for key, value in details.items():
        assert result.usage_metadata["input_token_details"][key] == value
    assert len(requests) == 1
    assert requests[0]["model"] == "test-model"
    assert requests[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert requests[0]["stream"] == (mode in ("stream", "astream"))
    if mode in ("stream", "astream"):
        assert requests[0]["stream_options"] == {"include_usage": True}
    else:
        assert "stream_options" not in requests[0]


@pytest.mark.parametrize("usage_only", [True, False])
@pytest.mark.parametrize("async_mode", [True, False])
@pytest.mark.parametrize("stream_usage", [True, False])
@pytest.mark.asyncio
async def test_chat_stream_usage_shapes(usage_client, usage_only, async_mode, stream_usage):
    payload, requests = usage_client
    usage = {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    payload.update(_chat_payload(usage, usage_only=usage_only))
    llm = ChatDatabricks(model="test-model", stream_usage=stream_usage)
    chunks = (
        [chunk async for chunk in llm.astream("Hello")] if async_mode else list(llm.stream("Hello"))
    )
    aggregate = reduce(lambda left, right: left + right, chunks)
    assert aggregate.content == "Hello world!"
    assert sum(chunk.usage_metadata is not None for chunk in chunks) == int(stream_usage)
    assert aggregate.usage_metadata == (
        {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18} if stream_usage else None
    )
    assert requests[0].get("stream_options") == ({"include_usage": True} if stream_usage else None)


@pytest.mark.parametrize("options", [{"include_usage": False}, {"include_usage": True}])
@pytest.mark.parametrize("async_mode", [True, False])
@pytest.mark.asyncio
async def test_chat_stream_preserves_explicit_stream_options(usage_client, options, async_mode):
    payload, requests = usage_client
    payload.update(_chat_payload({"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}))
    llm = ChatDatabricks(model="test-model")
    if async_mode:
        _ = [chunk async for chunk in llm.astream("Hello", stream_options=options)]
    else:
        list(llm.stream("Hello", stream_options=options))
    assert requests[0]["stream_options"] == options


def _responses_payload(cache_write_tokens):
    response = {
        "id": "resp-test",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": "completed",
        "output": [
            {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 100,
            "output_tokens": 8,
            "total_tokens": 108,
            "input_tokens_details": {"cached_tokens": 30},
            "output_tokens_details": {"reasoning_tokens": 3},
        },
    }
    if cache_write_tokens is not None:
        response["usage"]["input_tokens_details"]["cache_write_tokens"] = cache_write_tokens
    return {
        "response": response,
        "stream": _sse(
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg-test",
                    "output_index": 0,
                    "content_index": 0,
                    "sequence_number": 0,
                    "delta": "Hello",
                    "logprobs": [],
                },
                {"type": "response.completed", "sequence_number": 1, "response": response},
            ]
        ),
    }


@pytest.mark.parametrize("cache_write_tokens", [None, 0, 20])
@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.asyncio
async def test_responses_cache_writes(usage_client, cache_write_tokens, mode):
    payload, requests = usage_client
    payload.update(_responses_payload(cache_write_tokens))
    llm = ChatDatabricks(model="test-model", use_responses_api=True)
    if mode == "invoke":
        result = llm.invoke("Hello")
    elif mode == "ainvoke":
        result = await llm.ainvoke("Hello")
    else:
        chunks = (
            list(llm.stream("Hello"))
            if mode == "stream"
            else [chunk async for chunk in llm.astream("Hello")]
        )
        result = reduce(lambda left, right: left + right, chunks)
        assert sum(chunk.usage_metadata is not None for chunk in chunks) == 1
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_tokens"] == 100
    assert result.usage_metadata["output_tokens"] == 8
    assert result.usage_metadata["total_tokens"] == 108
    assert result.usage_metadata["input_token_details"]["cache_read"] == 30
    if cache_write_tokens is not None:
        assert result.usage_metadata["input_token_details"]["cache_creation"] == cache_write_tokens
    assert result.usage_metadata["output_token_details"]["reasoning"] == 3
    assert len(requests) == 1
    assert requests[0]["model"] == "test-model"
    assert requests[0]["input"][0]["role"] == "user"
    assert "stream_options" not in requests[0]


def test_langgraph_message_stream_preserves_authoritative_usage(usage_client):
    payload, _ = usage_client
    usage = {"prompt_tokens": 2697, "completion_tokens": 8, "total_tokens": 2705}
    payload.update(_chat_payload(usage))
    llm = ChatDatabricks(model="test-model")
    graph = StateGraph(cast(Any, MessagesState))
    graph.add_node("chat", lambda state: {"messages": [llm.invoke(state["messages"])]})
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    events = cast(
        Iterator[tuple[str, Any]],
        graph.compile().stream(
            {"messages": [HumanMessage(content="Hello")]}, stream_mode=["messages", "values"]
        ),
    )
    values = [event for mode, event in events if mode == "values"]
    assert values[-1]["messages"][-1].usage_metadata == {
        "input_tokens": 2697,
        "output_tokens": 8,
        "total_tokens": 2705,
    }
