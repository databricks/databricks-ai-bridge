"""Generic clients preserve caller cache controls and normalize returned usage."""

import json
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from databricks.sdk import WorkspaceClient

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI

MODEL = "system.ai.claude-sonnet-4-5"
USAGE = {
    "prompt_tokens": 5000,
    "completion_tokens": 2,
    "total_tokens": 5002,
    "cache_read_input_tokens": 4096,
    "cache_creation_input_tokens": 16,
}


@pytest.fixture
def endpoint(monkeypatch):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        response = {
            "id": "test",
            "created": 0,
            "model": body["model"],
            "usage": USAGE,
        }
        if body.get("stream"):
            response.update(
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}],
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n",
            )
        response.update(
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
        )
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(
        "databricks_openai.utils.clients._get_authorized_http_client",
        lambda *_args: httpx.Client(transport=transport),
    )
    monkeypatch.setattr(
        "databricks_openai.utils.clients._get_authorized_async_http_client",
        lambda *_args: httpx.AsyncClient(transport=transport),
    )
    workspace = MagicMock(spec=WorkspaceClient)
    workspace.config.host = "https://test.databricks.com"
    return workspace, requests


def assert_usage(usage):
    assert usage.prompt_tokens == 5000
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.prompt_tokens_details.cached_tokens == 4096
    assert usage.prompt_tokens_details.cache_write_tokens == 16
    assert usage.cache_read_input_tokens == 4096
    assert usage.cache_creation_input_tokens == 16


@pytest.mark.parametrize("location", ["none", "top", "block"])
def test_sync_preserves_caller_request_and_reports_cache_usage(endpoint, location):
    workspace, requests = endpoint
    messages: list[Any] = [
        {"role": "system", "content": "Stable"},
        {"role": "user", "content": "Hello"},
    ]
    extra_body = {}
    if location == "top":
        extra_body = {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    elif location == "block":
        messages[0]["content"] = [
            {"type": "text", "text": "Stable", "cache_control": {"type": "ephemeral"}}
        ]
    original = deepcopy(messages)
    with DatabricksOpenAI(workspace_client=workspace) as client:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, extra_body=extra_body
        )
    assert messages == original
    assert requests[0]["messages"] == original
    assert requests[0].get("cache_control") == extra_body.get("cache_control")
    assert_usage(response.usage)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_async_passes_top_level_control_and_normalizes_usage(endpoint, stream):
    workspace, requests = endpoint
    async with AsyncDatabricksOpenAI(workspace_client=workspace) as client:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            extra_body={"cache_control": {"type": "ephemeral"}},
            stream=stream,
        )
        if stream:
            async with response:
                chunks = [chunk async for chunk in response]
            assert response.response.is_closed
            usage = chunks[-1].usage
        else:
            usage = response.usage
    assert requests[0]["messages"] == [{"role": "user", "content": "Hello"}]
    assert requests[0]["cache_control"] == {"type": "ephemeral"}
    assert_usage(usage)


def test_sync_stream_preserves_context_manager_and_cache_usage(endpoint):
    workspace, _ = endpoint
    with DatabricksOpenAI(workspace_client=workspace) as client:
        with client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True
        ) as stream:
            chunks = list(stream)
        assert stream.response.is_closed
    assert_usage(chunks[-1].usage)


@pytest.mark.parametrize("model", [MODEL, "databricks-gpt-5-2"])
def test_generic_clients_do_not_infer_or_copy_caching_policy(endpoint, model):
    class UncopyableDict(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Generic connectors must not copy payloads for cache placement")

    workspace, requests = endpoint
    messages: list[Any] = [UncopyableDict(role="user", content="Hello")]
    with DatabricksOpenAI(workspace_client=workspace) as client:
        client.chat.completions.create(model=model, messages=messages)
    assert requests[0]["messages"] == messages
    assert "cache_control" not in requests[0]
