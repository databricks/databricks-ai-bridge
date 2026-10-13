"""Exercise cache controls and usage through the real SDK serializer and stream parser."""

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


def test_sync_cache_points_include_tools_prefix_and_leave_transcript_unchanged(endpoint):
    workspace, requests = endpoint
    messages: list[Any] = [
        {"role": "system", "content": "Stable instructions"},
        {"role": "user", "content": "Old turn"},
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "New turn"},
    ]
    tools: list[Any] = [
        {"type": "function", "function": {"name": "echo", "parameters": {}, "strict": True}}
    ]
    original = deepcopy((messages, tools))
    with DatabricksOpenAI(workspace_client=workspace, prompt_caching="auto") as client:
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    assert (messages, tools) == original
    assert "strict" not in requests[0]["tools"][0]["function"]
    for index in (0, 1, 3):
        assert requests[0]["messages"][index]["content"][0]["cache_control"] == {
            "type": "ephemeral"
        }
    assert requests[0]["messages"][2] == messages[2]
    assert_usage(response.usage)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_async_request_and_usage_normalization(endpoint, stream):
    workspace, requests = endpoint
    async with AsyncDatabricksOpenAI(workspace_client=workspace, prompt_caching="auto") as client:
        response = await client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=stream
        )
        if stream:
            async with response:
                chunks = [chunk async for chunk in response]
            usage = chunks[-1].usage
            assert response.response.is_closed
        else:
            usage = response.usage
    assert requests[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert_usage(usage)


def test_sync_stream_retains_context_manager_and_usage(endpoint):
    workspace, _requests = endpoint
    with DatabricksOpenAI(workspace_client=workspace, prompt_caching="auto") as client:
        with client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Hello"}], stream=True
        ) as stream:
            chunks = list(stream)
        assert stream.response.is_closed
    assert_usage(chunks[-1].usage)


@pytest.mark.parametrize("policy,model", [("off", MODEL), ("auto", "databricks-gpt-5-2")])
def test_disabled_and_other_providers_have_no_added_controls(endpoint, policy, model):
    workspace, requests = endpoint
    messages: list[Any] = [{"role": "user", "content": "Hello"}]
    with DatabricksOpenAI(workspace_client=workspace, prompt_caching=policy) as client:
        client.chat.completions.create(model=model, messages=messages)
    assert requests[0]["messages"] == messages


@pytest.mark.parametrize(
    "policy,model,override,provider",
    [
        ("off", MODEL, None, None),
        ("auto", "databricks-gpt-5-2", None, None),
        ("off", "my-endpoint", None, "anthropic"),
        ("auto", MODEL, "databricks-gpt-5-2", None),
    ],
)
def test_ineligible_requests_accept_payloads_that_cannot_be_deepcopied(
    endpoint, policy, model, override, provider
):
    class UncopyableDict(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Ineligible requests must not deepcopy caller payloads")

    workspace, requests = endpoint
    messages: list[Any] = [UncopyableDict(role="user", content="Hello")]
    tools: list[Any] = [
        UncopyableDict(type="function", function={"name": "echo", "parameters": {}})
    ]
    extra_body = UncopyableDict(custom_inputs={"marker": "unchanged"})
    if override is not None:
        extra_body["model"] = override
    with DatabricksOpenAI(
        workspace_client=workspace, prompt_caching=policy, prompt_cache_provider=provider
    ) as client:
        client.chat.completions.create(
            model=model, messages=messages, tools=tools, extra_body=extra_body
        )
    assert requests[0]["model"] == (override or model)
    assert requests[0]["messages"] == messages
    assert requests[0]["custom_inputs"] == {"marker": "unchanged"}


def test_extra_body_claude_model_override_is_annotated(endpoint):
    workspace, requests = endpoint
    messages: list[Any] = [{"role": "user", "content": "Hello"}]
    with DatabricksOpenAI(workspace_client=workspace, prompt_caching="auto") as client:
        client.chat.completions.create(
            model="databricks-gpt-5-2", messages=messages, extra_body={"model": MODEL}
        )
    assert requests[0]["model"] == MODEL
    assert requests[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert messages == [{"role": "user", "content": "Hello"}]


def test_custom_claude_endpoint_provider_override(endpoint):
    workspace, requests = endpoint
    with DatabricksOpenAI(
        workspace_client=workspace, prompt_caching="auto", prompt_cache_provider="anthropic"
    ) as client:
        client.chat.completions.create(
            model="my-endpoint", messages=[{"role": "user", "content": "Hello"}]
        )
    assert requests[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("location", ["messages", "extra_body", "tools"])
def test_caller_cache_controls_take_precedence(endpoint, location):
    workspace, requests = endpoint
    messages: list[Any] = [
        {"role": "system", "content": "Stable"},
        {"role": "user", "content": "Hello"},
    ]
    kwargs: dict[str, Any] = {"model": MODEL, "messages": messages}
    control = {"type": "ephemeral", "ttl": "1h"}
    if location == "messages":
        messages[-1]["content"] = [{"type": "text", "text": "Hello", "cache_control": control}]
    elif location == "extra_body":
        kwargs["extra_body"] = {"cache_control": control}
    else:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {"name": "echo", "parameters": {}},
                "cache_control": control,
            }
        ]
    original = deepcopy(kwargs)
    with DatabricksOpenAI(workspace_client=workspace, prompt_caching="auto") as client:
        client.chat.completions.create(**kwargs)
    assert kwargs == original
    assert requests[0]["messages"] == messages


def test_long_parallel_tool_batches_retain_previous_cache_boundary(endpoint):
    workspace, requests = endpoint
    messages: list[Any] = [
        {"role": "system", "content": "Stable"},
        {"role": "user", "content": "Start"},
    ]
    for batch in range(3):
        messages.append({"role": "assistant", "content": "Calling tools", "tool_calls": []})
        messages.extend(
            {"role": "tool", "tool_call_id": f"{batch}-{index}", "content": "Result"}
            for index in range(25)
        )
    with DatabricksOpenAI(workspace_client=workspace, prompt_caching="auto") as client:
        client.chat.completions.create(model=MODEL, messages=messages)
    marked = [
        message
        for message in requests[0]["messages"]
        if isinstance(message["content"], list) and "cache_control" in message["content"][-1]
    ]
    assert len(marked) == 4
    assert marked[0]["role"] == "system"
    assert [message["tool_call_id"] for message in marked[1:]] == ["0-24", "1-24", "2-24"]


def test_extra_body_message_override_is_annotated(endpoint):
    workspace, requests = endpoint
    override = {
        "messages": [{"role": "user", "content": "Actual message"}],
        "custom_inputs": {"id": "1"},
    }
    original = deepcopy(override)
    with DatabricksOpenAI(workspace_client=workspace, prompt_caching="auto") as client:
        client.chat.completions.create(model=MODEL, messages=[], extra_body=override)
    assert requests[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert requests[0]["custom_inputs"] == {"id": "1"}
    assert override == original


def test_cache_control_tool_parameter_is_not_an_explicit_breakpoint(endpoint):
    workspace, requests = endpoint
    tools: list[Any] = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "parameters": {
                    "type": "object",
                    "properties": {"cache_control": {"type": "string"}},
                },
            },
        }
    ]
    with DatabricksOpenAI(workspace_client=workspace, prompt_caching="auto") as client:
        client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Hello"}], tools=tools
        )
    assert requests[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("client_cls", [DatabricksOpenAI, AsyncDatabricksOpenAI])
@pytest.mark.parametrize(
    "kwargs", [{"prompt_caching": "invalid"}, {"prompt_cache_provider": "openai"}]
)
def test_invalid_configuration_fails_before_auth(client_cls, kwargs):
    with pytest.raises(ValueError, match="prompt_cach"):
        client_cls(**kwargs)
