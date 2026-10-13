"""Provider-specific template policy and its real OpenAI request serialization."""

import json
from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

pytest.importorskip("databricks_openai")

from databricks.sdk import WorkspaceClient

from databricks_agentkit.claude import ClaudeCachingOpenAI, prepare_claude_request

NATIVE = "system.ai.claude-sonnet-4-6"
LEGACY = "system.ai.claude-sonnet-4-5"


@pytest.mark.parametrize("model", [NATIVE, "databricks-claude-sonnet-4-6", LEGACY])
def test_native_history_placement_and_legacy_explicit_points(model):
    messages = [
        {"role": "system", "content": "Stable instructions"},
        {"role": "user", "content": "Previous request"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Current request"},
    ]
    original = deepcopy(messages)
    prepared = prepare_claude_request({"model": model, "messages": messages})
    assert messages == original
    assert prepared["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    if model != LEGACY:
        assert prepared["extra_body"]["cache_control"] == {"type": "ephemeral"}
        assert prepared["messages"][1:] == original[1:]
    else:
        assert "cache_control" not in prepared.get("extra_body", {})
        for index in (1, 3):
            assert prepared["messages"][index]["content"][0]["cache_control"] == {
                "type": "ephemeral"
            }


@pytest.mark.parametrize(
    "model,policy", [(NATIVE, "off"), ("databricks-gpt-5-2", "auto"), ("my-endpoint", "auto")]
)
def test_disabled_non_claude_and_unknown_alias_requests_are_untouched(model, policy):
    class UncopyableDict(dict):
        def __deepcopy__(self, memo):
            raise AssertionError("Ineligible requests must not be copied")

    request = {"model": model, "messages": [UncopyableDict(role="user", content="Hello")]}
    assert prepare_claude_request(request, policy) is request


@pytest.mark.parametrize("location", ["message", "tools", "top"])
def test_caller_controls_take_precedence(location):
    control = {"type": "ephemeral", "ttl": "1h"}
    messages: list[dict[str, Any]] = [{"role": "system", "content": "Stable"}]
    request: dict[str, Any] = {"model": NATIVE, "messages": messages}
    if location == "message":
        request["messages"][0]["content"] = [
            {"type": "text", "text": "Stable", "cache_control": control}
        ]
    elif location == "tools":
        request["tools"] = [
            {
                "type": "function",
                "function": {"name": "echo", "parameters": {}},
                "cache_control": control,
            }
        ]
    else:
        request["extra_body"] = {"cache_control": control}
    original = deepcopy(request)
    assert prepare_claude_request(request) == original
    assert request == original


def test_native_keeps_fixed_system_marker_after_long_history_growth():
    messages = [{"role": "system", "content": "Expensive stable context"}]
    for index in range(32):
        messages.append({"role": "user" if index % 2 else "assistant", "content": f"Turn {index}"})
    prepared = prepare_claude_request({"model": NATIVE, "messages": messages})
    assert prepared["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert prepared["messages"][1:] == messages[1:]
    assert prepared["extra_body"]["cache_control"] == {"type": "ephemeral"}


def test_legacy_parallel_tool_batches_keep_the_previous_boundaries():
    messages = [{"role": "system", "content": "Stable"}, {"role": "user", "content": "Start"}]
    for batch in range(3):
        messages.append({"role": "assistant", "content": "Calling tools"})
        messages.extend(
            {"role": "tool", "tool_call_id": f"{batch}-{index}", "content": "Result"}
            for index in range(25)
        )
    prepared = prepare_claude_request({"model": LEGACY, "messages": messages})
    marked = [
        m
        for m in prepared["messages"]
        if isinstance(m["content"], list) and "cache_control" in m["content"][-1]
    ]
    assert len(marked) == 4
    assert [m["tool_call_id"] for m in marked[1:]] == ["0-24", "1-24", "2-24"]


def test_extra_body_overrides_and_schema_parameter_named_cache_control():
    request = {
        "model": "databricks-gpt-5-2",
        "messages": [],
        "extra_body": {
            "model": NATIVE,
            "messages": [{"role": "system", "content": "Actual system"}],
            "custom_inputs": {"id": "1"},
        },
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "parameters": {"properties": {"cache_control": {"type": "string"}}},
                },
            }
        ],
    }
    original = deepcopy(request)
    prepared = prepare_claude_request(request)
    assert prepared["extra_body"]["cache_control"] == {"type": "ephemeral"}
    assert prepared["extra_body"]["messages"][0]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }
    assert prepared["extra_body"]["custom_inputs"] == {"id": "1"}
    assert request == original
    request["extra_body"]["model"] = "databricks-gpt-5-2"
    assert prepare_claude_request(request) is request


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", [NATIVE, LEGACY, "databricks-gpt-5-2"])
@pytest.mark.parametrize("policy", ["auto", "off"])
@pytest.mark.asyncio
async def test_application_adapter_serializes_through_the_generic_client(
    monkeypatch, model, stream, policy
):
    sent = []

    def handle(request):
        assert request.headers["X-Databricks-Org-Id"] == "123"
        assert str(request.url).startswith("https://test.databricks.com/ai-gateway/mlflow/v1/")
        body = json.loads(request.content)
        sent.append(body)
        result = {"id": "test", "created": 0, "model": model}
        if stream:
            result.update(
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}],
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n",
            )
        result.update(
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
        )
        return httpx.Response(200, json=result)

    monkeypatch.setattr(
        "databricks_openai.utils.clients._get_authorized_async_http_client",
        lambda *_: httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    workspace = MagicMock(spec=WorkspaceClient)
    workspace.config.host = "https://test.databricks.com"
    messages = [{"role": "system", "content": "Stable"}, {"role": "user", "content": "Hello"}]
    original = deepcopy(messages)
    async with ClaudeCachingOpenAI(
        workspace_client=workspace,
        use_ai_gateway=True,
        cache_policy=policy,
        default_headers={"X-Databricks-Org-Id": "123"},
    ) as client:
        client = client.with_options(max_retries=0)
        assert client.chat.completions is client.chat.completions
        response = await client.chat.completions.create(
            model=model, messages=messages, stream=stream
        )
        if stream:
            async with response:
                chunks = [chunk async for chunk in response]
            assert chunks[-1].choices[0].delta.content == "OK"
        else:
            assert response.choices[0].message.content == "OK"
    assert messages == original
    if model == NATIVE and policy == "auto":
        assert sent[0]["cache_control"] == {"type": "ephemeral"}
        assert sent[0]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert sent[0]["messages"][1] == original[1]
    elif model == LEGACY and policy == "auto":
        assert "cache_control" not in sent[0]
        assert sent[0]["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    else:
        assert sent[0]["messages"] == original
        assert "cache_control" not in sent[0]


def test_invalid_policy():
    with pytest.raises(ValueError, match="DATABRICKS_CLAUDE_PROMPT_CACHING"):
        ClaudeCachingOpenAI(cache_policy="invalid")
    with pytest.raises(ValueError, match="DATABRICKS_CLAUDE_PROMPT_CACHING"):
        prepare_claude_request({"model": NATIVE}, "invalid")
