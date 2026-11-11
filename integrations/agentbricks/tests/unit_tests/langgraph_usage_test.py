"""LangGraph outputs report total_tokens as input plus output, including Claude cache tokens."""

import asyncio
import copy
import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest
from template_fixtures import TEMPLATES, forget_template, import_template

pytest.importorskip("langchain_core")

CLAUDE5_BLOCKS: list[Any] = [
    {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "Greet back.", "signature": "c2ln"}],
    },
    {"type": "text", "text": "Hello!"},
]

# What databricks-langchain 0.21.0 reports for a cached Claude turn: the gateway's total, 1104,
# leaves out the 8892 cache-read tokens that input_tokens counts.
CACHED_USAGE: dict[str, Any] = {
    "input_tokens": 9828,
    "output_tokens": 168,
    "total_tokens": 1104,
    "input_token_details": {"cache_read": 8892, "cache_creation": 0},
}


@pytest.fixture
def clean_template_modules():
    yield
    forget_template()


def _cached_reply(content: Any = "Hello!", **extra: Any):
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=content, id="answer", usage_metadata=copy.deepcopy(CACHED_USAGE), **extra
    )


@pytest.mark.parametrize("content", ["Hello!", CLAUDE5_BLOCKS], ids=["text", "reasoning"])
def test_message_items_total_is_input_plus_output(content):
    from databricks_agentkit.langgraph.messages import message_items

    message = _cached_reply(content)
    usage = message_items(message)[-1]["usage_metadata"]

    assert usage == {**CACHED_USAGE, "total_tokens": 9828 + 168}
    # The checkpointed message keeps the raw usage.
    assert message.usage_metadata == CACHED_USAGE


def test_cache_creation_tokens_count_toward_total():
    from databricks_agentkit.langgraph.messages import message_items

    message = _cached_reply()
    message.usage_metadata = {
        "input_tokens": 5100,
        "output_tokens": 40,
        "total_tokens": 140,
        "input_token_details": {"cache_read": 0, "cache_creation": 5000},
    }
    [item] = message_items(message)
    assert item["usage_metadata"]["total_tokens"] == 5140
    assert item["usage_metadata"]["input_token_details"] == {
        "cache_read": 0,
        "cache_creation": 5000,
    }


def test_messages_without_usage_are_unchanged():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from databricks_agentkit.langgraph.messages import message_items, with_total_tokens

    for message in (
        AIMessage(content="hi"),
        HumanMessage(content="hi"),
        ToolMessage(content="hi", tool_call_id="t"),
    ):
        assert message_items(message) == [message.model_dump()]
    partial = {"type": "ai", "usage_metadata": {"input_tokens": 3}}
    assert with_total_tokens(partial) is partial
    assert with_total_tokens("text") == "text"


def test_real_claude_usage_conversion_is_consistent_after_message_items():
    pytest.importorskip("databricks_langchain")
    from databricks_langchain import ChatDatabricks
    from langchain_core.messages import AIMessage

    from databricks_agentkit.langgraph.messages import message_items

    usage = SimpleNamespace(
        prompt_tokens=936,
        completion_tokens=168,
        total_tokens=1104,
        prompt_tokens_details=None,
        cache_read_input_tokens=8892,
        cache_creation_input_tokens=0,
    )
    metadata = ChatDatabricks._convert_completion_usage_to_usage_metadata(cast(Any, usage))
    [item] = message_items(AIMessage(content="Hello!", usage_metadata=metadata))

    result = item["usage_metadata"]
    assert result["input_tokens"] == 9828
    assert result["total_tokens"] == result["input_tokens"] + result["output_tokens"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("clean_template_modules")
async def test_adapter_stream_and_invoke_output_report_corrected_total(monkeypatch):
    for dependency in ("databricks_langchain", "langgraph", "langchain"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[langgraph]")
    adapter, _agent = import_template(monkeypatch, TEMPLATES / "agent-langgraph")

    async def events():
        yield "updates", {"model": {"messages": [_cached_reply(CLAUDE5_BLOCKS)]}}

    out = [event async for event in adapter._serialize_events(events())]

    # The invocation output collects these same message events.
    answer = [event["message"] for event in out if event["type"] == "message"][-1]
    assert answer["usage_metadata"] == {**CACHED_USAGE, "total_tokens": 9996}


@pytest.mark.usefixtures("clean_template_modules")
def test_custom_langgraph_output_reports_corrected_total(monkeypatch):
    pytest.importorskip("databricks_langchain", reason="Requires databricks-agentbricks[langgraph]")
    # This template has no runtime.adapter, so import_template doesn't apply.
    monkeypatch.syspath_prepend(str(TEMPLATES / "custom-agent-langgraph"))
    forget_template()
    agent = importlib.import_module("agent.agent")
    monkeypatch.setattr(agent, "_model", lambda: None)

    class _Graph:
        async def ainvoke(self, _input):
            return {"messages": [_cached_reply()]}

    monkeypatch.setattr(agent, "create_agent", lambda **_kwargs: _Graph())

    result = asyncio.run(agent.invoke([{"role": "user", "content": "hi"}]))

    assert result["output"][0]["usage_metadata"]["total_tokens"] == 9996
    assert result["output"][0]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_history_reports_corrected_total(monkeypatch):
    pytest.importorskip("langgraph", reason="Requires databricks-agentbricks[langgraph]")
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    from databricks_agentkit.langgraph import read_history, session_store, thread_config

    saver = InMemorySaver()
    monkeypatch.setattr(session_store, "checkpointer", lambda: saver)
    builder = StateGraph(cast(Any, MessagesState))
    builder.add_node("answer", cast(Any, lambda _: {"messages": [_cached_reply()]}))
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)
    config = thread_config("conversation", "alice")
    await graph.ainvoke({"messages": [HumanMessage(content="hi", id="prompt")]}, config)

    history = await read_history("conversation", "alice")

    assert history["session_items"][-1]["data"]["usage_metadata"] == {
        **CACHED_USAGE,
        "total_tokens": 9996,
    }
    saved = (await graph.aget_state(config)).values["messages"][-1]
    assert saved.usage_metadata["total_tokens"] == 1104
