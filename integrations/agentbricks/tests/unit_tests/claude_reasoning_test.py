"""Reasoning blocks (Claude 5.x, Qwen, Gemini, gpt-oss) stay out of both templates' reply text."""

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"

SIGNATURE = "CAQSqwUKDAgBEAEYAiIEdGVzdA=="
REASONING = "The user greets me, so I greet them back."
ANSWER = "Hello! How can I help?"


def _reasoning(text: str, signature: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text, "signature": signature, **extra}],
    }


# What databricks-langchain returns for Claude 5.x: merged stream blocks keep their merge indexes.
CLAUDE5_BLOCKS: list[Any] = [
    "",
    {**_reasoning(REASONING, SIGNATURE, index=0), "index": 0},
    {"type": "text", "text": ANSWER, "index": 1},
]
# What databricks-langchain 0.21.0 returns: one JSON list per streamed reasoning chunk.
CLAUDE5_LEGACY = json.dumps([_reasoning("", "")]) + json.dumps([_reasoning("", SIGNATURE)]) + ANSWER


def _template(monkeypatch, name: str):
    monkeypatch.syspath_prepend(str(TEMPLATES / name))
    for module in list(sys.modules):
        if module in {"agent", "runtime"} or module.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, module)
    return importlib.import_module("agent.agent"), importlib.import_module("runtime.adapter")


@pytest.fixture
def langgraph_template(monkeypatch):
    for dependency in ("databricks_langchain", "langgraph", "langchain"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[langgraph]")
    yield _template(monkeypatch, "agent-langgraph")
    for module in list(sys.modules):
        if module in {"agent", "runtime"} or module.startswith(("agent.", "runtime.")):
            sys.modules.pop(module, None)


@pytest.fixture
def openai_template(monkeypatch):
    for dependency in ("agents", "databricks_openai"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[openai]")
    yield _template(monkeypatch, "agent-openai")
    for module in list(sys.modules):
        if module in {"agent", "runtime"} or module.startswith(("agent.", "runtime.")):
            sys.modules.pop(module, None)


@pytest.mark.parametrize("content", [CLAUDE5_BLOCKS, CLAUDE5_LEGACY])
def test_message_items_split_reasoning_from_answer(content):
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from databricks_agentkit.langgraph.messages import message_items

    message = AIMessage(content=content, id="answer")
    reasoning, answer = message_items(message)

    assert answer == {**message.model_dump(), "content": ANSWER}
    assert reasoning["role"] == "assistant"
    [block] = [block for block in reasoning["content"] if block["summary"][0]["signature"]]
    assert block == _reasoning(REASONING if content is CLAUDE5_BLOCKS else "", SIGNATURE)


def test_message_items_keep_tool_calls_on_a_reasoning_turn():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from databricks_agentkit.langgraph.messages import message_items

    message = AIMessage(
        content=[_reasoning("Look it up.", SIGNATURE)],
        tool_calls=[{"name": "lookup", "args": {}, "id": "t"}],
    )
    reasoning, call = message_items(message)
    assert reasoning == {"role": "assistant", "content": [_reasoning("Look it up.", SIGNATURE)]}
    assert call["content"] == ""
    assert call["tool_calls"][0]["name"] == "lookup"


def test_delta_contents_stream_reasoning_and_text_separately():
    from databricks_agentkit.langgraph.messages import delta_contents

    readable = {**_reasoning("Thinking", "", index=0), "index": 0}
    assert delta_contents([readable]) == [[_reasoning("Thinking", "")]]
    assert delta_contents([{**_reasoning("", SIGNATURE, index=0), "index": 0}]) == []
    assert delta_contents([{"type": "text", "text": "Hello", "index": 1}]) == ["Hello"]
    assert delta_contents(json.dumps([_reasoning("Thinking", "")])) == [
        [_reasoning("Thinking", "")]
    ]
    assert delta_contents(json.dumps([_reasoning("", "")])) == []


@pytest.mark.parametrize(
    "content",
    [
        ANSWER,
        "",
        '[{"type": "reasoning" is how this answer starts',
        # Lists with other parts (images) are not reasoning content.
        [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
    ],
)
def test_other_replies_are_unchanged(content):
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from databricks_agentkit.langgraph.messages import delta_contents, message_items

    for message in (
        AIMessage(content=content),
        HumanMessage(content=content),
        ToolMessage(content=content, tool_call_id="t"),
    ):
        assert message_items(message) == [message.model_dump()]
    assert delta_contents(content) == ([content] if content else [])


def test_legacy_reasoning_fragments_merge_into_one_block_per_signature():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from databricks_agentkit.langgraph.messages import message_items

    # databricks-langchain 0.21.0 streaming a summarized reasoning block, then the answer.
    fragments = [_reasoning("", ""), _reasoning("The user ", ""), _reasoning("greets me.", "")]
    content = "".join(json.dumps([fragment]) for fragment in fragments)
    content += json.dumps([_reasoning("", SIGNATURE)]) + ANSWER
    reasoning, answer = message_items(AIMessage(content=content))
    assert reasoning["content"] == [_reasoning("The user greets me.", SIGNATURE)]
    assert answer["content"] == ANSWER


def test_gateway_messages_restore_legacy_reasoning_for_the_next_request():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from databricks_agentkit.langgraph.messages import gateway_messages

    call = AIMessage(
        content=json.dumps([_reasoning("Look it up.", "")])
        + json.dumps([_reasoning("", SIGNATURE)]),
        tool_calls=[{"name": "lookup", "args": {}, "id": "t"}],
    )
    others = [
        HumanMessage(content=CLAUDE5_LEGACY),
        ToolMessage(content="12:00", tool_call_id="t"),
        AIMessage(content=ANSWER),
        AIMessage(content=CLAUDE5_BLOCKS),
    ]
    restored, *unchanged = gateway_messages([call, *others])
    assert restored.content == [_reasoning("Look it up.", SIGNATURE)]
    assert restored.tool_calls == call.tool_calls
    assert unchanged == others
    [answer] = gateway_messages([AIMessage(content=CLAUDE5_LEGACY)])
    assert answer.content == [_reasoning("", SIGNATURE), {"type": "text", "text": ANSWER}]


def test_langgraph_template_sends_reasoning_signature_after_a_tool_call(langgraph_template):
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    agent_module, _adapter = langgraph_template
    model = agent_module._RoutedChatDatabricks.model_construct(
        model="system.ai.claude-opus-5-5", extra_params={}
    )
    call = AIMessage(
        content=json.dumps([_reasoning("", SIGNATURE)]),
        tool_calls=[{"name": "get_current_time", "args": {}, "id": "toolu_1"}],
    )
    request = model._prepare_inputs(
        [
            HumanMessage(content="What time is it?"),
            call,
            ToolMessage("12:00", tool_call_id="toolu_1"),
        ]
    )
    assistant = request["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [_reasoning("", SIGNATURE)]
    assert assistant["tool_calls"][0]["id"] == "toolu_1"

    # Claude 4.x and other models' string replies go back exactly as before.
    request = model._prepare_inputs([HumanMessage(content="hi"), AIMessage(content=ANSWER)])
    assert request["messages"][1] == {"role": "assistant", "content": ANSWER}


@pytest.mark.asyncio
async def test_langgraph_adapter_streams_text_and_returns_reasoning_item(langgraph_template):
    from langchain_core.messages import AIMessage, AIMessageChunk

    _agent, adapter = langgraph_template

    async def events():
        yield "messages", (AIMessageChunk(content="", id="run"), {})
        yield "messages", (AIMessageChunk(content=[CLAUDE5_BLOCKS[1]], id="run"), {})
        yield "messages", (AIMessageChunk(content=[CLAUDE5_BLOCKS[2]], id="run"), {})
        yield "updates", {"model": {"messages": [AIMessage(content=CLAUDE5_BLOCKS, id="run")]}}

    out = [event async for event in adapter._serialize_events(events())]

    assert [event["content"] for event in out if event["type"] == "delta"] == [
        [_reasoning(REASONING, SIGNATURE)],
        ANSWER,
    ]
    reasoning, answer = [event["message"] for event in out if event["type"] == "message"]
    assert reasoning == {"role": "assistant", "content": [_reasoning(REASONING, SIGNATURE)]}
    assert answer["type"] == "ai"
    assert answer["content"] == ANSWER
    assert SIGNATURE not in json.dumps(answer)


@pytest.mark.asyncio
async def test_langgraph_history_lists_reasoning_before_answer_text(
    langgraph_template, monkeypatch
):
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph

    from databricks_agentkit.langgraph import read_history, session_store, thread_config

    saver = InMemorySaver()
    monkeypatch.setattr(session_store, "checkpointer", lambda: saver)
    builder = StateGraph(cast(Any, MessagesState))
    builder.add_node(
        "answer",
        cast(Any, lambda _: {"messages": [AIMessage(content=CLAUDE5_BLOCKS, id="answer")]}),
    )
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=saver)
    config = thread_config("conversation", "alice")
    await graph.ainvoke({"messages": [HumanMessage(content="hi", id="prompt")]}, config)

    history = await read_history("conversation", "alice")
    # The checkpoint keeps the signed blocks for the model's next turn.
    saved = (await graph.aget_state(config)).values["messages"][-1]
    assert saved.content == CLAUDE5_BLOCKS
    assert [item["item_id"] for item in history["session_items"]] == [
        "prompt",
        "answer:reasoning",
        "answer",
    ]
    assert history["session_items"][1]["data"]["content"] == [_reasoning(REASONING, SIGNATURE)]
    assert history["session_items"][2]["data"]["content"] == ANSWER


def _claude5_chunks(deltas: list[dict[str, Any]]):
    """Chunks as databricks-openai's AsyncDatabricksOpenAI passes them on for Claude 5.x."""
    from openai.types.chat import ChatCompletionChunk

    return [
        ChatCompletionChunk.construct(
            id="chatcmpl",
            object="chat.completion.chunk",
            created=0,
            model="claude-opus-5-5",
            choices=[
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": "stop" if index == len(deltas) - 1 else None,
                }
            ],
        )
        for index, delta in enumerate(deltas)
    ]


class _FakeCompletions:
    def __init__(self, deltas):
        self.deltas = deltas

    async def create(self, **_kwargs):
        async def stream():
            for chunk in _claude5_chunks(self.deltas):
                yield chunk

        return stream()


@pytest.mark.asyncio
async def test_openai_adapter_streams_text_and_returns_reasoning_item(openai_template):
    from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

    _agent, adapter = openai_template
    set_tracing_disabled(True)
    completions = _FakeCompletions(
        [
            {"role": "assistant", "content": ""},
            {"thinking_blocks": [{"type": "thinking", "thinking": REASONING}]},
            {"thinking_blocks": [{"type": "thinking", "thinking": "", "signature": SIGNATURE}]},
            {"content": "Hello! "},
            {"content": "How can I help?"},
        ]
    )
    client = cast(
        Any,
        types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions), base_url="https://gateway"
        ),
    )
    agent = Agent(
        name="Agent",
        model=OpenAIChatCompletionsModel(model="system.ai.claude-opus-5-5", openai_client=client),
    )

    out = [event async for event in adapter._serialize_events(Runner.run_streamed(agent, "hi"))]

    assert "".join(event["content"] for event in out if event["type"] == "delta") == ANSWER
    reasoning, answer = [event["message"] for event in out if event["type"] == "message"]
    assert answer == {"role": "assistant", "content": ANSWER}
    [block] = reasoning["content"]
    assert block["type"] == "reasoning"
    assert block["content"][0]["text"] == REASONING
    assert block["encrypted_content"] == SIGNATURE


def test_openai_template_uses_databricks_chat_model(openai_template, monkeypatch):
    from agents import OpenAIChatCompletionsModel

    from databricks_agentkit.openai import chat_model

    agent_module, _adapter = openai_template

    class DatabricksChatCompletionsModel(OpenAIChatCompletionsModel):
        pass

    monkeypatch.setitem(
        sys.modules,
        "databricks_openai.agents",
        types.SimpleNamespace(DatabricksChatCompletionsModel=DatabricksChatCompletionsModel),
    )
    client: Any = types.SimpleNamespace(base_url="https://gateway")
    monkeypatch.setattr("agents.models._openai_shared.get_default_openai_client", lambda: client)
    for tools in ("all_tools", "memory_tools", "genie_tools"):
        monkeypatch.setattr(agent_module, tools, lambda *_args, **_kwargs: [])
    model = agent_module.create_agent("alice", model="system.ai.claude-opus-5-5").model
    assert isinstance(model, DatabricksChatCompletionsModel)
    assert (model.model, model._client) == ("system.ai.claude-opus-5-5", client)

    # Without a configured client, or on releases without the class, the SDK resolves the name.
    monkeypatch.setattr("agents.models._openai_shared.get_default_openai_client", lambda: None)
    assert chat_model("system.ai.claude-opus-5-5") == "system.ai.claude-opus-5-5"
    monkeypatch.setitem(sys.modules, "databricks_openai.agents", types.SimpleNamespace())
    monkeypatch.setattr("agents.models._openai_shared.get_default_openai_client", lambda: client)
    assert chat_model("system.ai.claude-sonnet-4-5") == "system.ai.claude-sonnet-4-5"


# Other models: unsigned reasoning, and text parts with extra keys.

UNSIGNED_REASONING = "The user says hi."


def _unsigned(text: str) -> dict[str, Any]:
    """A Qwen / gpt-oss reasoning part: summary text, no signature."""
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


# `system.ai.qwen35-122b-a10b` through databricks-langchain with list content kept as blocks.
QWEN_BLOCKS: list[Any] = [_unsigned(UNSIGNED_REASONING), {"type": "text", "text": ANSWER}]
# The same reply through databricks-langchain 0.21.0: one JSON list per streamed reasoning chunk.
QWEN_LEGACY_STREAM = (
    json.dumps([_unsigned("The user ")]) + json.dumps([_unsigned("says hi.")]) + ANSWER
)
# ... and non-streamed: the whole list content JSON-encoded.
QWEN_LEGACY_MESSAGE = json.dumps(QWEN_BLOCKS)
# `system.ai.gemini-3-pro-image`'s answer after a tool call: text parts with a thought signature.
GEMINI_TEXT: list[Any] = [{"type": "text", "text": ANSWER, "thoughtSignature": "Z2VtaW5p"}]


@pytest.mark.parametrize("content", [QWEN_BLOCKS, QWEN_LEGACY_STREAM, QWEN_LEGACY_MESSAGE])
def test_unsigned_reasoning_is_a_separate_item_and_answer_is_text(content):
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from databricks_agentkit.langgraph.messages import message_items

    message = AIMessage(content=content, id="answer")
    reasoning, answer = message_items(message)

    assert answer == {**message.model_dump(), "content": ANSWER}
    assert reasoning == {"role": "assistant", "content": [_unsigned(UNSIGNED_REASONING)]}


def test_text_parts_with_extra_keys_are_answer_text():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from databricks_agentkit.langgraph.messages import delta_contents, message_items

    [answer] = message_items(AIMessage(content=GEMINI_TEXT))
    assert answer["content"] == ANSWER
    assert delta_contents(GEMINI_TEXT) == [ANSWER]


def test_unsigned_reasoning_deltas_stream_reasoning_then_text():
    from databricks_agentkit.langgraph.messages import delta_contents

    assert delta_contents(json.dumps([_unsigned("The user ")])) == [[_unsigned("The user ")]]
    assert delta_contents([{**_unsigned(UNSIGNED_REASONING), "index": 0}]) == [
        [_unsigned(UNSIGNED_REASONING)]
    ]
    assert delta_contents([{"type": "text", "text": "Hello", "index": 1}]) == ["Hello"]


def test_unsigned_reasoning_is_not_sent_back():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from databricks_agentkit.langgraph.messages import gateway_messages

    call = AIMessage(
        content=json.dumps([_unsigned("Look it up.")]),
        tool_calls=[{"name": "lookup", "args": {}, "id": "t"}],
    )
    replies = [AIMessage(content=QWEN_LEGACY_STREAM), AIMessage(content=QWEN_LEGACY_MESSAGE)]
    restored_call, *restored = gateway_messages([call, *replies])
    assert restored_call.content == ""
    assert restored_call.tool_calls == call.tool_calls
    assert [message.content for message in restored] == [ANSWER, ANSWER]


def test_signed_reasoning_is_still_sent_back_next_to_unsigned():
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from databricks_agentkit.langgraph.messages import gateway_messages

    content = json.dumps([_reasoning("Signed.", SIGNATURE)]) + ANSWER
    [message] = gateway_messages([AIMessage(content=content)])
    assert message.content == [_reasoning("Signed.", SIGNATURE), {"type": "text", "text": ANSWER}]


def test_langgraph_template_sends_qwen_history_as_answer_text(langgraph_template):
    from langchain_core.messages import AIMessage, HumanMessage

    agent_module, _adapter = langgraph_template
    model = agent_module._RoutedChatDatabricks.model_construct(
        model="system.ai.qwen35-122b-a10b", extra_params={}
    )
    request = model._prepare_inputs(
        [HumanMessage(content="hi"), AIMessage(content=QWEN_LEGACY_STREAM), HumanMessage("again")]
    )
    assert request["messages"][1] == {"role": "assistant", "content": ANSWER}


@pytest.mark.asyncio
async def test_langgraph_adapter_streams_qwen_reply(langgraph_template):
    from langchain_core.messages import AIMessage, AIMessageChunk

    _agent, adapter = langgraph_template

    async def events():
        yield (
            "messages",
            (AIMessageChunk(content=json.dumps([_unsigned(UNSIGNED_REASONING)]), id="r"), {}),
        )
        yield "messages", (AIMessageChunk(content=ANSWER, id="r"), {})
        yield "updates", {"model": {"messages": [AIMessage(content=QWEN_LEGACY_STREAM, id="r")]}}

    out = [event async for event in adapter._serialize_events(events())]

    assert [event["content"] for event in out if event["type"] == "delta"] == [
        [_unsigned(UNSIGNED_REASONING)],
        ANSWER,
    ]
    reasoning, answer = [event["message"] for event in out if event["type"] == "message"]
    assert reasoning == {"role": "assistant", "content": [_unsigned(UNSIGNED_REASONING)]}
    assert answer["content"] == ANSWER


@pytest.mark.asyncio
async def test_openai_adapter_streams_unsigned_reasoning_reply(openai_template):
    from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

    _agent, adapter = openai_template
    set_tracing_disabled(True)
    # What AsyncDatabricksOpenAI passes on for Qwen: reasoning moved to unsigned thinking blocks.
    completions = _FakeCompletions(
        [
            {"role": "assistant", "content": ""},
            {"thinking_blocks": [{"type": "thinking", "thinking": "The user "}]},
            {"thinking_blocks": [{"type": "thinking", "thinking": "says hi."}]},
            {"content": "Hello! "},
            {"content": "How can I help?"},
        ]
    )
    client = cast(
        Any,
        types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions), base_url="https://gateway"
        ),
    )
    agent = Agent(
        name="Agent",
        model=OpenAIChatCompletionsModel(model="system.ai.qwen35-122b-a10b", openai_client=client),
    )

    out = [event async for event in adapter._serialize_events(Runner.run_streamed(agent, "hi"))]

    assert "".join(event["content"] for event in out if event["type"] == "delta") == ANSWER
    reasoning, answer = [event["message"] for event in out if event["type"] == "message"]
    assert answer == {"role": "assistant", "content": ANSWER}
    [block] = reasoning["content"]
    assert block["type"] == "reasoning"
    assert block["content"][0]["text"] == UNSIGNED_REASONING
    assert not block.get("encrypted_content")
