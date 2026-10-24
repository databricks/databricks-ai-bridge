"""Claude 5.x reasoning blocks stay out of both templates' reply text."""

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
        # gpt-oss: JSON-encoded reasoning without Claude signatures stays as today.
        json.dumps(
            [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "hmm"}]},
                {"type": "text", "text": ANSWER},
            ]
        ),
        '[{"type": "reasoning" is how this answer starts',
        [{"type": "text", "text": ANSWER}],
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
    expected = ANSWER if isinstance(content, list) else content
    assert delta_contents(content) == ([expected] if expected else [])


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
