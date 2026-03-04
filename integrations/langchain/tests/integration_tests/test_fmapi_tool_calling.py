"""
End-to-end FMAPI tool calling tests for ChatDatabricks via LangGraph.

Prerequisites:
- FMAPI endpoints must be available on the test workspace
"""

from __future__ import annotations

import os

import pytest
from databricks_ai_bridge.test_utils.fmapi import (
    LANGCHAIN_CHAT_SKIP_MODELS,
    async_retry,
    discover_chat_models,
    discover_responses_models,
    max_tokens_for_model,
    retry,
)
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from databricks_langchain import ChatDatabricks

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FMAPI_TOOL_CALLING_TESTS") != "1",
    reason="FMAPI tool calling tests disabled. Set RUN_FMAPI_TOOL_CALLING_TESTS=1 to enable.",
)

_CHAT_MODELS = discover_chat_models(LANGCHAIN_CHAT_SKIP_MODELS)
_RESPONSES_MODELS = discover_responses_models()


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b


# =============================================================================
# Sync LangGraph Agent
# =============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("model", _CHAT_MODELS)
class TestLangGraphSync:
    """Sync LangGraph agent tests using ChatDatabricks + create_react_agent."""

    def test_single_turn(self, model):
        """Single-turn: agent calls tools and produces a final answer."""

        def _run():
            llm = ChatDatabricks(model=model, max_tokens=max_tokens_for_model(model))
            agent = create_react_agent(llm, [add, multiply])

            response = agent.invoke(
                {
                    "messages": [
                        (
                            "human",
                            "Use the add tool to compute 10 + 5, then use the multiply tool "
                            "to multiply the result by 3. You MUST use the tools.",
                        )
                    ]
                }
            )

            last_message = response["messages"][-1]
            assert isinstance(last_message, AIMessage)
            assert "45" in last_message.content

            tool_messages = [m for m in response["messages"] if isinstance(m, ToolMessage)]
            assert len(tool_messages) > 0, "Expected tool calls in conversation history"

        retry(_run)

    def test_multi_turn(self, model):
        """Multi-turn: agent maintains conversation context across turns."""

        def _run():
            llm = ChatDatabricks(model=model, max_tokens=max_tokens_for_model(model))
            agent = create_react_agent(llm, [add, multiply], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": f"test-sync-multi-turn-{model}"}}

            response = agent.invoke({"messages": [("human", "What is 10 + 5?")]}, config=config)
            last_message = response["messages"][-1]
            assert isinstance(last_message, AIMessage)
            assert "15" in last_message.content

            response = agent.invoke({"messages": [("human", "Multiply that by 3")]}, config=config)
            last_message = response["messages"][-1]
            assert isinstance(last_message, AIMessage)
            assert "45" in last_message.content

        retry(_run)

    def test_streaming(self, model):
        """Streaming: agent streams node updates and tool execution events."""

        def _run():
            llm = ChatDatabricks(model=model, max_tokens=max_tokens_for_model(model))
            agent = create_react_agent(llm, [add, multiply])

            events = list(
                agent.stream(
                    {
                        "messages": [
                            (
                                "human",
                                "Use the add tool to compute 10 + 5, then use the multiply tool "
                                "to multiply the result by 3. You MUST use the tools.",
                            )
                        ]
                    },
                    stream_mode="updates",
                )
            )

            assert len(events) > 0, "No stream events received"

            nodes_seen = set()
            for event in events:
                nodes_seen.update(event.keys())

            assert "agent" in nodes_seen, f"Expected 'agent' node, got: {nodes_seen}"
            assert "tools" in nodes_seen, f"Expected 'tools' node, got: {nodes_seen}"

            last_event = events[-1]
            last_messages = list(last_event.values())[0]["messages"]
            assert any("45" in str(m.content) for m in last_messages)

        retry(_run)


# =============================================================================
# Async LangGraph Agent
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("model", _CHAT_MODELS)
class TestLangGraphAsync:
    """Async LangGraph agent tests using ChatDatabricks + create_react_agent."""

    async def test_single_turn(self, model):
        """Single-turn via ainvoke."""

        async def _run():
            llm = ChatDatabricks(model=model, max_tokens=max_tokens_for_model(model))
            agent = create_react_agent(llm, [add, multiply])

            response = await agent.ainvoke(
                {
                    "messages": [
                        (
                            "human",
                            "Use the add tool to compute 10 + 5, then use the multiply tool "
                            "to multiply the result by 3. You MUST use the tools.",
                        )
                    ]
                }
            )

            last_message = response["messages"][-1]
            assert isinstance(last_message, AIMessage)
            assert "45" in last_message.content

            tool_messages = [m for m in response["messages"] if isinstance(m, ToolMessage)]
            assert len(tool_messages) > 0, "Expected tool calls in conversation history"

        await async_retry(_run)

    async def test_multi_turn(self, model):
        """Multi-turn via ainvoke with MemorySaver checkpointer."""

        async def _run():
            llm = ChatDatabricks(model=model, max_tokens=max_tokens_for_model(model))
            agent = create_react_agent(llm, [add, multiply], checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": f"test-async-multi-turn-{model}"}}

            response = await agent.ainvoke(
                {"messages": [("human", "What is 10 + 5?")]}, config=config
            )
            last_message = response["messages"][-1]
            assert isinstance(last_message, AIMessage)
            assert "15" in last_message.content

            response = await agent.ainvoke(
                {"messages": [("human", "Multiply that by 3")]}, config=config
            )
            last_message = response["messages"][-1]
            assert isinstance(last_message, AIMessage)
            assert "45" in last_message.content

        await async_retry(_run)

    async def test_streaming(self, model):
        """Streaming via astream with updates + messages stream modes."""

        async def _run():
            llm = ChatDatabricks(model=model, max_tokens=max_tokens_for_model(model))
            agent = create_react_agent(llm, [add, multiply])

            nodes_seen = set()
            got_message_chunks = False
            event_count = 0

            async for event in agent.astream(
                {
                    "messages": [
                        (
                            "human",
                            "Use the add tool to compute 10 + 5, then use the multiply tool "
                            "to multiply the result by 3. You MUST use the tools.",
                        )
                    ]
                },
                stream_mode=["updates", "messages"],
            ):
                event_count += 1
                mode, data = event
                if mode == "updates":
                    nodes_seen.update(data.keys())
                elif mode == "messages":
                    chunk, _metadata = data
                    if isinstance(chunk, AIMessageChunk):
                        got_message_chunks = True

            assert event_count > 0, "No stream events received"
            assert "agent" in nodes_seen, f"Expected 'agent' node, got: {nodes_seen}"
            assert "tools" in nodes_seen, f"Expected 'tools' node, got: {nodes_seen}"
            assert got_message_chunks, "Expected AIMessageChunk tokens in message stream"

        await async_retry(_run)


# =============================================================================
# Responses API — LangGraph (GPT models including codex)
# =============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("model", _RESPONSES_MODELS)
class TestLangGraphResponsesAPI:
    """LangGraph agent tests using ChatDatabricks(use_responses_api=True).

    Tests GPT models (including codex which only supports Responses API).
    """

    def test_single_turn(self, model):
        """Single-turn: agent calls tools and produces a final answer via Responses API."""
        llm = ChatDatabricks(model=model, use_responses_api=True)
        agent = create_react_agent(llm, [_echo_message, _get_weather])

        def _run():
            response = agent.invoke(
                {"messages": [("human", "Echo the message 'responses API test'")]}
            )
            tool_msgs = [m for m in response["messages"] if isinstance(m, ToolMessage)]
            assert len(tool_msgs) >= 1, "Agent should have called at least one tool"
            last = response["messages"][-1]
            assert isinstance(last, AIMessage)

        retry(_run)

    def test_multi_turn(self, model):
        """Multi-turn: agent maintains context across turns via Responses API."""
        llm = ChatDatabricks(model=model, use_responses_api=True)
        checkpointer = MemorySaver()
        agent = create_react_agent(llm, [_echo_message, _get_weather], checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "responses-api-test"}}

        def _run():
            r1 = agent.invoke({"messages": [("human", "Echo 'first turn'")]}, config=config)
            tool_msgs_1 = [m for m in r1["messages"] if isinstance(m, ToolMessage)]
            assert len(tool_msgs_1) >= 1

            r2 = agent.invoke({"messages": [("human", "Echo 'second turn'")]}, config=config)
            assert len(r2["messages"]) > len(r1["messages"]), "History should grow across turns"

        retry(_run)
