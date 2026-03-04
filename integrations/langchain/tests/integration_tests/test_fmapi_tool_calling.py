"""
End-to-end FMAPI tool calling tests for ChatDatabricks via LangGraph.

Prerequisites:
- FMAPI endpoints must be available on the test workspace
"""

from __future__ import annotations

import os

import pytest
from databricks_ai_bridge.test_utils.fmapi import (
    LANGCHAIN_SKIP_MODELS,
    async_retry,
    discover_foundation_models,
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

_FOUNDATION_MODELS = discover_foundation_models(LANGCHAIN_SKIP_MODELS)


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
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
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
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
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
