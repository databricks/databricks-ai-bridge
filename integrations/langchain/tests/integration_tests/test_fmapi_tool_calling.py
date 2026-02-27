"""
End-to-end FMAPI tool calling tests for LangGraph agents mirroring app-templates CUJs.

These tests replicate the exact user code patterns from app-templates
(agent-langgraph, agent-langgraph-short-term-memory) to verify that
single-turn, multi-turn, and streaming conversations don't break.

Prerequisites:
- FMAPI endpoints must be available on the test workspace
"""

from __future__ import annotations

import os

import pytest
from databricks.sdk import WorkspaceClient
from databricks_openai import DatabricksOpenAI
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from openai.types.chat import ChatCompletionToolParam

from databricks_langchain import ChatDatabricks

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FMAPI_TOOL_CALLING_TESTS") != "1",
    reason="FMAPI tool calling tests disabled. Set RUN_FMAPI_TOOL_CALLING_TESTS=1 to enable.",
)

# Models that pass the tool calling probe but have known issues in agent/test flows.
# These are skipped entirely to keep CI green. When a new model is added to FMAPI,
# it will be discovered and tested automatically — add it here only if it fails.
_SKIP_MODELS = {
    "databricks-gpt-5-nano",  # too small for reliable tool calling
    "databricks-gpt-oss-20b",  # hallucinates tool names in agent loop
    "databricks-gpt-oss-120b",  # hallucinates tool names in agent loop
    "databricks-llama-4-maverick",  # hallucinates tool names in agent loop
    "databricks-gemini-3-flash",  # requires thought_signature on function calls
    "databricks-gemini-3-pro",  # requires thought_signature on function calls
    "databricks-gemini-3-1-pro",  # requires thought_signature on function calls
}

# Max retries for flaky models (e.g. transient FMAPI errors, model non-determinism)
_MAX_RETRIES = 3

# Minimal tool definition used to probe whether a model supports tool calling
_PROBE_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "probe",
        "description": "probe",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        },
    },
}


def _supports_tool_calling(client: DatabricksOpenAI, model: str) -> bool:
    """Send a minimal tool call request to check if the model supports tools."""
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "call probe with x=test"}],
            tools=[_PROBE_TOOL],
            max_tokens=10,
        )
        return True
    except Exception:
        return False


def _discover_foundation_models() -> list:
    """Discover all FMAPI chat models that support tool calling.

    1. List all serving endpoints with databricks- prefix and llm/v1/chat task
    2. Probe each model with a minimal tool call to check if tools are supported
    3. Models in _XFAIL_MODELS are included but marked as expected failures
    """
    import logging

    log = logging.getLogger(__name__)

    try:
        w = WorkspaceClient()
        endpoints = list(w.serving_endpoints.list())
    except Exception as exc:
        log.warning("Could not discover FMAPI models, using fallback list: %s", exc)
        return _FALLBACK_MODELS

    # Filter to FMAPI chat endpoints
    chat_endpoints = [
        e
        for e in endpoints
        if e.name and e.name.startswith("databricks-") and e.task == "llm/v1/chat"
    ]

    # Probe each model to check if it accepts tool definitions
    client = DatabricksOpenAI(workspace_client=w)

    models = []
    for e in sorted(chat_endpoints, key=lambda e: e.name or ""):
        name = e.name or ""
        if not _supports_tool_calling(client, name):
            log.info("Skipping %s: does not support tool calling", name)
            continue
        if name in _SKIP_MODELS:
            log.info("Skipping %s: in skip list", name)
            continue
        models.append(name)

    log.info("Discovered %d FMAPI models with tool calling support", len(models))
    return models


# Fallback list if dynamic discovery fails (e.g. auth not configured at collection time)
_FALLBACK_MODELS = [
    "databricks-claude-sonnet-4-6",
    "databricks-claude-opus-4-6",
    "databricks-meta-llama-3-3-70b-instruct",
    "databricks-gpt-5-2",
    "databricks-gpt-5-1",
    "databricks-qwen3-next-80b-a3b-instruct",
]

_FOUNDATION_MODELS = _discover_foundation_models()


def retry(fn, retries=_MAX_RETRIES):
    """Retry a test function up to `retries` times. Only fails if all attempts fail."""
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                import logging

                logging.getLogger(__name__).warning(
                    "Attempt %d/%d failed: %s — retrying", attempt + 1, retries, exc
                )
    raise last_exc  # type: ignore[misc]


async def async_retry(fn, retries=_MAX_RETRIES):
    """Retry an async test function up to `retries` times."""
    last_exc = None
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                import logging

                logging.getLogger(__name__).warning(
                    "Attempt %d/%d failed: %s — retrying", attempt + 1, retries, exc
                )
    raise last_exc  # type: ignore[misc]


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
    """Sync LangGraph agent tests mirroring app-templates/agent-langgraph.

    Each test follows the pattern:
      ChatDatabricks -> create_react_agent -> agent.invoke / agent.stream
    """

    def test_single_turn(self, model):
        """Single-turn: agent calls tools and produces a final answer.

        Mirrors the basic app-template @invoke() handler.
        """

        def _run():
            llm = ChatDatabricks(model=model, max_tokens=200)
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
        """Multi-turn: agent maintains conversation context across turns.

        Mirrors app-templates/agent-langgraph-short-term-memory with MemorySaver
        checkpointer and thread_id for session continuity.
        """

        def _run():
            llm = ChatDatabricks(model=model, max_tokens=200)
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
        """Streaming: agent streams node updates and tool execution events.

        Mirrors the app-template @stream() handler pattern using agent.stream().
        """

        def _run():
            llm = ChatDatabricks(model=model, max_tokens=200)
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
    """Async LangGraph agent tests mirroring the app-templates @stream() handler.

    Each test follows the exact async pattern deployed in production:
      ChatDatabricks -> create_react_agent -> agent.ainvoke / agent.astream
    """

    async def test_single_turn(self, model):
        """Single-turn via ainvoke."""

        async def _run():
            llm = ChatDatabricks(model=model, max_tokens=200)
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
            llm = ChatDatabricks(model=model, max_tokens=200)
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
        """Streaming via astream — mirrors the exact app-templates production path.

        Uses agent.astream(stream_mode=["updates", "messages"]) which is the
        pattern in agent-langgraph and agent-langgraph-short-term-memory.
        """

        async def _run():
            llm = ChatDatabricks(model=model, max_tokens=200)
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
