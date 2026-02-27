"""
End-to-end FMAPI tool calling tests mirroring app-templates CUJs.

These tests replicate the exact user code patterns from app-templates
(agent-openai-agents-sdk) to verify that single-turn and multi-turn
conversations don't break.

Naturally exercises regressions like:
  - PR #269: Agents SDK adds strict:True -> our client strips it -> FMAPI
  - PR #333: Multi-turn agent loop replays assistant messages with empty
    content + tool_calls -> our client fixes content -> FMAPI

Prerequisites:
- FMAPI endpoints must be available on the test workspace
- echo_message UC function in integration_testing.databricks_ai_bridge_mcp_test
"""

from __future__ import annotations

import json
import os

import pytest
from databricks.sdk import WorkspaceClient
from openai.types.chat import ChatCompletionToolParam

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI

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


# MCP test infrastructure
_MCP_CATALOG = "integration_testing"
_MCP_SCHEMA = "databricks_ai_bridge_mcp_test"
_MCP_FUNCTION = "echo_message"


@pytest.fixture(scope="module")
def workspace_client():
    return WorkspaceClient()


@pytest.fixture(scope="module")
def sync_client(workspace_client):
    return DatabricksOpenAI(workspace_client=workspace_client)


@pytest.fixture(scope="module")
def async_client(workspace_client):
    return AsyncDatabricksOpenAI(workspace_client=workspace_client)


# =============================================================================
# Async DatabricksOpenAI
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
class TestAgentToolCalling:
    """End-to-end agent tests mirroring app-templates/agent-openai-agents-sdk.

    Each test follows the exact pattern users deploy:
      AsyncDatabricksOpenAI -> set_default_openai_client -> McpServer -> Agent -> Runner.run
    """

    async def test_single_turn(self, async_client, workspace_client, model):
        """Single-turn conversation: user sends one message, agent calls a tool and responds.

        Mirrors the basic app-template @invoke() handler.
        """
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem

        from databricks_openai.agents import McpServer

        async def _run():
            set_default_openai_client(async_client)
            set_default_openai_api("chat_completions")

            async with McpServer.from_uc_function(
                catalog=_MCP_CATALOG,
                schema=_MCP_SCHEMA,
                function_name=_MCP_FUNCTION,
                workspace_client=workspace_client,
                timeout=60,
            ) as server:
                agent = Agent(
                    name="echo-agent",
                    instructions="Use the echo_message tool to echo messages when asked.",
                    model=model,
                    mcp_servers=[server],
                )
                result = await Runner.run(agent, "Echo the message 'hello from FMAPI test'")

                assert result.final_output is not None
                assert "hello from FMAPI test" in result.final_output

                item_types = [type(item) for item in result.new_items]
                assert ToolCallItem in item_types, f"Expected a tool call, got: {item_types}"
                assert ToolCallOutputItem in item_types, f"Expected tool output, got: {item_types}"
                assert MessageOutputItem in item_types, f"Expected a message, got: {item_types}"

                input_list = result.to_input_list()
                assert len(input_list) > 1, "Expected multi-item conversation history"

        await async_retry(_run)

    async def test_multi_turn(self, async_client, workspace_client, model):
        """Multi-turn conversation: simulates a chat UI sending conversation history.

        First turn: user asks to echo a message, agent calls the tool.
        Second turn: user sends a followup with the full conversation history
        (including the assistant's prior tool-calling turn), agent calls the tool again.

        This is how the app-templates chat UI works: each request includes the
        full conversation history. The second FMAPI call replays the assistant
        message from the first turn, which may have empty content + tool_calls.
        """
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.items import ToolCallItem

        from databricks_openai.agents import McpServer

        async def _run():
            set_default_openai_client(async_client)
            set_default_openai_api("chat_completions")

            async with McpServer.from_uc_function(
                catalog=_MCP_CATALOG,
                schema=_MCP_SCHEMA,
                function_name=_MCP_FUNCTION,
                workspace_client=workspace_client,
                timeout=60,
            ) as server:
                agent = Agent(
                    name="echo-agent",
                    instructions="Use the echo_message tool to echo messages when asked.",
                    model=model,
                    mcp_servers=[server],
                )

                first_result = await Runner.run(agent, "Echo the message 'hello'")
                assert first_result.final_output is not None
                assert "hello" in first_result.final_output

                first_item_types = [type(item) for item in first_result.new_items]
                assert ToolCallItem in first_item_types

                history = first_result.to_input_list()
                history.append({"role": "user", "content": "Now echo the message 'world'"})

                second_result = await Runner.run(agent, history)
                assert second_result.final_output is not None
                assert "world" in second_result.final_output

                second_item_types = [type(item) for item in second_result.new_items]
                assert ToolCallItem in second_item_types

                second_history = second_result.to_input_list()
                assert len(second_history) > len(history), (
                    f"Expected history to grow: {len(history)} -> {len(second_history)}"
                )

        await async_retry(_run)

    async def test_streaming(self, async_client, workspace_client, model):
        """Streaming conversation: mirrors the app-template @stream() handler.

        Uses Runner.run_streamed() which is the streaming path in app-templates.
        Verifies that stream events arrive in the expected order and contain
        the expected item types.
        """
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.stream_events import RunItemStreamEvent

        from databricks_openai.agents import McpServer

        async def _run():
            set_default_openai_client(async_client)
            set_default_openai_api("chat_completions")

            async with McpServer.from_uc_function(
                catalog=_MCP_CATALOG,
                schema=_MCP_SCHEMA,
                function_name=_MCP_FUNCTION,
                workspace_client=workspace_client,
                timeout=60,
            ) as server:
                agent = Agent(
                    name="echo-agent",
                    instructions="Use the echo_message tool to echo messages when asked.",
                    model=model,
                    mcp_servers=[server],
                )
                result = Runner.run_streamed(agent, input="Echo the message 'streaming test'")

                run_item_events = []
                event_count = 0
                async for event in result.stream_events():
                    event_count += 1
                    if isinstance(event, RunItemStreamEvent):
                        run_item_events.append(event)

                assert event_count > 0, "No stream events received"

                event_names = [e.name for e in run_item_events]
                assert "tool_called" in event_names, (
                    f"Expected tool_called event, got: {event_names}"
                )
                assert "tool_output" in event_names, (
                    f"Expected tool_output event, got: {event_names}"
                )
                assert "message_output_created" in event_names, (
                    f"Expected message_output_created event, got: {event_names}"
                )

                assert result.final_output is not None
                assert "streaming test" in result.final_output

        await async_retry(_run)


# =============================================================================
# Sync DatabricksOpenAI — direct chat.completions.create()
# =============================================================================

# echo_message tool definition (mirrors the UC function signature)
_ECHO_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "echo_message",
        "description": "Echo back the provided message",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo back",
                }
            },
            "required": ["message"],
        },
    },
}


@pytest.mark.integration
@pytest.mark.parametrize("model", _FOUNDATION_MODELS)
class TestSyncClientToolCalling:
    """Sync DatabricksOpenAI tests using direct client.chat.completions.create().

    The Agents SDK requires an async client, so the sync DatabricksOpenAI CUJ
    is direct chat.completions.create() calls with tool definitions.

    This exercises DatabricksCompletions.create() (the sync counterpart to
    AsyncDatabricksCompletions.create() used by the Agents SDK).
    """

    def test_single_turn(self, sync_client, model):
        """Single-turn: model receives tool, produces a tool call."""

        def _run():
            response = sync_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Echo the message 'sync hello'"}],
                tools=[_ECHO_MESSAGE_TOOL],
                max_tokens=200,
            )
            message = response.choices[0].message
            assert message.tool_calls is not None
            assert len(message.tool_calls) >= 1

            tool_call = message.tool_calls[0]
            assert tool_call.id is not None
            assert tool_call.type == "function"
            assert tool_call.function.name == "echo_message"
            args = json.loads(tool_call.function.arguments)
            assert "message" in args

        retry(_run)

    def test_multi_turn(self, sync_client, model):
        """Multi-turn: tool_call -> tool result -> text response.

        The second FMAPI call replays the assistant message (potentially with
        empty content + tool_calls), exercising the PR #333 fix.
        """

        def _run():
            # Turn 1: get tool call
            response = sync_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Echo the message 'sync world'"}],
                tools=[_ECHO_MESSAGE_TOOL],
                max_tokens=200,
            )
            assistant_msg = response.choices[0].message
            assert assistant_msg.tool_calls is not None
            tool_call = assistant_msg.tool_calls[0]

            # Turn 2: send tool result back, get text response
            # Manually construct the assistant message to avoid extra fields
            # (e.g. "annotations") that model_dump() includes but FMAPI rejects
            assistant_dict = {
                "role": "assistant",
                "content": assistant_msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ],
            }
            response = sync_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Echo the message 'sync world'"},
                    assistant_dict,
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "sync world",
                    },
                ],
                tools=[_ECHO_MESSAGE_TOOL],
                max_tokens=200,
            )
            followup = response.choices[0].message
            assert followup.content is not None
            assert "sync world" in followup.content

        retry(_run)

    def test_streaming(self, sync_client, model):
        """Streaming: tool call arrives as chunked deltas."""

        def _run():
            stream = sync_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Echo the message 'sync stream'"}],
                tools=[_ECHO_MESSAGE_TOOL],
                max_tokens=200,
                stream=True,
            )
            chunks = list(stream)
            assert len(chunks) > 0

            # Reassemble tool call from streamed deltas
            tool_call_name = ""
            tool_call_args = ""
            tool_call_id = None
            for chunk in chunks:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.tool_calls:
                    tc = delta.tool_calls[0]
                    if tc.id:
                        tool_call_id = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_name += tc.function.name
                        if tc.function.arguments:
                            tool_call_args += tc.function.arguments

            assert tool_call_id is not None, "No tool call ID found in stream"
            assert tool_call_name == "echo_message"
            args = json.loads(tool_call_args)
            assert "message" in args

        retry(_run)
