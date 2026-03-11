"""
End-to-end FMAPI tool calling tests for DatabricksOpenAI (sync + async via Agents SDK).

Prerequisites:
- FMAPI endpoints must be available on the test workspace
- echo_message UC function in integration_testing.databricks_ai_bridge_mcp_test
"""

from __future__ import annotations

import json
import os

import pytest
from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.test_utils.fmapi import (
    SKIP_CHAT_COMPLETIONS,
    SKIP_RESPONSES_API,
    async_retry,
    discover_chat_models,
    discover_responses_models,
    retry,
)

from databricks_openai import AsyncDatabricksOpenAI, DatabricksOpenAI

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FMAPI_TOOL_CALLING_TESTS") != "1",
    reason="FMAPI tool calling tests disabled. Set RUN_FMAPI_TOOL_CALLING_TESTS=1 to enable.",
)

_CHAT_MODELS = discover_chat_models(SKIP_CHAT_COMPLETIONS)
_RESPONSES_MODELS = discover_responses_models(SKIP_RESPONSES_API)


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
@pytest.mark.parametrize("model", _CHAT_MODELS)
class TestAgentToolCalling:
    """Async agent tests using the OpenAI Agents SDK with MCP tools.

    Pattern: AsyncDatabricksOpenAI -> McpServer -> Agent -> Runner.run
    """

    async def test_single_turn(self, async_client, workspace_client, model):
        """Single-turn: user sends one message, agent calls a tool and responds."""
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
        """Multi-turn: two-turn conversation with full history replay."""
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
        """Streaming: verify stream events via Runner.run_streamed()."""
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

# echo_message tool definition for direct chat.completions.create() tests
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
@pytest.mark.parametrize("model", _CHAT_MODELS)
class TestSyncClientToolCalling:
    """Sync DatabricksOpenAI tests using direct chat.completions.create()."""

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
        """Multi-turn: tool_call -> tool result -> text response."""

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


# =============================================================================
# Responses API — Agents SDK (GPT models including codex)
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("model", _RESPONSES_MODELS)
class TestAgentToolCallingResponsesAPI:
    """Agents SDK tests using the Responses API path.

    Tests GPT models (including codex which only supports Responses API)
    via set_default_openai_api("responses").
    """

    async def test_single_turn(self, async_client, workspace_client, model):
        """Single-turn via Responses API."""
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.items import ToolCallItem, ToolCallOutputItem

        from databricks_openai.agents import McpServer

        async def _run():
            set_default_openai_client(async_client)
            set_default_openai_api("responses")

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
                result = await Runner.run(agent, "Echo the message 'hello from responses API'")

                assert result.final_output is not None
                assert "hello from responses API" in result.final_output

                item_types = [type(item) for item in result.new_items]
                assert ToolCallItem in item_types, f"Expected ToolCallItem, got: {item_types}"
                assert ToolCallOutputItem in item_types

        await async_retry(_run)

    async def test_multi_turn(self, async_client, workspace_client, model):
        """Multi-turn via Responses API."""
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.items import ToolCallItem

        from databricks_openai.agents import McpServer

        async def _run():
            set_default_openai_client(async_client)
            set_default_openai_api("responses")

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
                first_result = await Runner.run(agent, "Echo 'turn one'")
                assert first_result.final_output is not None

                history = first_result.to_input_list()
                second_result = await Runner.run(
                    agent, input=history + [{"role": "user", "content": "Now echo 'turn two'"}]
                )
                assert second_result.final_output is not None
                second_item_types = [type(item) for item in second_result.new_items]
                assert ToolCallItem in second_item_types

        await async_retry(_run)

    async def test_streaming(self, async_client, workspace_client, model):
        """Streaming via Responses API: verify stream events via Runner.run_streamed()."""
        from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
        from agents.stream_events import RunItemStreamEvent

        from databricks_openai.agents import McpServer

        async def _run():
            set_default_openai_client(async_client)
            set_default_openai_api("responses")

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

                event_count = 0
                run_item_events = []
                async for event in result.stream_events():
                    event_count += 1
                    if isinstance(event, RunItemStreamEvent):
                        run_item_events.append(event)

                assert event_count > 0, "No stream events received"
                event_names = [e.name for e in run_item_events]
                assert "tool_called" in event_names, (
                    f"Expected tool_called event, got: {event_names}"
                )

        await async_retry(_run)
