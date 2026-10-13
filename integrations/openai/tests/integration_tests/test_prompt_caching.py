"""Opt-in Claude cache reuse through Chat Completions and OpenAI Agents.

Set RUN_CLAUDE_PROMPT_CACHING_TESTS=1 and configure Databricks credentials. These tests make
billable calls through both the templates' AI Gateway route and Foundation Model APIs.
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLAUDE_PROMPT_CACHING_TESTS") != "1",
    reason="Set RUN_CLAUDE_PROMPT_CACHING_TESTS=1 to verify live Claude cache reuse.",
)


@pytest.fixture(params=["gateway", "fmapi"])
def route(request):
    gateway = request.param == "gateway"
    model = os.environ.get(
        "CLAUDE_CACHE_GATEWAY_MODEL" if gateway else "CLAUDE_CACHE_FMAPI_MODEL",
        "system.ai.claude-sonnet-4-5" if gateway else "databricks-claude-sonnet-4-6",
    )
    return gateway, model


@pytest.fixture
def context():
    return (
        str(uuid.uuid4())
        + "\n"
        + (
            "You are a data engineering assistant. Answer briefly and use the requested tool. "
            * 450
        )
    )


def _cache_read_tokens(usage):
    return getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0


def _assert_usage(usage):
    assert usage.prompt_tokens_details.cached_tokens > 0, "Repeated prefix did not read the cache"
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.prompt_tokens_details.cached_tokens == usage.cache_read_input_tokens


def test_sync_automatic_prompt_caching(route, context):
    from databricks.sdk import WorkspaceClient

    from databricks_openai import DatabricksOpenAI

    gateway, model = route
    with DatabricksOpenAI(
        workspace_client=WorkspaceClient(), use_ai_gateway=gateway, prompt_caching="auto"
    ) as client:
        responses = []
        # Gateway cache writes do not guarantee a read on the immediately following request.
        # Bound warmup while still requiring real reuse of the unchanged system prefix.
        for turn in range(4):
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": f"Reply OK: {turn}"},
                ],
                max_tokens=10,
            )
            responses.append(response)
            if turn > 0 and _cache_read_tokens(response.usage) > 0:
                break
    assert getattr(responses[0].usage, "cache_creation_input_tokens", 0) > 0, (
        "Fresh prefix did not write the cache"
    )
    _assert_usage(responses[-1].usage)


@pytest.mark.asyncio
async def test_async_stream_automatic_prompt_caching(route, context):
    from databricks.sdk import WorkspaceClient

    from databricks_openai import AsyncDatabricksOpenAI

    gateway, model = route
    usages = []
    async with AsyncDatabricksOpenAI(
        workspace_client=WorkspaceClient(), use_ai_gateway=gateway, prompt_caching="auto"
    ) as client:
        # Use the same bounded warmup as the synchronous client, with distinct user turns.
        for turn in range(4):
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": f"Reply OK: {turn}"},
                ],
                max_tokens=10,
                stream=True,
                stream_options={"include_usage": True},
            )
            async with stream:
                usage = None
                async for chunk in stream:
                    if chunk.usage is not None:
                        usage = chunk.usage
            assert usage is not None, "Streaming response omitted usage"
            usages.append(usage)
            if turn > 0 and _cache_read_tokens(usage) > 0:
                break
    assert usages[0].cache_creation_input_tokens > 0
    _assert_usage(usages[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_agents_automatic_prompt_caching_with_tool_and_session(route, context, stream):
    from agents import (
        Agent,
        ModelSettings,
        OpenAIChatCompletionsModel,
        RunConfig,
        Runner,
        SQLiteSession,
        function_tool,
    )
    from databricks.sdk import WorkspaceClient

    from databricks_openai import AsyncDatabricksOpenAI

    calls = []

    @function_tool
    def cache_echo(text: str) -> str:
        """Echo the supplied text."""
        calls.append(text)
        return text

    gateway, model = route
    session = SQLiteSession(str(uuid.uuid4()))
    async with AsyncDatabricksOpenAI(
        workspace_client=WorkspaceClient(), use_ai_gateway=gateway, prompt_caching="auto"
    ) as client:
        agent = Agent(
            name="cache-test",
            instructions=context,
            model=OpenAIChatCompletionsModel(model=model, openai_client=client),
            model_settings=ModelSettings(max_tokens=80),
            tools=[cache_echo],
        )
        for turn in range(2):
            text = f"cache verification {turn}"
            prompt = f"Call cache_echo with '{text}' and report its result."
            if stream:
                result = Runner.run_streamed(
                    agent, prompt, session=session, run_config=RunConfig(tracing_disabled=True)
                )
                async for _event in result.stream_events():
                    pass
            else:
                result = await Runner.run(
                    agent, prompt, session=session, run_config=RunConfig(tracing_disabled=True)
                )
            assert text in calls, "Agent did not execute the requested tool"
            assert result.context_wrapper.usage.input_tokens_details.cached_tokens > 0
            usage = result.context_wrapper.usage
            assert usage.total_tokens == usage.input_tokens + usage.output_tokens
