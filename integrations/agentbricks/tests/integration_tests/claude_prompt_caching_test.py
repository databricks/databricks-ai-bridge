"""Verify the template provider adapter against live Claude endpoints.

RUN_CLAUDE_PROMPT_CACHING_TESTS=1 enables billable calls. Both AgentKit framework extras
are required. The test requires actual reads after repeated system prefixes and tool turns.
"""

import os
import uuid

import pytest
from databricks.sdk import WorkspaceClient

from databricks_agentkit.claude import ClaudeCachingOpenAI, prepare_claude_request

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLAUDE_PROMPT_CACHING_TESTS") != "1",
    reason="Set RUN_CLAUDE_PROMPT_CACHING_TESTS=1 to verify live template caching",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["system.ai.claude-sonnet-4-5", "system.ai.claude-sonnet-4-6"])
@pytest.mark.parametrize("framework", ["openai", "langgraph"])
@pytest.mark.parametrize("stream", [False, True])
async def test_template_adapter_reuses_prefix_after_tool_turns(model, framework, stream):
    gateway = os.environ.get("CLAUDE_CACHE_ADAPTER_ROUTE", "gateway") == "gateway"
    if not gateway:
        model = model.replace("system.ai.", "databricks-")
    context = (
        str(uuid.uuid4())
        + "\n"
        + (
            "You are a data engineering assistant. Answer briefly and use the requested tool. "
            * 200
        )
    )
    calls = []
    reads = []
    if framework == "openai":
        from agents import (
            Agent,
            ModelSettings,
            OpenAIChatCompletionsModel,
            RunConfig,
            Runner,
            SQLiteSession,
            function_tool,
        )

        @function_tool
        def cache_echo(text: str) -> str:
            """Echo text."""
            calls.append(text)
            return text

        async with ClaudeCachingOpenAI(
            workspace_client=WorkspaceClient(), use_ai_gateway=gateway, max_retries=8
        ) as client:
            agent = Agent(
                name="cache-test",
                instructions=context,
                model=OpenAIChatCompletionsModel(model=model, openai_client=client),
                model_settings=ModelSettings(max_tokens=80),
                tools=[cache_echo],
            )
            session = SQLiteSession(str(uuid.uuid4()))
            for turn in range(4):
                text = f"cache verification {turn}"
                prompt = f"Call cache_echo with '{text}' and report its result."
                config = RunConfig(tracing_disabled=True)
                if stream:
                    result = Runner.run_streamed(agent, prompt, session=session, run_config=config)
                    async for _ in result.stream_events():
                        pass
                else:
                    result = await Runner.run(agent, prompt, session=session, run_config=config)
                usage = result.context_wrapper.usage
                assert usage.total_tokens == usage.input_tokens + usage.output_tokens
                reads.append(usage.input_tokens_details.cached_tokens or 0)
                assert text in calls
                if turn > 0 and any(reads):
                    break
    else:
        from databricks_langchain import ChatDatabricks
        from langchain.agents import create_agent
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableConfig
        from langchain_core.tools import tool
        from langgraph.checkpoint.memory import MemorySaver

        class TemplateChatDatabricks(ChatDatabricks):
            def _prepare_inputs(self, *args, **kwargs):
                return prepare_claude_request(super()._prepare_inputs(*args, **kwargs))

        @tool
        def cache_echo(text: str) -> str:
            """Echo text."""
            calls.append(text)
            return text

        graph = create_agent(
            model=TemplateChatDatabricks(
                endpoint=model,
                workspace_client=WorkspaceClient(),
                use_ai_gateway=gateway,
                max_tokens=80,
                max_retries=8,
            ),
            system_prompt=context,
            tools=[cache_echo],
            checkpointer=MemorySaver(),
        )
        config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
        for turn in range(4):
            text = f"cache verification {turn}"
            inputs = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Call cache_echo with '{text}' and report its result.",
                    }
                ]
            }
            if stream:
                usages = []
                async for message, _ in graph.astream(
                    inputs, config=config, stream_mode="messages"
                ):
                    if usage := getattr(message, "usage_metadata", None):
                        usages.append(usage)
            else:
                result = await graph.ainvoke(inputs, config=config)
                usages = [
                    m.usage_metadata
                    for m in result["messages"]
                    if isinstance(m, AIMessage) and m.usage_metadata
                ]
            assert usages, "Model omitted usage"
            for usage in usages:
                assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
                reads.append(usage.get("input_token_details", {}).get("cache_read", 0))
            assert text in calls
            if turn > 0 and any(reads):
                break
    assert len(calls) >= 2, "Two tool turns are required"
    assert any(reads), "Repeated template prefix did not read the cache"
