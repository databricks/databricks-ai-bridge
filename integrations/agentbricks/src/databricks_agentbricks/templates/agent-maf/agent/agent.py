"""A framework-native Microsoft Agent Framework agent with managed state."""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from agent_framework import Agent, AgentSession, Message
from agent_framework.openai import OpenAIChatClient
from databricks_openai import AsyncDatabricksOpenAI

from agent.tools import all_tools
from databricks_agentkit import workspace_client, workspace_headers
from databricks_agentkit.maf import (
    configure_tracing,
    genie_tools,
    mcp_tools,
    memory_tools,
    read_history,
    session_store,
    start_trace,
)
from databricks_agentkit.runtime.tool_manifest import resolve_model_service

MODEL = resolve_model_service() or "system.ai.claude-sonnet-4-5"


def configure() -> None:
    """Validate workspace authentication and configure tracing after loading .env."""
    global MODEL
    MODEL = resolve_model_service() or MODEL
    workspace_client()
    configure_tracing()


def create_agent(
    actor: str, *, model: str | None = None, mcp=(), instructions: str | None = None
) -> Agent:
    tools = [*all_tools(), *memory_tools(actor), *genie_tools(), *mcp]
    if any(
        getattr(tool, "approval_mode", None) not in {None, "never_require"}
        for tool in tools
    ):
        raise ValueError(
            "The Microsoft Agent Framework template does not support approval-requiring tools."
        )
    return Agent(
        name="Agent",
        instructions="You are a helpful assistant."
        + (f"\n{instructions}" if instructions else ""),
        client=OpenAIChatClient(
            model=model or resolve_model_service() or MODEL,
            async_client=AsyncDatabricksOpenAI(
                workspace_client=workspace_client(),
                default_headers=workspace_headers() or None,
                use_ai_gateway=True,
            ),
        ),
        context_providers=[session_store(actor)],
        tools=tools,
    )


@asynccontextmanager
async def run_agent(
    messages: list[Message],
    *,
    session_id: str,
    actor: str,
    model: str | None = None,
    instructions: str | None = None,
) -> AsyncIterator[Any]:
    """Yield the native response stream; callers can also use this outside the HTTP runtime."""
    async with AsyncExitStack() as stack:
        connections = [
            await stack.enter_async_context(connection) for connection in mcp_tools()
        ]
        agent = create_agent(
            actor, model=model, mcp=connections, instructions=instructions
        )
        with start_trace(
            "invoke", inputs=[m.to_dict() for m in messages], session_id=session_id
        ) as span:
            async with agent.run(
                messages, session=AgentSession(session_id=session_id), stream=True
            ) as stream:
                yield stream
                result = await stream.get_final_response()
                if span is not None:
                    span.set_outputs({"output": result.text})
