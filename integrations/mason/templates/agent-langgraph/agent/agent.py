import logging
import os
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessageChunk
from langgraph.types import Command

from databricks_mason import (
    configure_tracing,
    tag_session,
    workspace_client,
    workspace_headers,
)
from databricks_mason.langgraph import checkpointer, mcp_tools, memory_tools, thread_config
from databricks_mason.runtime import DurableAgentApp, DurableAgentContext

from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools

logger = logging.getLogger(__name__)

MODEL = "databricks-gpt-5-2"
app = DurableAgentApp()
_RUN_METADATA_KEY = "databricks_mason.run_id"

# Tools that require human approval before they run. Map a tool name to True to allow every decision
# (approve / edit / reject / respond), or to a config dict to restrict them (see HumanInTheLoopMiddleware).
# When a listed tool is about to run, the agent pauses and emits an `interrupt` event; the client
# resumes by sending `resume` with the same session id. Empty this dict to disable approval gating.
REQUIRE_APPROVAL = {"send_message": True}


class _RoutedChatDatabricks(ChatDatabricks):
    """Forward account-host workspace routing to the underlying OpenAI clients."""

    def _get_client_kwargs(self) -> dict[str, Any]:
        kwargs = super()._get_client_kwargs()
        if headers := workspace_headers():
            kwargs["default_headers"] = headers
        return kwargs


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    configure_tracing()


def _check_databricks_auth() -> None:
    """Fail fast at startup with a clear message if Databricks auth isn't configured.

    Without this, a missing/invalid profile only surfaces on the first model call — as a generic SDK
    error buried in a request traceback. Resolving a WorkspaceClient here validates the same config
    the model client uses, so the failure is immediate and actionable.
    """
    try:
        workspace_client()
    except Exception as e:
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
        target = (
            f"profile {profile!r}" if profile else "the DEFAULT profile / DATABRICKS_HOST+TOKEN"
        )
        raise RuntimeError(
            f"Databricks auth is not configured — the agent can't call the model. Tried {target}.\n"
            "Fix one of:\n"
            "  • set DATABRICKS_CONFIG_PROFILE in .env to a profile from `databricks auth profiles`, or\n"
            "  • run `databricks auth login --profile <name>` to create one, or\n"
            "  • set DATABRICKS_HOST and DATABRICKS_TOKEN in .env.\n"
            f"(underlying error: {e})"
        ) from e


async def create_agent_graph(actor: str):
    """Build the LangGraph agent: local tools + long-term-memory tools + any MCP tools.

    ``actor`` is the identity whose long-term memory the agent reads/writes; it's captured in the
    memory tools' closures (never exposed to the model). See ``_actor``.
    """
    # Join the manifest's MCP servers (from agent.toml) with your own hand-declared ones (mcps.py),
    # then fetch their tools. Edit build_mcp_servers in agent/mcps.py to add servers.
    mcp = await mcp_tools(build_mcp_servers())
    tools = [*all_tools(), *memory_tools(actor), *mcp]
    middleware = (
        [HumanInTheLoopMiddleware(interrupt_on=REQUIRE_APPROVAL)] if REQUIRE_APPROVAL else []
    )
    return create_agent(
        model=_RoutedChatDatabricks(endpoint=MODEL, workspace_client=workspace_client()),
        tools=tools,
        middleware=middleware,
        checkpointer=checkpointer(),
    )


@app.invoke
async def invoke(request: dict, context: DurableAgentContext) -> dict:
    """Translate an invocation payload into LangGraph input and run it to completion."""
    return await _run_agent(_invocation_input(request), context)


@app.recover
async def recover(request: dict, context: DurableAgentContext) -> dict:
    """Resume the same LangGraph session after the runtime replaces a failed worker."""
    saver = checkpointer()
    checkpoint = await saver.aget_tuple(thread_config(context.session_id, context.actor))
    current_run_checkpointed = bool(
        checkpoint and checkpoint.metadata.get(_RUN_METADATA_KEY) == context.run_id
    )
    agent_input = None if current_run_checkpointed else _invocation_input(request)
    return await _run_agent(agent_input, context)


def _invocation_input(request: dict) -> Any:
    """Translate an invocation payload into LangGraph's native input."""
    resume = request.get("resume")
    if resume is not None:
        return Command(resume=resume)
    return {"messages": request.get("input") or []}


async def _run_agent(agent_input: Any, context: DurableAgentContext) -> dict:
    """Run one turn while the server persists events and attempt state."""
    tag_session(context.session_id)
    outputs = [
        event
        async for event in _persisted_agent_events(agent_input, context)
        if event.get("type") in ("message", "interrupt")
    ]
    interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
    return {
        "output": [e["message"] if e["type"] == "message" else e for e in outputs],
        "status": "interrupted" if interrupted else "completed",
    }


async def _persisted_agent_events(
    agent_input: Any, context: DurableAgentContext
) -> AsyncGenerator[dict, None]:
    """Persist framework events before the server delivers or replays them."""
    async for event in _agent_events(agent_input, context):
        await context.emit(event)
        yield event


async def _agent_events(
    agent_input: Any, context: DurableAgentContext
) -> AsyncGenerator[dict, None]:
    """Translate one LangGraph event stream into the server's JSON event contract."""
    agent = await create_agent_graph(context.actor)
    async for event in _serialize_events(
        agent.astream(
            input=agent_input,
            config={
                **thread_config(context.session_id, context.actor),
                "metadata": {_RUN_METADATA_KEY: context.run_id},
            },
            stream_mode=["updates", "messages"],
            durability="sync",
        )
    ):
        yield event


async def _serialize_events(async_stream: AsyncIterator[Any]) -> AsyncGenerator[dict, None]:
    """Turn LangGraph's ``astream`` events into JSON dicts in LangChain's native shape (not reshaped).

    ``stream_mode=["updates", "messages"]`` yields completed node outputs (full LangChain messages,
    incl. tool calls/results) and token-level chunks. Completed messages become
    ``{"type": "message", "message": <dict>}`` and text chunks ``{"type": "delta", "content", "id"}``.
    A human-approval gate surfaces as an ``__interrupt__`` update, relayed as
    ``{"type": "interrupt", "id", "value"}``; the run is then paused on the session's thread until the
    client resumes with the same session id.
    """
    async for event in async_stream:
        mode, payload = event[0], event[1]
        if mode == "updates":
            if interrupts := payload.get("__interrupt__"):
                for it in interrupts:
                    yield {"type": "interrupt", "id": it.id, "value": it.value}
                continue
            for node_data in payload.values():
                messages = node_data.get("messages", []) if isinstance(node_data, dict) else []
                for msg in messages:
                    yield {"type": "message", "message": msg.model_dump()}
        elif mode == "messages":
            try:
                chunk = payload[0]
                if isinstance(chunk, AIMessageChunk) and (content := chunk.content):
                    yield {"type": "delta", "content": content, "id": chunk.id}
            except Exception:
                logger.exception("Error processing agent stream chunk")
