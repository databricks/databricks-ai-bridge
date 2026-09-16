import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from agents import Agent, Runner, RunResultStreaming, RunState
from agents.mcp import MCPServerManager
from databricks_openai import AsyncDatabricksOpenAI

from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools
from databricks_mason import workspace_client, workspace_headers
from databricks_mason.openai import (
    configure_tracing,
    mcp_servers,
    memory_tools,
    session_store,
    start_trace,
)

logger = logging.getLogger(__name__)

MODEL = "databricks-gpt-5-2"

# Tools that require human approval before they run. Add a tool's name here and the agent pauses when
# the model calls it, emitting an `interrupt` event; the client resumes by sending `resume` with the
# same session id. The tools declare `needs_approval=True` themselves (see agent/tools/); this set is
# how the runtime knows which pending calls to surface. Empty it to disable approval gating.
REQUIRE_APPROVAL = {"send_message"}

# OpenAI Sessions persist transcript history, not a paused RunState. Keep pending approvals local.
_pending_runs: dict[str, RunState] = {}


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    from agents import set_default_openai_api, set_default_openai_client

    set_default_openai_client(
        AsyncDatabricksOpenAI(
            workspace_client=workspace_client(),
            default_headers=workspace_headers() or None,
        )
    )
    set_default_openai_api("chat_completions")
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


def create_agent(actor: str, mcp=None, model: str | None = None) -> Agent:
    """Build the OpenAI Agents SDK agent: tools, memory, MCP servers, and model."""
    return Agent(
        name="Agent",
        instructions="You are a helpful assistant.",
        model=model or MODEL,
        tools=[*all_tools(), *memory_tools(actor)],
        mcp_servers=mcp or [],
    )


def resume_agent(session_id: str, resume: dict[str, Any]) -> RunState:
    """Apply human decisions to a paused run and return the native RunState."""
    state = _pending_runs.pop(session_id, None)
    if state is None:
        raise RuntimeError(
            "No paused run for this session. HITL pauses are in-process only, so a restart or a "
            "different replica loses them; retry the turn."
        )
    decisions = resume.get("decisions") or []
    for decision, item in zip(decisions, state.get_interruptions(), strict=False):
        if decision.get("type") == "approve":
            state.approve(item)
        else:
            state.reject(item, rejection_message=decision.get("message"))
    return state


@asynccontextmanager
async def run_agent(
    agent_input: list[Any] | RunState,
    *,
    session_id: str,
    actor: str | None = None,
    model: str | None = None,
) -> AsyncIterator[RunResultStreaming]:
    """Run the agent and expose its native streaming result.

    This is the framework-native entrypoint. It has no dependency on Mason request or context types,
    so it can be called from another server, a notebook, or a test harness.
    """
    actor = actor or session_id
    servers = await mcp_servers(build_mcp_servers())
    async with MCPServerManager(servers) as manager:
        active_servers = []
        for server in manager.active_servers:
            tool_filter = server.tool_filter
            try:
                server.tool_filter = None
                server.cache_tools_list = True
                async with asyncio.timeout(manager.connect_timeout_seconds):
                    await server.list_tools()
            except Exception:
                logger.warning(
                    "Failed to list tools from MCP server %r; continuing without it.",
                    server.name,
                    exc_info=True,
                )
            else:
                active_servers.append(server)
            finally:
                server.tool_filter = tool_filter

        agent = create_agent(actor, active_servers, model=model)
        with start_trace(name="invoke", inputs=agent_input, session_id=session_id) as span:
            if isinstance(agent_input, RunState):
                result = Runner.run_streamed(agent, agent_input)
            else:
                result = Runner.run_streamed(
                    agent,
                    agent_input,
                    session=session_store(session_id, actor),
                )

            yield result

            if result.interruptions:
                _pending_runs[session_id] = result.to_state()
            if span is not None:
                span.set_outputs({"output": result.final_output})
