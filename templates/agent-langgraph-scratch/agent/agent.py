import os
from collections.abc import AsyncGenerator

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.types import Command

from agent.mason import mcp_runtime, tracing
from agent.mason.memory import memory_tools
from agent.mason.session_store import checkpointer, thread_config
from agent.mason.wire.inbound import get_resume, get_session_id
from agent.mason.wire.outbound import process_agent_astream_events

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools

MODEL = "databricks-gpt-5-2"

# Tools that require human approval before they run. Map a tool name to True to allow every decision
# (approve / edit / reject / respond), or to a config dict to restrict them (see HumanInTheLoopMiddleware).
# When a listed tool is about to run, the agent pauses and emits an `interrupt` event; the client
# resumes by POSTing `resume` with the same session_id. Empty this dict to disable approval gating.
REQUIRE_APPROVAL = {"send_message": True}


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    tracing.configure()


def _check_databricks_auth() -> None:
    """Fail fast at startup with a clear message if Databricks auth isn't configured.

    Without this, a missing/invalid profile only surfaces on the first model call — as a generic SDK
    error buried in a request traceback. Resolving a WorkspaceClient here validates the same config
    the model client uses, so the failure is immediate and actionable.
    """
    try:
        WorkspaceClient()
    except Exception as e:
        profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
        target = f"profile {profile!r}" if profile else "the DEFAULT profile / DATABRICKS_HOST+TOKEN"
        raise RuntimeError(
            f"Databricks auth is not configured — the agent can't call the model. Tried {target}.\n"
            "Fix one of:\n"
            "  • set DATABRICKS_CONFIG_PROFILE in .env to a profile from `databricks auth profiles`, or\n"
            "  • run `databricks auth login --profile <name>` to create one, or\n"
            "  • set DATABRICKS_HOST and DATABRICKS_TOKEN in .env.\n"
            f"(underlying error: {e})"
        ) from e


async def create_agent_graph():
    """Build the LangGraph agent: local tools + long-term-memory tools + any MCP tools."""
    tools = [*all_tools(), *memory_tools(), *await mcp_runtime.mcp_tools()]
    middleware = [HumanInTheLoopMiddleware(interrupt_on=REQUIRE_APPROVAL)] if REQUIRE_APPROVAL else []
    return create_agent(
        model=ChatDatabricks(endpoint=MODEL),
        tools=tools,
        middleware=middleware,
        checkpointer=checkpointer(),
    )


async def invoke_handler(request: dict) -> dict:
    """Run one turn to completion. Called by the server for POST /invocations and /responses.

    ``request`` is a dict with an ``input`` list of LangChain message dicts + optional
    ``session_id``; the returned dict carries the run's new messages (LangChain-native shape) and the
    ``session_id`` to pass back next turn. If a gated tool needs approval the run pauses: ``output``
    then ends with an ``interrupt`` event and ``status`` is ``"interrupted"`` — resume by calling
    again with the same ``session_id`` and a ``resume`` payload.
    """
    outputs = [
        event
        async for event in stream_handler(request)
        if event.get("type") in ("message", "interrupt")
    ]
    interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
    return {
        "output": [e["message"] if e["type"] == "message" else e for e in outputs],
        "session_id": get_session_id(request),
        "status": "interrupted" if interrupted else "completed",
    }


async def stream_handler(request: dict) -> AsyncGenerator[dict, None]:
    """Stream the agent's run events as JSON dicts. Called by the server when stream=true."""
    session_id = get_session_id(request)
    tracing.tag_session(session_id)

    agent = await create_agent_graph()
    # A `resume` payload continues a session paused awaiting approval; otherwise start a new turn from
    # `input`. Either way the checkpointer keys off session_id's thread for prior history / paused state.
    # LangChain accepts message dicts natively, so `input` is passed straight through (new turn only).
    resume = get_resume(request)
    agent_input = Command(resume=resume) if resume is not None else {"messages": request.get("input") or []}

    async for event in process_agent_astream_events(
        agent.astream(input=agent_input, config=thread_config(session_id), stream_mode=["updates", "messages"])
    ):
        yield event
