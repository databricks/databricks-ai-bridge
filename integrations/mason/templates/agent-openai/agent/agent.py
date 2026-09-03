import logging
import os
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from typing import Any

from agents import Agent, RunResultStreaming, Runner, RunState
from agents.items import ToolApprovalItem
from databricks_openai import AsyncDatabricksOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from databricks_mason import tag_session, workspace_client
from databricks_mason.openai import configure_tracing, mcp_servers, memory_tools, session_store

from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools

logger = logging.getLogger(__name__)

MODEL = "databricks-gpt-5-2"

# Tools that require human approval before they run. Add a tool's name here and the agent pauses when
# the model calls it, emitting an `interrupt` event; the client resumes by sending `resume` with the
# same session id. The tools declare `needs_approval=True` themselves (see agent/tools/); this set is
# how the runtime knows which pending calls to surface. Empty it to disable approval gating.
REQUIRE_APPROVAL = {"send_message"}

# Paused runs awaiting human approval, keyed by session id. In-process only — a paused run does NOT
# survive a restart or reach another replica, even with AGENT_SESSION_STORE set: unlike a LangGraph
# checkpoint, an Agents SDK Session persists the transcript but not paused RunState. Durable HITL
# would stash RunState.to_json() separately; this template keeps it simple and in-memory.
_pending_runs: dict[str, RunState] = {}


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    # Route the Agents SDK's default OpenAI client at the Databricks model endpoint (account-host
    # routing and auth handled by the SDK), so `Agent(model=MODEL)` resolves to a Databricks model.
    from agents import set_default_openai_api, set_default_openai_client

    set_default_openai_client(AsyncDatabricksOpenAI())
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
        tried = f"profile {profile!r}" if profile else "no profile (Databricks default resolution)"
        raise RuntimeError(
            f"Databricks auth is not configured — the agent can't call the model (tried {tried}).\n"
            "Authenticate once with `mason login --profile <name>` "
            "(see `databricks auth profiles` for names), then make sure the agent process sees it: "
            "set DATABRICKS_CONFIG_PROFILE=<name> in .env for local `mason dev`, or "
            "DATABRICKS_HOST + DATABRICKS_TOKEN for a deployed app.\n"
            f"(underlying error: {e})"
        ) from e


def create_agent(mcp=None) -> Agent:
    """Build the OpenAI Agents SDK agent: local tools + long-term-memory tools + any MCP servers."""
    return Agent(
        name="Agent",
        instructions="You are a helpful assistant.",
        model=MODEL,
        tools=[*all_tools(), *memory_tools()],
        mcp_servers=mcp or [],
    )


def _session_id(request: dict) -> str:
    """Return the session id derived by the runtime from the Apps routing cookie.

    Clients do not send ``session_id`` in the body. The runtime makes the cookie value available to
    the handler after resolving the deployed Apps cookie or the local-development fallback cookie.
    """
    return str(request["session_id"])


async def invoke_handler(request: dict) -> dict:
    """Run one turn to completion. Called by the runtime for POST /invocations.

    ``request`` is a dict with an ``input`` list of Responses message dicts; the returned dict carries
    the run's new items (normalized message shape) and the ``session_id`` to pass back next turn. If a
    gated tool needs approval the run pauses: ``output`` then ends with an ``interrupt`` event and
    ``status`` is ``"interrupted"`` — resume by calling again with the same session id and a ``resume``
    payload.
    """
    request = {**request, "session_id": _session_id(request)}
    outputs = [
        event
        async for event in stream_handler(request)
        if event.get("type") in ("message", "interrupt")
    ]
    interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
    return {
        "output": [e["message"] if e["type"] == "message" else e for e in outputs],
        "session_id": request["session_id"],
        "status": "interrupted" if interrupted else "completed",
    }


async def stream_handler(request: dict) -> AsyncGenerator[dict, None]:
    """Stream the agent's run events as JSON dicts. Called by the runtime when stream=true."""
    session_id = _session_id(request)
    tag_session(session_id)

    # The agent runs inside an AsyncExitStack so any MCP servers stay connected for the whole run —
    # the Agents SDK lists each server's tools lazily inside Runner.run.
    async with AsyncExitStack() as stack:
        # Join the manifest's MCP servers (from agent.toml) with your own hand-declared ones
        # (mcps.py), then connect them for the life of the run.
        servers = await mcp_servers(build_mcp_servers())
        mcp = [await stack.enter_async_context(server) for server in servers]
        agent = create_agent(mcp)

        # A `resume` payload continues a session paused awaiting approval; otherwise start a new turn
        # from `input`. A resumed run re-runs the stashed RunState (with decisions applied); a new
        # turn passes the messages plus the session store so prior history is loaded automatically.
        resume = request.get("resume")
        if resume is not None:
            run_input: Any = _apply_decisions(session_id, resume)
            result = Runner.run_streamed(agent, run_input)
        else:
            result = Runner.run_streamed(
                agent, request.get("input") or [], session=session_store(session_id)
            )

        async for event in _serialize_events(result, session_id):
            yield event


def _apply_decisions(session_id: str, resume: dict) -> RunState:
    """Apply human decisions to the session's paused run and return the RunState to re-run.

    ``resume`` mirrors the LangGraph contract: ``{"decisions": [{"type": "approve"|"reject", ...}]}``,
    one decision per pending approval, in interruption order. Raises if no paused run is loaded for
    the session (in-process only — a restart or another replica drops it).
    """
    state = _pending_runs.pop(session_id, None)
    if state is None:
        raise RuntimeError(
            "No paused run for this session. HITL pauses are in-process only, so a restart or a "
            "different replica loses them; retry the turn."
        )
    decisions = resume.get("decisions") or []
    for decision, item in zip(decisions, state.get_interruptions()):
        if decision.get("type") == "approve":
            state.approve(item)
        else:
            state.reject(item, rejection_message=decision.get("message"))
    return state


async def _serialize_events(
    result: RunResultStreaming, session_id: str
) -> AsyncGenerator[dict, None]:
    """Turn the Agents SDK run's stream events into the runtime's JSON envelope.

    Emits the same shape the LangGraph template does — ``{"type": "delta", ...}`` for token chunks,
    ``{"type": "message", "message": {...}}`` for completed items, ``{"type": "interrupt", ...}`` for a
    human-approval pause — so the SDK-agnostic runtime and browser UI are identical across frameworks.
    Message dicts are normalized to ``{role, content, tool_calls?}`` regardless of the SDK's native
    item type.
    """
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                yield {"type": "delta", "content": event.data.delta, "id": event.data.item_id}
        elif event.type == "run_item_stream_event":
            if message := _normalize_item(event.item):
                yield {"type": "message", "message": message}

    # After the stream drains, a paused run surfaces as pending interruptions. Stash the RunState
    # (in-process) so a later resume can apply the decisions, and relay each pending call as an
    # interrupt event on the session's thread.
    if result.interruptions:
        _pending_runs[session_id] = result.to_state()
        for item in result.interruptions:
            yield {"type": "interrupt", "id": item.call_id, "value": _approval_value(item)}


def _approval_value(item: ToolApprovalItem) -> dict:
    """The interrupt payload for one pending approval, matching the LangGraph HITL event shape."""
    return {
        "action_requests": [{"name": item.tool_name, "args": _tool_args(item)}],
    }


def _tool_args(item: ToolApprovalItem) -> Any:
    import json

    args = item.arguments
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"arguments": args}
    return args or {}


def _normalize_item(item: Any) -> dict | None:
    """Normalize one Agents SDK run item to the UI's ``{role, content, tool_calls?}`` message shape.

    The browser renders LangChain-native message dicts; normalizing here keeps the frontend identical
    across frameworks. Only user/assistant/tool items become messages; other run items are dropped.
    """
    from agents import ItemHelpers
    from agents.items import MessageOutputItem, ToolCallItem, ToolCallOutputItem

    if isinstance(item, MessageOutputItem):
        return {"role": "assistant", "content": ItemHelpers.text_message_output(item)}
    if isinstance(item, ToolCallItem):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": item.tool_name, "args": _tool_args_from_call(item)}],
        }
    if isinstance(item, ToolCallOutputItem):
        return {"role": "tool", "name": _tool_call_name(item), "content": str(item.output)}
    return None


def _tool_args_from_call(item: Any) -> Any:
    import json

    raw = item.raw_item
    args = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"arguments": args}
    return args or {}


def _tool_call_name(item: Any) -> str | None:
    raw = item.raw_item
    return raw.get("name") if isinstance(raw, dict) else getattr(raw, "name", None)
