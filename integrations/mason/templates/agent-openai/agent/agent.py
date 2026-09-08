import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from agents import Agent, Runner, RunResultStreaming, RunState
from agents.items import ToolApprovalItem
from agents.mcp import MCPServerManager
from databricks_openai import AsyncDatabricksOpenAI
from openai.types.responses import ResponseTextDeltaEvent

from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools
from databricks_mason import (
    DurableAgentContext,
    tag_session,
    workspace_client,
    workspace_headers,
)
from databricks_mason.openai import configure_tracing, mcp_servers, memory_tools, session_store

logger = logging.getLogger(__name__)

MODEL = "databricks-gpt-5-2"

# Tools that require human approval before they run. Add a tool's name here and the agent pauses when
# the model calls it, emitting an `interrupt` event; the client resumes by sending `resume` with the
# same session id. The tools declare `needs_approval=True` themselves (see agent/tools/); this set is
# how the runtime knows which pending calls to surface. Empty it to disable approval gating.
REQUIRE_APPROVAL = {"send_message"}

# Paused runs awaiting human approval, keyed by session id. In-process only — a paused run does NOT
# survive a restart or reach another replica, even with AGENT_SESSION_STORE set: an Agents SDK
# Session persists the transcript but not the paused RunState. Durable HITL would stash
# RunState.to_json() separately; this template keeps it simple and in-memory.
_pending_runs: dict[str, RunState] = {}


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    # Route the Agents SDK's default OpenAI client at the Databricks model endpoint (account-host
    # routing and auth handled by the SDK), so `Agent(model=MODEL)` resolves to a Databricks model.
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
    """Build the OpenAI Agents SDK agent: local tools + long-term-memory tools + any MCP servers.

    ``actor`` is the identity whose long-term memory the agent reads/writes; it's captured in the
    memory tools' closures (never exposed to the model). See ``_actor``.

    ``model`` selects the serving endpoint for this run; the chat UI passes the picker's choice and
    everything else falls back to ``MODEL``. The agent is rebuilt per turn, so the endpoint can vary
    request to request.
    """
    return Agent(
        name="Agent",
        instructions="You are a helpful assistant.",
        model=model or MODEL,
        tools=[*all_tools(), *memory_tools(actor)],
        mcp_servers=mcp or [],
    )


def _payload(value: Any) -> dict[str, Any]:
    """Normalize the application payload carried inside the durable request's ``input`` field."""
    if isinstance(value, list):
        return {"messages": value}
    if not isinstance(value, dict):
        raise ValueError("input must be a message list or an object")
    return value


def _session_id(payload: dict[str, Any], context: DurableAgentContext) -> str:
    value = payload.get("session_id") or context.session_id
    if not isinstance(value, str) or not value:
        raise ValueError("session_id must be a non-empty string")
    return value


def _actor(payload: dict[str, Any], session_id: str) -> str:
    value = payload.get("actor") or session_id
    if not isinstance(value, str) or not value:
        raise ValueError("actor must be a non-empty string")
    return value


async def invoke(value: Any, context: DurableAgentContext) -> dict:
    """Run the first attempt for one durable invocation."""
    return await _run_agent(_payload(value), context)


async def on_recovery(value: Any, context: DurableAgentContext) -> dict:
    """Replay the persisted application input after the runtime replaces a stale worker."""
    return await _run_agent(_payload(value), context)


async def _run_agent(payload: dict[str, Any], context: DurableAgentContext) -> dict:
    session_id = _session_id(payload, context)
    actor = _actor(payload, session_id)
    tag_session(session_id)

    outputs = [
        event
        async for event in _persisted_agent_events(payload, context, session_id, actor)
        if event.get("type") in ("message", "interrupt")
    ]
    interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
    return {
        "output": [event["message"] if event["type"] == "message" else event for event in outputs],
        "session_id": session_id,
        "status": "interrupted" if interrupted else "completed",
    }


async def _persisted_agent_events(
    payload: dict[str, Any],
    context: DurableAgentContext,
    session_id: str,
    actor: str,
) -> AsyncGenerator[dict, None]:
    async for event in _agent_events(payload, session_id, actor):
        await context.emit(event)
        yield event


async def _agent_events(
    payload: dict[str, Any], session_id: str, actor: str
) -> AsyncGenerator[dict, None]:
    """Translate one Agents SDK run into persisted runtime events."""

    servers = await mcp_servers(build_mcp_servers())
    async with MCPServerManager(servers) as manager:
        mcp = []
        for server in manager.active_servers:
            # Cache raw tools now; the SDK applies any context-dependent filter during the run.
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
                mcp.append(server)
            finally:
                server.tool_filter = tool_filter

        model = payload.get("model")
        agent = create_agent(actor, mcp, model=model if isinstance(model, str) else None)

        # A `resume` payload continues a session paused awaiting approval; otherwise start a new turn
        # from `messages`. A resumed run re-runs the stashed RunState (with decisions applied); a new
        # turn passes the messages plus the session store so prior history is loaded automatically.
        resume = payload.get("resume")
        if resume is not None:
            if not isinstance(resume, dict):
                raise ValueError("resume must be an object")
            run_input: Any = _apply_decisions(session_id, resume)
            result = Runner.run_streamed(agent, run_input)
        else:
            messages = payload.get("messages") or []
            if not isinstance(messages, list):
                raise ValueError("messages must be a list")
            result = Runner.run_streamed(
                agent,
                messages,
                session=session_store(session_id, actor),
            )

        async for event in _serialize_events(result, session_id):
            yield event


def _apply_decisions(session_id: str, resume: dict) -> RunState:
    """Apply human decisions to the session's paused run and return the RunState to re-run.

    ``resume`` is the runtime's approval contract: ``{"decisions": [{"type": "approve"|"reject", ...}]}``,
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
    for decision, item in zip(decisions, state.get_interruptions(), strict=False):
        if decision.get("type") == "approve":
            state.approve(item)
        else:
            state.reject(item, rejection_message=decision.get("message"))
    return state


async def _serialize_events(
    result: RunResultStreaming, session_id: str
) -> AsyncGenerator[dict, None]:
    """Turn the Agents SDK run's stream events into the runtime's JSON envelope.

    Emits the runtime's framework-neutral event shape — ``{"type": "delta", ...}`` for token chunks,
    ``{"type": "message", "message": {...}}`` for completed items, ``{"type": "interrupt", ...}`` for a
    human-approval pause — so the SDK-agnostic runtime and browser UI are identical across templates.
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
    """The interrupt payload for one pending approval, in the runtime's framework-neutral shape."""
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

    That shape is framework-neutral — the browser renders it the same regardless of which SDK
    produced the run — so normalizing here keeps the frontend identical across templates. Only
    user/assistant/tool items become messages; other run items are dropped.
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
