import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any
from uuid import uuid4

from agents import Agent, Runner, RunResultStreaming, RunState
from agents.mcp import MCPServerManager
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI

from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools
from databricks_agentkit import workspace_client, workspace_headers
from databricks_agentkit.openai import (
    configure_tracing,
    genie_tools,
    mcp_servers,
    memory_tools,
    session_store,
    start_trace,
)
from databricks_agentkit.runtime.auth import AuthError

logger = logging.getLogger(__name__)

# A Unity Catalog AI Gateway model service, served from the `system.ai` schema and queried through
# the gateway (see `use_ai_gateway=True` in configure()). Swap for any `system.ai.*` model service
# your workspace exposes — the demo chat app's picker lists what's available.
MODEL = "system.ai.claude-sonnet-4-5"

# Tools that require human approval before they run. Add a tool's name here and the agent pauses when
# the model calls it, emitting an `interrupt` event; the client resumes by sending `resume` with the
# same session id. The tools declare `needs_approval=True` themselves (see agent/tools/); this set is
# how the runtime knows which pending calls to surface. Empty it to disable approval gating.
REQUIRE_APPROVAL = {"send_message"}

# Native callers keep pauses local; the runtime supplies durable, attempt-fenced state callbacks.
_pending_runs: dict[str, dict[str, Any]] = {}


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_databricks_auth()
    from agents import set_default_openai_api, set_default_openai_client

    # use_ai_gateway routes to the Unity Catalog AI Gateway (`<host>/ai-gateway/mlflow/v1`), so
    # `MODEL` is a `system.ai.*` model name rather than a serving-endpoint name.
    set_default_openai_client(
        AsyncDatabricksOpenAI(
            workspace_client=workspace_client(),
            default_headers=workspace_headers() or None,
            use_ai_gateway=True,
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


def create_agent(
    actor: str,
    mcp=None,
    model: str | None = None,
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> Agent:
    """Build the OpenAI Agents SDK agent: tools, memory, MCP servers, and model."""
    return Agent(
        name="Agent",
        instructions="You are a helpful assistant.",
        model=model or MODEL,
        tools=[
            *all_tools(),
            *memory_tools(actor),
            *genie_tools(workspace_client_for=workspace_client_for),
        ],
        mcp_servers=mcp or [],
    )


def resume_agent(state: RunState, resume: dict[str, Any]) -> RunState:
    """Apply human decisions to a paused run and return the native RunState."""
    decisions = resume.get("decisions")
    interruptions = state.get_interruptions()
    if not isinstance(decisions, list) or not decisions or len(decisions) != len(interruptions):
        raise ValueError("Provide exactly one decision for each pending tool call.")
    # Validate the whole batch before applying even its first decision.
    for decision, item in zip(decisions, interruptions, strict=True):
        if (
            not isinstance(decision, dict)
            or decision.get("type") not in ("approve", "reject")
            or ("message" in decision and not isinstance(decision["message"], str))
            or decision.get("call_id") != item.call_id
        ):
            raise ValueError("Invalid approval decision or mismatched pending tool call.")
    for decision, item in zip(decisions, interruptions, strict=True):
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
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
    resume: dict[str, Any] | None = None,
    load_state: Callable[[], Awaitable[Any]] | None = None,
    save_state: Callable[[Any], Awaitable[None]] | None = None,
    invocation_id: str | None = None,
) -> AsyncIterator[RunResultStreaming]:
    """Run the agent and expose its native streaming result.

    This is the framework-native entrypoint. It has no dependency on Agent Bricks request or context types,
    so it can be called from another server, a notebook, or a test harness.
    """
    actor = actor or session_id
    invocation_id = invocation_id or str(uuid4())
    snapshot = await load_state() if load_state else _pending_runs.get(session_id)
    if snapshot is not None and snapshot["actor"] != actor:
        raise ValueError("The pending approval belongs to a different actor.")
    if snapshot is not None and snapshot.get("status") == "completed":
        if resume is not None or snapshot["invocation_id"] == invocation_id:
            raise RuntimeError("This invocation already completed; use its saved response.")
        snapshot = None
    if resume is not None:
        if not isinstance(resume, dict):
            raise ValueError("resume must be an object")
        if snapshot is None:
            raise RuntimeError("No paused run for this session.")
        if snapshot["status"] == "running" and snapshot["invocation_id"] != invocation_id:
            raise RuntimeError("This approval was already accepted; check its invocation status.")
        if snapshot.get("status") in ("running", "failed") and resume.get(
            "decisions"
        ) != snapshot.get("decisions"):
            raise ValueError("Retry the continuation with its previously accepted decisions.")
    elif snapshot is not None and snapshot.get("invocation_id") != invocation_id:
        raise ValueError("Resolve the pending approval before starting another turn.")
    if snapshot is not None:
        model = snapshot["model"]

    async def persist(value: dict[str, Any]) -> None:
        if save_state:
            await save_state(value)
        else:
            _pending_runs[session_id] = value

    auth_kwargs = {"workspace_client_for": workspace_client_for} if workspace_client_for else {}
    servers = await mcp_servers(build_mcp_servers(), **auth_kwargs)
    async with MCPServerManager(servers) as manager:
        for server, error in manager.errors.items():
            if getattr(server, "_agentbricks_request_user", False) is True or isinstance(
                error, AuthError
            ):
                raise error
        active_servers = []
        for server in manager.active_servers:
            tool_filter = server.tool_filter
            try:
                server.tool_filter = None
                server.cache_tools_list = True
                async with asyncio.timeout(manager.connect_timeout_seconds):
                    await server.list_tools()
            except Exception as error:
                if getattr(server, "_agentbricks_request_user", False) is True or isinstance(
                    error, AuthError
                ):
                    raise
                logger.warning(
                    "Failed to list tools from MCP server %r; continuing without it.",
                    server.name,
                    exc_info=True,
                )
            else:
                active_servers.append(server)
            finally:
                server.tool_filter = tool_filter

        agent = create_agent(
            actor,
            active_servers,
            model=model,
            workspace_client_for=workspace_client_for,
        )
        if snapshot is not None:
            state = await RunState.from_json(agent, snapshot["run_state"])
            agent_input = (
                resume_agent(state, resume)
                if resume is not None
                and snapshot["status"] == "pending"
                and snapshot["invocation_id"] != invocation_id
                else state
            )
            if resume is not None:
                await persist(
                    {
                        **snapshot,
                        "run_state": agent_input.to_json(),
                        "status": "running",
                        "invocation_id": invocation_id,
                        "decisions": resume["decisions"],
                    }
                )
        with start_trace(name="invoke", inputs=agent_input, session_id=session_id) as span:
            result = Runner.run_streamed(
                agent,
                agent_input,
                session=session_store(session_id, actor),
            )

            async def persist_result(status: str) -> None:
                await persist(
                    {
                        "run_state": result.to_state().to_json(),
                        "model": model or MODEL,
                        "actor": actor,
                        "invocation_id": invocation_id,
                        "status": status,
                        "decisions": resume["decisions"] if resume is not None else None,
                    }
                )

            try:
                yield result
            except Exception:
                if resume is not None and result.is_complete and result.final_output is None:
                    # Keep completed tool outputs, so retrying a model failure does not rerun tools.
                    await persist_result("failed")
                raise
            finally:
                if not result.is_complete:
                    result.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        async for _ in result.stream_events():
                            pass

            if result.interruptions:
                if workspace_client_for is not None:
                    raise AuthError(
                        "MCP_USER_AUTH_HITL_UNSUPPORTED",
                        "Request-user invocations do not support paused approvals.",
                        400,
                    )
                await persist_result("pending")
            elif resume is not None and result.is_complete:
                await persist_result("completed")
            if span is not None:
                span.set_outputs({"output": result.final_output})
