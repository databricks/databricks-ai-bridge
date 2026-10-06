import asyncio
import configparser
import logging
import os
import sys
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

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

# OpenAI Sessions persist transcript history, not a paused RunState. Keep pending approvals local.
_pending_runs: dict[str, RunState] = {}


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


# Resolving auth can block forever on an external-browser OAuth flow (a browser plus a local
# callback server), so the startup check is bounded by this timeout instead.
_AUTH_CHECK_TIMEOUT_S = 20.0


def _check_databricks_auth() -> None:
    """Fail fast at startup with a clear message if Databricks auth isn't configured.

    Without this, a missing/invalid profile only surfaces on the first model call — as a generic SDK
    error buried in a request traceback. Resolving a WorkspaceClient here validates the same config
    the model client uses, so the failure is immediate and actionable. The resolution is bounded:
    an external-browser profile with no cached token would otherwise block forever on a browser
    flow that a server process can never complete.
    """
    problem: BaseException | None = None
    if _external_browser_without_cached_token():
        problem = RuntimeError("external-browser auth with no cached token; it needs an interactive login")
    else:
        outcome: list[BaseException | None] = []

        def _resolve() -> None:
            try:
                workspace_client()
                outcome.append(None)
            except BaseException as e:
                outcome.append(e)

        worker = threading.Thread(target=_resolve, daemon=True, name="databricks-auth-check")
        worker.start()
        worker.join(_AUTH_CHECK_TIMEOUT_S)
        problem = outcome[0] if outcome else TimeoutError("auth check timed out")
        if problem is None:
            return
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    target = f"profile {profile!r}" if profile else "the DEFAULT profile / DATABRICKS_HOST+TOKEN"
    message = (
        f"Databricks auth is not configured — the agent can't call the model. Tried {target}.\n"
        "Fix one of:\n"
        "  • set DATABRICKS_CONFIG_PROFILE in .env to a profile from `databricks auth profiles`, or\n"
        "  • run `databricks auth login --profile <name>` to create one, or\n"
        "  • set DATABRICKS_HOST and DATABRICKS_TOKEN in .env.\n"
        f"(underlying error: {problem})"
    )
    # stderr too: depending on the server, a raised exception can surface as a bare traceback.
    print(message, file=sys.stderr)
    raise RuntimeError(message) from problem


def _external_browser_without_cached_token() -> bool:
    """Whether the active profile needs an interactive browser login this process can't complete.

    Best-effort: the profile's auth_type says external-browser, and the Python SDK's OAuth token
    cache (what lets that flow skip the browser) holds no tokens at all.
    """
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    if not profile:
        return False
    config = configparser.ConfigParser(interpolation=None)
    config.read(os.path.expanduser(os.getenv("DATABRICKS_CONFIG_FILE", "~/.databrickscfg")))
    if config.get(profile, "auth_type", fallback="") != "external-browser":
        return False
    try:
        with os.scandir(os.path.expanduser("~/.config/databricks-sdk-py/oauth")) as entries:
            return not any(entry.is_file() and entry.stat().st_size > 0 for entry in entries)
    except OSError:
        return True  # no cache directory at all -> no cached token


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
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> AsyncIterator[RunResultStreaming]:
    """Run the agent and expose its native streaming result.

    This is the framework-native entrypoint. It has no dependency on Agent Bricks request or context types,
    so it can be called from another server, a notebook, or a test harness.
    """
    actor = actor or session_id
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
        with start_trace(name="invoke", inputs=agent_input, session_id=session_id) as span:
            if isinstance(agent_input, RunState):
                result = Runner.run_streamed(agent, agent_input)
            else:
                result = Runner.run_streamed(
                    agent,
                    agent_input,
                    session=session_store(session_id, actor),
                )

            try:
                yield result
            finally:
                if workspace_client_for is not None and not result.is_complete:
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
                _pending_runs[session_id] = result.to_state()
            if span is not None:
                span.set_outputs({"output": result.final_output})
