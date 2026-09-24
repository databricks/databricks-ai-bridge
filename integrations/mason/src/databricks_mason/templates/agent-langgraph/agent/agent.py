import os
from collections.abc import AsyncGenerator, Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools
from databricks_mason import workspace_client, workspace_headers
from databricks_mason.langgraph import (
    checkpointer,
    configure_tracing,
    genie_tools,
    mcp_tools,
    memory_tools,
    start_trace,
    thread_config,
)
from databricks_mason.runtime.auth import AuthError

# A Unity Catalog AI Gateway model service, served from the `system.ai` schema and queried through
# the gateway (see `use_ai_gateway=True` below). Swap for any `system.ai.*` model service your
# workspace exposes — the demo chat app's picker lists what's available.
MODEL = "system.ai.claude-sonnet-4-5"
_INVOCATION_METADATA_KEY = "databricks_mason.invocation_id"

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


async def create_agent_graph(
    actor: str,
    model: str | None = None,
    *,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
):
    """Build the LangGraph agent: local tools + long-term-memory tools + any MCP tools.

    ``actor`` is the identity whose long-term memory the agent reads/writes; it's captured in the
    memory tools' closures (never exposed to the model).

    ``model`` selects the gateway model for this run; the chat UI passes the picker's choice and
    everything else falls back to ``MODEL``. The agent is rebuilt per turn, so the model can vary
    request to request.
    """
    mcp = await mcp_tools(build_mcp_servers(), workspace_client_for=workspace_client_for)
    tools = [
        *all_tools(),
        *memory_tools(actor),
        *genie_tools(workspace_client_for=workspace_client_for),
        *mcp,
    ]
    middleware = (
        [HumanInTheLoopMiddleware(interrupt_on=REQUIRE_APPROVAL)] if REQUIRE_APPROVAL else []
    )
    endpoint = model or MODEL
    return create_agent(
        # use_ai_gateway routes to the Unity Catalog AI Gateway (`<host>/ai-gateway/mlflow/v1`), so
        # `endpoint` is a `system.ai.*` model name rather than a serving-endpoint name.
        model=_RoutedChatDatabricks(
            endpoint=endpoint, workspace_client=workspace_client(), use_ai_gateway=True
        ),
        tools=tools,
        middleware=middleware,
        checkpointer=checkpointer(),
    )


async def recovery_input(
    agent_input: Any,
    *,
    session_id: str,
    actor: str,
    invocation_id: str,
) -> Any:
    """Return the input that should be passed to ``run_agent`` after worker loss.

    A checkpoint tagged with this invocation means LangGraph can continue by receiving ``None``.
    Otherwise recovery replays the original application input.
    """
    checkpoint = await checkpointer().aget_tuple(thread_config(session_id, actor))
    current_invocation_checkpointed = bool(
        checkpoint and checkpoint.metadata.get(_INVOCATION_METADATA_KEY) == invocation_id
    )
    return None if current_invocation_checkpointed else agent_input


async def run_agent(
    agent_input: Any,
    *,
    session_id: str,
    actor: str | None = None,
    model: str | None = None,
    invocation_id: str | None = None,
    workspace_client_for: Callable[[str], WorkspaceClient] | None = None,
) -> AsyncGenerator[Any, None]:
    """Run the agent and yield native LangGraph stream events.

    This is the framework-native entrypoint. It has no dependency on Mason request or context types,
    so it can be called from another server, a notebook, or a test harness.
    """
    actor = actor or session_id
    graph = await create_agent_graph(actor, model, workspace_client_for=workspace_client_for)
    config = thread_config(session_id, actor)
    if invocation_id:
        config["metadata"] = {_INVOCATION_METADATA_KEY: invocation_id}

    last_update: Any = None
    with start_trace(name="invoke", inputs=agent_input, session_id=session_id) as span:
        async for event in graph.astream(
            input=agent_input,
            config=config,
            stream_mode=["updates", "messages"],
            durability="sync",
        ):
            if event[0] == "updates":
                last_update = event[1]
                if workspace_client_for is not None and last_update.get("__interrupt__"):
                    raise AuthError(
                        "MCP_USER_AUTH_HITL_UNSUPPORTED",
                        "Request-user invocations do not support paused approvals.",
                        400,
                    )
            yield event
        if span is not None and last_update is not None:
            span.set_outputs(last_update)
