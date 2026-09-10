import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

from agent.mcps import build_mcp_servers

# Importing the tools package auto-registers every tool module.
from agent.tools import all_tools
from databricks_mason import DurableAgentContext, tag_session
from databricks_mason.claude import (
    client as claude_client,
    configure_tracing,
    mcp_servers,
    memory_tools,
    session_history,
)
from databricks_mason.claude.mcp import MCP_CONNECTOR_BETA

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# Tools that require human approval before they run. When the model calls one, the agent stops before
# executing it and emits an `interrupt`; the client resumes by sending `resume` with the same session
# id. Empty it to disable approval gating.
REQUIRE_APPROVAL = {"send_message"}


def configure() -> None:
    """Wire up global state; call once at server startup (not at import)."""
    _check_anthropic_auth()
    configure_tracing()


def _check_anthropic_auth() -> None:
    """Fail fast at startup with a clear message if no Anthropic credential is configured."""
    from databricks_mason.claude.client import use_bedrock

    if use_bedrock() or os.getenv("ANTHROPIC_API_KEY"):
        return
    raise RuntimeError(
        "Anthropic auth is not configured — the agent can't call Claude.\n"
        "Fix one of:\n"
        "  • set ANTHROPIC_API_KEY (an Anthropic API key) in .env / the app env, or\n"
        "  • set CLAUDE_CODE_USE_BEDROCK=1 to route through Amazon Bedrock with the app's AWS creds."
    )


def _model(payload: dict[str, Any]) -> str:
    """The serving model for this run: request ``model`` → ``AGENT_CLAUDE_MODEL`` env → default."""
    model = payload.get("model")
    if isinstance(model, str) and model:
        return model
    return os.getenv("AGENT_CLAUDE_MODEL", DEFAULT_MODEL)


def create_tools(actor: str) -> list[Any]:
    """The agent's tools: local ``@beta_tool`` tools + long-term-memory tools for ``actor``."""
    return [*all_tools(), *memory_tools(actor)]


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
    """Replay the persisted application input after the runtime replaces a stale worker.

    The Tool Runner owns its loop, so a mid-run pause can't be restored; recovery replays the input
    against the persisted transcript (same limitation as the other Mason templates' HITL).
    """
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
    """Translate one Tool Runner run into persisted runtime events.

    The runner is stateless, so prior turns come from the durable transcript and new turns are
    appended after the run. History persists the user turn and the assistant's final text (not
    intermediate tool calls), keeping the stored transcript a valid user/assistant alternation.
    """
    history = session_history(session_id, actor)
    prior = history.load()

    resume = payload.get("resume")
    approve_all = False
    new_messages: list[dict[str, Any]] = []
    if resume is not None:
        if not isinstance(resume, dict):
            raise ValueError("resume must be an object")
        approve_all, new_messages = _decisions(resume)
    else:
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        new_messages = messages

    tools = create_tools(actor)
    kwargs: dict[str, Any] = {
        "model": _model(payload),
        "max_tokens": MAX_TOKENS,
        "tools": tools,
        "messages": [*prior, *new_messages],
    }
    servers = mcp_servers(build_mcp_servers())
    if servers:
        kwargs["mcp_servers"] = servers
        kwargs["betas"] = [MCP_CONNECTOR_BETA]

    interrupted = False
    answer_parts: list[str] = []
    # The runner is a sync iterator; each __next__ produces one assistant turn and, before the next,
    # executes any (non-gated) tool calls. Breaking on a gated call stops before it runs.
    for message in claude_client().beta.messages.tool_runner(**kwargs):
        content = list(getattr(message, "content", []) or [])
        text = "".join(b.text for b in content if getattr(b, "type", None) == "text")
        if text:
            answer_parts.append(text)
            yield {"type": "delta", "content": text, "id": getattr(message, "id", None)}
        yield {"type": "message", "message": _normalize(content)}

        gated = [
            block
            for block in content
            if getattr(block, "type", None) == "tool_use"
            and block.name in REQUIRE_APPROVAL
            and not approve_all
        ]
        if gated:
            interrupted = True
            for block in gated:
                yield {
                    "type": "interrupt",
                    "id": block.id,
                    "value": {"action_requests": [{"name": block.name, "args": block.input}]},
                }
            break

    persisted = list(new_messages)
    answer = "\n".join(answer_parts)
    if answer and not interrupted:
        persisted.append({"role": "assistant", "content": answer})
    history.append(persisted)


def _decisions(resume: dict) -> tuple[bool, list[dict[str, Any]]]:
    """Read the approval contract ``{"decisions": [{"type": "approve"|"reject", "message"?}]}``.

    First cut: any ``approve`` lets the gated tool(s) run on the re-run; each ``reject`` adds a user
    note so the model backs off. Approval isn't mapped to a specific pending call id.
    """
    approve_all = False
    rejections: list[dict[str, Any]] = []
    for decision in resume.get("decisions") or []:
        if decision.get("type") == "approve":
            approve_all = True
        else:
            rejections.append(
                {"role": "user", "content": decision.get("message") or "The user declined that action."}
            )
    return approve_all, rejections


def _normalize(content: list) -> dict:
    """Normalize one assistant turn's content blocks to the UI's ``{role, content, tool_calls?}`` shape."""
    text = "".join(b.text for b in content if getattr(b, "type", None) == "text")
    tool_calls = [
        {"name": b.name, "args": b.input}
        for b in content
        if getattr(b, "type", None) == "tool_use"
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message
