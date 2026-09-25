"""Translate between Agent Bricks invocations and the framework-native agent entrypoint."""

from typing import Any

from agent import agent as agent_module
from agent.agent import recovery_input, run_agent
from langgraph.types import Command

from databricks_agentkit import InvocationContext
from databricks_agentkit.langgraph.responses import checkpointed_messages, collect_response
from databricks_agentkit.langgraph.session_store import thread_config
from databricks_agentkit.runtime.auth import AuthError


def _payload(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {"messages": value}
    if not isinstance(value, dict):
        raise ValueError("input must be a message list or an object")
    return value


def _session_id(payload: dict[str, Any], context: InvocationContext) -> str:
    value = payload.get("session_id") or context.session_id
    if not isinstance(value, str) or not value:
        raise ValueError("session_id must be a non-empty string")
    return value


def _actor(payload: dict[str, Any], session_id: str) -> str:
    value = payload.get("actor") or session_id
    if not isinstance(value, str) or not value:
        raise ValueError("actor must be a non-empty string")
    return value


def _agent_input(payload: dict[str, Any]) -> Any:
    if (resume := payload.get("resume")) is not None:
        return Command(resume=resume)
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    return {"messages": messages}


async def invoke(value: Any, context: InvocationContext) -> dict:
    payload = _payload(value)
    _check_user_input(payload, context)
    return await _invoke_agent(_agent_input(payload), payload, context)


async def recover(value: Any, context: InvocationContext) -> dict:
    payload = _payload(value)
    _check_user_input(payload, context)
    if getattr(context, "request_auth", None) is not None:
        raise AuthError(
            "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
            "Request-user invocations cannot be recovered in the background.",
            400,
        )
    session_id = _session_id(payload, context)
    actor = _actor(payload, session_id)
    agent_input = await recovery_input(
        _agent_input(payload),
        session_id=session_id,
        actor=actor,
        invocation_id=context.invocation_id,
    )
    return await _invoke_agent(agent_input, payload, context)


def _check_user_input(payload: dict[str, Any], context: InvocationContext) -> None:
    auth = getattr(context, "request_auth", None)
    if auth is not None and any(
        payload.get(key) is not None for key in ("resume", "approval", "approvals")
    ):
        raise AuthError(
            "MCP_USER_AUTH_HITL_UNSUPPORTED",
            "Request-user invocations do not support approval or resume input.",
            400,
        )


async def _invoke_agent(
    agent_input: Any,
    payload: dict[str, Any],
    context: InvocationContext,
) -> dict:
    session_id = _session_id(payload, context)
    actor = _actor(payload, session_id)
    auth = getattr(context, "request_auth", None)
    internal_session_id = (
        auth.namespace("session", session_id) if auth and payload.get("session_id") else session_id
    )
    actor = auth.namespace("actor", actor) if auth else actor
    user_auth = auth is not None
    run_kwargs = {"workspace_client_for": auth.client_for} if user_auth else {}
    model = payload.get("model")
    model = model if isinstance(model, str) else None
    restored_messages = []
    if agent_input is None:
        graph = await agent_module.create_agent_graph(actor, model, **run_kwargs)
        restored_messages = await checkpointed_messages(
            graph, thread_config(internal_session_id, actor), context.invocation_id
        )
        run_kwargs["graph"] = graph

    async def emit(event):
        if user_auth and event.get("type") == "interrupt":
            raise AuthError(
                "MCP_USER_AUTH_HITL_UNSUPPORTED",
                "Request-user invocations do not support paused approvals.",
                400,
            )
        await context.emit(event)

    response = await collect_response(
        run_agent(
            agent_input,
            session_id=internal_session_id,
            actor=actor,
            model=model,
            invocation_id=context.invocation_id,
            **run_kwargs,
        ),
        emit,
        restored_messages,
    )

    return {
        **response,
        **({"session_id": session_id} if not user_auth or payload.get("session_id") else {}),
    }
