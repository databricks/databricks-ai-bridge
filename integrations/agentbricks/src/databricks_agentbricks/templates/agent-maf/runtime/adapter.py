"""Translate Agent Bricks invocations to native Microsoft Agent Framework messages."""

from typing import Any

from agent_framework import Message

from agent.agent import run_agent
from databricks_agentkit import InvocationContext
from databricks_agentkit.maf.messages import serialize_message
from databricks_agentkit.runtime.auth import AuthError

_RECOVERY_INSTRUCTION = (
    "A previous worker stopped during this invocation. Some tools or external side effects may "
    "already have completed. Check current state before repeating side effects and avoid duplicates."
)


async def invoke(value: Any, context: InvocationContext) -> dict:
    return await _invoke(value, context)


async def recover(value: Any, context: InvocationContext) -> dict:
    return await _invoke(value, context, recovery=True)


async def _invoke(
    value: Any, context: InvocationContext, *, recovery: bool = False
) -> dict:
    payload = {"messages": value} if isinstance(value, list) else value
    if not isinstance(payload, dict):
        raise ValueError("input must be a message list or an object")
    if payload.get("resume") is not None:
        raise ValueError(
            "The Microsoft Agent Framework template does not support approval continuation."
        )
    session_id = context.session_id
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be provided as a top-level invocation field")
    actor = payload.get("actor") or session_id
    if not isinstance(actor, str) or not actor:
        raise ValueError("actor must be a non-empty string")
    auth = getattr(context, "request_auth", None)
    if auth and recovery:
        raise AuthError(
            "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
            "Request-user invocations cannot be recovered in the background.",
            400,
        )
    if auth and payload.get("actor"):
        actor = auth.namespace("actor", actor)
    raw = payload.get("messages", [])
    if not isinstance(raw, list) or not raw:
        raise ValueError("messages must be a non-empty list")
    messages = []
    for item in raw:
        if (
            not isinstance(item, dict)
            or item.get("role") != "user"
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError(
                "Each input message requires role='user' and string content"
            )
        messages.append(Message(item["role"], [item["content"]]))
    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise ValueError("model must be a non-empty string")
    async with run_agent(
        messages,
        session_id=session_id,
        actor=actor,
        model=model,
        instructions=_RECOVERY_INSTRUCTION if recovery else None,
    ) as stream:
        async for update in stream:
            if update.text:
                await context.emit(
                    {"type": "delta", "content": update.text, "id": update.message_id}
                )
        result = await stream.get_final_response()
        if any(
            content.user_input_request
            for message in result.messages
            for content in message.contents
        ):
            raise ValueError(
                "The Microsoft Agent Framework template does not support approval continuation."
            )
        # Persisted native messages retain tool calls/results. The wire format is framework-neutral.
        output = [serialize_message(message) for message in result.messages]
        for message in output:
            await context.emit({"type": "message", "message": message})
    return {"output": output, "status": "completed"}
