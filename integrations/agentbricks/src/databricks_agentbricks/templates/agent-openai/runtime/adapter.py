"""Translate between Agent Bricks invocations and the framework-native agent entrypoint."""

from collections.abc import AsyncGenerator
from typing import Any

from agent.agent import resume_agent, run_agent
from agents import RunResultStreaming
from agents.items import ToolApprovalItem
from openai.types.responses import ResponseTextDeltaEvent

from databricks_agentkit import InvocationContext
from databricks_agentkit.runtime.auth import AuthError

_RECOVERY_INSTRUCTION = (
    "This is a recovery attempt after a previous worker stopped before completing this invocation. "
    "Continue the same task, but assume some tool calls or external side effects may already have "
    "completed or may still be in progress. Check current state before repeating side effects and "
    "avoid duplicate actions when possible."
)


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


def _agent_input(
    payload: dict[str, Any],
    session_id: str,
    *,
    recovery: bool = False,
) -> list[Any] | Any:
    if (resume := payload.get("resume")) is not None:
        if not isinstance(resume, dict):
            raise ValueError("resume must be an object")
        return resume_agent(session_id, resume)
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if recovery:
        return [{"role": "developer", "content": _RECOVERY_INSTRUCTION}, *messages]
    return messages


async def invoke(value: Any, context: InvocationContext) -> dict:
    return await _invoke_agent(_payload(value), context)


async def recover(value: Any, context: InvocationContext) -> dict:
    return await _invoke_agent(_payload(value), context, recovery=True)


async def _invoke_agent(
    payload: dict[str, Any],
    context: InvocationContext,
    *,
    recovery: bool = False,
) -> dict:
    session_id = _session_id(payload, context)
    actor = _actor(payload, session_id)
    auth = getattr(context, "request_auth", None)
    user_auth = auth is not None
    if user_auth and any(
        payload.get(key) is not None for key in ("resume", "approval", "approvals")
    ):
        raise AuthError(
            "MCP_USER_AUTH_HITL_UNSUPPORTED",
            "Request-user invocations do not support approval or resume input.",
            400,
        )
    if user_auth and recovery:
        raise AuthError(
            "MCP_USER_AUTH_BACKGROUND_UNSUPPORTED",
            "Request-user invocations cannot be recovered in the background.",
            400,
        )
    internal_session_id = (
        auth.namespace("session", session_id) if auth and payload.get("session_id") else session_id
    )
    actor = auth.namespace("actor", actor) if auth else actor
    auth_kwargs = {"workspace_client_for": auth.client_for} if user_auth else {}
    model = payload.get("model")
    outputs = []
    async with run_agent(
        _agent_input(payload, internal_session_id, recovery=recovery),
        session_id=internal_session_id,
        actor=actor,
        model=model if isinstance(model, str) else None,
        **auth_kwargs,
    ) as result:
        async for event in _serialize_events(result):
            if user_auth and event.get("type") == "interrupt":
                raise AuthError(
                    "MCP_USER_AUTH_HITL_UNSUPPORTED",
                    "Request-user invocations do not support paused approvals.",
                    400,
                )
            await context.emit(event)
            if event.get("type") in ("message", "interrupt"):
                outputs.append(event)

    interrupted = bool(outputs and outputs[-1].get("type") == "interrupt")
    return {
        "output": [event["message"] if event["type"] == "message" else event for event in outputs],
        **({"session_id": session_id} if not user_auth or payload.get("session_id") else {}),
        "status": "interrupted" if interrupted else "completed",
    }


async def _serialize_events(result: RunResultStreaming) -> AsyncGenerator[dict, None]:
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent) and event.data.delta:
                yield {"type": "delta", "content": event.data.delta, "id": event.data.item_id}
        elif event.type == "run_item_stream_event":
            if message := _normalize_item(event.item):
                yield {"type": "message", "message": message}

    for item in result.interruptions:
        yield {"type": "interrupt", "id": item.call_id, "value": _approval_value(item)}


def _approval_value(item: ToolApprovalItem) -> dict:
    return {"action_requests": [{"name": item.tool_name, "args": _tool_args(item)}]}


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
