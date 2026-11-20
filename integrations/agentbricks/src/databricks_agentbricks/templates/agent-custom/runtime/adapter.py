"""Map the managed invocation protocol to plain messages and public AgentKit store APIs.

The native function receives session history and actor-scoped recalled memories.
"""

import asyncio
import copy
import os
from typing import Any

from agent.agent import run_agent

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit import (
    AgentKitClient,
    InvocationContext,
    Session,
    start_trace,
    workspace_client,
)

# Credential-free development fallback. Bind a Session Store for persistence across restarts.
_local_sessions: dict[str, tuple[str, list[dict[str, Any]]]] = {}
_RECORD_TYPE = "agentbricks.custom.turn.v1"


def _input(value: Any) -> dict[str, Any]:
    payload = {"messages": value} if isinstance(value, list) else value
    if not isinstance(payload, dict):
        raise ValueError("input must be an object or a list of user messages")
    if payload.keys() - {"messages", "actor", "remember", "memory_query"}:
        raise ValueError("supported input fields: messages, actor, remember, memory_query")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of user messages")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.keys() != {"role", "content"}
            or message.get("role") != "user"
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("each message must have role 'user' and non-empty string content")
    for field in ("actor", "remember", "memory_query"):
        if field in payload and (not isinstance(payload[field], str) or not payload[field].strip()):
            raise ValueError(f"{field} must be a non-empty string")
    if ("remember" in payload or "memory_query" in payload) and not os.getenv("AGENT_MEMORY_STORE"):
        raise ValueError("bind a Memory Store before using remember or memory_query")
    return copy.deepcopy(payload)


def _load(session_id: str, actor: str) -> tuple[list[dict[str, Any]], Session | None]:
    store_name = os.getenv("AGENT_SESSION_STORE")
    if not store_name:
        owner, records = _local_sessions.setdefault(session_id, (actor, []))
        if owner != actor:
            raise ValueError("the session belongs to another actor")
        return copy.deepcopy(records), None
    client = AgentKitClient(workspace_client())
    store = client.session_stores.get(store_name)
    try:
        session = store.get(session_id)
    except AgentCliError as exc:
        if exc.error_code != "NOT_FOUND":
            raise
        try:
            session = store.add(actor_id=actor, session_id=session_id)
        except AgentCliError as create_error:
            if create_error.error_code != "ALREADY_EXISTS":
                raise
            session = store.get(session_id)
    if session.actor_id != actor:
        raise ValueError("the session belongs to another actor")
    records = [
        item.data
        for item in session.list_items(order_by="create_time asc")
        if isinstance(item.data, dict) and item.data.get("type") == _RECORD_TYPE
    ]
    return records, session


def _append(session: Session | None, session_id: str, record: dict[str, Any]) -> None:
    if session is None:
        _local_sessions[session_id][1].append(copy.deepcopy(record))
    else:
        session.append_items([record])


def _memories(payload: dict[str, Any], actor: str, invocation_id: str) -> list[str]:
    store_id = os.getenv("AGENT_MEMORY_STORE")
    if not store_id:
        return []
    client = AgentKitClient(workspace_client())
    store = client.memory_stores.get(store_id)
    if "remember" in payload:
        # An invocation-specific path makes retrying this example write idempotent.
        # Omit session_id so an authenticated actor can recall the fact in a new conversation.
        store.put(
            actor_id=actor,
            path=f"/invocations/{invocation_id}.md",
            content=payload["remember"],
        )
    results = store.search(
        actor_id=actor,
        query=payload.get("memory_query", payload["messages"][-1]["content"]),
        limit=5,
    )
    return [result.memory.content for result in results if result.memory.content]


async def invoke(value: Any, context: InvocationContext) -> dict[str, Any]:
    payload = _input(value)
    session_id = context.session_id
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be provided as a top-level invocation field")
    # Runtime namespaces context.session_id when request-user authentication is enabled.
    # An explicit actor is only an application partition, never proof of identity.
    actor = payload.get("actor", session_id)
    if context.request_auth is not None and "actor" in payload:
        actor = context.request_auth.namespace("actor", actor)
    records, session = await asyncio.to_thread(_load, session_id, actor)
    history = [message for record in records for message in record["messages"]]
    history.extend(payload["messages"])
    with start_trace(name="custom_agent", inputs=payload, session_id=session_id):
        memories = await asyncio.to_thread(_memories, payload, actor, context.invocation_id)
        message = await run_agent(copy.deepcopy(history), memories=memories)
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or not isinstance(message.get("content"), str)
    ):
        raise ValueError("run_agent must return an assistant message with string content")
    output = {**message, "type": "ai", "id": context.invocation_id}
    response = {"output": [output], "status": "completed"}
    await asyncio.to_thread(
        _append,
        session,
        session_id,
        {
            "type": _RECORD_TYPE,
            "messages": [
                *payload["messages"],
                {"role": message["role"], "content": message["content"]},
            ],
        },
    )
    await context.emit(copy.deepcopy({"type": "message", "message": output}))
    return response
