"""Read message-oriented LangGraph history without constructing a live agent."""

from typing import Any


async def read_history(session_id: str, actor: str) -> dict[str, Any]:
    """Restore messages and interrupts from the configured checkpoint backend.

    User-authenticated callers supply invocation's private session and actor identifiers.
    Supports the generated message-oriented StateGraph and its in-memory or managed saver;
    checkpoint decoding and pending-write reduction remain framework-specific.
    """
    from langgraph.graph.message import add_messages

    from databricks_agentkit.langgraph.session_store import history_checkpoint, thread_config

    saved = await history_checkpoint(thread_config(session_id, actor))
    messages = []
    interrupts = []
    if saved is not None:
        messages = add_messages([], saved.checkpoint["channel_values"].get("messages", []))
        # Successful task outputs can coexist with paused/failed work in the latest checkpoint.
        for _task_id, channel, value in saved.pending_writes or []:
            if channel == "messages":
                messages = add_messages(messages, value)
            elif channel == "__interrupt__":
                interrupts.extend({"id": item.id, "value": item.value} for item in value)
    items = []
    for index, message in enumerate(messages):
        data = message.model_dump() if hasattr(message, "model_dump") else message
        items.append(
            {
                "item_id": str(getattr(message, "id", None) or index),
                "data": data if isinstance(data, dict) else {"content": str(data)},
            }
        )
    return {"session_id": session_id, "session_items": items, "interrupts": interrupts}
