"""Read native Agents SDK history without constructing a live agent."""

from typing import Any


async def read_history(session_id: str, actor: str | None = None) -> dict[str, Any]:
    """Restore transcript items using invocation's native session backend.

    User-authenticated callers supply private session and actor identifiers. A paused OpenAI
    run is held separately in process, so this transcript reader never reports durable interrupts.
    """
    from databricks_agentkit.openai.sessions import session_store

    session = session_store(session_id, actor)
    items = []
    for index, message in enumerate(await session.get_items()):
        data = message if isinstance(message, dict) else {"content": str(message)}
        items.append({"item_id": str(data.get("id") or index), "data": data})
    return {"session_id": session_id, "session_items": items, "interrupts": []}
