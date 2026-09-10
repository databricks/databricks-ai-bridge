"""Long-term memory tools — opt-in, gated on ``AGENT_MEMORY_STORE``.

Exposes two Tool Runner tools — ``remember`` and ``recall`` — over the Databricks managed memory
store's ``agents/v1`` entries API, so facts persist across conversations. ``memory_tools(actor)``
returns the tools when a memory store is configured, else an empty list.

``actor`` is the identity whose memory these tools read and write; it is **closed over**, not a tool
argument, so the model can't set or spoof it. Pass a fixed value for one shared memory.
"""

from __future__ import annotations

from typing import Any

from anthropic import beta_tool

from databricks_mason.runtime.workspace import workspace_client

_AGENTS_V1 = "/api/agents/v1"


def _api():
    # Build the client lazily (needs workspace auth) so importing this module stays cheap.
    return workspace_client().api_client


def memory_tools(actor: str, store: str | None = None) -> list[Any]:
    """The long-term-memory tools for ``actor`` when a memory store is configured, else none.

    The store resolves ``store`` arg → ``AGENT_MEMORY_STORE`` env → the ``[memory_store]`` binding in
    agent.toml (`mason memory bind`) → none (no tools). ``actor`` partitions the store; it is captured
    in the tools' closures (not exposed to the model).
    """
    from databricks_mason.runtime.tool_manifest import resolve_memory_store

    store = resolve_memory_store(store)
    if not store:
        return []
    store_path = f"{_AGENTS_V1}/memory-stores/{store}"

    @beta_tool
    def remember(fact: str, topic: str) -> str:
        """Persist a durable fact about the user in long-term memory.

        Args:
            fact: The fact to remember.
            topic: A short topic to file the fact under.
        """
        _api().do(
            "POST",
            f"{store_path}/entries",
            body={"actor_id": actor, "path": f"/{topic}/{fact[:8]}.md", "content": fact},
        )
        return "stored"

    @beta_tool
    def recall(query: str) -> str:
        """Search the user's long-term memory for facts relevant to the query.

        Args:
            query: What to search the user's long-term memory for.
        """
        data = _api().do(
            "POST",
            f"{store_path}/entries:search",
            body={"actor_id": actor, "query": query, "limit": 5},
        )
        entries = data.get("managed_memory_entries") or []
        return "\n".join(f"- {e.get('content')}" for e in entries) or "No relevant memories."

    return [remember, recall]
