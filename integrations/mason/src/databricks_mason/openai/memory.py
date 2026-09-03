"""Long-term memory tools — opt-in, gated on ``AGENT_MEMORY_STORE``.

Unlike the session store (short-term transcript for one conversation), long-term memory is exposed
to the model as two tools — ``remember`` and ``recall`` — over the Databricks managed memory store's
``agents/v1`` entries API. Facts persist across conversations.

``memory_tools(actor)`` returns the tools when ``AGENT_MEMORY_STORE`` is set, else an empty list, so
the model never sees them when memory is unconfigured. The agent composes them into its tool list.
This is a stand-in for a future ``databricks-openai`` memory helper — when the SDK provides one, swap
the import in ``agent.py`` and delete this file.

Memory entries are per-actor. ``actor`` is the identity whose memory these tools read and write — the
caller (the agent server) supplies it, typically the signed-in user, so each user gets their own
long-term memory. It is **closed over**, not a tool argument: the model only ever sees ``remember``'s
and ``recall``'s own parameters and cannot set or spoof the actor. Pass a fixed value for one shared
memory (e.g. a team knowledge base).
"""

import os

from agents import FunctionTool, function_tool

from databricks_mason.runtime.workspace import workspace_client

_AGENTS_V1 = "/api/agents/v1"


def _store_path() -> str:
    return f"{_AGENTS_V1}/memory-stores/{os.environ['AGENT_MEMORY_STORE']}"


def _api():
    # Build the client lazily (needs workspace auth) so importing this module stays cheap.
    return workspace_client().api_client


def memory_tools(actor: str) -> list[FunctionTool]:
    """The long-term-memory tools for ``actor`` when ``AGENT_MEMORY_STORE`` is set, else none.

    ``actor`` partitions the memory store; it is captured in the tools' closures (not exposed to the
    model). Call this per request with the identity whose memory to use (e.g. the signed-in user).
    """
    if not os.getenv("AGENT_MEMORY_STORE"):
        return []

    @function_tool
    def remember(fact: str, topic: str) -> str:
        """Persist a durable fact about the user in long-term memory."""
        _api().do(
            "POST",
            f"{_store_path()}/entries",
            body={"actor_id": actor, "path": f"/{topic}/{fact[:8]}.md", "content": fact},
        )
        return "stored"

    @function_tool
    def recall(query: str) -> str:
        """Search the user's long-term memory for facts relevant to the query."""
        data = _api().do(
            "POST",
            f"{_store_path()}/entries:search",
            body={"actor_id": actor, "query": query, "limit": 5},
        )
        entries = data.get("managed_memory_entries") or []
        return "\n".join(f"- {e.get('content')}" for e in entries) or "No relevant memories."

    return [remember, recall]
