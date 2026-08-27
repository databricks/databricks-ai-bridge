"""Long-term memory tools — opt-in, gated on ``AGENT_MEMORY_STORE``.

Unlike the session store (short-term transcript for one conversation), long-term memory is exposed
to the model as two tools — ``remember`` and ``recall`` — over the Databricks managed memory store's
``agents/v1`` entries API. Facts persist across conversations.

``memory_tools()`` returns the tools when ``AGENT_MEMORY_STORE`` is set, else an empty list, so the
model never sees them when memory is unconfigured. ``create_agent_graph`` composes them into its
tool list. This is a stand-in for a future ``databricks-langchain`` memory helper — when the SDK
provides one, swap the import in ``agent.py`` and delete this file.

Memory entries are per-actor. This uses the store name as the actor id, giving the agent one shared
long-term memory; change ``_actor_id`` to scope per user (e.g. from request context) if needed.
"""

import os
from functools import lru_cache

from databricks.sdk import WorkspaceClient
from langchain_core.tools import BaseTool, tool

_AGENTS_V1 = "/api/agents/v1"
_PROFILE_CONFLICT_ENV = ("DATABRICKS_CONFIG_PROFILE", "DATABRICKS_HOST", "DATABRICKS_TOKEN")


def _actor_id() -> str:
    return os.getenv("AGENT_MEMORY_ACTOR_ID", "agent")


def _store_path() -> str:
    store = os.environ["AGENT_MEMORY_STORE"].strip().strip("/")
    store = store.removeprefix("memory-stores/")
    return f"{_AGENTS_V1}/memory-stores/{store}"


@lru_cache(maxsize=1)
def _workspace_client() -> WorkspaceClient:
    # Build the client lazily (needs workspace auth) so importing this module stays cheap.
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    if not profile:
        return WorkspaceClient()
    inherited = {name: os.environ.pop(name, None) for name in _PROFILE_CONFLICT_ENV}
    try:
        return WorkspaceClient(profile=profile)
    finally:
        for name, value in inherited.items():
            if value is not None:
                os.environ[name] = value


def _do(method: str, path: str, *, body: dict) -> dict:
    workspace = _workspace_client()
    headers = None
    if workspace.config.workspace_id:
        headers = {"X-Databricks-Workspace-Id": str(workspace.config.workspace_id)}
    return workspace.api_client.do(method, path, body=body, headers=headers)


@tool
def remember(fact: str, topic: str) -> str:
    """Persist a durable fact about the user in long-term memory."""
    _do(
        "POST",
        f"{_store_path()}/entries",
        body={"actor_id": _actor_id(), "path": f"/{topic}/{fact[:8]}.md", "content": fact},
    )
    return "stored"


@tool
def recall(query: str) -> str:
    """Search the user's long-term memory for facts relevant to the query."""
    data = _do(
        "POST",
        f"{_store_path()}/entries:search",
        body={"actor_id": _actor_id(), "query": query, "limit": 5},
    )
    entries = data.get("managed_memory_entries") or []
    return "\n".join(f"- {e.get('content')}" for e in entries) or "No relevant memories."


def memory_tools() -> list[BaseTool]:
    """The long-term-memory tools when ``AGENT_MEMORY_STORE`` is set, else none."""
    return [remember, recall] if os.getenv("AGENT_MEMORY_STORE") else []
