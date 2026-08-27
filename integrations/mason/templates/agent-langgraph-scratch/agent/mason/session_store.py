"""Conversation session store for the agent.

LangGraph persists conversation state through a **checkpointer** keyed by a ``thread_id`` (passed in
the run config), not through a session object. ``checkpointer()`` returns the checkpointer the agent
is built with, and ``thread_config(session_id)`` maps a session id onto that thread.

Default (no config): an in-memory checkpointer (``InMemorySaver``) — multi-turn history is preserved
within a single running process, no database. It does NOT survive restarts or span replicas.

Durable (``AGENT_SESSION_STORE`` set): a ``DatabricksMemorySaver`` bound to that managed Session
Store. The Session Store is backed by a service-managed Lakebase Postgres database; the saver runs
LangGraph's real ``PostgresSaver`` against it, so full graph state — including human-in-the-loop
pauses (pending writes + interrupts) — is durable across restarts and replicas. Setting the env var
is the only change; the agent code is identical.
"""

import os
from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

_SESSION_STORE_ENV = "AGENT_SESSION_STORE"

# Checkpoint tables live in their own schema so they never collide with the Session Store's own
# message-history tables (sessions / session_items) in the shared Lakebase database.
_CHECKPOINT_SCHEMA = "langgraph_checkpoints"


@lru_cache(maxsize=1)
def checkpointer() -> BaseCheckpointSaver:
    """The checkpointer the agent persists conversation state to.

    In-memory by default; a durable ``DatabricksMemorySaver`` when ``AGENT_SESSION_STORE`` names a
    managed Session Store. Cached so every request shares one saver — that's what makes multi-turn
    work in-process, and (for the durable saver) reuses a single Lakebase connection pool.
    """
    store = os.getenv(_SESSION_STORE_ENV)
    if store:
        return _durable_checkpointer(store)
    return InMemorySaver()


def _durable_checkpointer(session_store_name: str) -> BaseCheckpointSaver:
    """Open a Lakebase-backed ``CheckpointSaver`` for ``session_store_name`` (pool + tables ready).

    Delegates to ``databricks_langchain.checkpoint.CheckpointSaver`` — LangGraph's ``PostgresSaver``
    over the Session Store's service-managed Lakebase, with a connection pool that rotates the
    Lakebase OAuth token. Because it's a genuine Postgres checkpointer, HITL paused runs survive
    restarts, unlike the in-memory default.
    """
    # Lazy import: the durable path needs databricks-langchain[memory]; the base template runs
    # in-memory without it.
    from databricks_langchain.checkpoint import CheckpointSaver

    saver = CheckpointSaver(**_resolve_lakebase(session_store_name))
    saver.__enter__()  # open the connection pool (process-lived; no explicit close)
    saver.setup()  # create checkpoint tables if absent
    return saver


def _resolve_lakebase(session_store_name: str) -> dict:
    """Resolve a Session Store name to the Lakebase kwargs for ``CheckpointSaver``.

    The Session Store is backed by a service-managed Lakebase database; this maps the store name to
    that instance. The public ``GetSessionStore`` response does not yet expose the backing Lakebase
    instance — that resolver is a pending fast-follow in the Session Store API. Until it ships, set
    the Lakebase instance explicitly via ``LAKEBASE_INSTANCE_NAME``.
    """
    instance = os.getenv("LAKEBASE_INSTANCE_NAME")
    if not instance:
        raise NotImplementedError(
            f"{_SESSION_STORE_ENV}={session_store_name!r} is set, but resolving a Session Store to "
            "its backing Lakebase instance is not yet available in the public API. As an interim, "
            "set LAKEBASE_INSTANCE_NAME to the store's Lakebase instance; once the Session Store API "
            "exposes its backend, this resolver will use the store name alone."
        )
    return {"instance_name": instance, "schema": _CHECKPOINT_SCHEMA}


def thread_config(session_id: str) -> dict:
    """Run config that anchors this request to ``session_id``'s conversation thread."""
    return {"configurable": {"thread_id": session_id}}
