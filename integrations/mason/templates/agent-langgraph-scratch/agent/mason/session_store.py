"""Conversation session store for the agent.

LangGraph persists conversation state through a **checkpointer** keyed by a ``thread_id`` (passed in
the run config), not through a session object. ``checkpointer()`` returns the checkpointer the agent
is built with, and ``thread_config(session_id)`` maps a session id onto that thread.

Default (no config): an in-memory checkpointer (``InMemorySaver``) — multi-turn history is preserved
within a single running process, no database. It does NOT survive restarts or span replicas.

Durable (``AGENT_SESSION_STORE`` set): a Lakebase-backed ``AsyncCheckpointSaver``. A managed Session
Store is provisioned into a service-managed Lakebase Postgres project; the saver runs LangGraph's real
``AsyncPostgresSaver`` against it, so full graph state — including human-in-the-loop pauses (pending
writes + interrupts) — is durable across restarts and replicas. Setting the env var is the only
change; the agent code is identical.

``checkpointer()`` is async because the agent drives the graph with ``astream`` (async), and the
durable saver must expose the async checkpoint methods (``aget_tuple`` / ``aput`` / ``aput_writes``)
that async execution calls — the *sync* ``CheckpointSaver`` leaves those unimplemented, so it can't
back an async graph.
"""

import os

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

_SESSION_STORE_ENV = "AGENT_SESSION_STORE"

# Managed Session Stores are provisioned into one shared per-workspace Lakebase project, on its
# default "production" branch (autoscaling). These match the service's provisioning convention, so
# the durable checkpointer is derivable from AGENT_SESSION_STORE alone — no extra connection config.
_SHARED_PROJECT = "databricks-internal-lakebase-agent-session-store"
_DEFAULT_BRANCH = "production"

# Checkpoint tables live in their own schema so they never collide with the Session Store's own
# message-history tables (sessions / session_items).
_CHECKPOINT_SCHEMA = "langgraph_checkpoints"

# One saver per process, opened lazily on first use and shared thereafter — that's what makes
# multi-turn work in-process and (for the durable saver) reuses a single Lakebase connection pool.
_saver: BaseCheckpointSaver | None = None


async def checkpointer() -> BaseCheckpointSaver:
    """The checkpointer the agent persists conversation state to (opened once, then shared).

    In-memory by default; a durable Lakebase-backed ``AsyncCheckpointSaver`` when
    ``AGENT_SESSION_STORE`` names a managed Session Store.
    """
    global _saver
    if _saver is None:
        _saver = await _durable_checkpointer() if os.getenv(_SESSION_STORE_ENV) else InMemorySaver()
    return _saver


async def _durable_checkpointer() -> BaseCheckpointSaver:
    """A Lakebase-backed ``AsyncCheckpointSaver`` for the store's shared project (pool + tables ready).

    Delegates to ``databricks_langchain.checkpoint.AsyncCheckpointSaver`` — LangGraph's
    ``AsyncPostgresSaver`` over the managed Session Store's Lakebase project, with a pool that rotates
    the Lakebase OAuth token. Being a genuine async Postgres checkpointer, it works with the async
    graph and keeps HITL paused runs durable across restarts.

    NOTE: the saver connects to the shared project's *default* Lakebase database, not the per-store
    database named after ``AGENT_SESSION_STORE``. The Session Store's message items live in that
    per-store database; the current ``databricks-langchain`` connection pool hardcodes the default
    database name and can't target another, so checkpoints land alongside — in the same project, a
    different database. When the pool allows overriding the database, point this at the per-store one
    to co-locate checkpoints with the session's items (and match its access grants).
    """
    # Imported here (not at module load) so the in-memory default path doesn't pull in the Lakebase
    # connection-pool machinery it never uses.
    from databricks_langchain.checkpoint import AsyncCheckpointSaver

    # Uses default SDK auth: in a deployed Databricks App that resolves to the app's service
    # principal (which the store's Lakebase project grants). Do NOT pass a user-scoped
    # WorkspaceClient here — end-user credentials have no grant on the store's database.
    saver = AsyncCheckpointSaver(
        project=_SHARED_PROJECT,
        branch=_DEFAULT_BRANCH,
        schema=_CHECKPOINT_SCHEMA,
    )
    try:
        await saver.__aenter__()  # open the async connection pool (process-lived; no explicit close)
        await saver.setup()  # create checkpoint tables if absent
    except Exception as e:
        # The store's Lakebase project is granted to the deployed app's service principal (via the
        # app resource binding), not to individual human users. Running locally under your own
        # credentials, that grant doesn't exist, so Postgres rejects the connection. Surface that
        # rather than a raw driver error — the durable path is meant to run in the deployed app.
        if "authentication failed" in str(e).lower():
            raise RuntimeError(
                f"Connected to the Lakebase project for {_SESSION_STORE_ENV!r} but Postgres auth "
                "was rejected. The managed Session Store grants access to the deployed app's "
                "service principal, not to human users — so the durable checkpointer works in the "
                "deployed app but not under local user credentials. For local dev, leave "
                f"{_SESSION_STORE_ENV} unset to use the in-process checkpointer."
            ) from e
        raise
    return saver


def thread_config(session_id: str) -> dict:
    """Run config that anchors this request to ``session_id``'s conversation thread."""
    return {"configurable": {"thread_id": session_id}}
