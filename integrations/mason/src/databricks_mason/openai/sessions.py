"""Conversation session store for the agent.

The OpenAI Agents SDK persists conversation state through a **Session** — an object with
``get_items``/``add_items``/``pop_item``/``clear_session`` over a list of Responses input items.
``Runner.run(..., session=session_store(session_id))`` reads prior turns from it and appends the new
ones. ``session_store(session_id)`` returns the Session for a given conversation.

Default (no config): an in-process ``SQLiteSession`` backed by ``:memory:``, cached per session id —
multi-turn history is preserved within a single running process, no database. It does NOT survive
restarts or span replicas.

Durable (``AGENT_SESSION_STORE`` set): a ``DatabricksSessionStore``. Instead of a database the app
connects to directly, it stores each Responses item as one **session item** through the managed
Session Store REST API (over RPCs only — no Lakebase/Postgres connection), so the transcript is
durable across restarts and replicas. Setting the env var is the only change; the agent code is
identical.

Note — unlike the LangGraph checkpointer, a Session persists only the conversation transcript, not
paused human-in-the-loop run state. Durable HITL would need the paused ``RunState`` stashed
separately; ``agent.py`` keeps pending runs in-process only (see ``PendingRuns``).

The durable store uses a vendored REST client (``session_store_client.py``) so the template needs no
unpublished dependency; swap it for the published package when it lands.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from agents import SQLiteSession, TResponseInputItem
from agents.memory import SessionABC

from databricks_mason.runtime.session_store_client import Session as _StoreSession
from databricks_mason.runtime.session_store_client import SessionStoreClient

_SESSION_STORE_ENV = "AGENT_SESSION_STORE"
_SESSION_ACTOR_ENV = "AGENT_SESSION_ACTOR_ID"

# Fetch items oldest-first so the transcript replays in write order.
_ORDER_BY = "create_time asc"

# One in-process Session per session id, built lazily and shared — that's what makes multi-turn work
# in-process (the durable store needs no cache; each call resolves the same REST-backed session).
_local_sessions: dict[str, SQLiteSession] = {}


def _session_actor(session_id: str) -> str:
    """Actor partition for the durable store. Defaults to the session id (one actor per session)."""
    return os.getenv(_SESSION_ACTOR_ENV) or session_id


def session_store(session_id: str) -> SessionABC:
    """The Session the agent persists conversation state to for ``session_id``.

    In-memory ``SQLiteSession`` by default; a durable ``DatabricksSessionStore`` when
    ``AGENT_SESSION_STORE`` names a managed Session Store.
    """
    store = os.getenv(_SESSION_STORE_ENV)
    if store:
        return DatabricksSessionStore(session_id, store, actor_id=_session_actor(session_id))
    cached = _local_sessions.get(session_id)
    if cached is None:
        cached = SQLiteSession(session_id)  # ":memory:" — process-local, non-durable
        _local_sessions[session_id] = cached
    return cached


class DatabricksSessionStore(SessionABC):
    """An Agents SDK ``Session`` over the Databricks Session Store REST API.

    Each Responses input item is stored as one Session Store item whose ``data`` is the item dict
    (opaque JSON to the store). ``get_items`` replays them in write order; ``pop_item`` and
    ``clear_session`` are supported for the SDK's retry rollback path. The session is created on first
    use (get-or-create), keyed by the conversation's ``session_id`` under the configured actor.
    """

    def __init__(
        self,
        session_id: str,
        session_store_name: str,
        *,
        actor_id: str,
        client: Optional[SessionStoreClient] = None,
        workspace_client: Optional[Any] = None,
    ) -> None:
        if not session_store_name:
            raise ValueError("session_store_name is required")
        self.session_id = session_id
        self._actor_id = actor_id
        self._client = (client or SessionStoreClient(workspace_client)).set_session_store(
            session_store_name
        )
        self._session: _StoreSession | None = None

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        items = await _run_sync(self._read_items)
        return items[-limit:] if limit is not None else items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        if not items:
            return
        await _run_sync(lambda: self._client.append_items(self._resolve(), items=list(items)))

    async def pop_item(self) -> TResponseInputItem | None:
        # The REST store has no pop; re-list, clear, and re-append all but the last. Only used on the
        # SDK's best-effort retry rollback, so the extra round-trip is acceptable and rare.
        def _pop() -> TResponseInputItem | None:
            items = self._read_items()
            if not items:
                return None
            session = self._resolve()
            self._client.clear_items(session)
            if items[:-1]:
                self._client.append_items(session, items=items[:-1])
            return items[-1]

        return await _run_sync(_pop)

    async def clear_session(self) -> None:
        await _run_sync(lambda: self._client.clear_items(self._resolve()))

    # ----- internals ----------------------------------------------------------

    def _resolve(self) -> _StoreSession:
        if self._session is not None:
            return self._session
        try:
            self._session = self._client.get_session(session_id=self.session_id)
        except tuple(_not_found_errors()) as _:  # type: ignore[misc]
            self._session = self._client.create_session(
                actor_id=self._actor_id,
                session_id=self.session_id,
                metadata={"client": "mason-openai-agent"},
            )
        return self._session

    def _read_items(self) -> list[TResponseInputItem]:
        return [item.data for item in self._client.list_items(self._resolve(), order_by=_ORDER_BY)]


def _not_found_errors() -> tuple[type, ...]:
    """Exception types that mean 'session does not exist yet'."""
    try:
        from databricks.sdk.errors import NotFound

        return (NotFound,)
    except ImportError:  # pragma: no cover - SDK always present in practice
        return (_SessionNotFound,)


class _SessionNotFound(Exception):
    """Fallback 'not found' used only when databricks.sdk is unavailable."""


async def _run_sync(fn):
    import asyncio

    return await asyncio.get_running_loop().run_in_executor(None, fn)
