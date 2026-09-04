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

from typing import Any, Optional

from agents import SQLiteSession, TResponseInputItem
from agents.memory import SessionABC

from databricks_mason.runtime.session_store_client import Session as _StoreSession
from databricks_mason.runtime.session_store_client import SessionStoreClient

# Fetch items oldest-first so the transcript replays in write order.
_ORDER_BY = "create_time asc"

# One in-process Session per session id, built lazily and shared — that's what makes multi-turn work
# in-process (the durable store needs no cache; each call resolves the same REST-backed session).
_local_sessions: dict[str, _SanitizingSession] = {}


def _prune_dangling_tool_calls(
    items: list[TResponseInputItem],
) -> list[TResponseInputItem]:
    """Drop tool-call items that would replay with an unmatched ``call_id``.

    When a run interrupts for human approval, the Agents SDK persists the ``function_call`` item
    immediately but writes its matching ``function_call_output`` only once the run resumes. If that
    resume never lands cleanly — the user declines, or the paused (in-process) ``RunState`` is lost to
    a restart or another replica so the next prompt starts a fresh turn — the transcript keeps the
    ``function_call`` with no output. Replaying that makes the Responses API reject the whole request
    with BAD_REQUEST for the dangling ``call_id``. So keep a ``function_call`` only if its output was
    also stored, and a ``function_call_output`` only if its call was — every other item passes
    through untouched. (A plain tool exception does NOT trigger this: the SDK stores an error output
    as a matched pair.)
    """
    call_ids = {
        item["call_id"]
        for item in items
        if isinstance(item, dict) and item.get("type") == "function_call" and item.get("call_id")
    }
    output_ids = {
        item["call_id"]
        for item in items
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and item.get("call_id")
    }
    kept: list[TResponseInputItem] = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            if item.get("call_id") not in output_ids:
                continue
        elif isinstance(item, dict) and item.get("type") == "function_call_output":
            if item.get("call_id") not in call_ids:
                continue
        kept.append(item)
    return kept


class _SanitizingSession(SessionABC):
    """Wrap an Agents SDK ``Session`` and prune dangling tool calls when its history is replayed.

    ``get_items`` is the only read the SDK makes when rebuilding a run's input, so pruning there keeps
    a transcript left inconsistent by an earlier failed run (see ``_prune_dangling_tool_calls``) from
    poisoning the next turn. Writes and rollback pass straight through to the wrapped session.
    """

    def __init__(self, inner: SessionABC) -> None:
        self._inner = inner
        self.session_id = getattr(inner, "session_id", None)

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        # Prune before applying `limit` so a dropped call/output can't leave a partial pair at the
        # window edge, then take the last `limit` items in write order.
        items = _prune_dangling_tool_calls(await self._inner.get_items())
        return items[-limit:] if limit is not None else items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        await self._inner.add_items(items)

    async def pop_item(self) -> TResponseInputItem | None:
        return await self._inner.pop_item()

    async def clear_session(self) -> None:
        await self._inner.clear_session()


def session_store(
    session_id: str, actor: str | None = None, store: str | None = None
) -> SessionABC:
    """The Session the agent persists conversation state to for ``session_id``.

    In-memory ``SQLiteSession`` by default; a durable ``DatabricksSessionStore`` when a managed store
    is configured. The store name resolves ``store`` arg → ``AGENT_SESSION_STORE`` env → the
    ``[session_store]`` binding in agent.toml (`mason sessions bind`) → none. ``actor`` partitions the
    durable store — the caller supplies it (typically the signed-in user), so each user's transcripts
    stay separate; it defaults to ``session_id`` and is ignored by the in-memory store.
    """
    from databricks_mason.runtime.tool_manifest import resolve_session_store

    store = resolve_session_store(store)
    if store:
        durable = DatabricksSessionStore(session_id, store, actor_id=actor or session_id)
        return _SanitizingSession(durable)
    cached = _local_sessions.get(session_id)
    if cached is None:
        # ":memory:" — process-local, non-durable; wrapped so a failed run's dangling tool call can't
        # break the next turn within the process either.
        cached = _SanitizingSession(SQLiteSession(session_id))
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
