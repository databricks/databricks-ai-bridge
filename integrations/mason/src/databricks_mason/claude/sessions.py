"""Conversation history for the agent.

The Anthropic Tool Runner is stateless: each turn you pass the full ``messages`` list and read back
the new ones. ``session_history(session_id, actor)`` returns a small store you ``load()`` prior
messages from and ``append()`` new ones to.

Default (no config): an in-process dict keyed by session id — multi-turn works within one process,
not across restarts/replicas. Durable (``AGENT_SESSION_STORE`` set): each message is stored as one
item through the managed Session Store REST API (no Lakebase/Postgres connection), durable across
restarts and replicas. Setting the env var is the only change; the agent code is identical.
"""

from __future__ import annotations

from typing import Any, Optional

from databricks_mason.runtime.session_store_client import Session as _StoreSession
from databricks_mason.runtime.session_store_client import SessionStoreClient

_ORDER_BY = "create_time asc"

# One in-process transcript per session id (the durable store needs no cache).
_local_history: dict[str, list[dict[str, Any]]] = {}


class _History:
    """Load/append a conversation transcript as a list of Anthropic message dicts."""

    def load(self) -> list[dict[str, Any]]:  # pragma: no cover - overridden
        raise NotImplementedError

    def append(self, messages: list[dict[str, Any]]) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


def session_history(
    session_id: str, actor: str | None = None, store: str | None = None
) -> _History:
    """The transcript store for ``session_id``: in-process by default, durable when a store is set.

    The store name resolves ``store`` arg → ``AGENT_SESSION_STORE`` env → the ``[session_store]``
    binding in agent.toml (`mason sessions bind`) → none. ``actor`` partitions the durable store.
    """
    from databricks_mason.runtime.tool_manifest import resolve_session_store

    store = resolve_session_store(store)
    if store:
        return _DurableHistory(session_id, store, actor or session_id)
    return _LocalHistory(session_id)


class _LocalHistory(_History):
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id

    def load(self) -> list[dict[str, Any]]:
        return list(_local_history.get(self._session_id, []))

    def append(self, messages: list[dict[str, Any]]) -> None:
        if messages:
            _local_history.setdefault(self._session_id, []).extend(messages)


class _DurableHistory(_History):
    """Transcript backed by the Databricks Session Store REST API (one item per message)."""

    def __init__(
        self,
        session_id: str,
        session_store_name: str,
        actor_id: str,
        client: Optional[SessionStoreClient] = None,
    ) -> None:
        self._session_id = session_id
        self._actor_id = actor_id
        self._client = (client or SessionStoreClient()).set_session_store(session_store_name)
        self._session: _StoreSession | None = None

    def load(self) -> list[dict[str, Any]]:
        return [item.data for item in self._client.list_items(self._resolve(), order_by=_ORDER_BY)]

    def append(self, messages: list[dict[str, Any]]) -> None:
        if messages:
            self._client.append_items(self._resolve(), items=list(messages))

    def _resolve(self) -> _StoreSession:
        if self._session is not None:
            return self._session
        try:
            self._session = self._client.get_session(session_id=self._session_id)
        except tuple(_not_found_errors()):
            self._session = self._client.create_session(
                actor_id=self._actor_id,
                session_id=self._session_id,
                metadata={"client": "mason-claude-agent"},
            )
        return self._session


def _not_found_errors() -> tuple[type, ...]:
    try:
        from databricks.sdk.errors import NotFound

        return (NotFound,)
    except ImportError:  # pragma: no cover - SDK always present in practice
        return (_SessionNotFound,)


class _SessionNotFound(Exception):
    """Fallback 'not found' used only when databricks.sdk is unavailable."""
