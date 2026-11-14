"""Native Microsoft Agent Framework history backed by the managed Session Store."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from typing import Any

from agent_framework import HistoryProvider, Message
from databricks.sdk.errors import AlreadyExists, NotFound, ResourceAlreadyExists

from databricks_agentkit.maf.messages import serialize_message
from databricks_agentkit.runtime.session_store_client import SessionStoreClient
from databricks_agentkit.runtime.tool_manifest import resolve_session_store

_local_messages: dict[tuple[str, str], list[dict[str, Any]]] = {}


class DatabricksHistoryProvider(HistoryProvider):
    """Store complete native messages, including tool calls/results, in write order.

    Create a provider per invocation with the actor chosen by the application. The actor is
    captured by the provider and never exposed as a model-controlled tool parameter.
    """

    def __init__(
        self, actor: str, store: str | None = None, *, client: SessionStoreClient | None = None
    ) -> None:
        super().__init__("databricks-session-store")
        self._actor = actor
        selected = resolve_session_store(store)
        self._client = (
            (client or SessionStoreClient()).set_session_store(selected) if selected else None
        )

    def _session(self, session_id: str):
        assert self._client is not None
        try:
            session = self._client.get_session(session_id=session_id)
        except NotFound:
            try:
                session = self._client.create_session(
                    actor_id=self._actor,
                    session_id=session_id,
                    metadata={"client": "agentbricks-maf"},
                )
            except (AlreadyExists, ResourceAlreadyExists):
                session = self._client.get_session(session_id=session_id)
        if session.actor_id != self._actor:
            raise ValueError("This session belongs to a different actor.")
        return session

    @staticmethod
    def _validate_session_id(session_id: str | None) -> str:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        return session_id

    async def get_messages(
        self, session_id: str | None, *, state: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Message]:
        session_id = self._validate_session_id(session_id)

        def read():
            if self._client is None:
                return copy.deepcopy(_local_messages.get((self._actor, session_id), []))
            return [
                item.data
                for item in self._client.list_items(
                    self._session(session_id), order_by="create_time asc"
                )
            ]

        return [Message.from_dict(data) for data in await asyncio.to_thread(read)]

    async def save_messages(
        self,
        session_id: str | None,
        messages: Sequence[Message],
        *,
        state: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        session_id = self._validate_session_id(session_id)
        serialized = [message.to_dict() for message in messages]
        if not serialized:
            return

        def append():
            if self._client is None:
                _local_messages.setdefault((self._actor, session_id), []).extend(
                    copy.deepcopy(serialized)
                )
            else:
                self._client.append_items(self._session(session_id), items=serialized)

        await asyncio.to_thread(append)


def session_store(actor: str, store: str | None = None) -> DatabricksHistoryProvider:
    """Build an actor-scoped HistoryProvider for Agent(context_providers=[...]).

    The store resolves explicit argument, environment, then agent.toml binding. Without a managed
    store, conversation history is retained only within this process.
    """
    return DatabricksHistoryProvider(actor, store)


async def read_history(session_id: str, actor: str | None = None) -> dict[str, Any]:
    """Return the persisted native transcript in the browser's common message shape."""
    messages = await session_store(actor or session_id).get_messages(session_id)
    return {
        "session_id": session_id,
        "session_items": [
            {
                "item_id": message.message_id or str(index),
                "data": serialize_message(message),
            }
            for index, message in enumerate(messages)
        ],
        "interrupts": [],
    }
