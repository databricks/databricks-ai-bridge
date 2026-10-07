"""REST-only transport replacement shared by framework history regressions."""

import json
from collections.abc import Iterator, Sequence
from typing import Any

from databricks.sdk.errors import NotFound

from databricks_agentkit.runtime.session_store_client import (
    Session,
    SessionItem,
    SessionStoreClient,
)


class SessionStoreTransport(SessionStoreClient):
    """Keep native saver/session behavior while storing REST payloads locally."""

    def __init__(self):
        self.items = {}
        self.sessions = {}
        self._store_name = None

    def get_session(self, *, session_id: str) -> Session:
        if session_id not in self.sessions:
            raise NotFound("Session does not exist")
        return self.sessions[session_id]

    def create_session(
        self, *, actor_id: str, session_id: str | None = None, metadata=None
    ) -> Session:
        assert self._store_name and session_id
        session = Session(self._store_name, session_id, actor_id, metadata or {})
        self.sessions[session_id] = session
        return session

    def append_items(self, session: Session, *, items: Sequence[Any]) -> None:
        self.items.setdefault(session.session_id, []).extend(json.loads(json.dumps(items)))

    def list_items(self, session: Session, *, order_by: str | None = None) -> Iterator[SessionItem]:
        assert order_by == "create_time asc"
        for index, item in enumerate(self.items.get(session.session_id, [])):
            yield SessionItem(str(index), item)
