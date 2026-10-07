"""REST-only transport replacement shared by framework history regressions."""

import json
from collections.abc import Iterator, Sequence
from typing import Any

from databricks_agentkit.runtime.session_store_client import (
    Session,
    SessionItem,
    SessionStoreClient,
)


class SessionStoreTransport(SessionStoreClient):
    """Keep native saver/session behavior while storing REST payloads locally."""

    def __init__(self):
        self.items = {}
        self._store_name = None

    def get_session(self, *, session_id: str) -> Session:
        return Session("test", session_id, "actor")

    def append_items(self, session: Session, *, items: Sequence[Any]) -> None:
        self.items.setdefault(session.session_id, []).extend(json.loads(json.dumps(items)))

    def list_items(self, session: Session, *, order_by: str | None = None) -> Iterator[SessionItem]:
        assert order_by == "create_time asc"
        for index, item in enumerate(self.items.get(session.session_id, [])):
            yield SessionItem(str(index), item)
