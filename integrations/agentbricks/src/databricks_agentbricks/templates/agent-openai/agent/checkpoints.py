"""The example's OpenAI checkpoints, stored separately from its conversation transcript."""

import asyncio
import copy
import hashlib
import json

from databricks.sdk.errors import AlreadyExists, NotFound, ResourceAlreadyExists

from databricks_agentkit.runtime.session_store_client import SessionStoreClient
from databricks_agentkit.runtime.tool_manifest import resolve_session_store

_local_items: dict[str, list[dict]] = {}


class Checkpoints:
    def __init__(self, session_id, actor, invocation_id, attempt, emit):
        self._key = hashlib.sha256(
            json.dumps(["openai-checkpoints", session_id, actor]).encode()
        ).hexdigest()
        self._session_id, self._actor = session_id, actor
        self._invocation_id, self._attempt, self._emit = invocation_id, attempt, emit
        store = resolve_session_store()
        self._client = SessionStoreClient().set_session_store(store) if store else None
        self._loaded = False
        self._snapshot = None
        self._generation = self._index = 0
        self._reserved = False

    async def load(self):
        if not self._loaded:
            items = await asyncio.to_thread(self._read)
            generations = [item["generation"] for item in items]
            own = [
                item["generation"] for item in items if item["invocation_id"] == self._invocation_id
            ]
            self._generation = max(own) if own else max(generations, default=0) + 1
            self._reserved = bool(own)
            checkpoints = [item for item in items if "state" in item]
            latest = max(checkpoints, key=lambda item: item["version"], default=None)
            self._snapshot = latest["state"] if latest else None
            self._index = max(
                (
                    item["version"][2]
                    for item in checkpoints
                    if item["version"][:2] == [self._generation, self._attempt]
                ),
                default=0,
            )
            self._loaded = True
        return copy.deepcopy(self._snapshot)

    async def save(self, state):
        await self.load()
        item = {"invocation_id": self._invocation_id, "generation": self._generation}
        if not self._reserved:
            # Reserve order durably before checking ownership. A late reservation from an old
            # worker contains no state, and its subsequent ownership check will reject it.
            await asyncio.to_thread(self._append, item)
            self._reserved = True
        self._index += 1
        item = {
            **item,
            "version": [self._generation, self._attempt, self._index],
            "state": copy.deepcopy(state),
        }
        # The runtime owns scheduling/fencing; the harness owns checkpoint contents and storage.
        await self._emit({"type": "checkpoint"})
        await asyncio.to_thread(self._append, item)
        self._snapshot = item["state"]

    def _read(self):
        if self._client is None:
            return copy.deepcopy(_local_items.get(self._key, []))
        try:
            session = self._client.get_session(session_id=self._key)
        except NotFound:
            return []
        return [item.data for item in self._client.list_items(session, order_by="create_time asc")]

    def _append(self, item):
        if self._client is None:
            _local_items.setdefault(self._key, []).append(copy.deepcopy(item))
            return
        try:
            session = self._client.get_session(session_id=self._key)
        except NotFound:
            try:
                session = self._client.create_session(
                    actor_id=self._actor,
                    session_id=self._key,
                    metadata={
                        "client": "agentbricks-openai-checkpoints",
                        "public_session_id": self._session_id,
                    },
                )
            except (AlreadyExists, ResourceAlreadyExists):
                session = self._client.get_session(session_id=self._key)
        self._client.append_items(session, items=[item])
