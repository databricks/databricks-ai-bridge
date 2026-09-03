"""Conversation session store for the agent.

LangGraph persists conversation state through a **checkpointer** keyed by a ``thread_id`` (passed in
the run config), not through a session object. ``checkpointer()`` returns the checkpointer the agent
is built with, and ``thread_config(session_id)`` maps a session id onto that thread.

Default (no config): an in-memory checkpointer (``InMemorySaver``) — multi-turn history is preserved
within a single running process, no database. It does NOT survive restarts or span replicas.

Durable (``AGENT_SESSION_STORE`` set): a ``DatabricksSessionStoreSaver``. Instead of a database the
app connects to directly, it serializes each LangGraph checkpoint into ordered **session items** and
stores them through the managed Session Store REST API. Full graph state — including human-in-the-loop
pauses (pending writes + interrupts) — is durable across restarts and replicas, over RPCs only (no
Lakebase/Postgres connection). Setting the env var is the only change; the agent code is identical.

The saver is adapted from the first-party ``databricks_agent_client.langgraph`` prototype, over a
vendored REST client (``session_store_client.py``) so the template needs no unpublished dependency;
swap both for the published package when it lands.
"""

from __future__ import annotations

import base64
import builtins
import hashlib
import os
import random
from collections import defaultdict
from typing import Any, Iterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    SerializerProtocol,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.memory import InMemorySaver

from databricks_mason.runtime.session_store_client import Session, SessionStoreClient

_SESSION_STORE_ENV = "AGENT_SESSION_STORE"

# Discriminators stored inside each session item's `data` JSON.
_EVENT_CHECKPOINT = "checkpoint"
_EVENT_CHANNEL_DATA = "channel_data"
_EVENT_WRITES = "writes"

# Sentinel for a channel that has a version bump but no materialized value yet.
_EMPTY_CHANNEL_VALUE = "__databricks_empty_channel__"

# Fetch items oldest-first so replaying them reconstructs state in write order.
_ORDER_BY = "create_time asc"

# One saver per process, opened lazily on first use and shared thereafter — that's what makes
# multi-turn work in-process (and, for the durable saver, reuses one client/session cache).
_saver: BaseCheckpointSaver | None = None


def checkpointer() -> BaseCheckpointSaver:
    """The checkpointer the agent persists conversation state to (built once, then shared).

    In-memory by default; a durable ``DatabricksSessionStoreSaver`` when ``AGENT_SESSION_STORE`` names
    a managed Session Store.
    """
    global _saver
    if _saver is None:
        store = os.getenv(_SESSION_STORE_ENV)
        _saver = DatabricksSessionStoreSaver(store) if store else InMemorySaver()
    return _saver


def thread_config(session_id: str, actor: str | None = None) -> dict:
    """Run config that anchors this request to ``session_id``'s conversation thread.

    Includes ``actor_id`` because the durable saver maps it onto the Session's actor; it's ignored by
    the in-memory default. ``actor`` partitions the durable store — the caller supplies it (typically
    the signed-in user), so each user's threads stay separate; it defaults to ``session_id`` (one
    actor per conversation).
    """
    return {"configurable": {"thread_id": session_id, "actor_id": actor or session_id}}


class DatabricksSessionStoreSaver(BaseCheckpointSaver[str]):
    """LangGraph ``BaseCheckpointSaver`` over the Databricks Session Store REST API.

    A single ``put`` fans out into several session items so large, immutable channel values are stored
    once per version rather than copied into every checkpoint: one ``checkpoint`` item (the checkpoint
    dict minus ``channel_values`` + metadata), one ``channel_data`` item per changed channel, and
    ``writes`` items (via ``put_writes``). Each item's ``data`` is plain JSON with an ``event_type``
    discriminator, because the Session Store treats item ``data`` as opaque JSON. Both ``thread_id``
    and ``actor_id`` are required in the run config's ``configurable`` section.
    """

    def __init__(
        self,
        session_store_name: str,
        *,
        client: Optional[SessionStoreClient] = None,
        workspace_client: Optional[Any] = None,
        serde: Optional[SerializerProtocol] = None,
    ) -> None:
        super().__init__(serde=serde)
        if not session_store_name:
            raise ValueError("session_store_name is required")
        self._client = (client or SessionStoreClient(workspace_client)).set_session_store(
            session_store_name
        )
        # Cache resolved Session objects per derived session_id to avoid re-resolving every put/get.
        self._sessions: dict[str, Session] = {}

    # ----- write path ---------------------------------------------------------

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Persist a checkpoint and its changed channel values."""
        thread_id, actor_id, checkpoint_ns = _parse_config(config)
        session = self._resolve_session(thread_id, actor_id, checkpoint_ns)

        checkpoint_copy = dict(checkpoint)
        channel_values = checkpoint_copy.pop("channel_values", {}) or {}

        items: list[dict[str, Any]] = [
            self._channel_data_item(
                channel, str(version), channel_values.get(channel, _EMPTY_CHANNEL_VALUE)
            )
            for channel, version in new_versions.items()
        ]
        items.append(
            self._checkpoint_item(
                checkpoint_id=checkpoint["id"],
                checkpoint_data=checkpoint_copy,
                metadata=dict(get_checkpoint_metadata(config, metadata)),  # type: ignore[arg-type]
                parent_checkpoint_id=get_checkpoint_id(config),
            )
        )
        self._client.append_items(session, items=items)

        return {
            "configurable": {
                "thread_id": thread_id,
                "actor_id": actor_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Persist pending intermediate writes for a checkpoint."""
        thread_id, actor_id, checkpoint_ns = _parse_config(config)
        checkpoint_id = get_checkpoint_id(config)
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required in config for put_writes")

        session = self._resolve_session(thread_id, actor_id, checkpoint_ns)
        item = {
            "event_type": _EVENT_WRITES,
            "checkpoint_id": checkpoint_id,
            "writes": [
                {
                    "task_id": task_id,
                    "task_path": task_path,
                    "channel": channel,
                    "value": self._serialize(value),
                }
                for channel, value in writes
            ],
        }
        self._client.append_items(session, items=[item])

    def delete_thread(self, thread_id: str, actor_id: str = "") -> None:
        """Remove all checkpoints and writes for a thread (via ClearSessionItems)."""
        session = self._resolve_session(thread_id, actor_id, "")
        self._client.clear_items(session)

    # ----- read path ----------------------------------------------------------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Return the checkpoint named in ``config``, or the latest one."""
        thread_id, actor_id, checkpoint_ns = _parse_config(config)
        session = self._resolve_session(thread_id, actor_id, checkpoint_ns)
        checkpoints, writes_by_ckpt, channel_data = self._read(session)
        if not checkpoints:
            return None

        wanted = get_checkpoint_id(config)
        if wanted:
            event = checkpoints.get(wanted)
            if event is None:
                return None
        else:
            event = checkpoints[max(checkpoints)]

        return self._build_tuple(
            event,
            writes_by_ckpt.get(event["checkpoint_id"], []),
            channel_data,
            thread_id,
            actor_id,
            checkpoint_ns,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: RunnableConfig | None = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """Yield stored checkpoints newest-first, honoring ``before``/``limit``."""
        if config is None:
            raise ValueError("config with a thread_id is required to list checkpoints")
        thread_id, actor_id, checkpoint_ns = _parse_config(config)
        session = self._resolve_session(thread_id, actor_id, checkpoint_ns)
        checkpoints, writes_by_ckpt, channel_data = self._read(session)

        before_id = get_checkpoint_id(before) if before else None
        count = 0
        for checkpoint_id in sorted(checkpoints, reverse=True):
            if before_id and checkpoint_id >= before_id:
                continue
            if limit is not None and count >= limit:
                break
            yield self._build_tuple(
                checkpoints[checkpoint_id],
                writes_by_ckpt.get(checkpoint_id, []),
                channel_data,
                thread_id,
                actor_id,
                checkpoint_ns,
            )
            count += 1

    def get_next_version(self, current: Optional[Any], channel: Optional[str] = None) -> str:
        """Monotonic version string, matching LangGraph's default scheme."""
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(str(current).split(".")[0])
        return f"{current_v + 1:032}.{random.random():016}"

    # ----- async wrappers (the Session Store client is synchronous) -----------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        return await _run_sync(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: Optional[dict[str, Any]] = None,
        before: RunnableConfig | None = None,
        limit: Optional[int] = None,
    ):
        for item in await _run_sync(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        ):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await _run_sync(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return await _run_sync(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str, actor_id: str = "") -> None:
        await _run_sync(self.delete_thread, thread_id, actor_id)

    # ----- internals ----------------------------------------------------------

    def _resolve_session(self, thread_id: str, actor_id: str, checkpoint_ns: str) -> Session:
        session_id = _session_id(thread_id, checkpoint_ns)
        cached = self._sessions.get(session_id)
        if cached is not None:
            return cached
        try:
            session = self._client.get_session(session_id=session_id)
        except tuple(_not_found_errors()) as _:  # type: ignore[misc]
            session = self._client.create_session(
                actor_id=actor_id,
                session_id=session_id,
                metadata={"langgraph_thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
            )
        self._sessions[session_id] = session
        return session

    def _read(
        self, session: Session
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, builtins.list[dict[str, Any]]],
        dict[tuple[str, str], Any],
    ]:
        """List items and group them into checkpoints, writes, and channel data."""
        checkpoints: dict[str, dict[str, Any]] = {}
        writes_by_ckpt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        channel_data: dict[tuple[str, str], Any] = {}

        for item in self._client.list_items(session, order_by=_ORDER_BY):
            data = item.data
            event_type = data.get("event_type")
            if event_type == _EVENT_CHECKPOINT:
                checkpoints[data["checkpoint_id"]] = data
            elif event_type == _EVENT_WRITES:
                writes_by_ckpt[data["checkpoint_id"]].extend(data["writes"])
            elif event_type == _EVENT_CHANNEL_DATA and data["value"] != _EMPTY_CHANNEL_VALUE:
                channel_data[(data["channel"], data["version"])] = self._deserialize(data["value"])
        return checkpoints, writes_by_ckpt, channel_data

    def _build_tuple(
        self,
        checkpoint_event: dict[str, Any],
        writes: builtins.list[dict[str, Any]],
        channel_data: dict[tuple[str, str], Any],
        thread_id: str,
        actor_id: str,
        checkpoint_ns: str,
    ) -> CheckpointTuple:
        checkpoint = dict(self._deserialize(checkpoint_event["checkpoint_data"]))
        channel_values: dict[str, Any] = {}
        for channel, version in checkpoint.get("channel_versions", {}).items():
            key = (channel, str(version))
            if key in channel_data:
                channel_values[channel] = channel_data[key]
        checkpoint["channel_values"] = channel_values

        pending_writes = [
            (w["task_id"], w["channel"], self._deserialize(w["value"])) for w in writes
        ]

        parent_config = None
        if checkpoint_event.get("parent_checkpoint_id"):
            parent_config = _config(
                thread_id, actor_id, checkpoint_ns, checkpoint_event["parent_checkpoint_id"]
            )

        return CheckpointTuple(
            config=_config(thread_id, actor_id, checkpoint_ns, checkpoint_event["checkpoint_id"]),
            checkpoint=checkpoint,  # type: ignore[arg-type]
            metadata=self._deserialize(checkpoint_event["metadata"]),
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def _checkpoint_item(
        self,
        *,
        checkpoint_id: str,
        checkpoint_data: dict[str, Any],
        metadata: dict[str, Any],
        parent_checkpoint_id: Optional[str],
    ) -> dict[str, Any]:
        return {
            "event_type": _EVENT_CHECKPOINT,
            "checkpoint_id": checkpoint_id,
            "checkpoint_data": self._serialize(checkpoint_data),
            "metadata": self._serialize(metadata),
            "parent_checkpoint_id": parent_checkpoint_id,
        }

    def _channel_data_item(self, channel: str, version: str, value: Any) -> dict[str, Any]:
        stored = value if value == _EMPTY_CHANNEL_VALUE else self._serialize(value)
        return {
            "event_type": _EVENT_CHANNEL_DATA,
            "channel": channel,
            "version": version,
            "value": stored,
        }

    def _serialize(self, value: Any) -> dict[str, str]:
        """Serialize via the LangGraph serde into JSON-safe ``{type, data}``."""
        type_tag, blob = self.serde.dumps_typed(value)
        return {"type": type_tag, "data": base64.b64encode(blob).decode("utf-8")}

    def _deserialize(self, serialized: dict[str, str]) -> Any:
        return self.serde.loads_typed((serialized["type"], base64.b64decode(serialized["data"])))


# ----- module-level helpers --------------------------------------------------


def _parse_config(config: RunnableConfig | dict[str, Any] | None) -> tuple[str, str, str]:
    configurable = (config or {}).get("configurable", {}) if config else {}
    thread_id = configurable.get("thread_id")
    actor_id = configurable.get("actor_id")
    if not thread_id:
        raise ValueError("config must contain configurable.thread_id")
    if not actor_id:
        raise ValueError("config must contain configurable.actor_id")
    return thread_id, actor_id, configurable.get("checkpoint_ns", "") or ""


def _session_id(thread_id: str, checkpoint_ns: str) -> str:
    """Deterministic session id for a (thread_id, checkpoint_ns) pair.

    With a namespace we hash the pair so the id stays within the store's id constraints and never
    collides with the bare ``thread_id`` used for the default namespace.
    """
    if checkpoint_ns:
        return hashlib.sha256(f"{thread_id}\x1f{checkpoint_ns}".encode()).hexdigest()
    return thread_id


def _config(
    thread_id: str, actor_id: str, checkpoint_ns: str, checkpoint_id: str
) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "actor_id": actor_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }


def _not_found_errors() -> tuple[type, ...]:
    """Exception types that mean 'session does not exist yet'."""
    try:
        from databricks.sdk.errors import NotFound

        return (NotFound,)
    except ImportError:  # pragma: no cover - SDK always present in practice
        return (_SessionNotFound,)


class _SessionNotFound(Exception):
    """Fallback 'not found' used only when databricks.sdk is unavailable."""


async def _run_sync(fn, *args):
    import asyncio

    loop = asyncio.get_running_loop()
    return (
        await loop.run_in_executor(None, fn, *args)
        if args
        else await loop.run_in_executor(None, fn)
    )
