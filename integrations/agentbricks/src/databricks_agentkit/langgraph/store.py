"""LangGraph long-term memory store on Databricks managed memory.

``DatabricksMemoryStore`` is a LangGraph ``BaseStore`` over the Agent Bricks managed memory API,
reached through the AgentKit client. Pass it as ``store=`` to ``create_agent`` or
``StateGraph.compile`` and any LangGraph or LangMem code that uses the store gets memory that
outlives one conversation, with no Postgres connection, credential rotation or pool tuning::

    from databricks_agentkit.langgraph import checkpointer, memory_store, thread_config

    agent = create_agent(
        model, tools=[...], checkpointer=checkpointer(), store=memory_store()
    )
    agent.invoke(inputs, config=thread_config(session_id, actor))

Who the user is. Every entry belongs to an actor. Inside a LangGraph run the store reads it per
call from the run config (``configurable["actor_id"]``, which ``thread_config(session_id, actor)``
sets and the checkpointer already reads), so one store serves many users. ``actor=`` is for calls
outside a run; inside a run it must match the run's actor, or the call raises. The actor
partitions memories but is not an access-control boundary: the memory store is. Set the actor from
a trusted identity (the signed-in user), never from a model-chosen or user-supplied field, and use
separate stores for separate tenants.

Each item is one cross-session memory (no ``session_id``) at ``{root}/{namespace...}/{key}``,
each segment percent-encoded. Entries are kept under the storage actor ``<actor>:langgraph``, not
the plain actor. Memory pipelines find the actors they process through sessions, and no session
uses that id, so automatic pipeline runs leave these entries alone; that's a convention the
session and memory helpers enforce, not the service. ``memory_tools`` and sessions, which use the
plain actor, don't see the entries. The content is a small JSON envelope, ``{"lg": 1, "v":
<value>}``, plus ``"x"`` (expiry, epoch milliseconds) when the item has a TTL. Entries outside
``root``, saved under a session, or not holding an envelope this store wrote are skipped, with a
one-time warning for the last kind.

Limits:
  - Search ranking is the service's. The managed memory docs describe it as full-text (BM25)
    combined with semantic matching over each entry's content and description; the response
    doesn't say which matching ran. Scores are the service's rank-based values: higher is better,
    but they aren't similarities and there's no cutoff, so a query can return entries that share
    no words with it. There are no client-side embeddings, so ``index`` configs and per-item
    ``index`` field lists are rejected.
  - One search returns at most 100 entries, with no pagination; a filter runs after ranking, so
    entries ranked below the first 100 can't be reached by search, and ``offset + limit`` past
    what the service returns comes back short. Expired entries count toward the 100 until they're
    swept.
  - ``put`` isn't atomic: concurrent writers to one key overwrite each other (last writer wins).
    A batch isn't a transaction, and a call that raised may still have been applied.
  - ``search`` without a query reads the namespace's entries, 100 per request, until the page is
    full; ``list_namespaces`` reads every entry the actor has. Their cost grows with the entries
    stored.
  - Values are JSON objects (dicts). Tuples come back as lists and non-string keys as strings.
  - TTL (``ttl=`` on ``put``, or ``TTLConfig.default_ttl``) counts from the last write, on this
    process's clock. Reads never refresh it. Expired items aren't returned, but they stay in the
    service until ``sweep_ttl`` deletes them or they are overwritten; there's no background
    sweeper.
  - Python 3.10: LangGraph's ``get_config()`` and ``get_store()`` don't work in async code, so
    build the store per request with ``actor=`` and use that object directly there.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import logging
import math
import os
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Optional, Protocol, cast

from langgraph.config import get_config
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)

from databricks_agentbricks.errors import AgentCliError
from databricks_agentkit._actors import STORE_ACTOR_SUFFIX, check_actor
from databricks_agentkit.client import AgentKitClient
from databricks_agentkit.runtime.store import RUNTIME_STORE_LOCAL_ENV
from databricks_agentkit.runtime.tool_manifest import resolve_memory_store
from databricks_agentkit.runtime.workspace import workspace_client

if TYPE_CHECKING:
    from databricks_agentkit.memory_store import Memory

logger = logging.getLogger(__name__)

_SEARCH_MAX = 100  # the service returns at most this many entries per search, with no pagination
_SEARCH_HEADROOM = 50  # extra results requested to cover entries dropped by client-side filters
_LIST_PAGE_SIZE = 100
# The service rejects request bodies over 1 MiB, and only after a long wait. The SDK escapes
# non-ASCII text and quotes in the JSON it sends, so measure the encoded request, not the value.
_REQUEST_MAX_BYTES = 900_000
# The service indexes actor and path together and rejects pairs over about 2.7 KB.
_KEY_MAX_BYTES = 2_400
_TTL_KEYS = frozenset({"default_ttl", "refresh_on_read", "sweep_interval_minutes", "omit_expired"})
_TTL_MAX_MINUTES = 100 * 365 * 24 * 60  # a century


def _encode_segment(segment: str, what: str) -> str:
    if not isinstance(segment, str) or not segment:
        raise ValueError(f"{what} must be a non-empty string, got {segment!r}")
    try:
        encoded = urllib.parse.quote(segment, safe="")
    except UnicodeEncodeError as e:  # eg a lone surrogate
        raise ValueError(f"{what} {segment!r} isn't valid Unicode text: {e.reason}") from e
    # quote() leaves dots alone; "." and ".." would read as path navigation, so encode them.
    return encoded.replace(".", "%2E") if encoded in (".", "..") else encoded


def _prefix_path(root: str, namespace: tuple[str, ...]) -> str:
    """The listing prefix for ``namespace``: ends in "/" so ("a",) never matches ("ab",)."""
    segments = [_encode_segment(label, "namespace label") for label in namespace]
    return "/".join([root, *segments, ""])


def _item_path(root: str, namespace: tuple[str, ...], key: str) -> str:
    return _prefix_path(root, namespace) + _encode_segment(key, "key")


def _split_path(root: str, path: str) -> Optional[tuple[tuple[str, ...], str]]:
    """``(namespace, key)`` for a path this store wrote, else None."""
    if not path.startswith(root + "/"):
        return None
    segments = path[len(root) + 1 :].split("/")
    if any(not segment for segment in segments):
        return None
    *namespace, key = (urllib.parse.unquote(segment) for segment in segments)
    return tuple(namespace), key


_ENVELOPE_VERSION = 1
_ENVELOPE_KEYS = frozenset({"lg", "v", "x"})
_FOREIGN = (
    "Skipped an entry under the store's root that doesn't hold a value this store wrote; a memory "
    "pipeline, the CLI or another writer may have changed it."
)
_NEWER = "Skipped an entry written by a newer version of DatabricksMemoryStore; upgrade to read it."


@dataclass(frozen=True)
class _Stored:
    value: dict[str, Any]
    expires_at_ms: Optional[int]


def _wrap(value: Mapping[str, Any], expires_at_ms: Optional[int]) -> str:
    envelope: dict[str, Any] = {"lg": _ENVELOPE_VERSION, "v": value}
    if expires_at_ms is not None:
        envelope["x"] = expires_at_ms
    # ensure_ascii=False keeps non-Latin text searchable by the service's full-text matching;
    # allow_nan=False keeps the content strict JSON for the CLI and other readers.
    return json.dumps(envelope, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _unwrap(memory: Memory) -> Optional[_Stored]:
    """The value and expiry of an entry this store wrote, else None (with a one-time warning)."""
    try:
        envelope = json.loads(memory.content or "")
    except (TypeError, ValueError, RecursionError):  # RecursionError: deeply nested JSON
        envelope = None
    if isinstance(envelope, dict):
        version, value, expires = envelope.get("lg"), envelope.get("v"), envelope.get("x")
        if type(version) is int and version > _ENVELOPE_VERSION:
            _warn_once(_NEWER)
            return None
        if (
            type(version) is int
            and version == _ENVELOPE_VERSION
            and set(envelope) <= _ENVELOPE_KEYS
            and isinstance(value, dict)
            and (expires is None or type(expires) is int)
        ):
            return _Stored(value, expires)
    _warn_once(_FOREIGN)
    return None


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _compare_values(item_value: Any, filter_value: Any) -> bool:
    """Filter matching with InMemoryStore semantics ($eq/$ne/$gt/$gte/$lt/$lte, nested dicts).

    ``_compare_values`` and ``_apply_operator`` are adapted from the functions of the same names in
    ``langgraph.store.memory`` (MIT License, Copyright (c) 2024 LangChain, Inc.).
    """
    if isinstance(filter_value, dict):
        if any(k.startswith("$") for k in filter_value):
            return all(
                _apply_operator(item_value, op, operand) for op, operand in filter_value.items()
            )
        if not isinstance(item_value, dict):
            return False
        return all(_compare_values(item_value.get(k), v) for k, v in filter_value.items())
    if isinstance(filter_value, (list, tuple)):
        return (
            isinstance(item_value, (list, tuple))
            and len(item_value) == len(filter_value)
            and all(
                _compare_values(iv, fv) for iv, fv in zip(item_value, filter_value, strict=True)
            )
        )
    return item_value == filter_value


def _apply_operator(value: Any, operator: str, operand: Any) -> bool:
    if operator == "$eq":
        return value == operand
    if operator == "$ne":
        return value != operand
    if operator in ("$gt", "$gte", "$lt", "$lte"):
        try:
            left, right = float(value), float(operand)
        except (TypeError, ValueError):
            return False
        return {
            "$gt": left > right,
            "$gte": left >= right,
            "$lt": left < right,
            "$lte": left <= right,
        }[operator]
    raise ValueError(f"Unsupported filter operator: {operator}")


def _matches(value: dict[str, Any], filter_: Optional[dict[str, Any]]) -> bool:
    return not filter_ or all(_compare_values(value.get(k), v) for k, v in filter_.items())


def _matches_condition(namespace: tuple[str, ...], condition: MatchCondition) -> bool:
    pattern = tuple(condition.path)
    if len(namespace) < len(pattern):
        return False
    window = (
        namespace[: len(pattern)]
        if condition.match_type == "prefix"
        else namespace[-len(pattern) :]
    )
    return all(p == "*" or p == label for p, label in zip(pattern, window, strict=True))


class _Memories(Protocol):
    """What the store needs from a managed memory store handle (``MemoryStore``)."""

    def put(
        self, *, actor_id: str, path: str, content: str, description: Optional[str] = None
    ) -> Any: ...

    def list(
        self,
        *,
        actor_id: str,
        path: Optional[str] = None,
        path_prefix: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> Iterable[Any]: ...

    def search(
        self,
        *,
        actor_id: str,
        query: str,
        limit: Optional[int] = None,
        path_prefix: Optional[str] = None,
    ) -> Iterable[Any]: ...


@dataclass
class _LocalMemory:
    path: str
    content: str
    create_time: datetime
    update_time: datetime
    _entries: dict[tuple[str, str], _LocalMemory] = field(repr=False)
    _key: tuple[str, str] = field(repr=False)
    session_id: Optional[str] = None

    def delete(self) -> None:
        self._entries.pop(self._key, None)


@dataclass
class _LocalResult:
    memory: _LocalMemory
    score: float


class _LocalMemories:
    """An in-process stand-in for a managed memory store, for local development only.

    Same actor partitioning and paths as the service; search ranks by shared words.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _LocalMemory] = {}
        self._lock = threading.Lock()

    def put(
        self, *, actor_id: str, path: str, content: str, description: Optional[str] = None
    ) -> _LocalMemory:
        now = datetime.now(timezone.utc)
        key = (actor_id, path)
        with self._lock:
            old = self._entries.get(key)
            memory = _LocalMemory(
                path, content, old.create_time if old else now, now, self._entries, key
            )
            self._entries[key] = memory
        return memory

    def list(
        self,
        *,
        actor_id: str,
        path: Optional[str] = None,
        path_prefix: Optional[str] = None,
        page_size: Optional[int] = None,
    ) -> builtins.list[_LocalMemory]:
        with self._lock:
            entries = [m for (a, _), m in self._entries.items() if a == actor_id]
        return [
            m
            for m in entries
            if (path is None or m.path == path)
            and (path_prefix is None or m.path.startswith(path_prefix))
        ]

    def search(
        self,
        *,
        actor_id: str,
        query: str,
        limit: Optional[int] = None,
        path_prefix: Optional[str] = None,
    ) -> builtins.list[_LocalResult]:
        terms = set(re.findall(r"\w+", query.lower()))
        hits = []
        for memory in self.list(actor_id=actor_id, path_prefix=path_prefix):
            score = len(terms & set(re.findall(r"\w+", memory.content.lower())))
            if score:
                hits.append(_LocalResult(memory, float(score)))
        hits.sort(key=lambda r: -r.score)
        return hits[:limit]


class DatabricksMemoryStore(BaseStore):
    """LangGraph ``BaseStore`` on an Agent Bricks managed memory store.

    Args:
        store: Memory store id. Resolves ``store`` → ``AGENT_MEMORY_STORE`` on first use (deployed
            Apps set it from the agent.toml binding; ``agentbricks dev`` doesn't, so export it in
            your shell to use a store locally). With neither, the first call raises.
        actor: The identity whose memories to use, for calls outside a LangGraph run. Inside a
            run the actor comes from ``configurable[actor_key]`` (see ``thread_config``); a run
            for a different actor than ``actor`` raises rather than mixing users. Where the run
            config isn't visible (threads you start, async code on Python 3.10), ``actor`` is
            used as is, so build such a store per request, for that request's user.
        actor_key: The run-config key holding the actor.
        strict_user_key: Optional run-config key that must be present and equal the actor, eg
            ``"langgraph_user_id"`` for LangMem namespaces templated on the user id.
        root: Path prefix for this store's entries, so they don't mix with other writers.
        ttl: A LangGraph ``TTLConfig``. ``default_ttl`` (minutes) applies to puts that don't pass
            ``ttl=``. ``omit_expired`` defaults to True here (LangGraph's default is False).
            Reads never refresh a TTL, so ``refresh_on_read`` is set to False, and True raises: a
            refresh would rewrite the entry and could undo a concurrent put. ``sweep_interval_minutes``
            raises, because the memory API lists one actor at a time, so there's no global sweep;
            call :meth:`sweep_ttl` per user instead. Unknown keys raise.
        client: An AgentKit client; built from the workspace client on first use when omitted.
    """

    # Plain BaseStore on purpose: LangGraph's batched base pools concurrent callers into one
    # batch, where no caller's identity is visible. Each batch() here runs for one actor.
    supports_ttl = True

    def __init__(
        self,
        store: Optional[str] = None,
        *,
        actor: Optional[str] = None,
        actor_key: str = "actor_id",
        strict_user_key: Optional[str] = None,
        root: str = "/langgraph",
        ttl: Optional[TTLConfig] = None,
        client: Optional[AgentKitClient] = None,
    ) -> None:
        if not root.startswith("/") or root.endswith("/"):
            raise ValueError(f"root must start with '/' and not end with one, got {root!r}")
        if actor is not None:
            _explicit_actor(actor)
        ttl_opts: dict[str, Any] = dict(cast(Mapping[str, Any], ttl or {}))
        unknown = set(ttl_opts) - _TTL_KEYS
        if unknown:
            raise ValueError(f"Unknown ttl keys {sorted(unknown)}; expected {sorted(_TTL_KEYS)}.")
        if ttl_opts.get("sweep_interval_minutes"):
            raise ValueError(
                "sweep_interval_minutes isn't supported: the memory API lists one actor at a time, "
                "so the store can't sweep every user. Call sweep_ttl(actor=...) instead."
            )
        if ttl_opts.get("refresh_on_read"):
            raise ValueError(
                "refresh_on_read isn't supported: a refresh rewrites the entry and could undo a "
                "concurrent put. A TTL counts from the last write."
            )
        if ttl_opts.get("default_ttl") is not None:
            _check_ttl(ttl_opts["default_ttl"])
        if ttl_opts:
            ttl_opts["refresh_on_read"] = False  # so LangGraph's own TTL defaults match the store
        self.ttl_config = cast(TTLConfig, ttl_opts) if ttl_opts else None
        self._omit_expired = ttl_opts.get("omit_expired", True)
        self._now_ms = _now_ms
        self._store = store
        self._actor = actor
        self._actor_key = actor_key
        self._strict_user_key = strict_user_key
        self._root = root
        self._client = client
        self._handle: Optional[_Memories] = None
        self._local_fallback = False  # memory_store() sets it: no store means a local stand-in
        self._lock = threading.Lock()

    @classmethod
    def _local(cls, **kwargs: Any) -> DatabricksMemoryStore:
        store = cls("local", **kwargs)
        store._handle = _LocalMemories()
        return store

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._run(list(ops), self._resolve_actor())

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Read the run config here, on the event loop, before hopping to a worker thread.
        actor = self._resolve_actor()
        return await asyncio.to_thread(self._run, list(ops), actor)

    def _run(self, ops: list[Op], actor: str) -> list[Result]:
        handle, partition = self._memory_store(), self.storage_actor(actor)
        return [self._apply(handle, partition, op) for op in ops]

    @staticmethod
    def storage_actor(actor: str) -> str:
        """The actor id the service keeps ``actor``'s entries under, eg for the CLI."""
        return check_actor(actor) + STORE_ACTOR_SUFFIX

    def sweep_ttl(self, *, actor: Optional[str] = None) -> int:
        """Delete one user's expired items and return how many were deleted.

        ``actor`` defaults to the run's actor, as for any other call. There's no sweep across
        users, because the memory API lists one actor at a time.
        """
        user = _explicit_actor(actor) if actor is not None else self._resolve_actor()
        return self._sweep(self.storage_actor(user))

    async def asweep_ttl(self, *, actor: Optional[str] = None) -> int:
        user = _explicit_actor(actor) if actor is not None else self._resolve_actor()
        return await asyncio.to_thread(self._sweep, self.storage_actor(user))

    def _sweep(self, partition: str) -> int:
        handle, now, deleted = self._memory_store(), self._now_ms(), 0
        for memory in list(self._under(handle, partition, self._root + "/")):
            path = memory.path or ""
            if _split_path(self._root, path) is None or not _expired(_unwrap(memory), now):
                continue
            # Re-read just before deleting, so an entry a put renewed in the meantime survives.
            for current in self._at(handle, partition, path):
                if _expired(_unwrap(current), now):
                    try:
                        current.delete()
                    except AgentCliError as e:
                        if e.error_code != "NOT_FOUND":
                            raise
                    else:
                        deleted += 1
        return deleted

    def _resolve_actor(self) -> str:
        try:
            configurable = get_config().get("configurable", {})
        except RuntimeError as e:
            if self._actor is not None and self._strict_user_key is None:
                return self._actor  # outside a LangGraph run, or where its config isn't visible
            raise ValueError(
                "DatabricksMemoryStore needs a run config: call it inside a LangGraph run with "
                "config=thread_config(session_id, actor), or pass actor= for calls outside one. "
                "Threads you start yourself don't inherit the run config (copy it with "
                "contextvars.copy_context()), and async code on Python 3.10 doesn't see it."
            ) from e
        run_actor = configurable.get(self._actor_key)
        if self._actor is None:
            actor = run_actor
            # thread_config(session_id) without an actor sets the actor to the session id.
            if run_actor is not None and run_actor == configurable.get("thread_id"):
                _warn_once(
                    "The run config's actor is the session id, so memories won't carry over to "
                    "other conversations; pass the user to thread_config(session_id, actor)."
                )
        else:
            actor = self._actor
            if run_actor is not None and run_actor != actor:
                raise ValueError(
                    f"This DatabricksMemoryStore was built for actor {actor!r} but the run is for "
                    f"{run_actor!r} (thread_config(session_id) without an actor uses the session "
                    "id). Build one store without actor= and pass each user through "
                    "thread_config(session_id, actor)."
                )
        if not actor:
            raise ValueError(
                f"DatabricksMemoryStore needs an actor: the run config has no "
                f"{self._actor_key!r}. Use config=thread_config(session_id, actor)."
            )
        if not isinstance(actor, str):
            raise ValueError(
                f"DatabricksMemoryStore needs a string actor, got {type(actor).__name__} "
                f"from the run config's {self._actor_key!r}."
            )
        if self._strict_user_key is not None:
            user = configurable.get(self._strict_user_key)
            if user != actor:
                raise ValueError(
                    f"Run config {self._strict_user_key!r}={user!r} doesn't match actor {actor!r}; "
                    "refusing to read or write another user's memory."
                )
        return actor

    def _memory_store(self) -> _Memories:
        # Built on first use and reused: the client resolves workspace auth, and the id-only
        # handle avoids fetching the store before every call.
        with self._lock:
            if self._handle is None:
                store_id = resolve_memory_store(self._store)
                if store_id:
                    client = self._client or AgentKitClient(workspace_client())
                    self._handle = client.memory_stores._handle(store_id)
                elif self._local_fallback and not _deployed():
                    _warn_once(
                        "No memory store configured (pass store=, set AGENT_MEMORY_STORE, or run "
                        "`agentbricks memory bind`); using an in-process store that keeps nothing "
                        "across restarts."
                    )
                    self._handle = _LocalMemories()
                else:
                    raise ValueError(
                        "No memory store configured: pass store=, set AGENT_MEMORY_STORE, or bind "
                        "one with `agentbricks memory bind`. A deployed App needs a store; the "
                        "in-process stand-in is for local development only."
                    )
            return self._handle

    def _apply(self, handle: _Memories, partition: str, op: Op) -> Result:
        if isinstance(op, GetOp):
            return self._get(handle, partition, op)
        if isinstance(op, PutOp):
            return self._put(handle, partition, op)
        if isinstance(op, SearchOp):
            return self._search(handle, partition, op)
        if isinstance(op, ListNamespacesOp):
            return self._list_namespaces(handle, partition, op)
        raise TypeError(f"Unsupported store operation: {type(op).__name__}")

    def _at(self, handle: _Memories, partition: str, path: str) -> list[Memory]:
        # The API ignores session_id alongside path, so recheck both.
        listed = handle.list(actor_id=partition, path=path, page_size=_LIST_PAGE_SIZE)
        return [m for m in listed if m.path == path and m.session_id is None]

    def _under(self, handle: _Memories, partition: str, prefix: str) -> Iterator[Memory]:
        listed = handle.list(actor_id=partition, path_prefix=prefix, page_size=_LIST_PAGE_SIZE)
        for memory in listed:
            if memory.session_id is None and (memory.path or "").startswith(prefix):
                yield memory

    def _live(self, memory: Memory, now: int) -> Optional[_Stored]:
        """The item an entry holds, unless it isn't this store's or has expired (and is omitted)."""
        stored = _unwrap(memory)
        if stored is None or (self._omit_expired and _expired(stored, now)):
            return None
        return stored

    def _get(self, handle: _Memories, partition: str, op: GetOp) -> Optional[Item]:
        now = self._now_ms()
        for memory in self._at(handle, partition, _item_path(self._root, op.namespace, op.key)):
            stored = self._live(memory, now)
            if stored is not None:
                return Item(
                    value=stored.value, key=op.key, namespace=op.namespace, **_times(memory)
                )
        return None

    def _put(self, handle: _Memories, partition: str, op: PutOp) -> None:
        if isinstance(op.index, list):
            raise NotImplementedError(
                "DatabricksMemoryStore doesn't support per-item index fields: the managed memory "
                "service ranks search results itself, with no client-side embeddings."
            )
        path = _item_path(self._root, op.namespace, op.key)
        if op.value is None:
            for memory in self._at(handle, partition, path):
                try:
                    memory.delete()
                except AgentCliError as e:
                    if e.error_code != "NOT_FOUND":  # already gone, eg a retried delete
                        raise
            return
        if not isinstance(op.value, dict):
            raise TypeError(
                f"Value for key {op.key!r} must be a dict (a JSON object), "
                f"got {type(op.value).__name__}."
            )
        key_bytes = len(partition.encode()) + len(path.encode())
        if key_bytes > _KEY_MAX_BYTES:
            raise ValueError(
                f"The actor, namespace and key {op.key!r} take {key_bytes:,} bytes together; "
                f"managed memory accepts about {_KEY_MAX_BYTES:,}."
            )
        expires = None if op.ttl is None else self._now_ms() + round(_check_ttl(op.ttl) * 60_000)
        try:
            content = _wrap(op.value, expires)
        except TypeError as e:
            raise TypeError(f"Value for key {op.key!r} isn't JSON-serializable: {e}") from e
        except (ValueError, RecursionError) as e:  # NaN or infinity, a huge int, deep nesting
            raise ValueError(f"Value for key {op.key!r} can't be stored as JSON: {e}") from e
        try:
            content.encode()
        except UnicodeEncodeError as e:  # eg a lone surrogate in a string
            raise ValueError(
                f"Value for key {op.key!r} isn't valid Unicode text: {e.reason}"
            ) from e
        description = _describe(op.value)
        size = len(json.dumps({"content": content, "description": description}))
        if size > _REQUEST_MAX_BYTES:
            raise ValueError(
                f"Value for key {op.key!r} needs a {size:,}-byte request; managed memory entries "
                f"must stay under about {_REQUEST_MAX_BYTES:,} bytes."
            )
        handle.put(actor_id=partition, path=path, content=content, description=description)

    def _search(self, handle: _Memories, partition: str, op: SearchOp) -> list[SearchItem]:
        prefix, now = _prefix_path(self._root, op.namespace_prefix), self._now_ms()
        query, want = (op.query or "").strip(), op.offset + op.limit
        found: list[Any] = []
        limit = 0
        candidates: Iterable[tuple[Memory, Optional[float]]]
        if query:
            # A filter runs after the service ranks, so ask for the most it returns.
            limit = _SEARCH_MAX if op.filter else min(_SEARCH_MAX, want + _SEARCH_HEADROOM)
            found = list(
                handle.search(actor_id=partition, query=query, limit=limit, path_prefix=prefix)
            )
            candidates = [(r.memory, r.score) for r in found]
        else:
            candidates = ((m, None) for m in self._under(handle, partition, prefix))
        items = []
        for memory, score in candidates:
            path = memory.path or ""
            if memory.session_id is not None or not path.startswith(prefix):
                continue
            parts = _split_path(self._root, path)
            stored = self._live(memory, now) if parts is not None else None
            if parts is None or stored is None or not _matches(stored.value, op.filter):
                continue
            namespace, key = parts
            items.append(SearchItem(namespace, key, stored.value, score=score, **_times(memory)))
            if not query and len(items) >= want:
                break  # listing order is fixed, so later pages can't change this one
        if query and len(found) >= limit and len(items) < want:
            _warn_once(
                "A search reached the most results managed memory returns per query; matches "
                "ranked lower aren't seen, so the page may be short. Narrow the namespace or query."
            )
        return items[op.offset : want]

    def _list_namespaces(
        self, handle: _Memories, partition: str, op: ListNamespacesOp
    ) -> list[tuple[str, ...]]:
        namespaces, now = set(), self._now_ms()
        for memory in self._under(handle, partition, self._root + "/"):
            parts = _split_path(self._root, memory.path or "")
            if parts is None or self._live(memory, now) is None:  # same entries get and search see
                continue
            namespace = parts[0]
            if op.match_conditions and not all(
                _matches_condition(namespace, c) for c in op.match_conditions
            ):
                continue
            namespaces.add(namespace[: op.max_depth] if op.max_depth is not None else namespace)
        return sorted(namespaces)[op.offset : op.offset + op.limit]


_DESCRIPTION_MAX = 500


def _describe(value: Mapping[str, Any]) -> str:
    """The value's text, for the CLI and UI listings and the service's search over descriptions."""
    texts: list[str] = []
    stack: list[Any] = [value]
    while stack:  # depth-first, in the value's own order, without recursion
        node = stack.pop()
        if isinstance(node, str):
            texts.append(node)
        elif isinstance(node, Mapping):
            stack.extend(reversed(list(node.values())))
        elif isinstance(node, (list, tuple)):
            stack.extend(reversed(node))
    # The service can't store NUL characters in a description.
    return " ".join(" ".join(texts).replace("\x00", " ").split())[:_DESCRIPTION_MAX]


def _check_ttl(minutes: Any) -> float:
    if (
        isinstance(minutes, bool)
        or not isinstance(minutes, (int, float))
        or not math.isfinite(minutes)
        or not 0 < minutes <= _TTL_MAX_MINUTES
    ):
        raise ValueError(
            f"ttl must be a positive number of minutes, at most {_TTL_MAX_MINUTES:,}, "
            f"got {minutes!r}"
        )
    if round(minutes * 60_000) < 1:
        raise ValueError(f"ttl must be at least one millisecond, got {minutes!r} minutes")
    return float(minutes)


def _expired(stored: Optional[_Stored], now: int) -> bool:
    return stored is not None and stored.expires_at_ms is not None and stored.expires_at_ms <= now


def _explicit_actor(actor: Any) -> str:
    if not isinstance(actor, str) or not actor:
        raise ValueError(f"actor must be a non-empty string, got {actor!r}")
    return check_actor(actor)


def _deployed() -> bool:
    """True in a deployed Databricks App, where an in-process stand-in would lose memories."""
    local = os.getenv(RUNTIME_STORE_LOCAL_ENV, "").lower() == "true"
    return bool(os.getenv("DATABRICKS_APP_NAME")) and not local


_warned: set[str] = set()
_warned_lock = threading.Lock()


def _warn_once(message: str) -> None:
    with _warned_lock:
        if message in _warned:
            return
        _warned.add(message)
    logger.warning(message)


def _times(memory: Any) -> dict[str, datetime]:
    now = datetime.now(timezone.utc)
    return {"created_at": memory.create_time or now, "updated_at": memory.update_time or now}


def memory_store(
    store: Optional[str] = None, *, actor: Optional[str] = None, **kwargs: Any
) -> BaseStore:
    """The LangGraph long-term memory store on managed memory. Build it once and reuse it.

    Resolves ``store`` → ``AGENT_MEMORY_STORE`` on first use (deployed Apps set it from the
    agent.toml binding; ``agentbricks dev`` doesn't). With neither, a local run logs a warning
    and uses an in-process stand-in with the same actor partitioning, which keeps nothing across
    restarts; a deployed App raises instead. ``actor`` and ``kwargs`` go to
    :class:`DatabricksMemoryStore`.
    """
    instance = DatabricksMemoryStore(store, actor=actor, **kwargs)
    instance._local_fallback = True
    return instance
