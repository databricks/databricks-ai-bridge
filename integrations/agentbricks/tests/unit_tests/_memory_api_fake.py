"""Wire-level fake of the managed memory entries API, shared by the LangGraph store tests.

The fake stands in for ``WorkspaceClient.api_client``, the one call ``_AgentBricksApiClient._do``
makes. Everything above it runs for real: AgentKitClient, _AgentBricksApiClient (route, query and
body building, error mapping), MemoryStores, MemoryStore, Memory and DatabricksMemoryStore.

Response shapes are the ones the client parses: lists return ``managed_memory_entries`` plus
``next_page_token``, and search returns ``results[].managed_memory_entry`` with a ``score``.
Server behaviour the client code cannot confirm is set by the constructor knobs, and the tests
that depend on it say so. Nothing here talks to a service.
"""

from __future__ import annotations

import copy
import re
import threading
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

STORE = "langgraph-store"
_API_PREFIX = "/api/2.0/agents/"
_ENTRY_PATH = re.compile(
    r"memory-stores/(?P<store>[^/]+)/entries(?P<rest>/(?P<entry>[^/]+)|:search)?"
)
_SEARCH_MAX = 100  # the server returns at most this many search hits per call
_LIST_DEFAULT_PAGE = 100


class WireError(Exception):
    """An HTTP error as the SDK raises it: a message and an ``error_code`` attribute."""

    def __init__(self, error_code: str, message: str = "", *, status: int = 400) -> None:
        super().__init__(message or error_code)
        self.error_code = error_code
        self.status = status


@dataclass
class Call:
    route: str  # create | list | get | update | delete | search
    store: str
    entry: Optional[str]
    query: dict
    body: Any
    thread: int = field(default_factory=threading.get_ident)

    @property
    def actor_id(self) -> Optional[str]:
        return self.query.get("actor_id") or (self.body or {}).get("actor_id")


@dataclass
class _Fault:
    route: str
    nth: int
    error: WireError
    commit: bool  # apply the request's effect before failing, like a lost response or a race


class FakeMemoryApi:
    """In-memory memory entries API behind the SDK's ``api_client.do`` seam.

    Knobs:
        server_page_size: the server's page size; a list page is ``min(page_size, this)``.
        ignore_filters: the server ignores ``path``, ``path_prefix`` and ``session_id`` (the
            actor is always applied), so the store's own client-side rechecks are exercised.

    Without ``ignore_filters``, an omitted ``session_id`` returns entries of every session. The
    server's real default is not confirmed here, so the store must recheck session scope either way.
    Search ranks by whole-word matches (case-insensitive) over content and description.
    """

    def __init__(self, *, server_page_size: int = 100, ignore_filters: bool = False) -> None:
        self.server_page_size = server_page_size
        self.ignore_filters = ignore_filters
        self.entries: dict[str, dict[str, Any]] = {}  # resource name -> server row
        self.calls: list[Call] = []
        # Called with the resource name at the start of update/delete, to simulate a race.
        self.hooks: dict[str, Callable[[str], None]] = {}
        self._route_counts: dict[str, int] = {}
        self._faults: list[_Fault] = []
        self._seq = 0
        self._lock = threading.Lock()

    # --- setup and inspection ------------------------------------------------

    def seed(
        self,
        path: str,
        content: Optional[str],
        *,
        actor_id: str = "alice:langgraph",  # the store's storage actor for "alice"
        session_id: Optional[str] = None,
        description: Optional[str] = None,
        store: str = STORE,
        create_time: Optional[str] = None,
        update_time: Optional[str] = None,
    ) -> str:
        """Write a row straight into server state (test setup, not a logged call)."""
        name = self._new_name(store)
        now = _rfc3339(datetime.now(timezone.utc))
        row: dict[str, Any] = {
            "name": name,
            "actor_id": actor_id,
            "path": path,
            "create_time": create_time or now,
            "update_time": update_time or now,
        }
        if content is not None:
            row["content"] = content
        if description is not None:
            row["description"] = description
        if session_id is not None:
            row["session_id"] = session_id
        self.entries[name] = row
        return name

    def fail(
        self,
        route: str,
        error_code: str,
        *,
        nth: int = 1,
        status: int = 400,
        commit: bool = False,
    ) -> None:
        """Fail the ``nth`` call to ``route`` with ``error_code``.

        With ``commit``, the call's effect is applied first, then the error is raised.
        """
        self._faults.append(_Fault(route, nth, WireError(error_code, status=status), commit))

    def rows(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(row) for row in self.entries.values()]

    def routes(self) -> list[str]:
        return [call.route for call in self.calls]

    def workspace_client(self) -> types.SimpleNamespace:
        """A stand-in ``WorkspaceClient`` whose ``api_client`` is this fake."""
        return types.SimpleNamespace(api_client=self, config=types.SimpleNamespace(host=None))

    # --- the SDK seam ---------------------------------------------------------

    def do(self, method: str, path: str, query: Optional[dict] = None, body: Any = None) -> Any:
        with self._lock:
            route, store, entry = _route(method, path)
            call = Call(route, store, entry, copy.deepcopy(dict(query or {})), copy.deepcopy(body))
            self.calls.append(call)
            self._route_counts[route] = self._route_counts.get(route, 0) + 1
            fault = self._fault_for(route)
            if fault is not None and not fault.commit:
                raise fault.error
            result = getattr(self, f"_{route}")(store, entry, call.query, call.body)
            if fault is not None:
                raise fault.error
            return result

    # --- handlers, one per route -----------------------------------------------

    def _create(self, store: str, _entry, _query: dict, body: dict) -> dict:
        if not body.get("actor_id"):
            raise _invalid("actor_id is required")
        _check_path(body.get("path") or "")
        key = (body["actor_id"], body.get("session_id"), body["path"])
        for row in self.entries.values():
            if (row["actor_id"], row.get("session_id"), row["path"]) == key:
                raise WireError(
                    "ALREADY_EXISTS", "a memory with this path already exists", status=409
                )
        name = self._new_name(store)
        now = _rfc3339(datetime.now(timezone.utc))
        row: dict[str, Any] = {"name": name, "create_time": now, "update_time": now}
        for field_name in (
            "actor_id",
            "path",
            "content",
            "description",
            "session_id",
            "source_type",
        ):
            if body.get(field_name) is not None:
                row[field_name] = body[field_name]
        self.entries[name] = row
        return copy.deepcopy(row)

    def _list(self, store: str, _entry, query: dict, _body) -> dict:
        rows = [row for row in self._store_rows(store) if self._visible(row, query)]
        size = min(_page_size(query.get("page_size"), _LIST_DEFAULT_PAGE), self.server_page_size)
        token = query.get("page_token") or "0"
        if not str(token).isdigit():
            raise _invalid("invalid page_token")
        start = int(token)
        page: dict[str, Any] = {"managed_memory_entries": copy.deepcopy(rows[start : start + size])}
        if start + size < len(rows):
            page["next_page_token"] = str(start + size)
        return page

    def _get(self, store: str, entry: str, _query: dict, _body) -> dict:
        return copy.deepcopy(self._row(_name(store, entry)))

    def _update(self, store: str, entry: str, query: dict, body: dict) -> dict:
        name = _name(store, entry)
        self._run_hook("update", name)
        row = self._row(name)
        if not query.get("update_mask"):
            raise _invalid("update_mask is required")
        mask = set(query["update_mask"].split(","))
        if "content" in mask and "content" in body:
            row["content"] = body["content"]
        if "description" in mask and "description" in body:
            row["description"] = body["description"]
        row["update_time"] = _rfc3339(datetime.now(timezone.utc))
        return copy.deepcopy(row)

    def _delete(self, store: str, entry: str, _query: dict, _body) -> dict:
        name = _name(store, entry)
        self._run_hook("delete", name)
        self._row(name)
        del self.entries[name]
        return {}

    def _search(self, store: str, _entry, _query: dict, body: dict) -> dict:
        if not (body.get("query") or "").strip():
            raise _invalid("query is required")
        terms = _tokens(body["query"])
        cap = _page_size(body.get("page_size") or body.get("limit"), _SEARCH_MAX)
        scored = []
        for row in self._store_rows(store):
            if not self._visible(row, body):
                continue
            score = _score(row, terms)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda hit: -hit[0])  # stable: insertion order breaks ties
        return {
            "results": [
                {"managed_memory_entry": copy.deepcopy(row), "score": float(score)}
                for score, row in scored[:cap]
            ]
        }

    # --- helpers --------------------------------------------------------------

    def _new_name(self, store: str) -> str:
        self._seq += 1
        return f"memory-stores/{store}/entries/e{self._seq}"

    def _store_rows(self, store: str) -> list[dict[str, Any]]:
        return [
            row
            for row in self.entries.values()
            if row["name"].startswith(f"memory-stores/{store}/")
        ]

    def _visible(self, row: dict[str, Any], query: dict) -> bool:
        """The server's filter. The actor always applies; the rest only without ``ignore_filters``."""
        if row["actor_id"] != query.get("actor_id"):
            return False
        if self.ignore_filters:
            return True
        if query.get("path") is not None:  # exact path, every session; session_id is ignored
            return row["path"] == query["path"]
        prefix = query.get("path_prefix")
        if prefix is not None and not row["path"].startswith(prefix):
            return False
        session = query.get("session_id")
        return session is None or row.get("session_id") == session

    def _row(self, name: str) -> dict[str, Any]:
        if name not in self.entries:
            raise WireError("NOT_FOUND", f"{name} not found", status=404)
        return self.entries[name]

    def _run_hook(self, route: str, name: str) -> None:
        hook = self.hooks.get(route)
        if hook is not None:
            hook(name)

    def _fault_for(self, route: str) -> Optional[_Fault]:
        nth = self._route_counts[route]
        for fault in self._faults:
            if fault.route == route and fault.nth == nth:
                return fault
        return None


def install(monkeypatch, fake: Optional[FakeMemoryApi] = None) -> FakeMemoryApi:
    """Route the store's real AgentKitClient at a fake, by patching its workspace client only."""
    fake = fake or FakeMemoryApi()
    monkeypatch.setattr(
        "databricks_agentkit.langgraph.store.workspace_client", fake.workspace_client
    )
    return fake


def _route(method: str, path: str) -> tuple[str, str, Optional[str]]:
    assert path.startswith(_API_PREFIX), f"unexpected path {path!r}"
    match = _ENTRY_PATH.fullmatch(path[len(_API_PREFIX) :])
    assert match is not None, f"unexpected memory route {method} {path!r}"
    store, rest, entry = match["store"], match["rest"] or "", match["entry"]
    if rest == ":search" and method == "POST":
        return "search", store, None
    if rest == "" and method == "POST":
        return "create", store, None
    if rest == "" and method == "GET":
        return "list", store, None
    verbs = {"GET": "get", "PATCH": "update", "DELETE": "delete"}
    assert entry is not None and method in verbs, f"unexpected memory route {method} {path!r}"
    return verbs[method], store, entry


def _name(store: str, entry: str) -> str:
    return f"memory-stores/{store}/entries/{entry}"


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _score(row: dict[str, Any], terms: list[str]) -> int:
    """How many distinct query terms appear as whole words in content or description."""
    words = set(_tokens(f"{row.get('content') or ''} {row.get('description') or ''}"))
    return len(words & set(terms))


def _invalid(message: str) -> WireError:
    """The service's answer to a malformed request."""
    return WireError("INVALID_PARAMETER_VALUE", message, status=400)


def _page_size(value: Any, default: int) -> int:
    """Page sizes run 1 to 100; the service rejects anything else rather than clamping it."""
    if value is None:
        return default
    size = int(value)
    if not 1 <= size <= _SEARCH_MAX:
        raise _invalid(f"page_size must be between 1 and {_SEARCH_MAX}, got {size}")
    return size


def _check_path(path: str) -> None:
    """Paths start with "/" and have no empty, "." or ".." segments; a trailing "/" is allowed."""
    segments = path[1:].rstrip("/").split("/") if path.startswith("/") else [""]
    if not path.startswith("/") or any(s in ("", ".", "..") for s in segments):
        raise _invalid(f"invalid path {path!r}")


def _rfc3339(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"  # milliseconds, as the service sends
