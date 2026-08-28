"""Session Store-backed durability metadata for the Mason stop/start demo."""

from __future__ import annotations

import asyncio
import hashlib
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

from databricks.sdk import WorkspaceClient

_EVENT_TYPE = "mason_demo_durability"
_SESSION_STORE_ENV = "AGENT_SESSION_STORE"
_SESSION_ACTOR_ENV = "AGENT_SESSION_ACTOR_ID"
_HEARTBEAT_SECONDS_ENV = "MASON_DEMO_HEARTBEAT_SECONDS"
_STALE_SECONDS_ENV = "MASON_DEMO_STALE_SECONDS"
_AGENTS_API = "/api/agents/v1"


@dataclass(frozen=True)
class _Session:
    session_id: str


@dataclass(frozen=True)
class _SessionItem:
    data: Any


class _SessionStoreClient:
    """Minimal Session Store client kept local to the generated durability demo."""

    def __init__(self, workspace_client: WorkspaceClient | None = None) -> None:
        self._workspace = workspace_client or WorkspaceClient()
        self._store_name = ""

    def set_session_store(self, session_store_name: str) -> "_SessionStoreClient":
        self._store_name = session_store_name
        return self

    def get_session(self, *, session_id: str) -> _Session:
        self._workspace.api_client.do("GET", f"{self._sessions_path()}/{session_id}")
        return _Session(session_id)

    def create_session(
        self,
        *,
        actor_id: str,
        session_id: str,
        metadata: dict[str, str],
    ) -> _Session:
        self._workspace.api_client.do(
            "POST",
            self._sessions_path(),
            query={"session_id": session_id},
            body={"actor_id": actor_id, "metadata": metadata},
        )
        return _Session(session_id)

    def append_items(self, session: _Session, *, items: list[dict[str, Any]]) -> None:
        self._workspace.api_client.do(
            "POST",
            f"{self._sessions_path()}/{session.session_id}/items:append",
            body={"items": [{"data": item} for item in items]},
        )

    def list_items(self, session: _Session, *, order_by: str) -> list[_SessionItem]:
        page_token = None
        items = []
        while True:
            query = {"order_by": order_by, "page_size": 100}
            if page_token:
                query["page_token"] = page_token
            response = self._workspace.api_client.do(
                "GET",
                f"{self._sessions_path()}/{session.session_id}/items",
                query=query,
            )
            if not isinstance(response, dict):
                raise RuntimeError(
                    "Expected an object response while listing durability events."
                )
            response_object = cast(dict[str, Any], response)
            items.extend(
                _SessionItem(item["data"])
                for item in response_object.get("session_items", [])
                if "data" in item
            )
            page_token = response_object.get("next_page_token")
            if not page_token:
                return items

    def _sessions_path(self) -> str:
        return f"{_AGENTS_API}/session-stores/{self._store_name}/sessions"


def execution_id(session_id: str) -> str:
    """Return the durable execution id associated with a public chat session."""
    return hashlib.sha256(f"mason-demo-recovery\x1f{session_id}".encode()).hexdigest()


def heartbeat_seconds() -> float:
    """How often an active process persists an ownership heartbeat."""
    return max(0.1, float(os.getenv(_HEARTBEAT_SECONDS_ENV, "3")))


def stale_seconds() -> float:
    """How old a heartbeat must be before another process may resume the run."""
    configured = max(0.2, float(os.getenv(_STALE_SECONDS_ENV, "10")))
    return max(configured, heartbeat_seconds() * 2)


@dataclass(frozen=True)
class DurabilityLease:
    session_id: str
    execution_id: str
    attempt: int
    lease_id: str
    owner_id: str


@dataclass(frozen=True)
class DurabilityState:
    session_id: str
    execution_id: str
    status: str
    attempt: int
    lease_id: str | None
    owner_id: str | None
    heartbeat_at: str | None
    heartbeat_age_seconds: float | None
    heartbeat_fresh: bool
    event_count: int
    recent_events: list[dict[str, Any]]


class SessionStoreDurabilityLog:
    """Append-only attempt and heartbeat log stored beside LangGraph checkpoints.

    Session Store does not expose an atomic compare-and-swap claim. This helper is
    intentionally a demo-grade, last-writer-wins lease. A process verifies ownership
    before each heartbeat, and the Databricks Apps affinity cookie keeps one browser
    routed to one replica, but a production runtime still needs a transactional claim.
    """

    def __init__(
        self,
        session_store_name: str | None = None,
        *,
        client: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store_name = (
            session_store_name or os.getenv(_SESSION_STORE_ENV, "").strip()
        )
        self._client = None
        if self._store_name:
            self._client = (client or _SessionStoreClient()).set_session_store(
                self._store_name
            )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sessions: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def configured(self) -> bool:
        return self._client is not None

    async def state(self, session_id: str) -> DurabilityState:
        events = await asyncio.to_thread(self._read_events, session_id)
        return self._state_from_events(session_id, events)

    async def claim(self, session_id: str, owner_id: str) -> DurabilityLease:
        async with self._lock(session_id):
            current = await self.state(session_id)
            if current.status == "running":
                raise ValueError(
                    f"Execution {current.execution_id} still has a fresh heartbeat from "
                    f"process {current.owner_id}."
                )
            lease = DurabilityLease(
                session_id=session_id,
                execution_id=execution_id(session_id),
                attempt=current.attempt + 1,
                lease_id=uuid.uuid4().hex,
                owner_id=owner_id,
            )
            await self._append(
                lease,
                "attempt_started",
                {
                    "claim_mode": "session_store_last_writer_wins",
                    "atomic_claim": False,
                },
            )
            verified = await self.state(session_id)
            if verified.lease_id != lease.lease_id:
                raise ValueError(
                    f"Another process acquired execution {lease.execution_id} before this "
                    "attempt started."
                )
            return lease

    async def heartbeat(self, lease: DurabilityLease) -> bool:
        async with self._lock(lease.session_id):
            current = await self.state(lease.session_id)
            if current.status != "running" or current.lease_id != lease.lease_id:
                return False
            await self._append(lease, "heartbeat")
            return True

    async def complete(self, lease: DurabilityLease) -> bool:
        return await self._finish(lease, "completed")

    async def fail(self, lease: DurabilityLease, error: str) -> bool:
        return await self._finish(lease, "failed", {"error": error})

    async def _finish(
        self,
        lease: DurabilityLease,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> bool:
        async with self._lock(lease.session_id):
            current = await self.state(lease.session_id)
            if current.lease_id != lease.lease_id:
                return False
            await self._append(lease, event, details)
            return True

    async def _append(
        self,
        lease: DurabilityLease,
        event: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "event_type": _EVENT_TYPE,
            "event": event,
            "session_id": lease.session_id,
            "execution_id": lease.execution_id,
            "attempt": lease.attempt,
            "lease_id": lease.lease_id,
            "owner_id": lease.owner_id,
            "timestamp": self._now().isoformat(),
            **(details or {}),
        }
        await asyncio.to_thread(self._append_event, lease.session_id, payload)

    def _append_event(self, session_id: str, payload: dict[str, Any]) -> None:
        client = self._required_client()
        client.append_items(self._session(session_id), items=[payload])

    def _read_events(self, session_id: str) -> list[dict[str, Any]]:
        client = self._required_client()
        return [
            item.data
            for item in client.list_items(
                self._session(session_id), order_by="create_time asc"
            )
            if isinstance(item.data, dict)
            and item.data.get("event_type") == _EVENT_TYPE
        ]

    def _session(self, session_id: str) -> Any:
        durable_id = execution_id(session_id)
        if cached := self._sessions.get(durable_id):
            return cached
        client = self._required_client()
        try:
            session = client.get_session(session_id=durable_id)
        except Exception as error:
            if not _is_not_found(error):
                raise
            try:
                session = client.create_session(
                    actor_id=os.getenv(_SESSION_ACTOR_ENV) or session_id,
                    session_id=durable_id,
                    metadata={
                        "client": "mason-demo-durability",
                        "public_session_id": session_id,
                    },
                )
            except Exception as create_error:
                if not _is_already_exists(create_error):
                    raise
                session = client.get_session(session_id=durable_id)
        self._sessions[durable_id] = session
        return session

    def _state_from_events(
        self, session_id: str, events: list[dict[str, Any]]
    ) -> DurabilityState:
        durable_id = execution_id(session_id)
        starts = [
            (index, event)
            for index, event in enumerate(events)
            if event.get("event") == "attempt_started"
        ]
        if not starts:
            return DurabilityState(
                session_id=session_id,
                execution_id=durable_id,
                status="not_started",
                attempt=0,
                lease_id=None,
                owner_id=None,
                heartbeat_at=None,
                heartbeat_age_seconds=None,
                heartbeat_fresh=False,
                event_count=len(events),
                recent_events=events[-8:],
            )

        start_index, started = starts[-1]
        lease_id = str(started["lease_id"])
        lease_events = [
            event
            for event in events[start_index:]
            if str(event.get("lease_id")) == lease_id
        ]
        terminal = next(
            (
                event
                for event in reversed(lease_events)
                if event.get("event") in {"completed", "failed"}
            ),
            None,
        )
        last_heartbeat = next(
            (
                event
                for event in reversed(lease_events)
                if event.get("event") in {"attempt_started", "heartbeat"}
            ),
            started,
        )
        heartbeat_at = str(last_heartbeat["timestamp"])
        heartbeat_age = max(
            0.0, (self._now() - _parse_time(heartbeat_at)).total_seconds()
        )
        heartbeat_fresh = terminal is None and heartbeat_age <= stale_seconds()
        status = (
            str(terminal["event"])
            if terminal
            else "running"
            if heartbeat_fresh
            else "stopped"
        )
        return DurabilityState(
            session_id=session_id,
            execution_id=durable_id,
            status=status,
            attempt=int(started.get("attempt", 1)),
            lease_id=lease_id,
            owner_id=str(started.get("owner_id") or "") or None,
            heartbeat_at=heartbeat_at,
            heartbeat_age_seconds=heartbeat_age,
            heartbeat_fresh=heartbeat_fresh,
            event_count=len(events),
            recent_events=events[-8:],
        )

    def _required_client(self) -> Any:
        if self._client is None:
            raise ValueError(f"Set {_SESSION_STORE_ENV} to enable durable recovery.")
        return self._client

    def _lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    def _now(self) -> datetime:
        value = self._clock()
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_not_found(error: Exception) -> bool:
    code = str(getattr(error, "error_code", "")).upper()
    return (
        code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}
        or "not found" in str(error).lower()
    )


def _is_already_exists(error: Exception) -> bool:
    code = str(getattr(error, "error_code", "")).upper()
    return (
        code in {"ALREADY_EXISTS", "RESOURCE_ALREADY_EXISTS"}
        or "already exists" in str(error).lower()
    )
