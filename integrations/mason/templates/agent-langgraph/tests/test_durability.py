from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from agent.mason.durability import SessionStoreDurabilityLog, execution_id


class _ApiError(Exception):
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code


class _FakeSessionStoreClient:
    def __init__(self):
        self.sessions = {}
        self.items = {}

    def set_session_store(self, name):
        self.store_name = name
        return self

    def get_session(self, *, session_id):
        if session_id not in self.sessions:
            raise _ApiError("not found", "NOT_FOUND")
        return self.sessions[session_id]

    def create_session(self, *, actor_id, session_id=None, metadata=None):
        if session_id in self.sessions:
            raise _ApiError("already exists", "ALREADY_EXISTS")
        session = SimpleNamespace(
            session_id=session_id, actor_id=actor_id, metadata=metadata or {}
        )
        self.sessions[session_id] = session
        self.items[session_id] = []
        return session

    def append_items(self, session, *, items):
        rows = self.items[session.session_id]
        rows.extend(SimpleNamespace(data=item) for item in items)

    def list_items(self, session, *, order_by=None):
        assert order_by == "create_time asc"
        yield from self.items[session.session_id]


@pytest.mark.asyncio
async def test_heartbeat_becomes_stale_and_new_attempt_takes_over(monkeypatch):
    monkeypatch.setenv("MASON_DEMO_HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("MASON_DEMO_STALE_SECONDS", "3")
    now = datetime(2026, 8, 28, tzinfo=timezone.utc)
    client = _FakeSessionStoreClient()
    log = SessionStoreDurabilityLog("sessions", client=client, clock=lambda: now)

    first = await log.claim("session-1", "process-1")
    running = await log.state("session-1")
    assert running.status == "running"
    assert running.attempt == 1
    assert running.owner_id == "process-1"
    assert running.heartbeat_fresh is True

    now += timedelta(seconds=4)
    stopped = await log.state("session-1")
    assert stopped.status == "stopped"
    assert stopped.heartbeat_fresh is False

    second = await log.claim("session-1", "process-2")
    assert second.attempt == 2
    assert await log.heartbeat(first) is False
    assert await log.heartbeat(second) is True
    assert await log.complete(second) is True

    completed = await log.state("session-1")
    assert completed.status == "completed"
    assert completed.attempt == 2
    assert completed.owner_id == "process-2"
    assert completed.execution_id == execution_id("session-1")


@pytest.mark.asyncio
async def test_fresh_heartbeat_blocks_another_claim():
    now = datetime(2026, 8, 28, tzinfo=timezone.utc)
    log = SessionStoreDurabilityLog(
        "sessions",
        client=_FakeSessionStoreClient(),
        clock=lambda: now,
    )

    await log.claim("session-1", "process-1")
    with pytest.raises(ValueError, match="fresh heartbeat"):
        await log.claim("session-1", "process-2")
