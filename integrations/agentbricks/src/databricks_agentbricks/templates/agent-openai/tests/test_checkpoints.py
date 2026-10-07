"""Checkpoint ordering and lease fencing over the example's append-only Session Store."""

import asyncio
import copy
import threading
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agent import checkpoints as module
from databricks.sdk.errors import AlreadyExists, NotFound, ResourceAlreadyExists


class _Store:
    def __init__(self):
        self.sessions, self.items = {}, {}

    def set_session_store(self, name):
        assert name == "test-store"
        return self

    def get_session(self, *, session_id):
        if session_id not in self.sessions:
            raise NotFound("missing")
        return self.sessions[session_id]

    def create_session(self, *, session_id, actor_id, metadata):
        if session_id in self.sessions:
            raise AlreadyExists("exists")
        session = SimpleNamespace(session_id=session_id, actor_id=actor_id, metadata=metadata)
        self.sessions[session_id], self.items[session_id] = session, []
        return session

    def append_items(self, session, *, items):
        self.items[session.session_id].extend(copy.deepcopy(items))

    def list_items(self, session, *, order_by=None):
        assert order_by == "create_time asc"
        # Physical append order includes delayed stale writes.
        return [
            SimpleNamespace(data=copy.deepcopy(item)) for item in self.items[session.session_id]
        ]


@pytest.fixture
def checkpoints(monkeypatch):
    store = _Store()
    monkeypatch.setattr(module, "SessionStoreClient", lambda: store)
    monkeypatch.setattr(module, "resolve_session_store", lambda: "test-store")
    monkeypatch.setattr(module, "_local_items", {})

    def create(invocation="first", attempt=1, *, session="session", actor="actor", emit=None):
        return module.Checkpoints(session, actor, invocation, attempt, emit or AsyncMock())

    return SimpleNamespace(create=create, store=store)


@asynccontextmanager
async def _delay_append(monkeypatch, checkpoint, *, state, reservation=False):
    """Let a newer owner write while an older HTTP append remains in flight."""
    loop, entered, release = asyncio.get_running_loop(), asyncio.Event(), threading.Event()
    append = checkpoint._append

    def delayed(item):
        if ("state" not in item) == reservation:
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), "timed out waiting to finish delayed append"
        append(item)

    monkeypatch.setattr(checkpoint, "_append", delayed)
    task = asyncio.create_task(checkpoint.save(state))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        yield task
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_checkpoint_namespace_isolates_sessions_actors_and_transcripts(checkpoints):
    variants = [("session", "actor"), ("other-session", "actor"), ("session", "other-actor")]
    for session, actor in variants:
        await checkpoints.create(session=session, actor=actor).save({"owner": [session, actor]})
    assert len(checkpoints.store.sessions) == 3
    for session, actor in variants:
        assert await checkpoints.create("reader", session=session, actor=actor).load() == {
            "owner": [session, actor]
        }
    for private in checkpoints.store.sessions.values():
        assert private.session_id not in {session for session, _ in variants}
        assert private.metadata["client"] == "agentbricks-openai-checkpoints"


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [AlreadyExists, ResourceAlreadyExists])
async def test_concurrent_first_session_creation_reuses_winner(checkpoints, monkeypatch, conflict):
    create = checkpoints.store.create_session

    def concurrent_create(**kwargs):
        create(**kwargs)  # Another caller wins between the initial GET and our POST.
        raise conflict("session already created")

    monkeypatch.setattr(checkpoints.store, "create_session", concurrent_create)
    await checkpoints.create().save({"value": "saved"})
    assert len(checkpoints.store.sessions) == 1
    assert await checkpoints.create("reader").load() == {"value": "saved"}


@pytest.mark.asyncio
@pytest.mark.parametrize("local", [False, True])
async def test_recreated_helper_continues_append_sequence(checkpoints, monkeypatch, local):
    if local:
        monkeypatch.setattr(module, "resolve_session_store", lambda: None)
    first = checkpoints.create()
    await first.save({"value": 1})
    recreated = checkpoints.create()
    loaded = await recreated.load()
    loaded["value"] = "mutated by caller"
    assert await recreated.load() == {"value": 1}
    await recreated.save({"value": 2})
    assert await checkpoints.create("reader").load() == {"value": 2}
    records = next(
        iter(module._local_items.values() if local else checkpoints.store.items.values())
    )
    assert [item["state"] for item in records if "state" in item] == [{"value": 1}, {"value": 2}]
    assert [item["version"] for item in records if "state" in item] == [[1, 1, 1], [1, 1, 2]]


@pytest.mark.asyncio
async def test_fencing_rejection_exposes_no_checkpoint(checkpoints):
    emit = AsyncMock(side_effect=RuntimeError("lost lease"))
    with pytest.raises(RuntimeError, match="lost lease"):
        await checkpoints.create(emit=emit).save({"secret": "not committed"})
    emit.assert_awaited_once_with({"type": "checkpoint"})
    assert await checkpoints.create("reader").load() is None
    assert all("state" not in item for items in checkpoints.store.items.values() for item in items)


@pytest.mark.asyncio
async def test_delayed_first_reservation_cannot_publish_after_losing_lease(
    checkpoints, monkeypatch
):
    stale = checkpoints.create(emit=AsyncMock(side_effect=RuntimeError("lost lease")))
    async with _delay_append(
        monkeypatch, stale, state={"value": "stale"}, reservation=True
    ) as task:
        await checkpoints.create("newer").save({"value": "newer"})
    with pytest.raises(RuntimeError, match="lost lease"):
        await task
    assert await checkpoints.create("reader").load() == {"value": "newer"}


@pytest.mark.asyncio
@pytest.mark.parametrize("later_invocation", [False, True])
async def test_delayed_state_never_overwrites_newer_checkpoint(
    checkpoints, monkeypatch, later_invocation
):
    old = checkpoints.create()
    async with _delay_append(monkeypatch, old, state={"value": "stale"}) as task:
        if later_invocation:
            # Recovery fails before committing state; a later invocation must still outrank the old write.
            recovery = checkpoints.create(
                attempt=2, emit=AsyncMock(side_effect=RuntimeError("lost lease"))
            )
            with pytest.raises(RuntimeError, match="lost lease"):
                await recovery.save({"value": "failed recovery"})
            newer = checkpoints.create("newer")
        else:
            newer = checkpoints.create(attempt=2)
        await newer.save({"value": "newer"})
    await task
    records = next(iter(checkpoints.store.items.values()))
    assert records[-1]["state"] == {"value": "stale"}  # Physically last, logically older.
    assert await checkpoints.create("reader").load() == {"value": "newer"}
