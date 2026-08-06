from __future__ import annotations

from typing import Any

import pytest

from databricks_ai_bridge.session_store.claude import DatabricksClaudeSessionStore


class FakeSessionStoreClient:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.deleted: list[tuple[str, bool]] = []

    async def ensure_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        session = {"session_id": session_id, **kwargs}
        self.sessions.setdefault(session_id, session)
        return self.sessions[session_id]

    async def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    async def append_events(
        self, session_id: str, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.events.setdefault(session_id, []).extend(events)
        return events

    async def list_events(self, session_id: str) -> list[dict[str, Any]]:
        return list(self.events.get(session_id, []))

    async def list_sessions(self, **kwargs: Any) -> list[dict[str, Any]]:
        parent_session = kwargs.get("parent_session")
        result = list(self.sessions.values())
        if parent_session is not None:
            result = [
                session for session in result if session.get("parent_session") == parent_session
            ]
        filter_expression = kwargs.get("filter_expression")
        if filter_expression is not None:
            project_hash = filter_expression.rsplit('"', 2)[1]
            result = [
                session
                for session in result
                if session.get("metadata", {}).get("claude_project_hash") == project_hash
            ]
        return result

    async def delete_session(self, session_id: str, *, force: bool = False) -> None:
        self.deleted.append((session_id, force))


@pytest.mark.asyncio
async def test_round_trips_entries_and_deduplicates_uuids() -> None:
    client = FakeSessionStoreClient()
    store = DatabricksClaudeSessionStore(client)  # type: ignore[arg-type]
    key = {"project_key": "project-a", "session_id": "session-1"}

    await store.append(
        key,
        [
            {"type": "user", "uuid": "duplicate", "message": {"role": "user"}},
            {"type": "assistant", "uuid": "new", "message": {"role": "assistant"}},
            {"type": "assistant", "uuid": "new", "message": {"role": "assistant"}},
        ],
    )
    await store.append(
        key,
        [
            {"type": "user", "uuid": "duplicate", "message": {"role": "user"}},
            {"type": "progress", "timestamp": "now"},
        ],
    )

    entries = await store.load(key)
    assert entries == [
        {"type": "user", "uuid": "duplicate", "message": {"role": "user"}},
        {"type": "assistant", "uuid": "new", "message": {"role": "assistant"}},
        {"type": "progress", "timestamp": "now"},
    ]
    physical_id = store._physical_id(key)
    assert client.events[physical_id][0]["role"] == "user"
    assert client.sessions[physical_id]["metadata"]["claude_session_id"] == "session-1"


@pytest.mark.asyncio
async def test_maps_subpaths_to_child_sessions() -> None:
    client = FakeSessionStoreClient()
    store = DatabricksClaudeSessionStore(client)  # type: ignore[arg-type]
    main_key = {"project_key": "project-a", "session_id": "session-1"}
    child_key = {**main_key, "subpath": "subagents/agent-a"}

    await store.append(child_key, [{"type": "assistant", "uuid": "child-1"}])

    main_id = store._main_id(main_key)
    child_id = store._physical_id(child_key)
    assert client.sessions[child_id]["parent_session"] == f"sessions/{main_id}"
    assert await store.list_subkeys(main_key) == ["subagents/agent-a"]
    assert await store.load(child_key) == [{"type": "assistant", "uuid": "child-1"}]

    await store.delete(child_key)
    await store.delete(main_key)
    assert client.deleted == [(child_id, False), (main_id, True)]


@pytest.mark.asyncio
async def test_project_key_is_part_of_physical_identity_and_listing_scope() -> None:
    client = FakeSessionStoreClient()
    store = DatabricksClaudeSessionStore(client)  # type: ignore[arg-type]
    first = {"project_key": "project-a", "session_id": "same-session"}
    second = {"project_key": "project-b", "session_id": "same-session"}

    assert store._physical_id(first) != store._physical_id(second)
    await store.append(first, [{"type": "user"}])
    await store.append(second, [{"type": "user"}])
    first_id = store._physical_id(first)
    client.sessions[first_id]["update_time"] = "2026-08-06T12:00:00Z"

    listed = await store.list_sessions("project-a")
    assert listed == [{"session_id": "same-session", "mtime": 1786017600000}]


@pytest.mark.asyncio
async def test_load_returns_none_for_unknown_key() -> None:
    store = DatabricksClaudeSessionStore(FakeSessionStoreClient())  # type: ignore[arg-type]
    assert await store.load({"project_key": "project", "session_id": "missing"}) is None


def test_rejects_empty_subpath() -> None:
    with pytest.raises(ValueError, match="subpath"):
        DatabricksClaudeSessionStore._physical_id(
            {"project_key": "project", "session_id": "session", "subpath": ""}
        )
