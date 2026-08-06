from __future__ import annotations

from typing import Any

import pytest
from agents.memory.session import Session

from databricks_openai.agents import DatabricksSession


class FakeSessionStoreClient:
    def __init__(self) -> None:
        self.ensure_calls: list[tuple[str, dict[str, Any]]] = []
        self.append_calls: list[tuple[str, list[dict[str, Any]]]] = []
        self.events: list[dict[str, Any]] = []
        self.popped: dict[str, Any] | None = None
        self.clear_calls: list[str] = []
        self.closed = False

    async def ensure_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        self.ensure_calls.append((session_id, kwargs))
        return {"session_id": session_id}

    async def list_events(self, session_id: str) -> list[dict[str, Any]]:
        return list(self.events)

    async def append_events(
        self, session_id: str, events: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        self.append_calls.append((session_id, events))
        return events

    async def pop_event(self, session_id: str) -> dict[str, Any] | None:
        return self.popped

    async def clear_events(self, session_id: str) -> None:
        self.clear_calls.append(session_id)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_maps_openai_items_to_opaque_events() -> None:
    client = FakeSessionStoreClient()
    session = DatabricksSession(
        "session-1",
        client=client,  # type: ignore[arg-type]
        display_name="Support",
        metadata={"app": "demo"},
        user_id="user-1",
        parent_session="sessions/parent",
    )
    items = [
        {"type": "message", "role": "user", "content": "Hello"},
        {"type": "function_call", "name": "lookup"},
    ]

    await session.add_items(items)
    await session.add_items([])

    assert client.ensure_calls == [
        (
            "session-1",
            {
                "display_name": "Support",
                "metadata": {"sdk": "openai-agents", "app": "demo"},
                "user_id": "user-1",
                "parent_session": "sessions/parent",
            },
        )
    ]
    assert client.append_calls == [
        (
            "session-1",
            [
                {"type": "message", "role": "user", "data": items[0]},
                {"type": "function_call", "data": items[1]},
            ],
        )
    ]


@pytest.mark.asyncio
async def test_get_items_respects_explicit_and_session_limits() -> None:
    class Settings:
        limit = 2

    client = FakeSessionStoreClient()
    client.events = [{"data": {"type": "message", "content": str(index)}} for index in range(4)]
    session = DatabricksSession(
        "session-1",
        client=client,  # type: ignore[arg-type]
        session_settings=Settings(),
    )

    assert [item["content"] for item in await session.get_items()] == ["2", "3"]
    assert [item["content"] for item in await session.get_items(1)] == ["3"]
    assert await session.get_items(0) == []
    assert len(client.ensure_calls) == 1


@pytest.mark.asyncio
async def test_pop_clear_and_close_injected_client() -> None:
    client = FakeSessionStoreClient()
    client.popped = {"data": {"type": "message", "content": "latest"}}
    session = DatabricksSession("session-1", client=client)  # type: ignore[arg-type]

    assert await session.pop_item() == {"type": "message", "content": "latest"}
    await session.clear_session()
    await session.aclose()

    assert client.clear_calls == ["session-1"]
    assert not client.closed


def test_generates_session_id_when_omitted() -> None:
    session = DatabricksSession(client=FakeSessionStoreClient())  # type: ignore[arg-type]
    assert session.session_id.startswith("session-")
    assert isinstance(session, Session)
