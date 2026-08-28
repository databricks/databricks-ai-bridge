"""Shared payload builders for typed Mason resource tests."""

from typing import Any, Optional, cast
from unittest import mock

from databricks_mason import MasonClient
from databricks_mason.memory_store import MemoryStoreClient
from databricks_mason.session_store import SessionStoreClient

STORE_ID = "15402663-997b-4300-b695-46913ad90c9f"
MEM_STORE_NAME = f"memory-stores/{STORE_ID}"
ENTRY_ID = "absc-2edddvd"
ENTRY_NAME = f"{MEM_STORE_NAME}/entries/{ENTRY_ID}"
SESSION_STORE = "support-agent-sessions"
SESSION_ID = "case-456"


def mem_store_payload(
    *,
    name: str = MEM_STORE_NAME,
    display_name: str = "coding_agent_memory",
    description: str = "d",
) -> dict[str, Any]:
    return {
        "name": name,
        "display_name": display_name,
        "workspace_id": 123,
        "owner_user_id": "456",
        "created_at": 1770000000000,
        "updated_at": 1770000600000,
        "description": description,
    }


def entry_payload(
    *,
    name: str = ENTRY_NAME,
    actor_id: str = "alice",
    session_id: Optional[str] = "s1",
    path: str = "/m/p.md",
    content: str = "c",
) -> dict[str, Any]:
    payload = {
        "name": name,
        "actor_id": actor_id,
        "path": path,
        "content": content,
        "description": "desc",
        "source_type": "MANAGED_MEMORY_ENTRY_SOURCE_TYPE_AGENT",
        "create_time": "2026-08-14T01:02:03Z",
        "update_time": "2026-08-14T02:03:04.500Z",
    }
    if session_id is not None:
        payload["session_id"] = session_id
    return payload


def session_store_payload(
    *,
    name: str = SESSION_STORE,
    description: str = "Support history",
    metadata: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    return {
        "session_store_name": name,
        "session_store_id": STORE_ID,
        "creator_user_id": "123",
        "create_time": "2026-08-14T01:02:03Z",
        "update_time": "2026-08-14T02:03:04.500Z",
        "description": description,
        "metadata": metadata or {"environment": "poc"},
    }


def session_payload(
    *,
    session_id: str = SESSION_ID,
    actor_id: str = "customer-123",
    metadata: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    return {
        "session_store_name": SESSION_STORE,
        "session_id": session_id,
        "actor_id": actor_id,
        "root_session_id": session_id,
        "metadata": metadata or {"channel": "chat"},
        "create_time": "2026-08-14T03:04:05Z",
        "update_time": "2026-08-14T04:05:06Z",
        "last_activity_time": "2026-08-14T05:06:07Z",
    }


def item_payload(*, item_id: str = "item-1", data: Any = None) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "data": data or {"type": "message", "role": "user", "content": "Help"},
        "create_time": "2026-08-14T06:07:08Z",
    }


def resource_client() -> tuple[MasonClient, mock.MagicMock]:
    api = mock.MagicMock(spec=MasonClient)
    client = cast(MasonClient, api)
    client.memory_stores = MemoryStoreClient(client)
    client.session_stores = SessionStoreClient(client)
    return client, api
