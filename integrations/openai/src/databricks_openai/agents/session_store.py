"""OpenAI Agents SDK adapter for the experimental Databricks Session Store."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from databricks.sdk import WorkspaceClient
from databricks_ai_bridge.session_store import DatabricksSessionStoreClient
from databricks_ai_bridge.utils.annotations import experimental


@experimental
class DatabricksSession:
    """OpenAI Agents SDK ``Session`` backed by Databricks.

    The adapter preserves each Agents SDK item as opaque JSON in a
    ``SessionEvent``. Databricks owns event IDs and sequence numbers; the SDK
    continues to see its original item objects.

    Args:
        session_id: Stable session ID. A client-generated ID is used when
            omitted.
        client: Shared Session Store client. When omitted, the adapter creates
            and owns one.
        workspace_client: Workspace client used when creating ``client``.
        base_url: Optional Session Store API prefix.
        traffic_id: Optional ``x-databricks-traffic-id`` value for LiteSwap.
        display_name: Display name assigned when the session is first created.
        metadata: Metadata assigned when the session is first created.
        user_id: Optional application-level grouping key.
        parent_session: Optional parent resource name for a subagent session.
        session_settings: Optional OpenAI Agents SDK session settings object.
    """

    def __init__(
        self,
        session_id: str | None = None,
        *,
        client: DatabricksSessionStoreClient | None = None,
        workspace_client: WorkspaceClient | None = None,
        base_url: str | None = None,
        traffic_id: str | None = None,
        display_name: str = "OpenAI Agents SDK session",
        metadata: dict[str, str] | None = None,
        user_id: str | None = None,
        parent_session: str | None = None,
        session_settings: Any | None = None,
    ) -> None:
        self.session_id = session_id or f"session-{uuid4().hex}"
        self.session_settings = session_settings
        self._client = client or DatabricksSessionStoreClient(
            workspace_client=workspace_client,
            base_url=base_url,
            traffic_id=traffic_id,
        )
        self._owns_client = client is None
        self._display_name = display_name
        self._metadata = {"sdk": "openai-agents", **(metadata or {})}
        self._user_id = user_id
        self._parent_session = parent_session
        self._session_created = False
        self._create_lock = asyncio.Lock()

    async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return the latest items in chronological order."""
        await self._ensure_session()
        events = await self._client.list_events(self.session_id)
        items = [self._event_data(event) for event in events]
        effective_limit = limit
        if effective_limit is None:
            effective_limit = getattr(self.session_settings, "limit", None)
        if effective_limit is None:
            return items
        if effective_limit <= 0:
            return []
        return items[-effective_limit:]

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        """Append OpenAI Agents SDK items to the remote transcript."""
        if not items:
            return
        await self._ensure_session()
        events: list[dict[str, Any]] = []
        for item in items:
            event: dict[str, Any] = {
                "type": str(item.get("type") or "openai_item"),
                "data": item,
            }
            role = item.get("role")
            if isinstance(role, str):
                event["role"] = role
            events.append(event)
        await self._client.append_events(self.session_id, events)

    async def pop_item(self) -> dict[str, Any] | None:
        """Remove and return the latest SDK item."""
        await self._ensure_session()
        event = await self._client.pop_event(self.session_id)
        return self._event_data(event) if event is not None else None

    async def clear_session(self) -> None:
        """Remove all SDK items while preserving the session resource."""
        await self._ensure_session()
        await self._client.clear_events(self.session_id)

    async def aclose(self) -> None:
        """Close the internally created Session Store client, if any."""
        if self._owns_client:
            await self._client.aclose()

    async def _ensure_session(self) -> None:
        if self._session_created:
            return
        async with self._create_lock:
            if self._session_created:
                return
            await self._client.ensure_session(
                self.session_id,
                display_name=self._display_name,
                metadata=self._metadata,
                user_id=self._user_id,
                parent_session=self._parent_session,
            )
            self._session_created = True

    @staticmethod
    def _event_data(event: dict[str, Any]) -> dict[str, Any]:
        data = event.get("data")
        if not isinstance(data, dict):
            raise TypeError("OpenAI session event data must be a JSON object")
        return data
