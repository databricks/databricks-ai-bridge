"""Claude Agent SDK adapter for the experimental Databricks Session Store."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from databricks.sdk import WorkspaceClient

from databricks_ai_bridge.session_store.client import DatabricksSessionStoreClient
from databricks_ai_bridge.utils.annotations import experimental


@experimental
class DatabricksClaudeSessionStore:
    """Claude Agent SDK ``SessionStore`` backed by Databricks.

    This class is intentionally duck typed so importing it does not require the
    Claude Agent SDK. Pass an instance as ``ClaudeAgentOptions(session_store=...)``.

    Claude's ``project_key`` and ``session_id`` identify the main Databricks
    session. Each Claude ``subpath`` is represented as a direct child session,
    which lets resume discover subagent transcripts via ``list_subkeys``.

    Args:
        client: Shared Session Store REST client. When omitted, the adapter
            creates and owns one.
        workspace_client: Workspace client used when creating ``client``.
        base_url: Optional Session Store API prefix.
        traffic_id: Optional ``x-databricks-traffic-id`` value for LiteSwap.
        display_name: Display name assigned to newly created main sessions.
        metadata: Additional metadata copied onto main and child sessions.
    """

    def __init__(
        self,
        client: DatabricksSessionStoreClient | None = None,
        *,
        workspace_client: WorkspaceClient | None = None,
        base_url: str | None = None,
        traffic_id: str | None = None,
        display_name: str = "Claude Agent SDK session",
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._client = client or DatabricksSessionStoreClient(
            workspace_client=workspace_client,
            base_url=base_url,
            traffic_id=traffic_id,
        )
        self._owns_client = client is None
        self._display_name = display_name
        self._metadata = dict(metadata or {})

    async def append(self, key: dict[str, str], entries: list[dict[str, Any]]) -> None:
        """Append opaque Claude JSONL entries to a main or subagent transcript."""
        if not entries:
            return
        physical_id = await self._ensure_key(key)

        # Claude retries UUID-bearing lines. Until the Session Store contract exposes a
        # caller idempotency key per event, avoid ordinary duplicates with a read check.
        # This is best effort and does not make concurrent writers atomic.
        existing = await self._client.list_events(physical_id)
        existing_uuids = {
            data.get("uuid") for event in existing if isinstance((data := event.get("data")), dict)
        }
        new_entries: list[dict[str, Any]] = []
        for entry in entries:
            entry_uuid = entry.get("uuid")
            if entry_uuid and entry_uuid in existing_uuids:
                continue
            new_entries.append(entry)
            if entry_uuid:
                existing_uuids.add(entry_uuid)
        if not new_entries:
            return

        events: list[dict[str, Any]] = []
        for entry in new_entries:
            event: dict[str, Any] = {
                "type": str(entry.get("type") or "claude_transcript_line"),
                "data": entry,
            }
            message = entry.get("message")
            if isinstance(message, dict) and isinstance(message.get("role"), str):
                event["role"] = message["role"]
            events.append(event)
        await self._client.append_events(physical_id, events)

    async def load(self, key: dict[str, str]) -> list[dict[str, Any]] | None:
        """Load an opaque Claude transcript for resume."""
        physical_id = self._physical_id(key)
        if not await self._client.session_exists(physical_id):
            return None
        events = await self._client.list_events(physical_id)
        return [self._event_data(event) for event in events]

    async def list_sessions(self, project_key: str) -> list[dict[str, Any]]:
        """List main Claude sessions for one project."""
        project_hash = self._project_hash(project_key)
        sessions = await self._client.list_sessions(
            filter_expression=f'metadata.claude_project_hash = "{project_hash}"'
        )
        result: list[dict[str, Any]] = []
        for session in sessions:
            metadata = session.get("metadata")
            if not isinstance(metadata, dict) or "claude_subpath" in metadata:
                continue
            session_id = metadata.get("claude_session_id")
            if isinstance(session_id, str):
                result.append(
                    {
                        "session_id": session_id,
                        "mtime": self._epoch_millis(session.get("update_time")),
                    }
                )
        return result

    async def delete(self, key: dict[str, str]) -> None:
        """Delete one subagent transcript or cascade-delete a main transcript."""
        await self._client.delete_session(self._physical_id(key), force="subpath" not in key)

    async def list_subkeys(self, key: dict[str, str]) -> list[str]:
        """Return Claude subpaths stored beneath a main transcript."""
        main_id = self._main_id(key)
        sessions = await self._client.list_sessions(parent_session=f"sessions/{main_id}")
        result: list[str] = []
        for session in sessions:
            metadata = session.get("metadata")
            if isinstance(metadata, dict) and isinstance(metadata.get("claude_subpath"), str):
                result.append(metadata["claude_subpath"])
        return result

    async def main_session_exists(self, project_key: str, session_id: str) -> bool:
        """Return whether the main Claude transcript has been created."""
        return await self._client.session_exists(
            self._main_id({"project_key": project_key, "session_id": session_id})
        )

    async def aclose(self) -> None:
        """Close the internally created Session Store client, if any."""
        if self._owns_client:
            await self._client.aclose()

    async def _ensure_key(self, key: dict[str, str]) -> str:
        self._validate_key(key)
        main_id = self._main_id(key)
        project_hash = self._project_hash(key["project_key"])
        metadata = {
            **self._metadata,
            "sdk": "claude-agent-sdk",
            "claude_session_id": key["session_id"],
            "claude_project_hash": project_hash,
        }
        await self._client.ensure_session(
            main_id,
            display_name=self._display_name,
            metadata=metadata,
        )

        subpath = key.get("subpath")
        if subpath is None:
            return main_id
        child_id = self._physical_id(key)
        await self._client.ensure_session(
            child_id,
            display_name=f"Claude subagent transcript: {subpath}",
            metadata={**metadata, "claude_subpath": subpath},
            parent_session=f"sessions/{main_id}",
        )
        return child_id

    @classmethod
    def _main_id(cls, key: dict[str, str]) -> str:
        cls._validate_key(key)
        return f"claude-{cls._project_hash(key['project_key'])}-{key['session_id']}"

    @classmethod
    def _physical_id(cls, key: dict[str, str]) -> str:
        main_id = cls._main_id(key)
        subpath = key.get("subpath")
        if subpath is None:
            return main_id
        if not subpath:
            raise ValueError("Claude SessionKey subpath must not be empty")
        digest = hashlib.sha256(subpath.encode()).hexdigest()[:16]
        return f"{main_id}-sub-{digest}"

    @staticmethod
    def _validate_key(key: dict[str, str]) -> None:
        if not key.get("project_key"):
            raise ValueError("Claude SessionKey project_key must not be empty")
        if not key.get("session_id"):
            raise ValueError("Claude SessionKey session_id must not be empty")

    @staticmethod
    def _project_hash(project_key: str) -> str:
        return hashlib.sha256(project_key.encode()).hexdigest()[:24]

    @staticmethod
    def _event_data(event: dict[str, Any]) -> dict[str, Any]:
        data = event.get("data")
        if not isinstance(data, dict):
            raise TypeError("Claude session event data must be a JSON object")
        return data

    @staticmethod
    def _epoch_millis(timestamp: object) -> int:
        if not isinstance(timestamp, str) or not timestamp:
            return 0
        return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)
