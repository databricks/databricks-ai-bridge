"""Workspace facts used to associate local deployment records with an authenticated client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from databricks_agentkit._api_client import _AgentBricksApiClient


@dataclass(frozen=True)
class WorkspaceIdentity:
    host: str
    workspace_id: str

    @classmethod
    def from_client(cls, client: _AgentBricksApiClient) -> WorkspaceIdentity | None:
        try:
            host = client.host
            workspace_id = client.workspace_client.config.workspace_id
        except AttributeError:
            return None
        return cls.from_dict({"host": host, "workspace_id": workspace_id})

    @classmethod
    def from_dict(cls, value: Any) -> WorkspaceIdentity | None:
        if not isinstance(value, dict):
            return None
        host, workspace_id = value.get("host"), value.get("workspace_id")
        # A shared regional host alone does not establish which workspace a record belongs to.
        if not isinstance(host, str) or not isinstance(workspace_id, str) or not workspace_id:
            return None
        host = host.rstrip("/").lower()
        if not host or host == "unknown":
            return None
        return cls(host=host, workspace_id=workspace_id)
