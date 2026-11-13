"""Best-effort links to the workspace's gated custom-agent details UI."""

from __future__ import annotations

import queue
import threading
from typing import Any, cast
from urllib.parse import quote, urlencode

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.workspace_identity import WorkspaceIdentity

_DETAILS_LOOKUP_TIMEOUT_S = 2.0
_AGENT_UI_FLAG = "databricks.fe.customagents.agentCentricView.enable"
_AGENT_PREVIEW = "agent_memory_sessions"


def _enabled(snapshot: dict[str, Any], name: str) -> bool:
    items = snapshot.get("flag_items")
    item = items.get(name) if isinstance(items, dict) else None
    if isinstance(item, dict):
        value = item.get("valueJsonStr")
    else:
        flags = snapshot.get("flags")
        value = flags.get(name) if isinstance(flags, dict) else None
    return value is True or value == "true"


class AgentDetailsClient:
    def __init__(self, api_client_provider: ApiClientProvider) -> None:
        self._api_client_provider = api_client_provider

    def get_url(
        self, deployment: str, *, expected_workspace: WorkspaceIdentity | None = None
    ) -> str | None:
        return self.get_urls([deployment], expected_workspace=expected_workspace).get(deployment)

    def get_urls(
        self, deployments: list[str], *, expected_workspace: WorkspaceIdentity | None = None
    ) -> dict[str, str]:
        """Share one bounded flag lookup across all rows in a command's output."""
        if not deployments:
            return {}
        urls: queue.Queue[dict[str, str]] = queue.Queue(maxsize=1)

        def fetch() -> None:
            try:
                urls.put(self._fetch_urls(deployments, expected_workspace))
            except Exception:  # noqa: BLE001 - optional UI discovery must not change deploy outcomes
                urls.put({})

        threading.Thread(target=fetch, name="agentbricks-details-link", daemon=True).start()
        try:
            return urls.get(timeout=_DETAILS_LOOKUP_TIMEOUT_S)
        except queue.Empty:
            return {}

    def _fetch_urls(
        self, deployments: list[str], expected_workspace: WorkspaceIdentity | None
    ) -> dict[str, str]:
        client = self._api_client_provider.get()
        if (
            expected_workspace is not None
            and WorkspaceIdentity.from_client(client) != expected_workspace
        ):
            return {}
        workspace = client.workspace_client
        workspace_id = workspace.config.workspace_id
        # Request named items so SAFE's compact, hashed browser representation is unnecessary.
        query = {"shouldHashAndEncodeFlagItems": "false"}
        if workspace_id:
            query["o"] = str(workspace_id)
        snapshot = workspace.api_client.do("GET", "/api/2.0/ui-flags", query=query)
        if not isinstance(snapshot, dict):
            return {}
        flags = cast(dict[str, Any], snapshot)
        if not all(_enabled(flags, flag) for flag in (_AGENT_UI_FLAG, _AGENT_PREVIEW)):
            return {}
        host = client.host
        if not host or host == "unknown":
            return {}
        suffix = f"?{urlencode({'o': workspace_id})}" if workspace_id else ""
        return {
            deployment: f"{host.rstrip('/')}/ml/agents/details/{quote(deployment, safe='')}{suffix}"
            for deployment in deployments
        }
