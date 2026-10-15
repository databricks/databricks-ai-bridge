"""Ergonomic resource wrapper over the generated service API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from databricks_agentkit._api_client import _AgentBricksApiClient
from databricks_agentkit.memory_store import MemoryStores
from databricks_agentkit.session_store import SessionStores

if TYPE_CHECKING:
    # Imported lazily elsewhere: pulling databricks.sdk costs ~0.7s, so keeping it out of the
    # import path lets local CLI commands (help, init, tools, bind, completion) start fast.
    from databricks.sdk import WorkspaceClient


class AgentKitClient:
    """High-level client for AgentKit memory and session APIs.

    ``WorkspaceClient.mason`` will replace the private transport once the generated SDK is
    released; the resource-oriented public surface remains unchanged.

    Args:
        workspace_client: An authenticated Databricks workspace client. When omitted,
            the Databricks SDK's default authentication resolution is used.
    """

    def __init__(self, workspace_client: Optional[WorkspaceClient] = None) -> None:
        api = _AgentBricksApiClient(workspace_client=workspace_client)
        self.memory_stores = MemoryStores(api)
        self.session_stores = SessionStores(api)
