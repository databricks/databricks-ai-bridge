"""Ergonomic resource wrapper over the generated Mason API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from databricks_mason._api_client import _MasonApiClient
from databricks_mason.memory_store import MemoryStores
from databricks_mason.session_store import SessionStores

if TYPE_CHECKING:
    # Imported lazily elsewhere: pulling databricks.sdk costs ~0.7s, so keeping it out of the
    # import path lets local CLI commands (help, init, tools, bind, completion) start fast.
    from databricks.sdk import WorkspaceClient


class MasonClient:
    """High-level Mason client.

    ``WorkspaceClient.mason`` will replace the private transport once the generated
    Mason SDK is released; the resource-oriented public surface remains unchanged.

    Args:
        workspace_client: An authenticated Databricks workspace client. When omitted,
            the Databricks SDK's default authentication resolution is used.
    """

    def __init__(self, workspace_client: Optional[WorkspaceClient] = None) -> None:
        api = _MasonApiClient(workspace_client=workspace_client)
        self.memory_stores = MemoryStores(api)
        self.session_stores = SessionStores(api)
