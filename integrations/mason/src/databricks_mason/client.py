"""Authenticated client for the agents/v1 memory and session APIs.

`MasonClient` is Mason's public Python entry point: construct it with a
`.databrickscfg` profile (or rely on the SDK's default auth) and call one method per
API operation. It wraps a databricks-sdk `WorkspaceClient` and returns typed models
over the JSON responses from `/api/agents/v1`. Account-routed profiles retain their
configured vanity host and add the resolved workspace id as a routing header, matching
the authentication behavior of the Databricks CLI.
Deployment is handled separately (deploy.py) since it wraps the `databricks apps` CLI.
"""

from __future__ import annotations

import configparser
import os
import pathlib
from typing import Any, Optional

from databricks.sdk import WorkspaceClient

from databricks_mason import models
from databricks_mason.errors import AgentCliError, wrap_api_error

_BASE = "/api/agents/v1"
_MCP_SERVICES_PATH = "/api/2.1/unity-catalog/mcp-services"


def _query(**kwargs: Any) -> dict[str, Any]:
    """Build a query dict, dropping None and empty values."""
    return {k: v for k, v in kwargs.items() if v is not None and v != ""}


def _as(cls: type, resp: Any) -> Any:
    """Wrap a JSON response in a typed model, passing non-dicts through unchanged."""
    return cls(resp) if isinstance(resp, dict) else resp


def memory_store_path(name: str) -> str:
    """Normalize a store id or name into the `memory-stores/{id}` resource segment.

    Validates client-side so an empty or wrong-typed argument raises a clear error
    instead of building a URL like `memory-stores/` and leaking a raw ENDPOINT_NOT_FOUND.
    """
    raw = (name or "").strip()
    if raw.startswith("memory-stores/"):
        raw = raw[len("memory-stores/") :]
    raw = raw.strip().strip("/")
    if not raw:
        raise AgentCliError("A memory store id or resource name is required.")
    if "/" in raw:
        raise AgentCliError(f"Invalid memory store id or resource name: {name!r}")
    return f"memory-stores/{raw}"


def session_store_path(name: str) -> str:
    """Normalize/validate a session store name into the `session-stores/{name}` segment."""
    raw = (name or "").strip()
    if raw.startswith("session-stores/"):
        raw = raw[len("session-stores/") :]
    raw = raw.strip().strip("/")
    if not raw:
        raise AgentCliError("A session store name is required.")
    return f"session-stores/{raw}"


def memory_entry_path(store: str, entry: str) -> str:
    entry = (entry or "").strip().strip("/")
    if entry.startswith("memory-stores/"):
        return entry
    if not entry:
        raise AgentCliError("A memory entry id or resource name is required.")
    return f"{memory_store_path(store)}/entries/{entry}"


def _profile_host(profile: str) -> Optional[str]:
    config_path = pathlib.Path(
        os.getenv("DATABRICKS_CONFIG_FILE", pathlib.Path.home() / ".databrickscfg")
    )
    parser = configparser.ConfigParser()
    try:
        parser.read(config_path)
    except (OSError, configparser.Error):
        return None
    return parser.get(profile, "host", fallback=None)


def _workspace_client(profile: Optional[str]) -> WorkspaceClient:
    client = WorkspaceClient(profile=profile)
    if not profile or not client.config.workspace_id:
        return client

    configured_host = _profile_host(profile)
    resolved_host = client.config.host
    if not configured_host or configured_host.rstrip("/") == (resolved_host or "").rstrip("/"):
        return client

    workspace_id = str(client.config.workspace_id)
    return WorkspaceClient(
        profile=profile,
        host=configured_host,
        custom_headers={"X-Databricks-Org-Id": workspace_id},
    )


class MasonClient:
    """Thin, authenticated wrapper over the agents/v1 REST surface.

    Example:
        >>> from databricks_mason import MasonClient
        >>> client = MasonClient(profile="my-workspace")
        >>> store = client.create_memory_store("my-store")
        >>> client.list_memory_stores()
    """

    def __init__(self, profile: Optional[str] = None):
        try:
            self._w = _workspace_client(profile)
        except Exception as exc:  # noqa: BLE001 - surfaced as a clean CLI error
            raise AgentCliError(
                f"Could not initialize Databricks auth: {exc}",
                hint="Check your profile (`databricks auth login --profile <name>`) "
                "or pass --profile.",
            ) from exc

    @property
    def host(self) -> str:
        return self._w.config.host or "unknown"

    @property
    def current_user(self) -> str:
        """The authenticated user's name (used to derive the app source workspace path)."""
        return str(self._w.current_user.me().user_name or "unknown")

    def ensure_workspace_dir(self, path: str) -> None:
        """Create a workspace directory (and parents), idempotently.

        MLflow's create_experiment won't create the intermediate folder for a nested experiment
        path, so callers make the parent dir first.
        """
        self._w.workspace.mkdirs(path)

    def _do(
        self, method: str, path: str, *, query: Optional[dict] = None, body: Optional[dict] = None
    ) -> Any:
        try:
            return self._w.api_client.do(method, path, query=query, body=body)
        except Exception as exc:  # noqa: BLE001 - normalized to AgentCliError
            raise wrap_api_error(exc) from exc

    # --- Unity Catalog MCP Services -----------------------------------------

    def list_mcp_services(
        self, schema: str = "system.ai", page_token: Optional[str] = None
    ) -> dict:
        """List MCP Services visible to the user in a Unity Catalog schema."""
        return self._do(
            "GET",
            _MCP_SERVICES_PATH,
            query=_query(parent=f"schemas/{schema}", page_token=page_token),
        )

    # --- memory stores -------------------------------------------------------

    def create_memory_store(
        self, display_name: str, description: Optional[str] = None
    ) -> models.MemoryStore:
        body = _query(display_name=display_name, description=description)
        return _as(models.MemoryStore, self._do("POST", f"{_BASE}/memory-stores", body=body))

    def get_memory_store(self, name: str) -> models.MemoryStore:
        return _as(models.MemoryStore, self._do("GET", f"{_BASE}/{memory_store_path(name)}"))

    def list_memory_stores(
        self, page_size: Optional[int] = None, page_token: Optional[str] = None
    ) -> models.MemoryStoreList:
        return _as(
            models.MemoryStoreList,
            self._do(
                "GET",
                f"{_BASE}/memory-stores",
                query=_query(page_size=page_size, page_token=page_token),
            ),
        )

    def update_memory_store(
        self, name: str, display_name: Optional[str] = None, description: Optional[str] = None
    ) -> models.MemoryStore:
        body = _query(display_name=display_name, description=description)
        if not body:
            raise AgentCliError("No fields to update. Provide a display name and/or description.")
        mask = ",".join(body.keys())
        return _as(
            models.MemoryStore,
            self._do(
                "PATCH",
                f"{_BASE}/{memory_store_path(name)}",
                query=_query(update_mask=mask),
                body=body,
            ),
        )

    def delete_memory_store(self, name: str) -> dict:
        return self._do("DELETE", f"{_BASE}/{memory_store_path(name)}")

    # --- memory entries ------------------------------------------------------

    def create_memory_entry(
        self,
        store: str,
        actor_id: str,
        path: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
        session_id: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> models.MemoryEntry:
        body = _query(
            actor_id=actor_id,
            path=path,
            content=content,
            description=description,
            session_id=session_id,
            source_type=source_type,
        )
        return _as(
            models.MemoryEntry,
            self._do("POST", f"{_BASE}/{memory_store_path(store)}/entries", body=body),
        )

    def get_memory_entry(self, store: str, entry: str) -> models.MemoryEntry:
        return _as(
            models.MemoryEntry, self._do("GET", f"{_BASE}/{memory_entry_path(store, entry)}")
        )

    def list_memory_entries(
        self,
        store: str,
        actor_id: str,
        path_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> models.MemoryEntryList:
        return _as(
            models.MemoryEntryList,
            self._do(
                "GET",
                f"{_BASE}/{memory_store_path(store)}/entries",
                query=_query(
                    actor_id=actor_id,
                    path_prefix=path_prefix,
                    session_id=session_id,
                    page_size=page_size,
                    page_token=page_token,
                ),
            ),
        )

    def search_memory_entries(
        self, store: str, actor_id: str, query: str, limit: Optional[int] = None
    ) -> models.MemorySearchResult:
        body = _query(actor_id=actor_id, query=query, limit=limit)
        return _as(
            models.MemorySearchResult,
            self._do("POST", f"{_BASE}/{memory_store_path(store)}/entries:search", body=body),
        )

    def update_memory_entry(
        self,
        store: str,
        entry: str,
        content: Optional[str] = None,
        description: Optional[str] = None,
    ) -> models.MemoryEntry:
        body = _query(content=content, description=description)
        if not body:
            raise AgentCliError("No fields to update. Provide content and/or a description.")
        return _as(
            models.MemoryEntry,
            self._do("PATCH", f"{_BASE}/{memory_entry_path(store, entry)}", body=body),
        )

    def delete_memory_entry(self, store: str, entry: str) -> dict:
        return self._do("DELETE", f"{_BASE}/{memory_entry_path(store, entry)}")

    # --- session stores ------------------------------------------------------

    def create_session_store(
        self, name: str, description: Optional[str] = None, metadata: Optional[dict] = None
    ) -> models.SessionStore:
        body = _query(description=description, metadata=metadata)
        return _as(
            models.SessionStore,
            self._do(
                "POST", f"{_BASE}/session-stores", query={"session_store_name": name}, body=body
            ),
        )

    def get_session_store(self, name: str) -> models.SessionStore:
        return _as(models.SessionStore, self._do("GET", f"{_BASE}/{session_store_path(name)}"))

    def list_session_stores(
        self, page_size: Optional[int] = None, page_token: Optional[str] = None
    ) -> models.SessionStoreList:
        return _as(
            models.SessionStoreList,
            self._do(
                "GET",
                f"{_BASE}/session-stores",
                query=_query(page_size=page_size, page_token=page_token),
            ),
        )

    def update_session_store(
        self, name: str, description: Optional[str] = None, metadata: Optional[dict] = None
    ) -> models.SessionStore:
        body = _query(description=description, metadata=metadata)
        if not body:
            raise AgentCliError("No fields to update. Provide a description and/or metadata.")
        mask = ",".join(body.keys())
        return _as(
            models.SessionStore,
            self._do(
                "PATCH",
                f"{_BASE}/{session_store_path(name)}",
                query=_query(update_mask=mask),
                body=body,
            ),
        )

    def delete_session_store(self, name: str) -> dict:
        return self._do("DELETE", f"{_BASE}/{session_store_path(name)}")

    # --- sessions ------------------------------------------------------------

    def create_session(
        self,
        store: str,
        actor_id: str,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> models.Session:
        body = _query(actor_id=actor_id, parent_session_id=parent_session_id, metadata=metadata)
        return _as(
            models.Session,
            self._do(
                "POST",
                f"{_BASE}/session-stores/{store}/sessions",
                query=_query(session_id=session_id),
                body=body,
            ),
        )

    def list_sessions(
        self,
        store: str,
        filter: Optional[str] = None,
        order_by: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> models.SessionList:
        return _as(
            models.SessionList,
            self._do(
                "GET",
                f"{_BASE}/session-stores/{store}/sessions",
                query=_query(
                    filter=filter, order_by=order_by, page_size=page_size, page_token=page_token
                ),
            ),
        )

    def get_session(self, session_id: str, store: Optional[str] = None) -> models.Session:
        if store:
            path = f"{_BASE}/session-stores/{store}/sessions/{session_id}"
        else:
            path = f"{_BASE}/sessions/{session_id}"
        return _as(models.Session, self._do("GET", path))

    def update_session(self, store: str, session_id: str, metadata: dict) -> models.Session:
        return _as(
            models.Session,
            self._do(
                "PATCH",
                f"{_BASE}/session-stores/{store}/sessions/{session_id}",
                query={"update_mask": "metadata"},
                body=_query(metadata=metadata),
            ),
        )

    def delete_session(self, store: str, session_id: str, force: bool = False) -> dict:
        return self._do(
            "DELETE",
            f"{_BASE}/session-stores/{store}/sessions/{session_id}",
            query=_query(force=force or None),
        )

    def fork_session(
        self,
        store: str,
        source_session_id: str,
        actor_id: str,
        up_to_item_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> models.Session:
        body = _query(
            source_session_id=source_session_id,
            actor_id=actor_id,
            up_to_item_id=up_to_item_id,
            session_id=session_id,
            metadata=metadata,
        )
        return _as(
            models.Session,
            self._do("POST", f"{_BASE}/session-stores/{store}/sessions:fork", body=body),
        )

    # --- session items -------------------------------------------------------

    def list_session_items(
        self,
        store: str,
        session_id: str,
        order_by: Optional[str] = None,
        page_size: Optional[int] = None,
        page_token: Optional[str] = None,
    ) -> models.SessionItemList:
        return _as(
            models.SessionItemList,
            self._do(
                "GET",
                f"{_BASE}/session-stores/{store}/sessions/{session_id}/items",
                query=_query(order_by=order_by, page_size=page_size, page_token=page_token),
            ),
        )

    def append_session_items(
        self, store: str, session_id: str, items: list[dict]
    ) -> models.SessionItemList:
        body = {"items": [{"data": item} for item in items]}
        return _as(
            models.SessionItemList,
            self._do(
                "POST",
                f"{_BASE}/session-stores/{store}/sessions/{session_id}/items:append",
                body=body,
            ),
        )

    def pop_session_item(self, store: str, session_id: str) -> models.PoppedSessionItem:
        return _as(
            models.PoppedSessionItem,
            self._do(
                "POST", f"{_BASE}/session-stores/{store}/sessions/{session_id}/items:pop", body={}
            ),
        )

    def clear_session_items(self, store: str, session_id: str) -> dict:
        return self._do(
            "POST", f"{_BASE}/session-stores/{store}/sessions/{session_id}/items:clear", body={}
        )


# Backwards-compatible alias for the pre-1.0 internal name.
AgentApiClient = MasonClient
