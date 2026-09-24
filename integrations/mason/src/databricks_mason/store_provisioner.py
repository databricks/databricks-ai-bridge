"""Provisions and grants access to the memory/session stores declared in agent.toml.

Wraps the managed-store API client so deploy (and tests) inject it once via the constructor
rather than threading `client` through free functions and monkeypatching module globals. The
class owns all store-lifecycle operations: resolving a store by display name, ensuring it exists
(creating when absent, reusing when already present), and granting the app's service principal
read/write access through the managed-store REST API - which arranges the underlying Lakebase
grant server-side, so no direct Lakebase ownership or MANAGE privilege is required here.
"""

from __future__ import annotations

from typing import Optional

from databricks_mason import render
from databricks_mason.errors import AgentCliError
from databricks_mason.render import field

_MEMORY_STORE_PAGE_SIZE = 100  # the memory-stores list API caps page_size at 100

_LAKEBASE_PERMISSION_DOCS = (
    "https://docs.databricks.com/aws/en/oltp/projects/manage-project-permissions"
)


def _store_create_permission_error(name: str, kind: str, cause: AgentCliError) -> AgentCliError:
    """PERMISSION_DENIED on create: the workspace admin has restricted Lakebase project creation."""
    return AgentCliError(
        f"You don't have permission to create {kind} store '{name}'.",
        error_code=cause.error_code,
        hint=(
            "Creating a managed store provisions a Lakebase project, which your workspace admin "
            "has restricted. Ask your workspace admin to grant you permission to create Lakebase "
            f"projects ({_LAKEBASE_PERMISSION_DOCS}), or bind an existing store you can access "
            "with --no-create-stores."
        ),
    )


def _store_access_error(name: str, kind: str) -> AgentCliError:
    """The store already exists but isn't accessible to the caller."""
    return AgentCliError(
        f"{kind.capitalize()} store '{name}' already exists but you don't have access to it.",
        hint=(
            "Ask the store's owner or your workspace admin to grant you access, or bind a "
            "different store you can access with --no-create-stores."
        ),
    )


class StoreProvisioner:
    """Provisions and grants the memory/session stores declared in agent.toml."""

    def __init__(self, client) -> None:
        self._client = client

    def resolve_memory_store(self, display_name: str) -> Optional[dict]:
        """Find a memory store by display name, paging through the list, or None if none matches.

        `get_memory_store` looks up by resource id (`memory-stores/<uuid>`), not the display name users
        pass, so resolving a name means listing and matching on `display_name`. The list API caps
        `page_size` at 100, so page through with the `next_page_token` rather than requesting all at once.
        """
        page_token: Optional[str] = None
        while True:
            listing = self._client.list_memory_stores(
                page_size=_MEMORY_STORE_PAGE_SIZE, page_token=page_token
            )
            for store in field(listing, "managed_memory_stores") or []:
                if field(store, "display_name") == display_name:
                    return store
            page_token = field(listing, "next_page_token")
            if not page_token:
                return None

    def ensure_memory_store(self, display_name: str) -> tuple[dict, bool]:
        """Create the memory store, or resolve it if it already exists. Returns (store, created)."""
        try:
            return self._client.create_memory_store(display_name, retry_transient=True), True
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_create_permission_error(display_name, "memory", exc) from exc
            if exc.error_code != "ALREADY_EXISTS":
                raise
        store = self.resolve_memory_store(display_name)
        if store is None:
            # ALREADY_EXISTS but not in the caller's listing: the store isn't accessible to them.
            raise _store_access_error(display_name, "memory")
        return store, False

    def ensure_session_store(self, name: str) -> tuple[dict, bool]:
        """Create the session store, or resolve it if it already exists. Returns (store, created)."""
        try:
            return self._client.create_session_store(name, retry_transient=True), True
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_create_permission_error(name, "session", exc) from exc
            if exc.error_code != "ALREADY_EXISTS":
                raise
        try:
            return self._client.get_session_store(name), False
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_access_error(name, "session") from exc
            raise

    def reconcile_declared_stores(
        self, memory_store: Optional[str], session_store: Optional[str]
    ) -> Optional[str]:
        """Create any store DECLARED in agent.toml that doesn't exist yet; return the memory store's id.

        `mason deploy` is the only verb that provisions stores. It reconciles to the names declared in
        agent.toml (by `mason init` or `mason memory/sessions bind`) - never inventing a name and never
        writing bindings back into the manifest. A store created here gets a one-line notice. The memory
        store's bare id is returned so the caller can wire AGENT_MEMORY_STORE (the entries API is keyed
        by id, not display name); session stores resolve by name and need nothing here.
        """
        memory_store_id: Optional[str] = None
        if memory_store:
            with render.status(f"Reconciling memory store '{memory_store}'…"):
                resolved, created = self.ensure_memory_store(memory_store)
            memory_store_id = (field(resolved, "name") or "").split("/", 1)[-1] or None
            if created:
                render.console().print(f"[green]✓[/] Created memory store {memory_store!r}")
        if session_store:
            with render.status(f"Reconciling session store '{session_store}'…"):
                _, created = self.ensure_session_store(session_store)
            if created:
                render.console().print(f"[green]✓[/] Created session store {session_store!r}")
        return memory_store_id

    def grant_store_access(
        self,
        sp: str,
        session_store: Optional[str],
        memory_store: Optional[str],
    ) -> Optional[str]:
        """Grant the app's service principal read/write on its bound stores, via the managed store API.

        The conversation-store service owns the (service-managed) store Lakebase, so it provisions the
        SP's role and runs the GRANTs itself. Unlike a direct Lakebase grant, this needs neither store
        ownership nor MANAGE on the store's Lakebase project, so it works for non-admin deployers.
        """
        try:
            if session_store:
                self._client.grant_session_store_permission(session_store, sp)
            if memory_store:
                store = self.resolve_memory_store(memory_store)
                if store is None:
                    return f"memory store {memory_store!r} could not be resolved."
                self._client.grant_memory_store_permission(field(store, "name"), sp)
        except AgentCliError as exc:
            return exc.hint or str(exc)
        return None
