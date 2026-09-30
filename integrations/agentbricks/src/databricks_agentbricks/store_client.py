"""Provisions and grants the memory and session stores declared in agent.toml.

`MemoryStoreClient` and `SessionStoreClient` each wrap the managed-store API for their own
store, mirroring how `RuntimeStoreClient` and `TracingClient` own their resource. Both are
stateless - the workspace client is passed to each method rather than captured at construction - so
`agentbricks deploy` injects one instance of each (symmetrically with the other resource
collaborators) and uses it once the client is opened. The two share only the error-mapping helpers
below, since a create denied/inaccessible looks the same for either store.
"""

from __future__ import annotations

from typing import Optional

from databricks_agentbricks import render
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.render import field

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


class MemoryStoreClient:
    """Provisions and grants access to the memory store declared in agent.toml.

    Stateless - every method takes the workspace client - so a single instance is injected into the
    deploy service and driven by ``MemoryStoreProvisioner``.
    """

    def resolve(self, client, display_name: str) -> Optional[dict]:
        """Find the memory store by display name, paging through the list, or None if none matches.

        `get_memory_store` looks up by resource id (`memory-stores/<uuid>`), not the display name users
        pass, so resolving a name means listing and matching on `display_name`. The list API caps
        `page_size` at 100, so page through with the `next_page_token` rather than requesting all at once.
        """
        page_token: Optional[str] = None
        while True:
            listing = client.list_memory_stores(
                page_size=_MEMORY_STORE_PAGE_SIZE, page_token=page_token
            )
            for store in field(listing, "managed_memory_stores") or []:
                if field(store, "display_name") == display_name:
                    return store
            page_token = field(listing, "next_page_token")
            if not page_token:
                return None

    def ensure(self, client, display_name: str) -> tuple[dict, bool]:
        """Create the memory store, or resolve it if it already exists. Returns (store, created)."""
        try:
            return client.create_memory_store(display_name, retry_transient=True), True
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_create_permission_error(display_name, "memory", exc) from exc
            if exc.error_code != "ALREADY_EXISTS":
                raise
        store = self.resolve(client, display_name)
        if store is None:
            # ALREADY_EXISTS but not in the caller's listing: the store isn't accessible to them.
            raise _store_access_error(display_name, "memory")
        return store, False

    def reconcile(self, client, display_name: str) -> Optional[str]:
        """Create the declared memory store if it doesn't exist yet; return its bare id.

        `agentbricks deploy` is the only verb that provisions stores. It reconciles to the name declared
        in agent.toml (by `agentbricks init` or `agentbricks memory bind`) - never inventing a name and
        never writing bindings back into the manifest. A store created here gets a one-line notice. The
        bare id is returned so the caller can wire AGENT_MEMORY_STORE (the entries API is keyed by id,
        not display name).
        """
        with render.status(f"Reconciling memory store '{display_name}'…"):
            resolved, created = self.ensure(client, display_name)
        if created:
            render.console().print(f"[green]✓[/] Created memory store {display_name!r}")
        return (field(resolved, "name") or "").split("/", 1)[-1] or None

    def grant(self, client, sp: str, display_name: str) -> Optional[str]:
        """Grant the app's service principal read/write on the memory store; return any error hint.

        Goes through the managed store API, so the store service owns the SP's Lakebase role and runs
        the GRANT itself - no store ownership or Lakebase MANAGE required of the deployer. Best-effort:
        a failure is returned, not raised, so a missing grant is reported as a next step.
        """
        try:
            store = self.resolve(client, display_name)
            if store is None:
                return f"memory store {display_name!r} could not be resolved."
            client.grant_memory_store_permission(field(store, "name"), sp)
        except AgentCliError as exc:
            return exc.hint or str(exc)
        return None


class SessionStoreClient:
    """Provisions and grants access to the session store declared in agent.toml.

    The memory store's sibling: same stateless, client-per-method shape, driven by
    ``SessionStoreProvisioner``. Session stores resolve by name, so there is no id to return.
    """

    def ensure(self, client, name: str) -> tuple[dict, bool]:
        """Create the session store, or resolve it if it already exists. Returns (store, created)."""
        try:
            return client.create_session_store(name, retry_transient=True), True
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_create_permission_error(name, "session", exc) from exc
            if exc.error_code != "ALREADY_EXISTS":
                raise
        try:
            return client.get_session_store(name), False
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_access_error(name, "session") from exc
            raise

    def reconcile(self, client, name: str) -> None:
        """Create the declared session store if it doesn't exist yet.

        Like the memory store, reconciled to the name in agent.toml and never written back. Session
        stores resolve by name, so nothing is returned - a created store gets a one-line notice.
        """
        with render.status(f"Reconciling session store '{name}'…"):
            _, created = self.ensure(client, name)
        if created:
            render.console().print(f"[green]✓[/] Created session store {name!r}")

    def grant(self, client, sp: str, name: str) -> Optional[str]:
        """Grant the app's service principal read/write on the session store; return any error hint.

        Same managed-store-API path and best-effort contract as the memory grant.
        """
        try:
            client.grant_session_store_permission(name, sp)
        except AgentCliError as exc:
            return exc.hint or str(exc)
        return None
