"""Managed conversation-store clients for memory and session stores.

Both use the same API client provider and error mapping.
Deployment-specific Runtime Store coordination lives in ``services.deployment.provisioners``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.errors import AgentCliError

_MEMORY_STORE_PAGE_SIZE = 100  # the memory-stores list API caps page_size at 100

_LAKEBASE_PERMISSION_DOCS = (
    "https://docs.databricks.com/aws/en/oltp/projects/manage-project-permissions"
)


def _field(payload: Any, name: str) -> Any:
    """Read snake_case or camelCase fields without importing the CLI rendering layer."""
    if isinstance(payload, dict):
        if name in payload:
            return payload[name]
        parts = name.split("_")
        return payload.get(parts[0] + "".join(part.title() for part in parts[1:]))
    return getattr(payload, name, None)


@dataclass(frozen=True)
class MemoryStoreReconcileResult:
    """Facts produced while reconciling a memory store."""

    store_id: Optional[str]
    resource_name: Optional[str]
    created: bool


@dataclass(frozen=True)
class SessionStoreReconcileResult:
    """Facts produced while reconciling a session store."""

    store_name: str
    created: bool


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

    Holds the per-command API client provider, invoked lazily on first use. The deployment
    workflow supplies the App identity and reconciled store resource name to ``grant``.
    """

    def __init__(self, api_client_provider: ApiClientProvider) -> None:
        self._api_client_provider = api_client_provider

    def resolve(self, display_name: str) -> Optional[dict]:
        """Find the memory store by display name, paging through the list, or None if none matches.

        `get_memory_store` looks up by resource id (`memory-stores/<uuid>`), not the display name users
        pass, so resolving a name means listing and matching on `display_name`. The list API caps
        `page_size` at 100, so page through with the `next_page_token` rather than requesting all at once.
        """
        page_token: Optional[str] = None
        while True:
            listing = self._api_client_provider.get().list_memory_stores(
                page_size=_MEMORY_STORE_PAGE_SIZE, page_token=page_token
            )
            for store in _field(listing, "managed_memory_stores") or []:
                if _field(store, "display_name") == display_name:
                    return store
            page_token = _field(listing, "next_page_token")
            if not page_token:
                return None

    def ensure(self, display_name: str) -> tuple[dict, bool]:
        """Create the memory store, or resolve it if it already exists. Returns (store, created)."""
        try:
            return self._api_client_provider.get().create_memory_store(
                display_name, retry_transient=True
            ), True
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_create_permission_error(display_name, "memory", exc) from exc
            if exc.error_code != "ALREADY_EXISTS":
                raise
        store = self.resolve(display_name)
        if store is None:
            # ALREADY_EXISTS but not in the caller's listing: the store isn't accessible to them.
            raise _store_access_error(display_name, "memory")
        return store, False

    def reconcile(self, display_name: str) -> MemoryStoreReconcileResult:
        """Create the declared memory store if absent and return its identifiers and status.

        `agentbricks deploy` is the only verb that provisions stores. It reconciles to the name declared
        in agent.toml (by `agentbricks init` or `agentbricks memory bind`) - never inventing a name and
        never writing bindings back into the manifest. Presentation belongs to the provisioner; this
        client returns facts only. The bare id wires AGENT_MEMORY_STORE (the entries API is keyed
        by id), while the full resource name is retained for the later grant.
        """
        resolved, created = self.ensure(display_name)
        resource_name = _field(resolved, "name") or None
        store_id = resource_name.split("/", 1)[-1] if resource_name else None
        return MemoryStoreReconcileResult(
            store_id=store_id, resource_name=resource_name, created=created
        )

    def grant(self, resource_name: str, service_principal_id: str) -> Optional[str]:
        """Grant a service principal read/write on a resolved memory store.

        The managed store API owns the SP's Lakebase role and runs the GRANT itself - no store
        ownership or Lakebase MANAGE required of the deployer. A failure is returned, not raised,
        so a missing grant can be reported as a next step.
        """
        try:
            self._api_client_provider.get().grant_memory_store_permission(
                resource_name, service_principal_id
            )
        except AgentCliError as exc:
            return exc.hint or str(exc)
        return None


class SessionStoreClient:
    """Provisions and grants access to the session store declared in agent.toml.

    The memory store's sibling, driven by ``SessionStoreProvisioner``. Session stores resolve by
    name, so that name is their grant identifier.
    """

    def __init__(self, api_client_provider: ApiClientProvider) -> None:
        self._api_client_provider = api_client_provider

    def ensure(self, name: str) -> tuple[dict, bool]:
        """Create the session store, or resolve it if it already exists. Returns (store, created)."""
        try:
            return self._api_client_provider.get().create_session_store(
                name, retry_transient=True
            ), True
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_create_permission_error(name, "session", exc) from exc
            if exc.error_code != "ALREADY_EXISTS":
                raise
        try:
            return self._api_client_provider.get().get_session_store(name), False
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_access_error(name, "session") from exc
            raise

    def reconcile(self, name: str) -> SessionStoreReconcileResult:
        """Create the declared session store if absent and return its name and creation status.

        Like the memory store, reconciled to the name in agent.toml and never written back. Session
        stores resolve by name, so no resource id is needed. Presentation belongs to the provisioner.
        """
        store, created = self.ensure(name)
        return SessionStoreReconcileResult(
            store_name=_field(store, "name") or name, created=created
        )

    def grant(self, store_name: str, service_principal_id: str) -> Optional[str]:
        """Grant a service principal read/write on a resolved session store.

        Same managed-store-API path and best-effort contract as the memory grant.
        """
        try:
            self._api_client_provider.get().grant_session_store_permission(
                store_name, service_principal_id
            )
        except AgentCliError as exc:
            return exc.hint or str(exc)
        return None
