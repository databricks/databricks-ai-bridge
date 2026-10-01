"""The store clients a deploy drives, behind a common "store client" role.

Two different store families live in this module:

- The memory/session stores declared in agent.toml, reached over the managed conversation-store
  REST API: ``MemoryStoreClient`` and ``SessionStoreClient`` each wrap that API for their own
  store. The two share only the error-mapping helpers below, since a create denied/inaccessible
  looks the same for either store.
- The Runtime Store backed by Lakebase/Postgres: ``RuntimeStoreClient`` get-or-creates the
  deployment's Runtime Store backend, shelling out to the ``databricks`` CLI for the legacy
  per-app path.

All three hold their connections - the caching workspace-client factory, called lazily on first
use, and the apps client their grants resolve the app's service principal through - so
`agentbricks deploy` injects one instance of each (symmetrically with the other resource
collaborators) and drives it through its provisioner.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import databricks_agentbricks.lakebase_runtime_store as managed_runtime_store
import databricks_agentbricks.legacy_lakebase_runtime_store as legacy_runtime_store
from databricks_agentbricks import render
from databricks_agentbricks.app_resources import LakebaseBackend, apply_postgres_resources
from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.lakebase_runtime_store import RuntimeStoreBackend
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

    Holds its connections - the caching workspace-client factory, invoked lazily on first use,
    and the apps client the grant resolves the app's service principal through - so its methods
    take only resource names, and a single instance is injected into the deploy service and
    driven by ``MemoryStoreProvisioner``.
    """

    def __init__(self, api_client_factory: Callable[[], Any], apps_client: AppsClient) -> None:
        self._api_client_factory = api_client_factory
        self._apps = apps_client

    def resolve(self, display_name: str) -> Optional[dict]:
        """Find the memory store by display name, paging through the list, or None if none matches.

        `get_memory_store` looks up by resource id (`memory-stores/<uuid>`), not the display name users
        pass, so resolving a name means listing and matching on `display_name`. The list API caps
        `page_size` at 100, so page through with the `next_page_token` rather than requesting all at once.
        """
        page_token: Optional[str] = None
        while True:
            listing = self._api_client_factory().list_memory_stores(
                page_size=_MEMORY_STORE_PAGE_SIZE, page_token=page_token
            )
            for store in field(listing, "managed_memory_stores") or []:
                if field(store, "display_name") == display_name:
                    return store
            page_token = field(listing, "next_page_token")
            if not page_token:
                return None

    def ensure(self, display_name: str) -> tuple[dict, bool]:
        """Create the memory store, or resolve it if it already exists. Returns (store, created)."""
        try:
            return self._api_client_factory().create_memory_store(
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

    def reconcile(self, display_name: str) -> Optional[str]:
        """Create the declared memory store if it doesn't exist yet; return its bare id.

        `agentbricks deploy` is the only verb that provisions stores. It reconciles to the name declared
        in agent.toml (by `agentbricks init` or `agentbricks memory bind`) - never inventing a name and
        never writing bindings back into the manifest. A store created here gets a one-line notice. The
        bare id is returned so the caller can wire AGENT_MEMORY_STORE (the entries API is keyed by id,
        not display name).
        """
        with render.status(f"Reconciling memory store '{display_name}'…"):
            resolved, created = self.ensure(display_name)
        if created:
            render.console().print(f"[green]✓[/] Created memory store {display_name!r}")
        return (field(resolved, "name") or "").split("/", 1)[-1] or None

    def grant(self, app_name: str, store_name: str) -> Optional[str]:
        """Grant the app's service principal read/write on the memory store; return any error hint.

        Resolves the service principal itself, through the injected apps client. Goes through the
        managed store API, so the store service owns the SP's Lakebase role and runs the GRANT
        itself - no store ownership or Lakebase MANAGE required of the deployer. Best-effort:
        a failure is returned, not raised, so a missing grant is reported as a next step.
        """
        sp = self._apps.get_service_principal(app_name)
        if sp is None:
            return "could not resolve the app's service principal."
        try:
            store = self.resolve(store_name)
            if store is None:
                return f"memory store {store_name!r} could not be resolved."
            self._api_client_factory().grant_memory_store_permission(field(store, "name"), sp)
        except AgentCliError as exc:
            return exc.hint or str(exc)
        return None


class SessionStoreClient:
    """Provisions and grants access to the session store declared in agent.toml.

    The memory store's sibling: same connection-holding shape, driven by
    ``SessionStoreProvisioner``. Session stores resolve by name, so there is no id to return.
    """

    def __init__(self, api_client_factory: Callable[[], Any], apps_client: AppsClient) -> None:
        self._api_client_factory = api_client_factory
        self._apps = apps_client

    def ensure(self, name: str) -> tuple[dict, bool]:
        """Create the session store, or resolve it if it already exists. Returns (store, created)."""
        try:
            return self._api_client_factory().create_session_store(name, retry_transient=True), True
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_create_permission_error(name, "session", exc) from exc
            if exc.error_code != "ALREADY_EXISTS":
                raise
        try:
            return self._api_client_factory().get_session_store(name), False
        except AgentCliError as exc:
            if exc.error_code == "PERMISSION_DENIED":
                raise _store_access_error(name, "session") from exc
            raise

    def reconcile(self, name: str) -> None:
        """Create the declared session store if it doesn't exist yet.

        Like the memory store, reconciled to the name in agent.toml and never written back. Session
        stores resolve by name, so nothing is returned - a created store gets a one-line notice.
        """
        with render.status(f"Reconciling session store '{name}'…"):
            _, created = self.ensure(name)
        if created:
            render.console().print(f"[green]✓[/] Created session store {name!r}")

    def grant(self, app_name: str, store_name: str) -> Optional[str]:
        """Grant the app's service principal read/write on the session store; return any error hint.

        Same managed-store-API path and best-effort contract as the memory grant, including
        resolving the service principal itself through the injected apps client.
        """
        sp = self._apps.get_service_principal(app_name)
        if sp is None:
            return "could not resolve the app's service principal."
        try:
            self._api_client_factory().grant_session_store_permission(store_name, sp)
        except AgentCliError as exc:
            return exc.hint or str(exc)
        return None


class RuntimeStoreClient:
    """Get-or-create the Runtime Store backend for a deployment, for a ``profile``.

    A temporary rollout switch chooses between the legacy per-app Lakebase project and the
    service-managed database; the switch is injected at construction (``use_managed``) so callers
    ask intent-level questions (``is_managed``) instead of reading the flag themselves. Holds its
    connections - the caching workspace-client factory, invoked lazily on first use, and the apps
    client the managed paths resolve the app's service principal through.
    """

    def __init__(
        self,
        api_client_factory: Callable[[], Any],
        profile: Optional[str],
        apps_client: AppsClient,
        use_managed: bool,
    ) -> None:
        self._api_client_factory = api_client_factory
        self._profile = profile
        self._apps = apps_client
        self._use_managed = use_managed

    def is_managed(self) -> bool:
        """Whether the service-managed Runtime Store rollout switch is on."""
        return self._use_managed

    def legacy_backend(self, name: str) -> LakebaseBackend:
        """Reuse or create the dedicated per-app Lakebase project used by the legacy path."""
        return legacy_runtime_store.get_or_create_backend(name, self._profile)

    def apply_legacy_resource(self, name: str, backend: LakebaseBackend) -> Optional[str]:
        """Bind the legacy backend's database as a ``postgres`` app resource.

        Returns None on success or a human-readable reason on failure.
        """
        return apply_postgres_resources(name, [backend], self._profile)

    def managed_backend(self, name: str) -> RuntimeStoreBackend:
        """Create or reuse the service-managed Runtime Store owned by the app's service principal."""
        sp = self._apps.get_service_principal(name)
        return managed_runtime_store.get_or_create_backend(self._api_client_factory(), name, sp)

    def delete_managed(self, name: str) -> None:
        """Drop the deployment's service-managed Runtime Store and its data.

        Called before the app itself is deleted: the delete needs the app's service principal, which
        stops resolving once the app is gone. If that identity can't be read we refuse outright
        rather than delete the app and orphan its data.
        """
        sp = self._apps.get_service_principal(name)
        if not sp:
            raise AgentCliError(
                "Could not resolve the app's service principal for Runtime Store cleanup.",
                hint="The deployment was retained. Check access to the app and retry deletion.",
            )
        managed_runtime_store.delete(self._api_client_factory(), name, sp)
