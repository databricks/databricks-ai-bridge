"""Stateless resource provisioners used by the agentbricks deployment system.

Provisioners hold only injected dependencies, not per-deploy outcomes. Resource phases
return frozen, typed state; the deploy
service keeps that state between phases and owns manifest writes and result assembly.  This keeps a
single provisioner instance safe to reuse for successive deploys.

The contexts passed to phases are immutable data, split small on purpose (no kitchen-sink object,
no methods, no collaborators): :class:`ProjectContext` is the deploy identity and
:class:`ResourceContext` adds the bindings read from ``agent.toml`` plus whether the app pre-existed
this deploy. The shared Apps client caches service-principal lookups needed by grant phases.

The module also holds :class:`AppProvisioner`, which owns the deployed app itself and is deliberately
separate from the typed resource provisioners.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Optional

import databricks_agentbricks.clients.legacy_runtime_store as legacy_runtime_store
import databricks_agentbricks.clients.managed_runtime_store as managed_runtime_store
from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.conversation_store_client import (
    MemoryStoreClient,
    SessionStoreClient,
)
from databricks_agentbricks.clients.legacy_runtime_store import LakebaseBackend
from databricks_agentbricks.clients.tracing_client import TraceTable, TracingClient
from databricks_agentbricks.deployment.config import (
    _AGENT_COMPUTE_OUTPUT,
    _AGENTKIT_RUNTIME_STORE_SCHEMA,
    mlflow_tracing_config,
)
from databricks_agentbricks.deployment.names import _DEPLOYMENT_PREFIX, DeploymentName
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentbricks.reporting import Reporter
from databricks_agentkit.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
    RUNTIME_STORE_SCHEMA_ENV,
    RUNTIME_STORE_USERNAME_ENV,
)
from databricks_agentkit.runtime.tool_manifest import MEMORY_STORE_ENV, SESSION_STORE_ENV


@dataclass(frozen=True)
class ProjectContext:
    """What is being deployed: the identity shared by every resource, and nothing else."""

    source_dir: pathlib.Path
    name: DeploymentName
    agent_project: Optional[AgentProject]


@dataclass(frozen=True)
class ResourceContext:
    """What a provisioner phase is handed: the project identity plus the resource bindings.

    Pure data - immutable, method-free, and holding no collaborators. A phase reads what it needs
    and returns typed state; the deploy service owns that state and all manifest writes.
    """

    project: ProjectContext
    memory_store: Optional[str]  # the memory store bound in agent.toml, if any
    session_store: Optional[str]  # the session store bound in agent.toml, if any
    experiment_name: Optional[str]  # the tracing experiment bound in agent.toml, if any
    deployment_exists: bool = False  # whether the app pre-existed this deploy


@dataclass(frozen=True)
class ManifestPatch:
    """One resource's contribution to the deployment manifest."""

    env: Mapping[str, str]
    env_removals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Copy before freezing so neither the producer nor caller can change a completed phase.
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


@dataclass(frozen=True)
class MemoryStoreState:
    """Facts retained by the deploy service between memory-store phases."""

    store_name: Optional[str]
    manifest: ManifestPatch


@dataclass(frozen=True)
class SessionStoreState:
    """Facts retained by the deploy service between session-store phases."""

    store_name: Optional[str]
    manifest: ManifestPatch


@dataclass(frozen=True)
class TracingState:
    """Facts retained by the deploy service between tracing phases."""

    experiment_id: Optional[str]
    otel_tables: tuple[TraceTable, ...]
    setup_error: Optional[str]
    manifest: ManifestPatch


@dataclass(frozen=True)
class RuntimeStoreState:
    """Facts retained by the deploy service between Runtime Store phases."""

    enabled: bool
    legacy_backend: Optional[LakebaseBackend]
    manifest: ManifestPatch


@dataclass(frozen=True)
class GrantOutcome:
    """Whether a best-effort post-rollout grant ran and, if so, how it ended."""

    attempted: bool
    error: Optional[str]

    @classmethod
    def skipped(cls) -> "GrantOutcome":
        """Return the outcome for a resource that was not bound or could not be reconciled."""
        return cls(attempted=False, error=None)


class MemoryStoreProvisioner:
    """The memory store declared in agent.toml, reconciled and granted on its own.

    Delegates API work to its own ``MemoryStoreClient``.  ``reconcile`` returns the bound store name
    and manifest patch; ``grant`` consumes that state and returns a :class:`GrantOutcome`.
    """

    def __init__(self, memory_store_client: MemoryStoreClient, reporter: Reporter) -> None:
        self._memory_store_client = memory_store_client
        self._reporter = reporter

    def reconcile(self, ctx: ResourceContext) -> MemoryStoreState:
        """Create the declared memory store if absent and wire ``AGENT_MEMORY_STORE``.

        `agentbricks deploy` is the only reconcile-to-cloud verb; agent.toml is the source of truth and
        is never rewritten. ``AGENT_MEMORY_STORE`` carries the store's bare id (not its display name),
        because the entries API is keyed by id.
        """
        store_name = ctx.memory_store
        if not store_name:
            return MemoryStoreState(store_name=None, manifest=ManifestPatch(env={}))
        with self._reporter.status(f"Reconciling memory store '{store_name}'…"):
            result = self._memory_store_client.reconcile(store_name)
        if result.created:
            self._reporter.note(f"Created memory store {store_name!r}")
        env: dict[str, str] = {}
        if result.store_id:
            env[MEMORY_STORE_ENV] = result.store_id
        return MemoryStoreState(store_name=store_name, manifest=ManifestPatch(env=env))

    def grant(self, ctx: ResourceContext, state: MemoryStoreState) -> GrantOutcome:
        """Grant the app's service principal read/write on the memory store (best-effort).

        Goes through the managed store API, so the store service performs the underlying Lakebase grant
        - no store ownership or Lakebase MANAGE required of the deployer. A failure is recorded, not
        raised: the deploy succeeded, and the CLI reports the missing grant as a next step.
        """
        if not state.store_name:
            return GrantOutcome.skipped()
        with self._reporter.status("Granting the app access to its memory store…"):
            error = self._memory_store_client.grant(ctx.project.name, state.store_name)
        return GrantOutcome(attempted=True, error=error)


class SessionStoreProvisioner:
    """The session store declared in agent.toml, reconciled and granted on its own.

    Delegates API work to its own ``SessionStoreClient``. Session stores resolve by name, so
    ``AGENT_SESSION_STORE`` carries the name rather than a resolved id.
    """

    def __init__(self, session_store_client: SessionStoreClient, reporter: Reporter) -> None:
        self._session_store_client = session_store_client
        self._reporter = reporter

    def reconcile(self, ctx: ResourceContext) -> SessionStoreState:
        """Create the declared session store if absent and wire ``AGENT_SESSION_STORE``."""
        store_name = ctx.session_store
        if not store_name:
            return SessionStoreState(store_name=None, manifest=ManifestPatch(env={}))
        with self._reporter.status(f"Reconciling session store '{store_name}'…"):
            result = self._session_store_client.reconcile(store_name)
        if result.created:
            self._reporter.note(f"Created session store {store_name!r}")
        return SessionStoreState(
            store_name=store_name,
            manifest=ManifestPatch(env={SESSION_STORE_ENV: store_name}),
        )

    def grant(self, ctx: ResourceContext, state: SessionStoreState) -> GrantOutcome:
        """Grant the app's service principal read/write on the session store (best-effort).

        Same managed-store-API path and best-effort contract as the memory grant; the service principal
        is resolved once and cached in ``AppsClient``, so this and the memory grant share the one lookup.
        """
        if not state.store_name:
            return GrantOutcome.skipped()
        with self._reporter.status("Granting the app access to its session store…"):
            error = self._session_store_client.grant(ctx.project.name, state.store_name)
        return GrantOutcome(attempted=True, error=error)


class TracingProvisioner:
    """The MLflow experiment a deployed agent traces to, and the app resources granting write access.

    Best-effort throughout: tracing is an add-on, so neither a failed resolve nor a failed grant
    blocks a deploy - both are recorded for the CLI to report.
    """

    def __init__(self, tracing_client: TracingClient, reporter: Reporter) -> None:
        self._tracing_client = tracing_client
        self._reporter = reporter

    def reconcile(self, ctx: ResourceContext) -> TracingState:
        """Get-or-create the experiment bound in agent.toml and wire the two env vars the runtime reads.

        Resolved by experiment NAME (never a stored id), and nothing is written back to agent.toml.
        (`agentbricks dev` traces to a local MLflow server instead and never touches this experiment.)
        The env is set when tracing resolves; on a CLEAN unbind - resolved to None with no setup error
        - the stale ``MLFLOW_*`` keys are pruned instead, so the manifest stops pointing the runtime at
        an experiment whose grant is about to be pruned too. On a resolve ERROR neither the env nor the
        trace resources are touched: a transient failure must not look like an unbind.

        (Store env is still upsert-only, a separate follow-up - unbinding a store leaves its env behind.)
        """
        trace_provision = None
        setup_error: Optional[str] = None
        try:
            if ctx.experiment_name:
                # Show progress while the experiment is get-or-created (a workspace round-trip),
                # matching the store reconcile spinners so deploy isn't silent about tracing.
                with self._reporter.status(
                    f"Reconciling tracing experiment '{ctx.experiment_name}'…"
                ):
                    trace_provision = self._tracing_client.ensure_experiment(ctx.experiment_name)
            else:
                trace_provision = self._tracing_client.ensure_experiment(None)
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort; never block a deploy
            setup_error = str(exc)
        experiment_id = trace_provision.experiment_id if trace_provision else None
        # No resolved experiment means no UC OTEL tables to grant - the same as an empty table set.
        otel_tables = tuple(trace_provision.tables.otel_tables()) if trace_provision else ()
        if experiment_id:
            manifest = ManifestPatch(env=mlflow_tracing_config(experiment_id).env())
        elif setup_error is None:
            manifest = ManifestPatch(
                env={},
                env_removals=tuple(mlflow_tracing_config("").env()),
            )  # the MLFLOW_* keys to prune
        else:
            manifest = ManifestPatch(env={})
        return TracingState(
            experiment_id=experiment_id,
            otel_tables=otel_tables,
            setup_error=setup_error,
            manifest=manifest,
        )

    def grant(self, ctx: ResourceContext, state: TracingState) -> GrantOutcome:
        """Reconcile the agentbricks-owned trace resources whenever tracing resolved cleanly.

        A resolved experiment grants that set (the ``experiment`` resource, CAN_EDIT, plus MODIFY on
        any UC OTEL tables via ``uc_securable`` resources); a cleanly-unbound project prunes the
        agentbricks-trace-* resources an earlier bound deploy left behind. When resolving the BOUND
        experiment errored instead we don't know the intended state, so the reconcile is skipped
        rather than run - a flaky deploy must not silently revoke trace access the way an unbind does.

        (Whether removing a ``uc_securable`` resource also revokes the underlying UC MODIFY grant is
        platform behavior - documented but not yet verified live.)
        """
        if state.setup_error is not None:
            return GrantOutcome.skipped()
        with self._reporter.status("Granting the agent runtime access to its trace experiment…"):
            error = self._tracing_client.reconcile_app_resources(
                ctx.project.name, state.experiment_id, state.otel_tables
            )
        return GrantOutcome(attempted=True, error=error)


class RuntimeStoreProvisioner:
    """The persistent Runtime Store behind an Agent Bricks Runtime deployment.

    The only resource that splits across both create phases, because a temporary rollout switch picks
    between two backends with different requirements: the legacy per-app Lakebase project can be
    provisioned before the app exists (only its resource attachment has to wait), while the
    service-managed database is owned by the app's service principal and so can't even be created
    until the app is there. Custom-server projects have no Runtime Store and skip every phase.
    Backend selection and manifest contributions are deployment policy, not client behavior.
    """

    def __init__(
        self,
        api_client_provider: ApiClientProvider,
        apps_client: AppsClient,
        profile: Optional[str],
        use_managed: bool,
        reporter: Reporter,
    ) -> None:
        self._api_client_provider = api_client_provider
        self._apps = apps_client
        self._profile = profile
        self._use_managed = use_managed
        self._reporter = reporter

    @staticmethod
    def _uses_runtime_store(ctx: ResourceContext) -> bool:
        """Only Agent Bricks Runtime projects get a Runtime Store; a custom server manages its own."""
        return (
            ctx.project.agent_project is not None
            and ctx.project.agent_project.server == AgentServer.AGENTBRICKS
        )

    def reconcile(self, ctx: ResourceContext) -> RuntimeStoreState:
        """Prepare the selected backend and collect any env available before App creation."""
        if not self._uses_runtime_store(ctx):
            return RuntimeStoreState(
                enabled=False, legacy_backend=None, manifest=ManifestPatch(env={})
            )
        with self._reporter.status("Reconciling Runtime Store…"):
            if self._use_managed:
                return RuntimeStoreState(
                    enabled=True, legacy_backend=None, manifest=ManifestPatch(env={})
                )
            backend = legacy_runtime_store.get_or_create_backend(ctx.project.name, self._profile)
            return RuntimeStoreState(
                enabled=True,
                legacy_backend=backend,
                manifest=ManifestPatch(
                    env={
                        RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV: backend.endpoint_path,
                        RUNTIME_STORE_SCHEMA_ENV: backend.schema,
                    }
                ),
            )

    def after_app_ready(self, ctx: ResourceContext, state: RuntimeStoreState) -> ManifestPatch:
        """Attach the legacy backend to the app, or provision the managed store the app now owns.

        The managed branch contributes late env after the app exists. The deploy service owns the
        corresponding second manifest write.
        """
        if not state.enabled:
            return ManifestPatch(env={})
        with self._reporter.status("Reconciling Runtime Store…"):
            if state.legacy_backend is not None:
                resource_error = self._apps.attach_postgres_backends(
                    ctx.project.name, [state.legacy_backend]
                )
                if resource_error:
                    raise AgentCliError(
                        "Could not attach the Lakebase resource required for the Runtime Store.",
                        hint=resource_error,
                    )
                return ManifestPatch(env={})

            sp = self._apps.get_service_principal(ctx.project.name)
            backend = managed_runtime_store.get_or_create_backend(
                self._api_client_provider.get(), ctx.project.name, sp
            )
            env = {
                RUNTIME_STORE_LAKEBASE_BRANCH_ENV: backend.branch,
                RUNTIME_STORE_DATABASE_ENV: backend.database_id,
                RUNTIME_STORE_USERNAME_ENV: backend.username,
            }
            if not ctx.deployment_exists and ctx.project.name.startswith(_DEPLOYMENT_PREFIX):
                env[RUNTIME_STORE_SCHEMA_ENV] = _AGENTKIT_RUNTIME_STORE_SCHEMA
            return ManifestPatch(env=env)

    def manages_persistent_data(self) -> bool:
        """Whether the Runtime Store is service-managed, so a delete must tear it down first."""
        return self._use_managed

    def delete_managed(self, name: DeploymentName) -> None:
        """Drop the deployment's service-managed Runtime Store and its data."""
        sp = self._apps.get_service_principal(name)
        if not sp:
            raise AgentCliError(
                "Could not resolve the app's service principal for Runtime Store cleanup.",
                hint="The deployment was retained. Check access to the app and retry deletion.",
            )
        managed_runtime_store.delete(self._api_client_provider.get(), name, sp)


class AppProvisioner:
    """Create-or-scale the deployed App, wait for its compute, then sync and roll out the source.

    Owns the Databricks App that a deploy creates and rolls source out to. Its operations cover App
    creation, compute readiness, and source rollout; bound stores and tracing have their own
    provisioners.

    Like the rest of the services layer it talks to the terminal only through the injected
    :class:`Reporter`; it shells out through the injected ``AppsClient`` and opens a workspace
    client only through the explicit per-command ``ApiClientProvider``.
    """

    def __init__(
        self,
        apps_client: AppsClient,
        api_client_provider: ApiClientProvider,
        reporter: Reporter,
    ) -> None:
        self._apps_client = apps_client
        self._api_client_provider = api_client_provider
        self._reporter = reporter

    def ensure_app_ready(
        self,
        ctx: ResourceContext,
        *,
        app_reconciled_by_auth: bool,
        instance_count: Optional[int],
    ) -> None:
        """Ensure the App exists, apply an explicit scale, and wait for ACTIVE compute.

        Request-user auth may already have created or updated the App through the SDK; the explicit
        boolean prevents a duplicate CLI mutation without leaking the SDK update plan into this
        collaborator. An omitted instance count preserves an existing App's scale.
        """
        name = ctx.project.name
        deployment_exists = ctx.deployment_exists
        #    `apps create` itself blocks for minutes (it provisions and waits for compute) and we capture
        #    its output to relabel "App compute" → "Agent compute", so nothing streams meanwhile. Wrap it
        #    in progress (persistent line + spinner) so the CLI isn't silent for the whole provision.
        if not app_reconciled_by_auth and not deployment_exists:
            with self._reporter.progress(
                "Creating the agent and starting its compute (this can take a few minutes)…"
            ):
                out = self._apps_client.create(name, instance_count)
            old, new = _AGENT_COMPUTE_OUTPUT
            self._reporter.echo(out.replace(old, new), add_newline=False)
        # Preserve an existing App's scale when --instances was omitted. Request-user auth already
        # applied an explicit count as part of its SDK update.
        elif not app_reconciled_by_auth and instance_count is not None:
            out = self._apps_client.create_update_instances(name, instance_count)
            old, new = _AGENT_COMPUTE_OUTPUT
            self._reporter.echo(out.replace(old, new), add_newline=False)
        # `apps deploy` requires the app's compute to be ACTIVE — a just-created app may still be
        # starting, and an existing one may be STOPPED — so wait either way. Returns immediately when
        with self._reporter.progress(
            "Waiting for agent compute to start (this can take a few minutes)…"
        ):
            self._apps_client.wait_for_active(name)

    def resolve_workspace_path(self, ctx: ResourceContext, requested_path: Optional[str]) -> str:
        """Return the explicit destination for source upload, deriving the default lazily."""
        if requested_path:
            return requested_path
        current_user = self._api_client_provider.get().current_user
        return f"/Workspace/Users/{current_user}/agentbricks_deployments/{ctx.project.name}"

    def deploy_source(self, ctx: ResourceContext, workspace_path: str) -> None:
        """Sync source to an already-resolved workspace destination and activate it.

        Distinct from :meth:`AppsClient.deploy`, which is the single ``databricks apps deploy``
        call this drives once the source is synced.
        """
        self._apps_client.sync_source(ctx.project.name, ctx.project.source_dir, workspace_path)
        self._apps_client.deploy(ctx.project.name, workspace_path)
