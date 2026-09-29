"""The phased resource provisioners `agentbricks deploy` drives, one per deployed resource.

A deploy has to reconcile several independent resources - the stores declared in agent.toml, the
tracing experiment, the Runtime Store - and each of them needs work at a different point in the
rollout: some before the app exists, some only after (its service principal isn't resolvable until
then), and some only after the source is live (the grants). Expressing that as inline blocks made
``deploy()`` a long script in which each resource's three moments were pages apart, and let the
asymmetries drift: a resource's reconcile sat next to its env contribution, while its grant sat in a
separate helper shared with the other resources' grants.

So each resource implements one uniform :class:`ResourceProvisioner` with the same three phases -
``reconcile`` (pre-create), ``after_app_ready`` (post-create), ``grant`` (post-rollout) - and
``deploy()`` becomes a driver that runs each phase across every provisioner. Adding a resource means
adding a provisioner, not editing the driver.

:class:`DeployContext` is the accumulator threaded through the phases: the run's inputs plus the
facts the phases record for the final ``DeployResult``. Env is the one thing a phase does *not* write
straight to the context: a provisioner records its contribution on itself and the driver merges them
in a fixed order, because the order env lands in ``app.yaml`` is user-visible and does not match the
order the resources are reconciled in (see ``DeployService.deploy``).

Every provisioner here delegates the actual work to the render-free collaborator that already owns it
(``StoreProvisioner``, ``TracingProvisioner``, ``RuntimeStoreProvisioner``); this module holds only
the phasing, the env contributions, and the deploy-level error policy. Like the service, it is free
of any CLI framework: progress goes through the injected :class:`Reporter`.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol

from databricks_agentbricks.app_manifest import upsert_env_file
from databricks_agentbricks.app_resources import LakebaseBackend
from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.deployment import (
    _AGENTKIT_RUNTIME_STORE_SCHEMA,
    _DEPLOYMENT_PREFIX,
    DeploymentName,
    mlflow_tracing_config,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_types import AgentServer
from databricks_agentbricks.runtime_store_provisioner import RuntimeStoreProvisioner
from databricks_agentbricks.services.interaction import Reporter
from databricks_agentbricks.store_provisioner import StoreProvisioner
from databricks_agentbricks.trace_tables import TraceTable
from databricks_agentkit.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
    RUNTIME_STORE_SCHEMA_ENV,
    RUNTIME_STORE_USERNAME_ENV,
)
from databricks_agentkit.runtime.tool_manifest import MEMORY_STORE_ENV, SESSION_STORE_ENV

if TYPE_CHECKING:
    # Typing-only: ``TracingProvisioner`` imports ``cli.tracing``, whose package ``__init__`` eagerly
    # imports the command tree - including the ``deploy`` command, which reaches this module. The
    # collaborator is injected, so the type is all this module needs.
    from databricks_agentbricks.tracing_provisioner import TracingProvisioner


@dataclass
class DeployContext:
    """One deploy run, as the provisioners see it: its inputs, plus what the phases record.

    Mutable and shared: the phases read the inputs and accumulate onto the same object, which is what
    lets a later phase depend on an earlier one's result (the managed Runtime Store branch needs
    ``deployment_exists``; the trace grant needs the reconcile's ``trace_setup_error``) without the
    driver passing tuples around. ``DeployService.deploy`` turns the accumulated facts into its
    ``DeployResult``.
    """

    # --- inputs ---------------------------------------------------------------
    source_dir: pathlib.Path
    name: DeploymentName
    project: Any  # the loaded AgentProject, or None when the source has no agent.toml
    client: Any  # the workspace client, opened once after the pre-flight/auth phase
    apps_client: AppsClient
    reporter: Reporter
    use_managed_runtime_store: bool
    # Known already from the name/scope pre-flight, else None until the driver probes for it (before
    # the app is ensured, so ``after_app_ready`` can rely on it).
    deployment_exists: Optional[bool] = None

    # --- accumulators ---------------------------------------------------------
    # The app.yaml env the deploy writes, assembled by the driver in a fixed order (see the module
    # docstring), plus the agentbricks-managed keys a clean unbind prunes.
    env: dict[str, str] = field(default_factory=dict)
    env_removals: list[str] = field(default_factory=list)
    scaffolded: bool = False  # True if a manifest write created app.yaml rather than patching it
    memory_store: Optional[str] = None
    session_store: Optional[str] = None
    trace_experiment_id: Optional[str] = None
    otel_tables: list[TraceTable] = field(default_factory=list)
    trace_setup_error: Optional[str] = None
    trace_grant_error: Optional[str] = None
    store_grant_error: Optional[str] = None
    grants_stores: bool = False


class ResourceProvisioner(Protocol):
    """One deployed resource, reconciled across the three phases of a deploy.

    A phase is a hook, not a step a caller has to remember: the driver runs all three for every
    provisioner, and a resource that has nothing to do in one leaves it at its no-op default (see
    :class:`_Provisioner`). ``env`` is the provisioner's contribution to ``app.yaml``, recorded during
    ``reconcile`` and merged by the driver in the order ``app.yaml`` expects.
    """

    env: dict[str, str]

    def reconcile(self, ctx: DeployContext) -> None:
        """Pre-create: create or resolve the resource, record its facts on ``ctx`` and its env here."""
        ...

    def after_app_ready(self, ctx: DeployContext) -> None:
        """Post-create: the phase for work that needs the app to exist (e.g. its service principal).

        Env contributed here goes straight onto ``ctx.env`` (and needs its own manifest write): the
        fixed-order merge has already run by now, and a post-create key is the last env a deploy emits.
        """
        ...

    def grant(self, ctx: DeployContext) -> None:
        """Post-rollout: grant the app's identity access to the resource, recording any failure."""
        ...


class _Provisioner:
    """Base with an empty env contribution and no-op phases, so a provisioner writes only its own.

    Nothing here is abstract on purpose: most resources use one or two of the three phases, and
    forcing the other one to be spelled out as ``pass`` would add noise without adding meaning.
    """

    def __init__(self) -> None:
        self.env: dict[str, str] = {}

    def reconcile(self, ctx: DeployContext) -> None:
        """No-op unless overridden."""

    def after_app_ready(self, ctx: DeployContext) -> None:
        """No-op unless overridden."""

    def grant(self, ctx: DeployContext) -> None:
        """No-op unless overridden."""


class DeclaredStoresProvisioner(_Provisioner):
    """The memory and session stores declared in agent.toml.

    One provisioner for both, not one each, because the managed-store API grants them as a unit: a
    single call covers both stores under a single service-principal resolution, and a deploy reports
    one store-grant outcome. They are still reconciled one at a time (memory, then session), each
    through ``StoreProvisioner``, which owns that store's progress spinner and its "Created …" notice.
    """

    def __init__(
        self,
        stores: StoreProvisioner,
        memory_store: Optional[str],
        session_store: Optional[str],
    ) -> None:
        super().__init__()
        self._stores = stores
        self._memory_store = memory_store
        self._session_store = session_store

    def reconcile(self, ctx: DeployContext) -> None:
        """Create any declared store that doesn't exist yet and wire the env the runtime reads.

        `agentbricks deploy` is the only reconcile-to-cloud verb; agent.toml is the source of truth
        and is never rewritten. The memory store's bare id (not its display name) is what
        ``AGENT_MEMORY_STORE`` carries, because the entries API is keyed by id.
        """
        ctx.memory_store = self._memory_store
        ctx.session_store = self._session_store
        memory_store_id = self._stores.reconcile_declared_stores(self._memory_store, None)
        self._stores.reconcile_declared_stores(None, self._session_store)
        if memory_store_id:
            self.env[MEMORY_STORE_ENV] = memory_store_id
        if self._session_store:
            self.env[SESSION_STORE_ENV] = self._session_store

    def grant(self, ctx: DeployContext) -> None:
        """Grant the app's service principal read/write on its bound stores (best-effort).

        Goes through the managed store API, so the store service performs the underlying Lakebase
        grant - no store ownership or Lakebase MANAGE required of the deployer. A failure is recorded,
        not raised: the deploy succeeded, and the CLI reports the missing grant as a next step.
        """
        ctx.grants_stores = bool(self._session_store or self._memory_store)
        if not ctx.grants_stores:
            return
        with ctx.reporter.status("Granting the app access to its stores…"):
            sp = ctx.apps_client.service_principal(ctx.name)
            if sp is None:
                ctx.store_grant_error = "could not resolve the app's service principal."
            else:
                ctx.store_grant_error = self._stores.grant_store_access(
                    sp, self._session_store, self._memory_store
                )


class TraceExperimentProvisioner(_Provisioner):
    """The MLflow experiment a deployed agent traces to, and the app resources granting write access.

    Best-effort throughout: tracing is an add-on, so neither a failed resolve nor a failed grant
    blocks a deploy - both are recorded for the CLI to report.
    """

    def __init__(self, tracing: TracingProvisioner, experiment_name: Optional[str]) -> None:
        super().__init__()
        self._tracing = tracing
        self._experiment_name = experiment_name

    def reconcile(self, ctx: DeployContext) -> None:
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
        try:
            if self._experiment_name:
                # Show progress while the experiment is get-or-created (a workspace round-trip),
                # matching the store reconcile spinners so deploy isn't silent about tracing.
                with ctx.reporter.status(
                    f"Reconciling tracing experiment '{self._experiment_name}'…"
                ):
                    trace_provision = self._tracing.get_or_create(ctx.source_dir, ctx.client)
            else:
                trace_provision = self._tracing.get_or_create(ctx.source_dir, ctx.client)
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort; never block a deploy
            ctx.trace_setup_error = str(exc)
        ctx.trace_experiment_id = trace_provision.experiment_id if trace_provision else None
        # No resolved experiment means no UC OTEL tables to grant - the same as an empty table set.
        ctx.otel_tables = list(trace_provision.tables.otel_tables()) if trace_provision else []
        if ctx.trace_experiment_id:
            self.env.update(mlflow_tracing_config(ctx.trace_experiment_id).env())
        elif ctx.trace_setup_error is None:
            ctx.env_removals = list(mlflow_tracing_config("").env())  # the MLFLOW_* keys to prune

    def grant(self, ctx: DeployContext) -> None:
        """Reconcile the agentbricks-owned trace resources whenever tracing resolved cleanly.

        A resolved experiment grants that set (the ``experiment`` resource, CAN_EDIT, plus MODIFY on
        any UC OTEL tables via ``uc_securable`` resources); a cleanly-unbound project prunes the
        agentbricks-trace-* resources an earlier bound deploy left behind. When resolving the BOUND
        experiment errored instead we don't know the intended state, so the reconcile is skipped
        rather than run - a flaky deploy must not silently revoke trace access the way an unbind does.

        (Whether removing a ``uc_securable`` resource also revokes the underlying UC MODIFY grant is
        platform behavior - documented but not yet verified live.)
        """
        if ctx.trace_setup_error is not None:
            return
        with ctx.reporter.status("Granting the agent runtime access to its trace experiment…"):
            ctx.trace_grant_error = self._tracing.apply_resources(
                ctx.name, ctx.trace_experiment_id, ctx.otel_tables
            )


class RuntimeStoreResourceProvisioner(_Provisioner):
    """The persistent Runtime Store behind an Agent Bricks Runtime deployment.

    The only resource that splits across both create phases, because a temporary rollout switch picks
    between two backends with different requirements: the legacy per-app Lakebase project can be
    provisioned before the app exists (only its resource attachment has to wait), while the
    service-managed database is owned by the app's service principal and so can't even be created
    until the app is there. Custom-server projects have no Runtime Store and skip every phase.
    """

    def __init__(self, runtime_store: RuntimeStoreProvisioner) -> None:
        super().__init__()
        self._runtime_store = runtime_store
        # None means the managed branch: `after_app_ready` keys off this to tell the two backends apart.
        self._legacy_backend: Optional[LakebaseBackend] = None

    @staticmethod
    def _uses_runtime_store(ctx: DeployContext) -> bool:
        """Only Agent Bricks Runtime projects get a Runtime Store; a custom server manages its own."""
        return ctx.project is not None and ctx.project.server == AgentServer.AGENTBRICKS

    def reconcile(self, ctx: DeployContext) -> None:
        """Legacy backend only: get-or-create the per-app Lakebase project and wire its env."""
        if not self._uses_runtime_store(ctx) or ctx.use_managed_runtime_store:
            return
        with ctx.reporter.status("Reconciling Runtime Store…"):
            self._legacy_backend = self._runtime_store.legacy_backend(ctx.name)
        self.env[RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV] = self._legacy_backend.endpoint_path
        self.env[RUNTIME_STORE_SCHEMA_ENV] = self._legacy_backend.schema

    def after_app_ready(self, ctx: DeployContext) -> None:
        """Attach the legacy backend to the app, or provision the managed store the app now owns.

        The managed branch writes its own env: app.yaml was already patched before the app was
        created, so these fields need a second manifest write, and they land last in the deploy's env.
        """
        if self._legacy_backend is not None:
            resource_error = self._runtime_store.apply_legacy_resource(
                ctx.name, self._legacy_backend
            )
            if resource_error:
                raise AgentCliError(
                    "Could not attach the Lakebase resource required for the Runtime Store.",
                    hint=resource_error,
                )
            return
        if not self._uses_runtime_store(ctx):
            return
        app_service_principal_id = ctx.apps_client.service_principal(ctx.name)
        with ctx.reporter.status("Reconciling Runtime Store…"):
            runtime_backend = self._runtime_store.managed_backend(
                ctx.client, ctx.name, app_service_principal_id
            )
        managed_env = {
            RUNTIME_STORE_LAKEBASE_BRANCH_ENV: runtime_backend.branch,
            RUNTIME_STORE_DATABASE_ENV: runtime_backend.database_id,
            RUNTIME_STORE_USERNAME_ENV: runtime_backend.username,
        }
        if not ctx.deployment_exists and ctx.name.startswith(_DEPLOYMENT_PREFIX):
            managed_env[RUNTIME_STORE_SCHEMA_ENV] = _AGENTKIT_RUNTIME_STORE_SCHEMA
        ctx.scaffolded = upsert_env_file(ctx.source_dir, managed_env) or ctx.scaffolded
        ctx.env.update(managed_env)
