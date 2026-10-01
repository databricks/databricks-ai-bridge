"""The phased resource provisioners `agentbricks deploy` drives, one per deployed resource.

A deploy has to reconcile several independent resources - the memory and session stores declared in
agent.toml, the tracing experiment, the Runtime Store - and each needs work at a different point in
the rollout: some before the app exists, some only after (its service principal isn't resolvable
until then), and some only after the source is live (the grants). Expressing that inline made
``deploy()`` a long script in which each resource's three moments were pages apart.

So each resource implements one uniform :class:`ResourceProvisioner` with the same three phases -
``reconcile`` (pre-create), ``after_app_ready`` (post-create), ``grant`` (post-rollout) - and
``deploy()`` becomes a driver that runs each phase across every provisioner. Adding a resource means
adding a provisioner, not editing the driver.

The contexts passed to the phases are immutable data, split small on purpose (no kitchen-sink
object, no methods, no collaborators): :class:`ProjectContext` is the deploy's identity (source
dir, name, agent project) and :class:`ResourceContext` is that plus the per-resource bindings read
out of agent.toml and whether the app pre-existed this deploy. A phase records nothing back onto
them - instead each provisioner accumulates its OWN outcome on itself (its ``env`` contribution,
the keys it wants pruned, whether its manifest write scaffolded, and its per-resource
facts/errors), and the driver reads those to assemble the ``DeployResult`` and to write
``app.yaml``. The app service principal several phases need is resolved and cached inside the
``AppsClient`` the low-level clients hold, so it needn't be threaded through here.

Each provisioner holds its own low-level client - ``MemoryStoreClient``, ``SessionStoreClient``,
``TracingClient``, ``RuntimeStoreClient`` - plus the :class:`Reporter` its progress spinners go
through, both injected at construction; this module holds only the phasing, the env contributions,
and the deploy-level error policy. (The store clients still render their reconcile notices
directly, a pre-existing wart left for a follow-up.)

The module also holds :class:`AppProvisioner`, which owns the deployed app itself and is
deliberately not one of the phased resource provisioners.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

from databricks_agentbricks.app_auth_client import AppUserScopeUpdatePlan
from databricks_agentbricks.app_manifest import AppManifest
from databricks_agentbricks.app_resources import LakebaseBackend
from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.deployment import (
    _AGENT_COMPUTE_OUTPUT,
    _AGENTKIT_RUNTIME_STORE_SCHEMA,
    _DEPLOYMENT_PREFIX,
    DeploymentName,
    mlflow_tracing_config,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_types import AgentServer
from databricks_agentbricks.services.interaction import Reporter
from databricks_agentbricks.store_client import (
    MemoryStoreClient,
    RuntimeStoreClient,
    SessionStoreClient,
)
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
    # Typing-only: ``TracingClient`` imports ``cli.tracing``, whose package ``__init__`` eagerly
    # imports the command tree - including the ``deploy`` command, which reaches this module. The
    # collaborator is injected, so the type is all this module needs.
    from databricks_agentbricks.tracing_client import TracingClient


@dataclass(frozen=True)
class ProjectContext:
    """What is being deployed: the identity shared by every resource, and nothing else."""

    source_dir: pathlib.Path
    name: DeploymentName
    agent_project: Any  # the loaded AgentProject, or None when the source has no agent.toml


@dataclass(frozen=True)
class ResourceContext:
    """What a provisioner phase is handed: the project identity plus the resource bindings.

    Pure data - immutable, method-free, and holding no collaborators. A phase reads what it needs
    and records its result on its own provisioner; the driver owns the rest.
    """

    project: ProjectContext
    memory_store: Optional[str]  # the memory store bound in agent.toml, if any
    session_store: Optional[str]  # the session store bound in agent.toml, if any
    experiment_name: Optional[str]  # the tracing experiment bound in agent.toml, if any
    deployment_exists: bool = False  # whether the app pre-existed this deploy


class ResourceProvisioner(Protocol):
    """One deployed resource, reconciled across the three phases of a deploy.

    A phase is a hook, not a step a caller has to remember: the driver runs all three for every
    provisioner, and a resource that has nothing to do in one leaves it at its no-op default (see
    :class:`_Provisioner`). Between phases a provisioner accumulates its contribution to the shared
    ``app.yaml`` on itself: ``env`` (keys to set), ``env_removals`` (keys to prune), and ``scaffolded``
    (whether its own manifest write created the file). The driver merges those in the order ``app.yaml``
    expects.
    """

    env: dict[str, str]
    env_removals: list[str]
    scaffolded: bool

    def reconcile(self, ctx: ResourceContext) -> None:
        """Pre-create: create or resolve the resource; record its env and facts on the provisioner."""
        ...

    def after_app_ready(self, ctx: ResourceContext) -> None:
        """Post-create: work that needs the app to exist (e.g. its service principal).

        ``ctx.deployment_exists`` tells a resource whether the app pre-existed this deploy. Env
        contributed here is the last a deploy emits, so a provisioner that adds it also writes it
        to ``app.yaml``.
        """
        ...

    def grant(self, ctx: ResourceContext) -> None:
        """Post-rollout: grant the app's identity access to the resource, recording any failure."""
        ...


class _Provisioner:
    """Base with empty manifest contributions and no-op phases, so a provisioner writes only its own.

    Nothing here is abstract on purpose: most resources use one or two of the three phases, and
    forcing the others to be spelled out as ``pass`` would add noise without adding meaning.
    """

    def __init__(self) -> None:
        self.env: dict[str, str] = {}
        self.env_removals: list[str] = []
        self.scaffolded: bool = False

    def reconcile(self, ctx: ResourceContext) -> None:
        """No-op unless overridden."""

    def after_app_ready(self, ctx: ResourceContext) -> None:
        """No-op unless overridden."""

    def grant(self, ctx: ResourceContext) -> None:
        """No-op unless overridden."""


class MemoryStoreProvisioner(_Provisioner):
    """The memory store declared in agent.toml, reconciled and granted on its own.

    A sibling of the session, tracing, and Runtime Store provisioners: same three phases, its own env
    contribution, its own grant. Delegates the API work to its own ``MemoryStoreClient``.
    """

    def __init__(self, memory_store_client: MemoryStoreClient, reporter: Reporter) -> None:
        super().__init__()
        self._memory_store_client = memory_store_client
        self._reporter = reporter
        self.store_name: Optional[str] = None  # the bound store name, surfaced on the DeployResult
        self.grant_error: Optional[str] = None
        self.grants_access: bool = False

    def reconcile(self, ctx: ResourceContext) -> None:
        """Create the declared memory store if absent and wire ``AGENT_MEMORY_STORE``.

        `agentbricks deploy` is the only reconcile-to-cloud verb; agent.toml is the source of truth and
        is never rewritten. ``AGENT_MEMORY_STORE`` carries the store's bare id (not its display name),
        because the entries API is keyed by id.
        """
        self.store_name = ctx.memory_store
        if not self.store_name:
            return
        memory_store_id = self._memory_store_client.reconcile(self.store_name)
        if memory_store_id:
            self.env[MEMORY_STORE_ENV] = memory_store_id

    def grant(self, ctx: ResourceContext) -> None:
        """Grant the app's service principal read/write on the memory store (best-effort).

        Goes through the managed store API, so the store service performs the underlying Lakebase grant
        - no store ownership or Lakebase MANAGE required of the deployer. A failure is recorded, not
        raised: the deploy succeeded, and the CLI reports the missing grant as a next step.
        """
        if not self.store_name:
            return
        with self._reporter.status("Granting the app access to its memory store…"):
            self.grant_error = self._memory_store_client.grant(ctx.project.name, self.store_name)
            self.grants_access = True


class SessionStoreProvisioner(_Provisioner):
    """The session store declared in agent.toml, reconciled and granted on its own.

    The memory store's sibling: same shape, delegating to its own ``SessionStoreClient``. Session
    stores resolve by name, so ``AGENT_SESSION_STORE`` carries the name rather than a resolved id.
    """

    def __init__(self, session_store_client: SessionStoreClient, reporter: Reporter) -> None:
        super().__init__()
        self._session_store_client = session_store_client
        self._reporter = reporter
        self.store_name: Optional[str] = None  # the bound store name, surfaced on the DeployResult
        self.grant_error: Optional[str] = None
        self.grants_access: bool = False

    def reconcile(self, ctx: ResourceContext) -> None:
        """Create the declared session store if absent and wire ``AGENT_SESSION_STORE``."""
        self.store_name = ctx.session_store
        if not self.store_name:
            return
        self._session_store_client.reconcile(self.store_name)
        self.env[SESSION_STORE_ENV] = self.store_name

    def grant(self, ctx: ResourceContext) -> None:
        """Grant the app's service principal read/write on the session store (best-effort).

        Same managed-store-API path and best-effort contract as the memory grant; the service principal
        is resolved once and cached in ``AppsClient``, so this and the memory grant share the one lookup.
        """
        if not self.store_name:
            return
        with self._reporter.status("Granting the app access to its session store…"):
            self.grant_error = self._session_store_client.grant(ctx.project.name, self.store_name)
            self.grants_access = True


class TracingProvisioner(_Provisioner):
    """The MLflow experiment a deployed agent traces to, and the app resources granting write access.

    Best-effort throughout: tracing is an add-on, so neither a failed resolve nor a failed grant
    blocks a deploy - both are recorded for the CLI to report.
    """

    def __init__(self, tracing_client: TracingClient, reporter: Reporter) -> None:
        super().__init__()
        self._tracing_client = tracing_client
        self._reporter = reporter
        self.experiment_id: Optional[str] = None
        self.otel_tables: list[TraceTable] = []
        self.setup_error: Optional[str] = None
        self.grant_error: Optional[str] = None

    def reconcile(self, ctx: ResourceContext) -> None:
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
            if ctx.experiment_name:
                # Show progress while the experiment is get-or-created (a workspace round-trip),
                # matching the store reconcile spinners so deploy isn't silent about tracing.
                with self._reporter.status(
                    f"Reconciling tracing experiment '{ctx.experiment_name}'…"
                ):
                    trace_provision = self._tracing_client.get_or_create(ctx.project.source_dir)
            else:
                trace_provision = self._tracing_client.get_or_create(ctx.project.source_dir)
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort; never block a deploy
            self.setup_error = str(exc)
        self.experiment_id = trace_provision.experiment_id if trace_provision else None
        # No resolved experiment means no UC OTEL tables to grant - the same as an empty table set.
        self.otel_tables = list(trace_provision.tables.otel_tables()) if trace_provision else []
        if self.experiment_id:
            self.env.update(mlflow_tracing_config(self.experiment_id).env())
        elif self.setup_error is None:
            self.env_removals = list(mlflow_tracing_config("").env())  # the MLFLOW_* keys to prune

    def grant(self, ctx: ResourceContext) -> None:
        """Reconcile the agentbricks-owned trace resources whenever tracing resolved cleanly.

        A resolved experiment grants that set (the ``experiment`` resource, CAN_EDIT, plus MODIFY on
        any UC OTEL tables via ``uc_securable`` resources); a cleanly-unbound project prunes the
        agentbricks-trace-* resources an earlier bound deploy left behind. When resolving the BOUND
        experiment errored instead we don't know the intended state, so the reconcile is skipped
        rather than run - a flaky deploy must not silently revoke trace access the way an unbind does.

        (Whether removing a ``uc_securable`` resource also revokes the underlying UC MODIFY grant is
        platform behavior - documented but not yet verified live.)
        """
        if self.setup_error is not None:
            return
        with self._reporter.status("Granting the agent runtime access to its trace experiment…"):
            self.grant_error = self._tracing_client.apply_resources(
                ctx.project.name, self.experiment_id, self.otel_tables
            )


class RuntimeStoreProvisioner(_Provisioner):
    """The persistent Runtime Store behind an Agent Bricks Runtime deployment.

    The only resource that splits across both create phases, because a temporary rollout switch picks
    between two backends with different requirements: the legacy per-app Lakebase project can be
    provisioned before the app exists (only its resource attachment has to wait), while the
    service-managed database is owned by the app's service principal and so can't even be created
    until the app is there. Custom-server projects have no Runtime Store and skip every phase. The
    switch itself is encapsulated in the ``RuntimeStoreClient`` - this provisioner asks
    ``is_managed()`` rather than holding the flag.
    """

    def __init__(self, runtime_store_client: RuntimeStoreClient, reporter: Reporter) -> None:
        super().__init__()
        self._runtime_store_client = runtime_store_client
        self._reporter = reporter
        # None means the managed branch: `after_app_ready` keys off this to tell the two backends apart.
        self._legacy_backend: Optional[LakebaseBackend] = None

    @staticmethod
    def _uses_runtime_store(ctx: ResourceContext) -> bool:
        """Only Agent Bricks Runtime projects get a Runtime Store; a custom server manages its own."""
        return (
            ctx.project.agent_project is not None
            and ctx.project.agent_project.server == AgentServer.AGENTBRICKS
        )

    def reconcile(self, ctx: ResourceContext) -> None:
        """Legacy backend only: get-or-create the per-app Lakebase project and wire its env."""
        if not self._uses_runtime_store(ctx) or self._runtime_store_client.is_managed():
            return
        with self._reporter.status("Reconciling Runtime Store…"):
            self._legacy_backend = self._runtime_store_client.legacy_backend(ctx.project.name)
        self.env[RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV] = self._legacy_backend.endpoint_path
        self.env[RUNTIME_STORE_SCHEMA_ENV] = self._legacy_backend.schema

    def after_app_ready(self, ctx: ResourceContext) -> None:
        """Attach the legacy backend to the app, or provision the managed store the app now owns.

        The managed branch writes its own env: app.yaml was already patched before the app was
        created, so these fields need a second manifest write, and they land last in the deploy's env.
        """
        if self._legacy_backend is not None:
            resource_error = self._runtime_store_client.apply_legacy_resource(
                ctx.project.name, self._legacy_backend
            )
            if resource_error:
                raise AgentCliError(
                    "Could not attach the Lakebase resource required for the Runtime Store.",
                    hint=resource_error,
                )
            return
        if not self._uses_runtime_store(ctx):
            return
        with self._reporter.status("Reconciling Runtime Store…"):
            runtime_backend = self._runtime_store_client.managed_backend(ctx.project.name)
        managed_env = {
            RUNTIME_STORE_LAKEBASE_BRANCH_ENV: runtime_backend.branch,
            RUNTIME_STORE_DATABASE_ENV: runtime_backend.database_id,
            RUNTIME_STORE_USERNAME_ENV: runtime_backend.username,
        }
        if not ctx.deployment_exists and ctx.project.name.startswith(_DEPLOYMENT_PREFIX):
            managed_env[RUNTIME_STORE_SCHEMA_ENV] = _AGENTKIT_RUNTIME_STORE_SCHEMA
        self.scaffolded = (
            AppManifest.upsert_env_file(ctx.project.source_dir, managed_env) or self.scaffolded
        )
        self.env.update(managed_env)

    def manages_persistent_data(self) -> bool:
        """Whether the Runtime Store is service-managed, so a delete must tear it down first."""
        return self._runtime_store_client.is_managed()

    def delete_managed(self, name: DeploymentName) -> None:
        """Drop the deployment's service-managed Runtime Store and its data."""
        self._runtime_store_client.delete_managed(name)


class AppProvisioner:
    """Create-or-scale the deployed App, wait for its compute, then sync and roll out the source.

    Owns the app the four :class:`ResourceProvisioner`\\ s above hang off - the Databricks App that
    a deploy creates and rolls source out to. It deliberately does NOT implement the three-phase
    ``ResourceProvisioner`` protocol: its moments are "create the app when new (or re-pin its scale)
    and wait for its compute to be ACTIVE" and "roll the source out", not
    ``reconcile``/``after_app_ready``/``grant``, so the naming does not imply it is one of the
    resource provisioners.

    Like the rest of the services layer it talks to the terminal only through the injected
    :class:`Reporter`; it shells out through the injected ``AppsClient`` and opens a workspace
    client only through ``api_client_factory``.
    """

    def __init__(
        self,
        apps_client: AppsClient,
        api_client_factory: Callable[[], Any],
        reporter: Reporter,
    ) -> None:
        self._apps_client = apps_client
        self._api_client_factory = api_client_factory
        self._reporter = reporter

    def create_and_wait_for_active(
        self,
        ctx: ResourceContext,
        user_scope_plan: Optional[AppUserScopeUpdatePlan],
        instance_count: int,
        instance_args: list[str],
    ) -> None:
        """Ensure the app exists at the requested scale, then block until its compute is ACTIVE.

        Creates the app when it is new; when the app already exists, re-pins its scale to the
        instance count this deploy asked for (via ``create_update_instances``). In both cases it
        then waits for the compute to reach ACTIVE - the name is not create-only: an existing app
        still gets its scale re-pinned and its compute waited on every deploy.
        """
        name = ctx.project.name
        deployment_exists = ctx.deployment_exists
        #    `apps create` itself blocks for minutes (it provisions and waits for compute) and we capture
        #    its output to relabel "App compute" → "Agent compute", so nothing streams meanwhile. Wrap it
        #    in progress (persistent line + spinner) so the CLI isn't silent for the whole provision.
        if user_scope_plan is None and not deployment_exists:
            with self._reporter.progress(
                "Creating the agent and starting its compute (this can take a few minutes)…"
            ):
                out = self._apps_client.create(name, instance_args)
            old, new = _AGENT_COMPUTE_OUTPUT
            self._reporter.echo(out.replace(old, new), add_newline=False)
        # An existing app has its scale re-pinned every deploy, so the count the deploy asked for wins
        # over whatever a previous deploy left behind. (When a scope plan ran it already applied the
        # count as part of the same Apps update.)
        elif user_scope_plan is None:
            out = self._apps_client.create_update_instances(name, instance_count)
            old, new = _AGENT_COMPUTE_OUTPUT
            self._reporter.echo(out.replace(old, new), add_newline=False)
        # `apps deploy` requires the app's compute to be ACTIVE — a just-created app may still be
        # starting, and an existing one may be STOPPED — so wait either way. Returns immediately when
        with self._reporter.progress(
            "Waiting for agent compute to start (this can take a few minutes)…"
        ):
            self._apps_client.wait_for_running(name)

    def deploy(self, ctx: ResourceContext, workspace_path: Optional[str]) -> str:
        """The app-level rollout: sync the source to the workspace, then ``apps deploy`` it.

        Distinct from :meth:`AppsClient.deploy`, which is the single ``databricks apps deploy``
        call this drives once the source is synced.
        """
        ws_path = (
            workspace_path
            or f"/Workspace/Users/{self._api_client_factory().current_user}/agentbricks_deployments/{ctx.project.name}"
        )
        self._apps_client.sync_source(ctx.project.name, ctx.project.source_dir, ws_path)
        self._apps_client.deploy(ctx.project.name, ws_path)
        return ws_path
