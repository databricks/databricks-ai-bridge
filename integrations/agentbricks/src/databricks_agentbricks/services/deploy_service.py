"""The framework-agnostic deployment workflow behind `agentbricks deploy` and `agentbricks deployments`.

``DeployService`` owns the deployment business logic. ``deploy`` is the big one: pre-flight + auth,
reconciling the stores and tracing bound in agent.toml, patching app.yaml, ensuring the app exists,
rolling out the source, and granting access. The lifecycle verbs (``list_deployments``, ``get``,
``logs``, ``start``, ``stop``, ``delete``) are thin, but they live here too so that the policy they
carry - what counts as an agent deployment, which name shapes are legal, what a destructive verb
confirms, and that a managed Runtime Store is torn down before its app - is stated once instead of
in each command.

It talks to the terminal only through the injected :class:`Reporter` and :class:`Prompter` ports and
hands back raw facts (a :class:`DeployResult`, or the Apps payloads as they came off the wire), so
the CLI layer owns every presentation decision and this module needs no CLI framework of its own.

Collaborators are injected so the command composes the service from the CLI context while tests
construct it with fakes. ``client_factory`` is called once, after the pre-flight/auth phase, so a
pre-flight failure never opens a workspace client.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.deployment import (
    _AGENT_COMPUTE_OUTPUT,
    _AGENTKIT_RUNTIME_STORE_SCHEMA,
    _DEPLOYMENT_PREFIX,
    _MAX_DEPLOYMENT_NAME_LEN,
    _PIP_INDEX_ENVS,
    _USE_MANAGED_RUNTIME_STORE,
    _instance_args,
    _prefixed_name,
    _validate_deployment_name,
    mlflow_tracing_config,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.manifest_manager import ManifestManager
from databricks_agentbricks.project_config import require_managed_tool_support
from databricks_agentbricks.project_resolver import ProjectResolver
from databricks_agentbricks.project_types import AgentServer
from databricks_agentbricks.runtime_store_provisioner import RuntimeStoreProvisioner
from databricks_agentbricks.services.interaction import Prompter, Reporter
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
    # Typing-only, because both of these reach the `cli` package (``TracingProvisioner`` imports
    # ``cli.tracing``), whose ``__init__`` eagerly imports the command tree — including the ``deploy``
    # command, which imports this module. Importing them at runtime would make importing the service
    # itself circular; the service only ever needs them as types, since the collaborator is injected.
    from databricks_agentbricks.cli.tracing import ResolvedTraceExperiment
    from databricks_agentbricks.tracing_provisioner import TracingProvisioner


class OperationAborted(Exception):
    """A destructive verb the user declined at the confirm prompt.

    The service's own signal, so declining doesn't require a CLI-framework exception here; the
    command translates it into whatever its framework uses to abort.
    """


def _field(obj: Any, name: str) -> Any:
    """Read a field off an Apps payload, tolerating snake_case or camelCase JSON keys.

    The `databricks` CLI's JSON uses either spelling depending on the version, so a lookup that
    assumed one would silently read None from the other. Duplicated (small) rather than imported
    from the display helpers, which reach for a CLI framework this module stays clear of.
    """
    if name in obj:
        return obj[name]
    parts = name.split("_")
    camel = parts[0] + "".join(p.title() for p in parts[1:])
    return obj.get(camel)


@dataclass(frozen=True)
class DeployRequest:
    """One `agentbricks deploy` invocation: the requested name plus the rollout options."""

    name: Optional[str]
    source: str
    pip_index_url: Optional[str]
    workspace_path: Optional[str]
    instances: Optional[int]
    allow_user_scope_update: bool


@dataclass(frozen=True)
class DeployResult:
    """What a deploy did, as raw facts: no formatted strings, no display-only derivations.

    The CLI turns this into either the JSON payload or the success panel; keeping it free of
    presentation choices means a non-CLI caller (or a test) can assert on the facts directly.
    """

    deployment: str
    source: str
    url: Optional[str]
    workspace_path: str
    env: dict[str, str]
    client_host: str
    memory_store: Optional[str]
    session_store: Optional[str]
    trace_experiment_id: Optional[str]
    uc_trace_tables: list[str]
    trace_setup_error: Optional[str]
    trace_grant_error: Optional[str]
    store_grant_error: Optional[str]
    grants_stores: bool
    scaffolded: bool
    pip_index_url: Optional[str]
    instances: Optional[int]
    uses_runtime_api: bool


@dataclass(frozen=True)
class _DeployPlan:
    """Resolved deploy identity + auth decisions from the pre-flight phase."""

    project: Any
    base_name: str
    name: str
    user_scope_plan: Any
    deployment_exists: Optional[bool]  # known already from the name/scope pre-flight, else None


class DeployService:
    """Owns the deployment verbs: `deploy` (pre-flight + auth, reconcile stores + tracing, patch
    app.yaml, ensure the app, roll out, grant access) and the lifecycle verbs that list, read, tail,
    start, stop, and delete what it deployed, then reports the facts back.
    """

    def __init__(
        self,
        *,
        project: ProjectResolver,
        apps_client: AppsClient,
        stores_factory: Callable[[Any], StoreProvisioner],
        manifest: ManifestManager,
        tracing: TracingProvisioner,
        runtime_store: RuntimeStoreProvisioner,
        client_factory: Callable[[], Any],
        profile: Optional[str],
        reporter: Reporter,
        prompter: Prompter,
    ) -> None:
        self._project = project
        self._apps_client = apps_client
        self._stores_factory = stores_factory
        self._manifest = manifest
        self._tracing = tracing
        self._runtime_store = runtime_store
        self._client_factory = client_factory
        self._profile = profile
        self._reporter = reporter
        self._prompter = prompter

    # --- lifecycle verbs ----------------------------------------------------

    def list_deployments(self) -> list[dict]:
        """Every agent deployment, as raw Apps payloads.

        Agent deployments are Apps named with the Agent Bricks prefix, so the workspace's other apps
        are filtered out here - the prefix convention is this service's policy, not the client's.
        """
        return [
            a
            for a in self._apps_client.list_all()
            if str(_field(a, "name") or "").startswith(_DEPLOYMENT_PREFIX)
        ]

    def get(self, name: str) -> dict:
        """One deployment's raw Apps payload."""
        _validate_deployment_name(name)
        return self._apps_client.get(name)

    def logs(self, name: str) -> None:
        """Stream the deployment's logs to the terminal until the user interrupts."""
        _validate_deployment_name(name)
        self._apps_client.logs(name)

    def start(self, name: str) -> None:
        _validate_deployment_name(name)
        self._apps_client.start(name)

    def stop(self, name: str, *, assume_yes: bool) -> None:
        """Stop a deployment, confirming first unless `assume_yes` (for scripts)."""
        _validate_deployment_name(name)
        self._confirm_destroy(f"Stop deployment '{name}'", assume_yes=assume_yes)
        self._apps_client.stop(name)

    def delete(self, name: str, *, assume_yes: bool) -> None:
        """Delete a deployment and, when managed provisioning is on, its Runtime Store first.

        The Runtime Store goes first because dropping it needs the app's service principal, which
        stops resolving once the app is gone. If that identity can't be read we refuse outright
        rather than delete the app and orphan its data.
        """
        _validate_deployment_name(name)
        use_managed_runtime_store = _USE_MANAGED_RUNTIME_STORE
        # Name the data loss in the prompt: with a managed store, deleting the app also drops the
        # agent's persisted memory/sessions, which the app name alone doesn't imply.
        target = (
            f"Delete deployment '{name}' and its Runtime Store data"
            if use_managed_runtime_store
            else f"Delete deployment '{name}'"
        )
        self._confirm_destroy(target, assume_yes=assume_yes)
        if use_managed_runtime_store:
            app_service_principal_id = self._apps_client.service_principal(name)
            if not app_service_principal_id:
                raise AgentCliError(
                    "Could not resolve the app's service principal for Runtime Store cleanup.",
                    hint="The deployment was retained. Check access to the app and retry deletion.",
                )
            with self._reporter.status("Deleting Runtime Store…"):
                self._runtime_store.delete_managed(
                    self._client_factory(), name, app_service_principal_id
                )
        self._apps_client.delete(name)

    def _confirm_destroy(self, target: str, *, assume_yes: bool) -> None:
        """Confirm before a destructive op; `assume_yes` skips the prompt (for scripts)."""
        if assume_yes:
            return
        if not self._prompter.confirm(f"{target}? This cannot be undone.", default=False):
            raise OperationAborted()

    # --- deploy -------------------------------------------------------------

    def deploy(self, request: DeployRequest) -> DeployResult:
        source_dir = pathlib.Path(request.source)
        plan = self._authorize(source_dir, request)
        name = plan.name
        instances = request.instances
        instance_args = _instance_args(instances)
        use_managed_runtime_store = _USE_MANAGED_RUNTIME_STORE
        client = self._client_factory()
        stores = self._stores_factory(client)

        # 1. Reconcile the stores DECLARED in agent.toml: create any that don't exist yet. `agentbricks deploy`
        #    is the only reconcile-to-cloud verb; agent.toml is the source of truth and is never rewritten.
        memory_store, session_store, experiment_name = self._project.resource_bindings(source_dir)
        memory_store_id = stores.reconcile_declared_stores(memory_store, session_store)

        # 2. Provision tracing when bound (`agentbricks init` binds a default experiment): get-or-create the
        #    experiment NAME from agent.toml and wire the two env vars the runtime reads. Resolved by name,
        #    never a stored id, and nothing is written back to agent.toml. (`agentbricks dev` traces to a local
        #    MLflow server instead and never touches this workspace experiment.) The app's SP is granted
        #    write access to it in step 5 (an experiment app resource, plus MODIFY on its UC OTEL tables
        #    when UC-backed). Best-effort: if it can't be set up
        #    (no mlflow, offline, permission), the deploy still proceeds without tracing.
        trace_provision: Optional[ResolvedTraceExperiment] = None
        trace_setup_error: Optional[str] = None
        try:
            if experiment_name:
                # Show progress while the experiment is get-or-created (a workspace round-trip), matching
                # the memory/session store reconcile spinners so deploy isn't silent about tracing.
                with self._reporter.status(f"Reconciling tracing experiment '{experiment_name}'…"):
                    trace_provision = self._tracing.get_or_create(source_dir, client)
            else:
                trace_provision = self._tracing.get_or_create(source_dir, client)
        except Exception as exc:  # noqa: BLE001 - tracing is best-effort; never block a deploy
            trace_setup_error = str(exc)
        trace_experiment_id = trace_provision.experiment_id if trace_provision else None
        # No resolved experiment means no UC OTEL tables to grant - the same as an empty table set.
        otel_tables = trace_provision.tables.otel_tables() if trace_provision else []
        env_updates: dict[str, str] = {}
        # Trace env: set it when bound; on a CLEAN unbind (tracing resolved to None, no setup error) remove
        # the stale MLFLOW_* keys so the manifest stops pointing the runtime at an experiment whose grant
        # was just pruned. On a resolve ERROR (trace_setup_error) we touch neither the env nor the trace
        # resources - a transient failure must not look like an unbind. (Store env is still upsert-only, a
        # separate follow-up.)
        trace_env_removals: list[str] = []
        if trace_experiment_id:
            env_updates.update(mlflow_tracing_config(trace_experiment_id).env())
        elif trace_setup_error is None:
            trace_env_removals = list(mlflow_tracing_config("").env())  # the MLFLOW_* keys to prune
        if memory_store_id:
            env_updates[MEMORY_STORE_ENV] = memory_store_id
        if session_store:
            env_updates[SESSION_STORE_ENV] = session_store

        legacy_runtime_backend = None
        if (
            plan.project is not None
            and plan.project.server == AgentServer.AGENTBRICKS
            and not use_managed_runtime_store
        ):
            with self._reporter.status("Reconciling Runtime Store…"):
                legacy_runtime_backend = self._runtime_store.legacy_backend(name)
            env_updates[RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV] = legacy_runtime_backend.endpoint_path
            env_updates[RUNTIME_STORE_SCHEMA_ENV] = legacy_runtime_backend.schema
        if request.pip_index_url:
            for env in _PIP_INDEX_ENVS:
                env_updates[env] = request.pip_index_url

        # 3. Patch app.yaml before creating the app. The managed Runtime Store fields are added after
        #    app creation because that API requires the app's service principal.
        scaffolded = (
            self._manifest.upsert_env(source_dir, env_updates, trace_env_removals)
            if (env_updates or trace_env_removals)
            else False
        )

        # 4. Ensure the app exists and its compute is active. Create only when new; the compute wait
        #    runs every deploy.
        deployment_exists = plan.deployment_exists
        if deployment_exists is None:
            deployment_exists = self._apps_client.exists(name)
        self._ensure_app(name, deployment_exists, plan.user_scope_plan, instances, instance_args)

        if legacy_runtime_backend is not None:
            resource_error = self._runtime_store.apply_legacy_resource(name, legacy_runtime_backend)
            if resource_error:
                raise AgentCliError(
                    "Could not attach the Lakebase resource required for the Runtime Store.",
                    hint=resource_error,
                )
        elif plan.project is not None and plan.project.server == AgentServer.AGENTBRICKS:
            app_service_principal_id = self._apps_client.service_principal(name)
            with self._reporter.status("Reconciling Runtime Store…"):
                runtime_backend = self._runtime_store.managed_backend(
                    client, name, app_service_principal_id
                )
            managed_env = {
                RUNTIME_STORE_LAKEBASE_BRANCH_ENV: runtime_backend.branch,
                RUNTIME_STORE_DATABASE_ENV: runtime_backend.database_id,
                RUNTIME_STORE_USERNAME_ENV: runtime_backend.username,
            }
            if not deployment_exists and name.startswith(_DEPLOYMENT_PREFIX):
                managed_env[RUNTIME_STORE_SCHEMA_ENV] = _AGENTKIT_RUNTIME_STORE_SCHEMA
            scaffolded = self._manifest.upsert_env(source_dir, managed_env) or scaffolded
            env_updates.update(managed_env)

        # 5. Upload the source and roll out the deployment.
        ws_path = (
            request.workspace_path
            or f"/Workspace/Users/{client.current_user}/agentbricks_deployments/{name}"
        )
        self._rollout(source_dir, name, ws_path)

        # 6. Grant the app's service principal (and the agent runtime) what they need to run.
        grants_stores, grant_error, trace_grant_error = self._grant_access(
            stores,
            name,
            memory_store,
            session_store,
            trace_experiment_id,
            otel_tables,
            trace_setup_error,
        )

        return DeployResult(
            deployment=name,
            source=request.source,
            url=self._apps_client.url(name),
            workspace_path=ws_path,
            env=env_updates,
            client_host=client.host,
            memory_store=memory_store,
            session_store=session_store,
            trace_experiment_id=trace_experiment_id,
            uc_trace_tables=[t.full_name for t in otel_tables],
            trace_setup_error=trace_setup_error,
            trace_grant_error=trace_grant_error,
            store_grant_error=grant_error,
            grants_stores=grants_stores,
            scaffolded=scaffolded,
            pip_index_url=request.pip_index_url,
            instances=instances,
            uses_runtime_api=bool(plan.project and plan.project.server == AgentServer.AGENTBRICKS),
        )

    def _authorize(self, source_dir: pathlib.Path, request: DeployRequest) -> _DeployPlan:
        # Imported here, not at module scope: the `cli` package's __init__ eagerly imports the whole
        # command tree — including the `deploy` command, which imports this module — so a top-level
        # import of anything under `cli` would make importing the service itself circular.
        from databricks_agentbricks.cli.app_auth import (  # noqa: PLC0415 - avoid import cycle
            apply_app_user_scope_update,
            plan_app_user_scope_update,
            required_user_api_scopes,
            requires_user_auth,
        )

        project = self._project.load(source_dir)
        if project is not None and project.tools:
            require_managed_tool_support(source_dir)
        user_auth = requires_user_auth(project)
        requested_name = request.name
        base_name = self._project.resolve_deployment_name(project, request.name)
        name = _prefixed_name(base_name)
        # Validate the shape before checking the Apps name length.
        _validate_deployment_name(name, check_length=False)
        deployment_exists: Optional[bool] = None
        # A project can store the unprefixed base name in agent.toml. When NAME is omitted, reuse the
        # matching Agent Bricks app if it exists.
        if (
            requested_name is None
            and project is not None
            and project.deployment_name
            and not base_name.startswith(_DEPLOYMENT_PREFIX)
        ):
            new_name_exists = len(name) <= _MAX_DEPLOYMENT_NAME_LEN and self._apps_client.exists(
                name
            )
            deployment_exists = new_name_exists
        _validate_deployment_name(name)
        if request.allow_user_scope_update and not user_auth:
            raise AgentCliError(
                "--allow-user-scope-update requires a managed tool with auth = 'user' in agent.toml."
            )
        # A request-user tool cannot use OBO until the App forwards request credentials and grants every
        # required user API scope. New Apps are configured automatically. For an existing App, adding a
        # missing scope requires --allow-user-scope-update; already-configured Apps need no flag.
        user_scope_plan = (
            plan_app_user_scope_update(
                name,
                self._profile,
                allow_existing_app_update=request.allow_user_scope_update,
                required_scopes=required_user_api_scopes(project),
            )
            if user_auth
            else None
        )
        if user_scope_plan is not None:
            self._reporter.note(
                "User auth: scope updates are not atomic; coordinate with other App owners. "
                "Users may need to sign out and re-consent after scope changes. "
                "Scopes are never removed automatically when tools change."
            )
            if deployment_exists is None:
                deployment_exists = user_scope_plan.existing_scopes is not None
            apply_app_user_scope_update(user_scope_plan, instances=request.instances)
        # Persist the base name so a later `agentbricks deploy` (no NAME) resolves to the same app.
        if project is not None and project.set_deployment_name(base_name):
            project.write()
        return _DeployPlan(project, base_name, name, user_scope_plan, deployment_exists)

    def _ensure_app(
        self,
        name: str,
        deployment_exists: bool,
        user_scope_plan,
        instances: Optional[int],
        instance_args: list[str],
    ) -> None:
        #    `apps create` itself blocks for minutes (it provisions and waits for compute) and we capture
        #    its output to relabel "App compute" → "Agent compute", so nothing streams meanwhile. Wrap it
        #    in progress (persistent line + spinner) so the CLI isn't silent for the whole provision.
        if user_scope_plan is None and not deployment_exists:
            with self._reporter.progress(
                "Creating the agent and starting its compute (this can take a few minutes)…"
            ):
                out = self._apps_client.create(name, instance_args)
            old, new = _AGENT_COMPUTE_OUTPUT
            self._reporter.echo(out.replace(old, new), newline=False)
        elif user_scope_plan is None and instances is not None:
            out = self._apps_client.create_update_instances(name, instances)
            old, new = _AGENT_COMPUTE_OUTPUT
            self._reporter.echo(out.replace(old, new), newline=False)
        # `apps deploy` requires the app's compute to be ACTIVE — a just-created app may still be
        # starting, and an existing one may be STOPPED — so wait either way. Returns immediately when
        with self._reporter.progress(
            "Waiting for agent compute to start (this can take a few minutes)…"
        ):
            self._apps_client.wait_for_running(name)

    def _rollout(self, source_dir: pathlib.Path, name: str, ws_path: str) -> None:
        self._apps_client.sync_source(name, source_dir, ws_path)
        self._apps_client.deploy(name, ws_path)

    def _grant_access(
        self,
        stores: StoreProvisioner,
        name: str,
        memory_store: Optional[str],
        session_store: Optional[str],
        trace_experiment_id: Optional[str],
        otel_tables: Sequence[TraceTable],
        trace_setup_error: Optional[str],
    ) -> tuple[bool, Optional[str], Optional[str]]:
        # Grant the app's service principal what it needs to run (best-effort):
        #    - stores: grant the SP read/write via the managed store API (the store service does the
        #      underlying Lakebase grant, so no store ownership / Lakebase MANAGE is required here);
        #    - tracing: bind the experiment as an `experiment` resource (CAN_EDIT) so it can write traces;
        #      a UC-backed experiment also needs MODIFY on its UC OTEL tables (`uc_securable` resources).
        #    The experiment resource is the platform-managed grant — no manual SQL grant needed.
        grants_stores = bool(session_store or memory_store)
        grant_error: Optional[str] = None
        if grants_stores:
            with self._reporter.status("Granting the app access to its stores…"):
                sp = self._apps_client.service_principal(name)
                if sp is None:
                    grant_error = "could not resolve the app's service principal."
                else:
                    grant_error = stores.grant_store_access(sp, session_store, memory_store)
        # Reconcile the agentbricks-owned trace resources whenever tracing resolved cleanly (`trace_setup_error
        # is None`): a resolved experiment grants that set, and a cleanly-unbound project (experiment_id
        # None) prunes stale agentbricks-trace-experiment / agentbricks-trace-table-* resources left by an earlier
        # bound deploy. If resolving the BOUND experiment errored instead (offline / permission /
        # transient), we don't know the intended state, so we skip the reconcile rather than prune - a
        # flaky deploy must not silently revoke the SP's trace access the way an unbind does. (Whether
        # removing a `uc_securable` resource also revokes the underlying UC MODIFY grant is platform
        # behavior - documented but not yet verified live.)
        trace_grant_error: Optional[str] = None
        if trace_setup_error is None:
            with self._reporter.status(
                "Granting the agent runtime access to its trace experiment…"
            ):
                trace_grant_error = self._tracing.apply_resources(
                    name, trace_experiment_id, otel_tables
                )
        return grants_stores, grant_error, trace_grant_error
