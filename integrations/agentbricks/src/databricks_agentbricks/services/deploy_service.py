"""The framework-agnostic deployment workflow behind `agentbricks deploy` and `agentbricks deployments`.

``DeployService`` owns the deployment business logic. ``deploy`` is the big one: pre-flight + auth,
reconciling the resources bound in agent.toml, patching app.yaml, ensuring the app exists, rolling
out the source, and granting access. Each resource's own work lives in its
:class:`~databricks_agentbricks.services.provisioners.ResourceProvisioner`, so ``deploy`` is the
sequence - the phase order every resource is driven through - rather than the sum of them. The
lifecycle verbs (``list_deployments``, ``get``, ``logs``, ``start``, ``stop``, ``delete``) are thin,
but they live here too so that the policy they carry - what counts as an agent deployment, what a
destructive verb confirms, and that a managed Runtime Store is torn down before its app - is stated
once instead of in each command. Which name shapes are legal is no longer among those policies: the
verbs require a :class:`~databricks_agentbricks.deployment.DeploymentName`, a value object that is
valid by construction, so a name is validated once at the boundary instead of re-checked in each verb.

It talks to the terminal only through the injected :class:`Reporter` and :class:`Prompter` ports and
hands back raw facts (a :class:`DeployResult`, or the Apps payloads as they came off the wire), so
the CLI layer owns every presentation decision and this module needs no CLI framework of its own.

Collaborators - including the four resource provisioners ``deploy`` drives - are injected so the
command composes the service from the CLI context while tests construct it with fakes.
``api_client_factory`` is called once, after the pre-flight/auth phase, so a pre-flight failure
never opens a workspace client.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from databricks_agentbricks.app_auth_client import AppAuthClient, AppUserScopeUpdatePlan
from databricks_agentbricks.app_manifest import upsert_env_file
from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.deployment import (
    _DEPLOYMENT_PREFIX,
    _MAX_DEPLOYMENT_NAME_LEN,
    _PIP_INDEX_ENVS,
    DeploymentName,
    _instance_args,
    _prefixed_name,
    _validate_deployment_name,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_config import require_managed_tool_support
from databricks_agentbricks.project_resolver import ProjectResolver
from databricks_agentbricks.project_types import AgentServer
from databricks_agentbricks.services.app_provisioner import AppProvisioner
from databricks_agentbricks.services.interaction import Prompter, Reporter
from databricks_agentbricks.services.provisioners import (
    MemoryStoreProvisioner,
    ProjectContext,
    ResourceContext,
    ResourceProvisioner,
    RuntimeStoreProvisioner,
    SessionStoreProvisioner,
    TracingProvisioner,
)


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
    memory_grant_error: Optional[str]
    session_grant_error: Optional[str]
    grants_memory: bool
    grants_session: bool
    scaffolded: bool
    pip_index_url: Optional[str]
    instances: Optional[int]
    uses_runtime_api: bool


@dataclass(frozen=True)
class _DeployPlan:
    """Resolved deploy identity + auth decisions from the pre-flight phase."""

    project: Any
    base_name: str
    name: DeploymentName
    user_scope_plan: Optional[AppUserScopeUpdatePlan]
    deployment_exists: Optional[bool]  # known already from the name/scope pre-flight, else None


class DeployService:
    """Owns the deployment verbs: `deploy` (pre-flight + auth, reconcile every bound resource, patch
    app.yaml, ensure the app, roll out, grant access) and the lifecycle verbs that list, read, tail,
    start, stop, and delete what it deployed, then reports the facts back.
    """

    def __init__(
        self,
        *,
        project: ProjectResolver,
        apps_client: AppsClient,
        api_client_factory: Callable[[], Any],
        app: AppProvisioner,
        app_auth: AppAuthClient,
        memory_store: MemoryStoreProvisioner,
        session_store: SessionStoreProvisioner,
        tracing: TracingProvisioner,
        runtime_store: RuntimeStoreProvisioner,
        profile: Optional[str],
        reporter: Reporter,
        prompter: Prompter,
    ) -> None:
        self._project = project
        self._apps_client = apps_client
        self._api_client_factory = api_client_factory
        self._app = app
        self._app_auth = app_auth
        self._memory_store = memory_store
        self._session_store = session_store
        self._tracing = tracing
        self._runtime_store = runtime_store
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

    def get(self, name: DeploymentName) -> dict:
        """One deployment's raw Apps payload."""
        return self._apps_client.get(name)

    def logs(self, name: DeploymentName) -> None:
        """Stream the deployment's logs to the terminal until the user interrupts."""
        self._apps_client.logs(name)

    def start(self, name: DeploymentName) -> None:
        self._apps_client.start(name)

    def stop(self, name: DeploymentName, *, assume_yes: bool) -> None:
        """Stop a deployment, confirming first unless `assume_yes` (for scripts)."""
        self._confirm_action(f"Stop deployment '{name}'", assume_yes=assume_yes)
        self._apps_client.stop(name)

    def delete(self, name: DeploymentName, *, assume_yes: bool) -> None:
        """Delete a deployment and, when managed provisioning is on, its Runtime Store first.

        The Runtime Store goes first because dropping it needs the app's service principal, which
        stops resolving once the app is gone. If that identity can't be read we refuse outright
        rather than delete the app and orphan its data.
        """
        manages_persistent_data = self._runtime_store.manages_persistent_data()
        # Name the data loss in the prompt: with a managed store, deleting the app also drops the
        # agent's persisted memory/sessions, which the app name alone doesn't imply.
        target = (
            f"Delete deployment '{name}' and its Runtime Store data"
            if manages_persistent_data
            else f"Delete deployment '{name}'"
        )
        self._confirm_action(target, assume_yes=assume_yes)
        if manages_persistent_data:
            with self._reporter.status("Deleting Runtime Store…"):
                self._runtime_store.delete_managed(name)
        self._apps_client.delete(name)

    def _confirm_action(self, target: str, *, assume_yes: bool) -> None:
        """Confirm before a destructive op; `assume_yes` skips the prompt (for scripts)."""
        if assume_yes:
            return
        if not self._prompter.confirm(f"{target}? This cannot be undone.", default=False):
            raise OperationAborted()

    # --- deploy -------------------------------------------------------------

    def deploy(self, request: DeployRequest) -> DeployResult:
        """Pre-flight + auth, reconcile every resource, patch app.yaml, ensure the app, roll out, grant.

        The phase order is the contract: each resource's work is split across the three
        :class:`ResourceProvisioner` hooks, and this drives them - so what lives here is only the
        sequence itself (and the steps no resource owns: the package index env, the app.yaml write,
        the app, the rollout), not any resource's logic.
        """
        source_dir = pathlib.Path(request.source)
        plan = self._authorize(source_dir, request)
        name = plan.name
        instances = request.instances
        instance_args = _instance_args(instances)
        client = self._api_client_factory()
        memory_store_name, session_store_name, experiment_name = self._project.resource_bindings(
            source_dir
        )
        ctx = ResourceContext(
            project=ProjectContext(source_dir=source_dir, name=name, agent_project=plan.project),
            memory_store=memory_store_name,
            session_store=session_store_name,
            experiment_name=experiment_name,
            deployment_exists=False,
        )
        # The order resources are reconciled in - each one's progress spinner appears here, so this is
        # the order the developer watches the deploy happen in. It also drives the later phases:
        # the stores grant (memory then session) before tracing does.
        provisioners: tuple[ResourceProvisioner, ...] = (
            self._memory_store,
            self._session_store,
            self._tracing,
            self._runtime_store,
        )
        # The order their env lands in app.yaml, which is NOT the reconcile order: tracing's MLFLOW_*
        # keys come first, then the memory/session store env. app.yaml's env list is a user-visible
        # file the developer reads and edits, so its key order is part of the CLI's output and is fixed
        # here rather than left to fall out of whichever order the resources happen to be reconciled in.
        env_order: tuple[ResourceProvisioner, ...] = (
            self._tracing,
            self._memory_store,
            self._session_store,
            self._runtime_store,
        )

        # 1. Reconcile every declared resource: create what doesn't exist yet; each records on itself
        #    the env the deployed runtime reads. agent.toml is the source of truth and is never rewritten.
        for provisioner in provisioners:
            provisioner.reconcile(ctx)
        env: dict[str, str] = {}
        for provisioner in env_order:
            env.update(provisioner.env)
        if request.pip_index_url:
            for key in _PIP_INDEX_ENVS:
                env[key] = request.pip_index_url
        env_removals = [key for provisioner in env_order for key in provisioner.env_removals]

        # 2. Patch app.yaml before creating the app. The managed Runtime Store fields are added after
        #    app creation because that API requires the app's service principal.
        scaffolded = (
            upsert_env_file(source_dir, env, env_removals) if (env or env_removals) else False
        )

        # 3. Ensure the app exists and its compute is active. Create only when new; the compute wait
        #    runs every deploy.
        deployment_exists = plan.deployment_exists
        if deployment_exists is None:
            deployment_exists = self._apps_client.exists(name)
        ctx = dataclasses.replace(ctx, deployment_exists=deployment_exists)
        self._app.ensure(ctx, plan.user_scope_plan, instances, instance_args)

        # 4. Finish the resources that needed the app to exist (its service principal is resolvable
        #    only now), then fold any env they added (the managed Runtime Store) in - appended after
        #    the pip keys, since re-merging leaves the already-present keys in place.
        for provisioner in provisioners:
            provisioner.after_app_ready(ctx)
        for provisioner in env_order:
            env.update(provisioner.env)

        # 5. Upload the source and roll out the deployment.
        ws_path = self._app.rollout(ctx, request.workspace_path)

        # 6. Grant the app's service principal (and the agent runtime) access to each resource. Every
        #    grant is best-effort: each provisioner records its own failure instead of raising, because
        #    the deploy itself succeeded and the CLI reports a missing grant as a next step.
        for provisioner in provisioners:
            provisioner.grant(ctx)

        return DeployResult(
            deployment=name,
            source=request.source,
            url=self._apps_client.url(name),
            workspace_path=ws_path,
            env=env,
            client_host=client.host,
            memory_store=self._memory_store.store_name,
            session_store=self._session_store.store_name,
            trace_experiment_id=self._tracing.experiment_id,
            uc_trace_tables=[t.full_name for t in self._tracing.otel_tables],
            trace_setup_error=self._tracing.setup_error,
            trace_grant_error=self._tracing.grant_error,
            memory_grant_error=self._memory_store.grant_error,
            session_grant_error=self._session_store.grant_error,
            grants_memory=self._memory_store.grants,
            grants_session=self._session_store.grants,
            scaffolded=scaffolded or any(provisioner.scaffolded for provisioner in provisioners),
            pip_index_url=request.pip_index_url,
            instances=instances,
            uses_runtime_api=bool(plan.project and plan.project.server == AgentServer.AGENTBRICKS),
        )

    def _authorize(self, source_dir: pathlib.Path, request: DeployRequest) -> _DeployPlan:
        project = self._project.load(source_dir)
        if project is not None and project.tools:
            require_managed_tool_support(source_dir)
        user_auth = self._app_auth.requires_user_auth(project)
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
        # Apply the length cap now (after the exists probe, which needs the raw string) and, in the
        # same step, promote the validated name to the DeploymentName the rest of deploy carries.
        name = DeploymentName(name)
        if request.allow_user_scope_update and not user_auth:
            raise AgentCliError(
                "--allow-user-scope-update requires a managed tool with auth = 'user' in agent.toml."
            )
        # A request-user tool cannot use OBO until the App forwards request credentials and grants every
        # required user API scope. New Apps are configured automatically. For an existing App, adding a
        # missing scope requires --allow-user-scope-update; already-configured Apps need no flag.
        user_scope_plan = (
            self._app_auth.plan_user_scope_update(
                name,
                allow_existing_app_update=request.allow_user_scope_update,
                required_scopes=self._app_auth.required_user_api_scopes(project),
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
            self._app_auth.apply_user_scope_update(user_scope_plan, instances=request.instances)
        # Persist the base name so a later `agentbricks deploy` (no NAME) resolves to the same app.
        if project is not None and project.set_deployment_name(base_name):
            project.write()
        return _DeployPlan(project, base_name, name, user_scope_plan, deployment_exists)
