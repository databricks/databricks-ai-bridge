"""The framework-agnostic deployment workflow behind `agentbricks deploy` and `agentbricks deployments`.

``DeployService`` owns the deployment business logic. ``deploy`` is the big one: pre-flight + auth,
reconciling the resources bound in agent.toml, patching app.yaml, ensuring the app exists, rolling
out the source, and granting access. Resource provisioners return the facts and manifest changes
they produce; this service holds those results between phases and controls their order. The
lifecycle verbs (``list_deployments``, ``get``, ``logs``, ``start``, ``stop``, ``delete``) are thin,
but they live here too so that the policy they carry - what counts as an agent deployment and that
a managed Runtime Store is torn down before its app - is stated once instead of in each command.
Which name shapes are legal is no longer among those policies: lifecycle verbs require a
:class:`~databricks_agentbricks.deployment.DeploymentName`, a value object that is valid by
construction, so a name is validated once at the boundary instead of re-checked in each verb.

It reports progress through the injected :class:`Reporter` port and hands back raw facts (a
:class:`DeployResult`, or the Apps payloads as they came off the wire). The CLI owns confirmation
and presentation; this module needs no CLI framework of its own.

Collaborators - including the four resource provisioners ``deploy`` drives - are injected so the
command composes the service from the CLI context while tests construct it with fakes.
``api_client_provider.get()`` is called once, after the pre-flight/auth phase, so a pre-flight
failure never opens a workspace client.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass
from typing import Any, Optional

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.app_auth_client import AppAuthClient
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.deployment.config import _PIP_INDEX_ENVS
from databricks_agentbricks.deployment.names import (
    _DEPLOYMENT_PREFIX,
    DeploymentName,
    _prefixed_name,
    _validate_deployment_name,
)
from databricks_agentbricks.deployment.provisioners import (
    AppProvisioner,
    MemoryStoreProvisioner,
    ProjectContext,
    ResourceContext,
    RuntimeStoreProvisioner,
    SessionStoreProvisioner,
    TracingProvisioner,
)
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.projects.app_manifest import AppManifest
from databricks_agentbricks.projects.config import require_managed_tool_support
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentbricks.reporting import Reporter


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
    # None preserves an existing App's scale; an explicit count pins min and max alike.
    instance_count: Optional[int]
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
    memory_grant_attempted: bool
    session_grant_attempted: bool
    # True only when deploy created a missing app.yaml with a placeholder command.
    created_app_yaml: bool
    pip_index_url: Optional[str]
    instance_count: Optional[int]
    uses_runtime_api: bool


@dataclass(frozen=True)
class _ResolvedDeployment:
    """Locally resolved project and deployment identity, before workspace mutations."""

    agent_project: Optional[AgentProject]
    base_name: str
    name: DeploymentName


@dataclass(frozen=True)
class _PreparedDeployment:
    """Deployment facts after tool/auth preflight has prepared any request-user App."""

    agent_project: Optional[AgentProject]
    name: DeploymentName
    app_reconciled_by_auth: bool
    deployment_exists: Optional[bool]


class DeployService:
    """Owns the deployment verbs: `deploy` (pre-flight + auth, reconcile every bound resource, patch
    app.yaml, ensure the app, roll out, grant access) and the lifecycle verbs that list, read, tail,
    start, stop, and delete what it deployed, then reports the facts back.
    """

    def __init__(
        self,
        *,
        project_resolver: ProjectResolver,
        apps_client: AppsClient,
        api_client_provider: ApiClientProvider,
        app_provisioner: AppProvisioner,
        app_auth_client: AppAuthClient,
        memory_store_provisioner: MemoryStoreProvisioner,
        session_store_provisioner: SessionStoreProvisioner,
        tracing_provisioner: TracingProvisioner,
        runtime_store_provisioner: RuntimeStoreProvisioner,
        reporter: Reporter,
    ) -> None:
        self._project_resolver = project_resolver
        self._apps_client = apps_client
        self._api_client_provider = api_client_provider
        self._app_provisioner = app_provisioner
        self._app_auth_client = app_auth_client
        self._memory_store_provisioner = memory_store_provisioner
        self._session_store_provisioner = session_store_provisioner
        self._tracing_provisioner = tracing_provisioner
        self._runtime_store_provisioner = runtime_store_provisioner
        self._reporter = reporter

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
        self._apps_client.stream_logs(name)

    def start(self, name: DeploymentName) -> None:
        """Start a stopped deployment's compute. Unconfirmed: starting one destroys nothing."""
        self._apps_client.start(name)

    def stop(self, name: DeploymentName) -> None:
        """Stop a deployment's compute. The caller owns any confirmation policy."""
        self._apps_client.stop(name)

    def deletes_runtime_store_data(self) -> bool:
        """Whether deleting a deployment also removes managed Runtime Store data."""
        return self._runtime_store_provisioner.manages_persistent_data()

    def delete(self, name: DeploymentName) -> None:
        """Delete a deployment and, when managed provisioning is on, its Runtime Store first.

        The Runtime Store goes first because dropping it needs the app's service principal, which
        stops resolving once the app is gone. If that identity can't be read we refuse outright
        rather than delete the app and orphan its data. The caller owns any confirmation policy.
        """
        if self.deletes_runtime_store_data():
            with self._reporter.status("Deleting Runtime Store…"):
                self._runtime_store_provisioner.delete_managed(name)
        self._apps_client.delete(name)

    # --- deploy -------------------------------------------------------------

    def deploy(self, request: DeployRequest) -> DeployResult:
        """Deploy an agent by coordinating its project, resources, App, rollout, and grants.

        The order matters:

        1. Resolve and validate the project/name, prepare request-user auth, persist the name,
           and read the resource bindings. Auth may have already created or updated the App.
        2. Reconcile the bound memory and session stores, tracing, and Runtime Store. Each
           provisioner returns its own facts and manifest changes; this service retains them.
        3. Write the initial resource env into app.yaml, then ensure the App exists and its
           compute is active. A managed Runtime Store needs the App's service principal, so
           finish that resource and write its late env only after the App is ready.
        4. Resolve the workspace destination, upload the source, and roll out the App.
        5. Attempt resource grants after rollout. Grant failures are reported in the result
           rather than undoing a successful deployment.

        The service owns the intermediate state, manifest writes, and phase order; provisioners
        own individual resource operations and return typed results for later phases.
        """
        source_dir = pathlib.Path(request.source)
        prepared = self._prepare_deployment(source_dir, request)
        name = prepared.name
        instance_count = request.instance_count
        memory_store_name, session_store_name, experiment_name = (
            self._project_resolver.resource_bindings(source_dir)
        )
        # The shared provider stays lazy through every local project/auth validation above.
        client = self._api_client_provider.get()
        ctx = ResourceContext(
            project=ProjectContext(
                source_dir=source_dir, name=name, agent_project=prepared.agent_project
            ),
            memory_store=memory_store_name,
            session_store=session_store_name,
            experiment_name=experiment_name,
            deployment_exists=False,
        )
        # 1. Reconcile resources in the order shown to the developer. Each return value is owned by
        #    this deployment run and passed explicitly to later phases.
        memory = self._memory_store_provisioner.reconcile(ctx)
        session = self._session_store_provisioner.reconcile(ctx)
        tracing = self._tracing_provisioner.reconcile(ctx)
        runtime = self._runtime_store_provisioner.reconcile(ctx)

        # app.yaml has a stable, user-visible env order that differs from reconcile order.
        manifest_patches = (
            tracing.manifest,
            memory.manifest,
            session.manifest,
            runtime.manifest,
        )
        env: dict[str, str] = {}
        for patch in manifest_patches:
            env.update(patch.env)
        if request.pip_index_url:
            for key in _PIP_INDEX_ENVS:
                env[key] = request.pip_index_url
        env_removals = [key for patch in manifest_patches for key in patch.env_removals]

        # 2. Patch app.yaml before creating the app. The managed Runtime Store fields are added after
        #    app creation because that API requires the app's service principal.
        created_app_yaml = (
            AppManifest.upsert_env_file(source_dir, env, env_removals)
            if (env or env_removals)
            else False
        )
        # 3. Ensure the app exists and its compute is active. Create only when new; the compute wait
        #    runs every deploy.
        deployment_exists = prepared.deployment_exists
        if deployment_exists is None:
            deployment_exists = self._apps_client.exists(name)
        ctx = dataclasses.replace(ctx, deployment_exists=deployment_exists)
        self._app_provisioner.ensure_app_ready(
            ctx,
            app_reconciled_by_auth=prepared.app_reconciled_by_auth,
            instance_count=instance_count,
        )

        # 4. The Runtime Store may need the App's service principal. Its late manifest contribution
        #    is written here, after App creation and after the earlier env keys.
        runtime_late = self._runtime_store_provisioner.after_app_ready(ctx, runtime)
        env.update(runtime_late.env)
        if runtime_late.env or runtime_late.env_removals:
            created_app_yaml = (
                AppManifest.upsert_env_file(
                    source_dir, runtime_late.env, list(runtime_late.env_removals)
                )
                or created_app_yaml
            )

        # 5. Resolve the upload destination explicitly, then roll out source to the app.
        workspace_path = self._app_provisioner.resolve_workspace_path(ctx, request.workspace_path)
        self._app_provisioner.deploy_source(ctx, workspace_path)

        # 6. Grants remain best-effort. The service retains each outcome for presentation.
        memory_grant = self._memory_store_provisioner.grant(ctx, memory)
        session_grant = self._session_store_provisioner.grant(ctx, session)
        trace_grant = self._tracing_provisioner.grant(ctx, tracing)

        return DeployResult(
            deployment=name,
            source=request.source,
            url=self._apps_client.get_app_url(name),
            workspace_path=workspace_path,
            env=env,
            client_host=client.host,
            memory_store=memory.store_name,
            session_store=session.store_name,
            trace_experiment_id=tracing.experiment_id,
            uc_trace_tables=[table.full_name for table in tracing.otel_tables],
            trace_setup_error=tracing.setup_error,
            trace_grant_error=trace_grant.error,
            memory_grant_error=memory_grant.error,
            session_grant_error=session_grant.error,
            memory_grant_attempted=memory_grant.attempted,
            session_grant_attempted=session_grant.attempted,
            created_app_yaml=created_app_yaml,
            pip_index_url=request.pip_index_url,
            instance_count=instance_count,
            uses_runtime_api=bool(
                prepared.agent_project and prepared.agent_project.server == AgentServer.AGENTBRICKS
            ),
        )

    def _resolve_deployment(
        self, source_dir: pathlib.Path, requested_name: Optional[str]
    ) -> _ResolvedDeployment:
        """Resolve and validate the local project/name identity without mutating workspace state."""
        agent_project = self._project_resolver.load(source_dir)
        base_name = self._project_resolver.resolve_deployment_name(agent_project, requested_name)
        name = _prefixed_name(base_name)
        _validate_deployment_name(name, check_length=False)
        return _ResolvedDeployment(
            agent_project=agent_project,
            base_name=base_name,
            name=DeploymentName(name),
        )

    def _prepare_deployment(
        self, source_dir: pathlib.Path, request: DeployRequest
    ) -> _PreparedDeployment:
        """Validate the project and prepare request-user auth before resource provisioning."""
        resolved = self._resolve_deployment(source_dir, request.name)
        agent_project = resolved.agent_project
        if agent_project is not None and agent_project.tools:
            require_managed_tool_support(source_dir)

        auth = self._app_auth_client.ensure_user_auth(
            resolved.name,
            agent_project,
            allow_existing_app_update=request.allow_user_scope_update,
            instance_count=request.instance_count,
        )
        if auth.required:
            self._reporter.note(
                "User auth: scope updates are not atomic; coordinate with other App owners. "
                "Users may need to sign out and re-consent after scope changes. "
                "Scopes are never removed automatically when tools change."
            )

        deployment_exists = auth.app_existed
        # A project can store the unprefixed base name in agent.toml. When NAME is omitted, reuse the
        # matching Agent Bricks app if it exists.
        if (
            deployment_exists is None
            and request.name is None
            and agent_project is not None
            and agent_project.deployment_name
            and not resolved.base_name.startswith(_DEPLOYMENT_PREFIX)
        ):
            deployment_exists = self._apps_client.exists(resolved.name)
        # Persist the base name so a later `agentbricks deploy` (no NAME) resolves to the same app.
        if agent_project is not None and agent_project.set_deployment_name(resolved.base_name):
            agent_project.write()
        return _PreparedDeployment(
            agent_project=agent_project,
            name=resolved.name,
            app_reconciled_by_auth=auth.app_reconciled,
            deployment_exists=deployment_exists,
        )
