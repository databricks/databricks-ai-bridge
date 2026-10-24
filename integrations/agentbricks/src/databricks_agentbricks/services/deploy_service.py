"""The framework-agnostic deployment workflow behind `agentbricks deploy` and `agentbricks deployments`.

``DeployService`` owns the deployment business logic. ``deploy`` is the big one: pre-flight + auth,
reconciling the resources bound in agent.toml, patching app.yaml, ensuring the app exists, rolling
out the source, and granting access. Resource provisioners return the facts and manifest changes
they produce; this service holds those results between phases and controls their order. The
lifecycle verbs (``list_deployments``, ``get``, ``logs``, ``start``, ``stop``, ``delete``) are thin,
but they live here too so that the policy they carry - what counts as an agent deployment and that
a managed Runtime Store is torn down before its app - is stated once instead of in each command.
Which name shapes are legal is no longer among those policies: lifecycle verbs require a
:class:`~databricks_agentbricks.services.deployment.DeploymentName`, a value object that is valid by
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
import json
import pathlib
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.app_probe import AppProbe, ProbeResult
from databricks_agentbricks.clients.apps_client import AppsClient, AppState
from databricks_agentbricks.clients.apps_user_auth_client import AppAuthPlan
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.projects.app_manifest import AppManifest
from databricks_agentbricks.projects.config import require_managed_tool_support
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentbricks.reporting import Heartbeat, Reporter, format_elapsed
from databricks_agentbricks.services.deployment import deploy_state
from databricks_agentbricks.services.deployment.app_provisioner import (
    AppProvisioner,
    recorded_source_code_path,
    source_delete_command,
)
from databricks_agentbricks.services.deployment.config import _PIP_INDEX_ENVS
from databricks_agentbricks.services.deployment.deploy_state import DeployStateTracker
from databricks_agentbricks.services.deployment.model_binding_provisioner import (
    ModelBindingProvisioner,
    ModelBindingState,
)
from databricks_agentbricks.services.deployment.names import (
    _DEPLOYMENT_PREFIX,
    _MAX_DEPLOYMENT_NAME_LEN,
    DeploymentName,
    _prefixed_name,
    fit_base_name,
    name_budget_hint,
)
from databricks_agentbricks.services.deployment.provisioners import (
    MemoryStoreProvisioner,
    MemoryStoreState,
    ProjectContext,
    ResourceContext,
    RuntimeStoreProvisioner,
    SessionStoreProvisioner,
    SessionStoreState,
    TracingProvisioner,
    TracingState,
)
from databricks_agentbricks.services.deployment.tool_access import ToolAccessPlan
from databricks_agentbricks.services.deployment.tool_access_provisioner import ToolAccessProvisioner
from databricks_agentkit.runtime.cors import CORS_ALLOWED_ORIGINS_ENV
from databricks_agentkit.runtime.tool_manifest import MODEL_SERVICE_ENV_PREFIX


def _error_text(exc: BaseException) -> str:
    """How a failed deploy's error reads in its state file: the message, then any hint."""
    if isinstance(exc, AgentCliError):
        return f"{exc.message}\n{exc.hint}" if exc.hint else exc.message
    if isinstance(exc, KeyboardInterrupt):
        return "deploy interrupted"
    return str(exc) or exc.__class__.__name__


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
    allowed_origins: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ToolAccessSummary:
    """Counts of explicit tool grant targets reconciled during one deployment."""

    app_resources: int
    uc_grants: int
    workspace_grants: int


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
    tool_access: Optional[ToolAccessSummary]
    # True only when deploy created a missing app.yaml with a placeholder command.
    created_app_yaml: bool
    pip_index_url: Optional[str]
    instance_count: Optional[int]
    uses_runtime_api: bool
    model_services: dict[str, str] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)
    model_service_grant_error: Optional[str] = None
    prompt_grant_error: Optional[str] = None
    # True once the App's server answered; None when this profile couldn't check.
    ready: Optional[bool] = None


# What happened to the source folder: deleted; already gone; kept because --delete-source wasn't
# given; kept because the App wasn't confirmed gone, so it may still need its source; failed to
# delete; or none recorded because the App was never rolled out.
SourceFolderStatus = Literal[
    "deleted", "already_gone", "kept", "kept_app_not_confirmed", "failed", "not_recorded"
]


@dataclass(frozen=True)
class DeleteTarget:
    """A deployment confirmed to exist, as read once before anything is deleted."""

    deployment: DeploymentName
    # The workspace folder the App's newest deployment came from.
    source_path: Optional[str]
    # The App's service principal client id, which owns its Runtime Store.
    service_principal: Optional[str]


@dataclass(frozen=True)
class DeleteResult:
    """What a delete did. The App delete was accepted; the folder is reported, not raised."""

    deployment: str
    source_path: Optional[str]
    source_folder_status: SourceFolderStatus
    source_folder_error: Optional[str]
    # The App was still deleting when the wait ran out.
    still_deleting: bool
    # Why the wait stopped early or, after it ran out, the last failed check.
    wait_error: Optional[str]

    @property
    def confirmed(self) -> bool:
        """Whether the App was seen gone."""
        return not self.still_deleting and self.wait_error is None


@dataclass(frozen=True)
class WaitResult:
    """Where a deploy stood when `deployments wait` returned, as raw facts.

    ``status`` is ``running`` (still deploying at the timeout), ``succeeded``, or ``failed``.
    ``log_tail`` holds the end of a failed background deploy's log, or the App's own status message
    when the outcome came from the workspace rather than a local deploy record; ``log`` is that
    log's path in the first case.
    """

    deployment: str
    status: str
    phase: str
    elapsed_s: int
    url: Optional[str]
    error: Optional[str]
    log_tail: list[str]
    uses_runtime_api: bool
    log: Optional[str] = None


@dataclass(frozen=True)
class _WaitSnapshot:
    """One poll of a deploy's progress: its status, phase, and when that phase began."""

    status: str
    phase: str
    phase_started: float
    elapsed_s: float
    url: Optional[str] = None
    error: Optional[str] = None
    log_tail: tuple[str, ...] = ()
    log: Optional[str] = None


# How often `deployments wait` re-reads a local deploy record, and the App when there is none.
_LOCAL_POLL_S = 2.0
_SERVER_POLL_S = 5.0
# How often and how long deploy probes a rolled-out App until its server answers.
_READY_POLL_S = 2.0
_READY_TIMEOUT_S = 300.0


class _Unverifiable(Exception):
    """Leaves the readiness wait, without its done message, when the App can't be checked."""

    def __init__(self, detail: Optional[str]) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True)
class _ResolvedDeployment:
    """Locally resolved project and deployment identity, before workspace mutations."""

    agent_project: Optional[AgentProject]
    base_name: str
    name: DeploymentName


@dataclass(frozen=True)
class PreparedDeployment:
    """Deployment facts after tool/auth preflight has planned any request-user App."""

    agent_project: Optional[AgentProject]
    name: DeploymentName
    auth_plan: AppAuthPlan
    deployment_exists: Optional[bool]
    # Read alongside ``deployment_exists`` when preflight had to look the App up itself.
    app_state: Optional[AppState] = None


@dataclass(frozen=True)
class _RolledOut:
    """Phase results retained past a successful rollout, for tool finalization and grants."""

    ctx: ResourceContext
    memory: MemoryStoreState
    session: SessionStoreState
    bindings: ModelBindingState
    tracing: TracingState
    env: dict[str, str]
    created_app_yaml: bool
    tool_plan: Optional[ToolAccessPlan]
    workspace_path: str


class _PhaseTimer:
    """Wall-clock time per deploy phase, summed when a phase name repeats, for debug timings."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._phases: dict[str, float] = {}

    @contextmanager
    def __call__(self, phase: str) -> Iterator[None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self._phases[phase] = self._phases.get(phase, 0.0) + time.monotonic() - start

    def phases(self) -> list[tuple[str, float]]:
        return list(self._phases.items())

    def total(self) -> float:
        return time.monotonic() - self._start


def _partial_app_note(name: str) -> str:
    """How to get out of a first deploy that stopped after it created the App."""
    return (
        f"App '{name}' was created but has no deployment yet. Re-run `agentbricks deploy` to "
        f"resume (the app is reused), or remove it with `agentbricks deployments delete {name}`."
    )


def _not_answering(name: str, last: ProbeResult) -> AgentCliError:
    """The error for an App whose server never answered within the readiness timeout."""
    last_seen = f" (last: {last.detail})" if last.detail else ""
    return AgentCliError(
        f"Agent '{name}' didn't answer within {format_elapsed(_READY_TIMEOUT_S)}{last_seen}.",
        hint=f"check `agentbricks deployments logs {name}`, fix the cause, then re-run "
        "`agentbricks deploy`",
    )


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
        memory_store_provisioner: MemoryStoreProvisioner,
        session_store_provisioner: SessionStoreProvisioner,
        tracing_provisioner: TracingProvisioner,
        runtime_store_provisioner: RuntimeStoreProvisioner,
        tool_access_provisioner: ToolAccessProvisioner,
        model_binding_provisioner: ModelBindingProvisioner,
        reporter: Reporter,
        state_tracker: Optional[DeployStateTracker] = None,
        app_probe: Optional[AppProbe] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._project_resolver = project_resolver
        self._apps_client = apps_client
        self._api_client_provider = api_client_provider
        self._app_provisioner = app_provisioner
        self._memory_store_provisioner = memory_store_provisioner
        self._session_store_provisioner = session_store_provisioner
        self._tracing_provisioner = tracing_provisioner
        self._runtime_store_provisioner = runtime_store_provisioner
        self._tool_access_provisioner = tool_access_provisioner
        self._model_binding_provisioner = model_binding_provisioner
        self._reporter = reporter
        self._state_tracker = state_tracker
        self._app_probe = app_probe
        self._clock = clock
        self._sleep = sleep

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

    def delete_target(self, name: DeploymentName) -> DeleteTarget:
        """Find what deleting ``name`` would delete, before the App - which records it - goes.

        Raises if the deployment doesn't exist, before anything is deleted.
        """
        app = self._app_provisioner.find(name)
        if app is None:
            raise AgentCliError(
                f"Deployment '{name}' does not exist.",
                hint="Run `agentbricks deployments list` to see the deployments in this workspace.",
            )
        return DeleteTarget(
            deployment=name,
            source_path=recorded_source_code_path(app),
            service_principal=app.service_principal_client_id,
        )

    def delete(self, target: DeleteTarget, *, delete_source: bool) -> DeleteResult:
        """Delete a deployment: its Runtime Store, then the app, then its source if asked.

        Returns once the app is gone, or reports it unconfirmed if the wait runs out or fails;
        `apps delete` returns while it is still deleting.

        The Runtime Store goes first because dropping it needs the app's service principal, which
        stops resolving once the app is gone. If that identity can't be read we refuse outright
        rather than delete the app and orphan its data. Once the app delete is accepted nothing
        raises: the wait and the folder delete report their failures in the result. The source
        folder goes only when asked for and only once the app is confirmed gone, so an app whose
        delete stalls keeps the source it would need to recover. The caller owns any confirmation
        policy.
        """
        name = target.deployment
        if self.deletes_runtime_store_data():
            with self._reporter.status("Deleting Runtime Store…"):
                self._runtime_store_provisioner.delete_managed(name, target.service_principal)
        self._apps_client.delete(name)
        try:
            with self._reporter.status("Waiting for the deployment to be deleted…"):
                wait = self._app_provisioner.wait_until_deleted(name)
        except KeyboardInterrupt:
            if delete_source and target.source_path:
                self._reporter.note(
                    f"Kept the source folder {target.source_path}: the wait was interrupted. Once "
                    f"the deployment is gone, run `{source_delete_command(target.source_path)}`."
                )
            raise
        confirmed = not wait.timed_out and wait.error is None
        status, error = self._delete_source_folder(
            target.source_path, requested=delete_source, app_gone=confirmed
        )
        return DeleteResult(
            deployment=name,
            source_path=target.source_path,
            source_folder_status=status,
            source_folder_error=error,
            still_deleting=wait.timed_out,
            wait_error=wait.error,
        )

    def _delete_source_folder(
        self, path: Optional[str], *, requested: bool, app_gone: bool
    ) -> tuple[SourceFolderStatus, Optional[str]]:
        if path is None:
            return "not_recorded", None
        if not requested:
            return "kept", None
        if not app_gone:
            return "kept_app_not_confirmed", None
        try:
            deleted = self._app_provisioner.delete_source(path)
        except AgentCliError as exc:
            return "failed", exc.message
        return ("deleted" if deleted else "already_gone"), None

    # --- wait ---------------------------------------------------------------

    def wait(self, name: Optional[str], source_dir: pathlib.Path, timeout_s: float) -> WaitResult:
        """Wait up to ``timeout_s`` for a deploy to finish, reporting its phases as it goes.

        NAME defaults to the deployment recorded in agent.toml. A deploy started from
        ``source_dir`` (foreground or ``--no-wait``) left a local record, which is followed;
        otherwise the App's own state in the workspace decides. Phase changes and elapsed-time
        heartbeats go through the reporter's notices (stderr). Never raises for a failed or
        unfinished deploy: the result says which it is.
        """
        project = self._project_resolver.load(source_dir)
        app = DeploymentName(
            _prefixed_name(self._project_resolver.resolve_deployment_name(project, name))
        )
        uses_runtime_api = bool(project and project.server == AgentServer.AGENTBRICKS)
        path = deploy_state.state_path(source_dir, app)
        local = deploy_state.read_state(path) is not None
        started = self._clock()
        deadline = started + timeout_s
        beat: Optional[Heartbeat] = None
        while True:
            now = self._clock()
            snap = (
                self._local_snapshot(path, now)
                if local
                else self._server_snapshot(app, now, started, beat)
            )
            if snap.status == deploy_state.RUNNING:
                if beat is None or snap.phase != beat.phase:
                    self._reporter.note(f"• {snap.phase}…")
                    beat = Heartbeat(snap.phase, start=snap.phase_started, now=now)
                elif line := beat.tick(now):
                    self._reporter.note(line)
            elif local and snap.status == deploy_state.SUCCEEDED:
                # A short last phase (the agent answering at once) can start and end between polls.
                if beat is None or snap.phase != beat.phase:
                    self._reporter.note(f"• {snap.phase}…")
            if snap.status != deploy_state.RUNNING or now >= deadline:
                return WaitResult(
                    deployment=app,
                    status=snap.status,
                    phase=snap.phase,
                    elapsed_s=int(snap.elapsed_s),
                    url=snap.url,
                    error=snap.error,
                    log_tail=list(snap.log_tail),
                    uses_runtime_api=uses_runtime_api,
                    log=snap.log,
                )
            poll = _LOCAL_POLL_S if local else _SERVER_POLL_S
            self._sleep(max(0.0, min(poll, deadline - now)))

    def _local_snapshot(self, path: pathlib.Path, now: float) -> _WaitSnapshot:
        """Read the local deploy record; a vanished or crashed deploy counts as failed."""
        state = deploy_state.read_state(path) or {
            "status": deploy_state.FAILED,
            "error": deploy_state.EXITED_UNEXPECTEDLY,
        }
        state = deploy_state.resolve_state(state, now=now)
        status = str(state.get("status") or deploy_state.FAILED)
        result = state.get("result")
        log = state.get("log")
        return _WaitSnapshot(
            status=status,
            phase=str(state.get("phase") or deploy_state.STARTING_PHASE),
            phase_started=now - deploy_state.seconds_since(state.get("phase_started_at"), now),
            elapsed_s=deploy_state.seconds_since(state.get("started_at"), now),
            url=result.get("url") if isinstance(result, dict) else None,
            error=state.get("error"),
            log_tail=tuple(deploy_state.log_tail(pathlib.Path(log)))
            if status == deploy_state.FAILED and log
            else (),
            log=str(log) if log else None,
        )

    def _server_snapshot(
        self, app: DeploymentName, now: float, started: float, beat: Optional[Heartbeat]
    ) -> _WaitSnapshot:
        """Classify the App's state; its phases are timed from when this wait first saw them."""
        outcome = deploy_state.classify_app(app, self._apps_client.find(app))
        # A SUCCEEDED App is still starting until its server answers.
        if (
            outcome.status == deploy_state.SUCCEEDED
            and outcome.url
            and self._app_probe is not None
            and self._app_probe.probe(outcome.url).outcome == "booting"
        ):
            outcome = dataclasses.replace(
                outcome, status=deploy_state.RUNNING, phase=deploy_state.READY_PHASE
            )
        same_phase = beat is not None and outcome.phase == beat.phase
        return _WaitSnapshot(
            status=outcome.status,
            phase=outcome.phase,
            phase_started=beat.start if beat is not None and same_phase else now,
            elapsed_s=now - started,
            url=outcome.url,
            error=outcome.error,
            log_tail=(outcome.message,)
            if outcome.status == deploy_state.FAILED and outcome.message
            else (),
        )

    # --- deploy -------------------------------------------------------------

    def check_auth(self) -> None:
        """Fail now, rather than in a background deploy, when the profile can't authenticate."""
        try:
            _ = self._api_client_provider.get().current_user
        except AgentCliError:
            raise
        except Exception as exc:  # noqa: BLE001 - any auth/transport failure reads the same way
            raise AgentCliError(
                "Could not authenticate with the Databricks workspace.",
                hint=f"{exc}\nRun `agentbricks login --profile <profile>` (or pass --profile), "
                "then rerun the command.",
            ) from exc

    def deploy(self, request: DeployRequest) -> DeployResult:
        """Deploy an agent by coordinating its project, resources, App, rollout, and grants.

        The order matters:

        1. Resolve and validate the project/name, plan request-user scopes without mutating the
           App, persist the name, and read the resource bindings.
        2. Get the App's compute booting: create a missing App, or start a stopped one, without
           waiting. The boot runs server-side while every step up to the join below runs here.
        3. Reconcile the bound memory and session stores, tracing, and Runtime Store. Each
           provisioner returns its own facts and manifest changes; this service retains them.
        4. Write the initial resource env into app.yaml. A managed Runtime Store needs the App's
           service principal (not its compute), so finish that resource and write its late env
           next, then upload the now-final source.
        5. Join: wait for ACTIVE compute, which every App update and `apps deploy` require. Attach
           a legacy Runtime Store and reconcile explicit tool access without pruning grants used by
           the running version. Roll out the source, then remove stale App tool resources.
        6. Attempt store and tracing grants after rollout. Their failures are reported in the
           result rather than undoing a successful deployment.
        7. Wait until the App's server answers requests, failing if it never does.

        A request-user App keeps the original order instead (see ``early_compute`` below). If a step
        fails after this run created the App, the error says the App was kept and how to resume or
        remove it. The service owns the intermediate state, manifest writes, and phase order;
        provisioners own individual resource operations and return typed results for later phases.
        """
        timer = _PhaseTimer()
        source_dir = pathlib.Path(request.source)
        with timer("preflight"):
            prepared = self.prepare(request)
        tracker = self._state_tracker
        if tracker is None:
            return self._roll_out(source_dir, request, prepared, timer)
        tracker.start(source_dir, prepared.name)
        try:
            result = self._roll_out(source_dir, request, prepared, timer)
        except BaseException as exc:
            tracker.failed(_error_text(exc))
            raise
        tracker.succeeded(result)
        return result

    def _roll_out(
        self,
        source_dir: pathlib.Path,
        request: DeployRequest,
        prepared: PreparedDeployment,
        timer: _PhaseTimer,
    ) -> DeployResult:
        """Everything after :meth:`prepare`: resources, the App, the rollout, and the grants."""
        with timer("preflight"):
            memory_store_name, session_store_name, experiment_name = (
                self._project_resolver.resource_bindings(source_dir)
            )
        name = prepared.name
        instance_count = request.instance_count
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
        # Out of scope for the early compute start: a request-user App is created or updated
        # through the SDK scope path (`_reconcile_scoped_app`) at its original position, after the
        # reconciles, and the wait follows immediately. A follow-up can move it once this lands.
        early_compute = prepared.auth_plan.scope_update is None
        created_app = False
        if early_compute:
            with timer("app_start"):
                # Reuse preflight's read only when it settled the question; an App known to exist
                # but without a state read is read here, so it is never mistaken for a missing one.
                app_state = (
                    prepared.app_state
                    if prepared.app_state is not None or prepared.deployment_exists is False
                    else self._apps_client.app_state(name)
                )
                # An App that exists but never rolled out (a first deploy that stopped partway) is
                # still a first deployment, so a resume writes the same env a clean first run would.
                ctx = dataclasses.replace(
                    ctx,
                    deployment_exists=app_state is not None and app_state.has_active_deployment,
                )
                created_app = self._app_provisioner.start_compute(ctx, app_state, instance_count)
        # From the early create until the rollout succeeds, the App exists without a deployment.
        # Say so on any failure, Ctrl-C included, and leave the exit behavior to the caller.
        try:
            rollout = self._provision_and_roll_out(
                request, prepared, ctx, client, timer, early_compute=early_compute
            )
        except AgentCliError as exc:
            if created_app:
                note = _partial_app_note(name)
                exc.hint = f"{exc.hint}\n{note}" if exc.hint else note
            raise
        except BaseException:
            if created_app:
                self._reporter.note(_partial_app_note(name))
            raise
        ctx = rollout.ctx
        memory, session, bindings, tracing = (
            rollout.memory,
            rollout.session,
            rollout.bindings,
            rollout.tracing,
        )
        tool_plan, env = rollout.tool_plan, rollout.env
        if tool_plan is not None:
            self._tool_access_provisioner.finalize_after_rollout(name, tool_plan)

        # 6. Grants remain best-effort. Resolve the App identity once and pass it with each
        #    reconciled store identifier; store clients need no knowledge of Apps.
        with timer("grants"):
            store_service_principal = (
                self._apps_client.get_service_principal(name)
                if memory.store_name
                or session.store_name
                or bindings.model_services
                or bindings.prompts
                else None
            )
            memory_grant = self._memory_store_provisioner.grant(memory, store_service_principal)
            session_grant = self._session_store_provisioner.grant(session, store_service_principal)
            model_grant, prompt_grant = self._model_binding_provisioner.grant(
                bindings, store_service_principal
            )
            trace_grant = self._tracing_provisioner.grant(ctx, tracing)
            url = self._apps_client.get_app_url(name)
        # 7. Finish only once the App's server answers, so the first call after deploy works.
        with timer("ready"):
            ready = self._wait_until_answering(name, url)
        for phase, seconds in timer.phases():
            self._reporter.timing(phase, seconds)
        self._reporter.timing("total", timer.total())

        return DeployResult(
            deployment=name,
            source=request.source,
            url=url,
            workspace_path=rollout.workspace_path,
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
            model_services=dict(bindings.model_services),
            prompts=dict(bindings.prompts),
            model_service_grant_error=model_grant.error,
            prompt_grant_error=prompt_grant.error,
            memory_grant_attempted=memory_grant.attempted,
            session_grant_attempted=session_grant.attempted,
            tool_access=ToolAccessSummary(
                app_resources=len(tool_plan.app_resources),
                uc_grants=len(tool_plan.uc_grants),
                workspace_grants=len(tool_plan.workspace_grants),
            )
            if tool_plan is not None
            else None,
            created_app_yaml=rollout.created_app_yaml,
            pip_index_url=request.pip_index_url,
            instance_count=instance_count,
            uses_runtime_api=bool(
                prepared.agent_project and prepared.agent_project.server == AgentServer.AGENTBRICKS
            ),
            ready=ready,
        )

    def _wait_until_answering(self, name: DeploymentName, url: Optional[str]) -> Optional[bool]:
        """Poll the rolled-out App until its server answers; None when it can't be checked.

        Apps reports SUCCEEDED once the start command launches, and its proxy returns 502 while
        the server installs packages and starts. Raises if the server never answers in time.
        """
        if self._app_probe is None:
            return None
        try:
            if url is None:
                raise _Unverifiable("the App has no URL")
            result = self._app_probe.probe(url)
            if result.outcome == "unverifiable":
                raise _Unverifiable(result.detail)
            started = self._clock()
            with self._reporter.progress(
                f"{deploy_state.READY_PHASE}…", done_message="Agent answered"
            ):
                while result.outcome == "booting":
                    if self._clock() - started >= _READY_TIMEOUT_S:
                        raise _not_answering(name, result)
                    self._sleep(_READY_POLL_S)
                    result = self._app_probe.probe(url)
                if result.outcome == "unverifiable":
                    raise _Unverifiable(result.detail)
        except _Unverifiable as exc:
            reason = f": {exc.detail.rstrip('.')}" if exc.detail else ""
            self._reporter.note(
                f"Couldn't check the agent answers{reason}. It may return 502 for a few seconds."
            )
            return None
        return True

    def _provision_and_roll_out(
        self,
        request: DeployRequest,
        prepared: PreparedDeployment,
        ctx: ResourceContext,
        client: Any,
        timer: _PhaseTimer,
        *,
        early_compute: bool,
    ) -> _RolledOut:
        """Reconcile resources, write app.yaml, and roll out source, in either path's order.

        With ``early_compute`` the compute is already booting, so everything that leaves the App
        untouched runs before the join on ACTIVE. Otherwise (a request-user App) the App is made
        ready after the initial app.yaml write, exactly as before the early start existed.
        """
        source_dir = ctx.project.source_dir
        name = ctx.project.name
        # 1. Reconcile resources in the order shown to the developer. Each return value is owned by
        #    this deployment run and passed explicitly to later phases.
        with timer("reconcile"):
            memory = self._memory_store_provisioner.reconcile(ctx)
            session = self._session_store_provisioner.reconcile(ctx)
            bindings = self._model_binding_provisioner.reconcile(ctx)
            tracing = self._tracing_provisioner.reconcile(ctx)
        with timer("runtime_store"):
            runtime = self._runtime_store_provisioner.reconcile(ctx)

        # app.yaml has a stable, user-visible env order that differs from reconcile order.
        manifest_patches = (
            tracing.manifest,
            memory.manifest,
            session.manifest,
            bindings.manifest,
            runtime.manifest,
        )
        env: dict[str, str] = {}
        for patch in manifest_patches:
            env.update(patch.env)
        if request.pip_index_url:
            for key in _PIP_INDEX_ENVS:
                env[key] = request.pip_index_url
        env_removals = [key for patch in manifest_patches for key in patch.env_removals]
        if prepared.agent_project is not None:
            origins = prepared.agent_project.allowed_origins
            if origins:
                env[CORS_ALLOWED_ORIGINS_ENV] = json.dumps(origins)
            else:
                env_removals.append(CORS_ALLOWED_ORIGINS_ENV)

        # 2. Patch app.yaml with the initial env. The managed Runtime Store fields are added after
        #    app creation because that API requires the app's service principal.
        with timer("app_yaml"):
            created_app_yaml = (
                AppManifest.upsert_env_file(
                    source_dir,
                    env,
                    env_removals,
                    prune_prefixes=(MODEL_SERVICE_ENV_PREFIX,)
                    if prepared.agent_project is not None
                    else (),
                )
                if (env or env_removals or prepared.agent_project is not None)
                else False
            )
        if not early_compute:
            # 3. A request-user App: ensure it exists and its compute is active. Create only when
            #    new; the compute wait runs every deploy.
            with timer("app_ready"):
                deployment_exists = prepared.deployment_exists
                if deployment_exists is None:
                    deployment_exists = self._apps_client.exists(name)
                ctx = dataclasses.replace(ctx, deployment_exists=deployment_exists)
                self._app_provisioner.ensure_app_ready(
                    ctx,
                    auth_plan=prepared.auth_plan,
                    instance_count=request.instance_count,
                )

        # 4. The Runtime Store may need the App's service principal. Its late manifest contribution
        #    is written here, after App creation and after the earlier env keys.
        with timer("runtime_store"):
            runtime_late = self._runtime_store_provisioner.after_app_created(ctx, runtime)
            if not early_compute:
                self._runtime_store_provisioner.attach_to_app(ctx, runtime)
        env.update(runtime_late.env)
        if runtime_late.env or runtime_late.env_removals:
            with timer("app_yaml"):
                created_app_yaml = (
                    AppManifest.upsert_env_file(
                        source_dir, runtime_late.env, list(runtime_late.env_removals)
                    )
                    or created_app_yaml
                )

        # 5. Resolve the upload destination explicitly. With the compute booting, app.yaml is now
        #    final, so the upload runs before the join; a request-user App uploads after its tools.
        if early_compute:
            with timer("sync"):
                workspace_path = self._app_provisioner.resolve_workspace_path(
                    ctx, request.workspace_path
                )
                self._app_provisioner.sync_source(ctx, workspace_path)
            # The join: everything below updates the App or rolls out, which needs ACTIVE compute.
            with timer("compute_wait"):
                self._app_provisioner.wait_for_compute(ctx)

        with timer("app_update"):
            if early_compute:
                self._runtime_store_provisioner.attach_to_app(ctx, runtime)
            # Direct tool grants must exist before the new source can use them. Delay removals and
            # permission downgrades until after a successful rollout so the old version keeps access.
            tool_plan = None
            if prepared.agent_project is not None:
                tool_plan = self._tool_access_provisioner.plan(prepared.agent_project.tools)
                self._tool_access_provisioner.reconcile_before_rollout(
                    name, tool_plan, client.workspace_client
                )

        if not early_compute:
            with timer("sync"):
                workspace_path = self._app_provisioner.resolve_workspace_path(
                    ctx, request.workspace_path
                )
                self._app_provisioner.sync_source(ctx, workspace_path)
        with timer("rollout"):
            self._app_provisioner.roll_out(ctx, workspace_path)
        return _RolledOut(
            ctx=ctx,
            memory=memory,
            session=session,
            bindings=bindings,
            tracing=tracing,
            env=env,
            created_app_yaml=created_app_yaml,
            tool_plan=tool_plan,
            workspace_path=workspace_path,
        )

    def _resolve_deployment(
        self, source_dir: pathlib.Path, requested_name: Optional[str]
    ) -> _ResolvedDeployment:
        """Resolve and validate the local project/name identity without mutating workspace state."""
        agent_project = self._project_resolver.load(source_dir)
        base_name = self._project_resolver.resolve_deployment_name(agent_project, requested_name)
        name = _prefixed_name(base_name)
        if len(name) > _MAX_DEPLOYMENT_NAME_LEN:
            # Same check DeploymentName makes, but the hint can name the deploy command that fixes it.
            raise AgentCliError(
                f"Deployment name {name!r} is too long ({len(name)} > {_MAX_DEPLOYMENT_NAME_LEN}).",
                hint=name_budget_hint(f"agentbricks deploy {fit_base_name(base_name)}"),
            )
        return _ResolvedDeployment(
            agent_project=agent_project,
            base_name=base_name,
            name=DeploymentName(name),
        )

    def prepare(self, request: DeployRequest) -> PreparedDeployment:
        """Validate the project and plan request-user auth before resource provisioning.

        These are the quick, synchronous checks `deploy --no-wait` runs in the foreground. A deploy
        of the same App still running from this project is refused before anything changes.
        """
        source_dir = pathlib.Path(request.source)
        resolved = self._resolve_deployment(source_dir, request.name)
        running = deploy_state.running_deploy(
            deploy_state.state_path(source_dir, resolved.name), now=self._clock()
        )
        if running is not None:
            raise AgentCliError(
                f"A deploy of {resolved.name} is already running (pid {running.get('pid')}).",
                hint=f"run `agentbricks deployments wait {resolved.name}`",
            )
        agent_project = resolved.agent_project
        origins_changed = False
        if request.allowed_origins is not None:
            if agent_project is None:
                raise AgentCliError("CORS options require an agent.toml Agent Bricks project.")
            origins_changed = agent_project.set_allowed_origins(request.allowed_origins)
        if agent_project is not None and agent_project.tools:
            require_managed_tool_support(source_dir)

        auth = self._app_provisioner.prepare_app_auth(
            resolved.name,
            agent_project,
            allow_existing_app_update=request.allow_user_scope_update,
        )
        if auth.required:
            self._reporter.note(
                "User auth: scope updates are not atomic; coordinate with other App owners. "
                "Users may need to sign out and re-consent after scope changes. "
                "Scopes are never removed automatically when tools change."
            )

        deployment_exists = auth.app_existed
        app_state = None
        # A project can store the unprefixed base name in agent.toml. When NAME is omitted, reuse the
        # matching Agent Bricks app if it exists. The full state read here also tells deploy whether
        # the compute needs starting, so it isn't read twice.
        if (
            deployment_exists is None
            and request.name is None
            and agent_project is not None
            and agent_project.deployment_name
            and not resolved.base_name.startswith(_DEPLOYMENT_PREFIX)
        ):
            app_state = self._apps_client.app_state(resolved.name)
            deployment_exists = app_state is not None
        # Persist the base name so a later `agentbricks deploy` (no NAME) resolves to the same app.
        if agent_project is not None:
            name_changed = agent_project.set_deployment_name(resolved.base_name)
            if name_changed or origins_changed:
                agent_project.write()
        return PreparedDeployment(
            agent_project=agent_project,
            name=resolved.name,
            auth_plan=auth,
            deployment_exists=deployment_exists,
            app_state=app_state,
        )
