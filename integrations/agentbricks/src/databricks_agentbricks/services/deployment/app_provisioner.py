"""The deployed Databricks App: create or scale it, roll out its source, and delete it.

Deliberately separate from the typed resource provisioners in ``provisioners``: the App is the
deployment itself, and deleting it can also cover the source folder deploy uploaded to.
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import requests
from databricks.sdk.errors import (
    DatabricksError,
    DeadlineExceeded,
    InternalError,
    NotFound,
    TemporarilyUnavailable,
    TooManyRequests,
)
from databricks.sdk.service.apps import App

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.apps_user_auth_client import (
    AppAuthPlan,
    AppsUserAuthClient,
    AppUserScopeUpdatePlan,
    validate_app_user_scope_drift,
    wait_for_app_user_scopes,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deployment.config import _AGENT_COMPUTE_OUTPUT
from databricks_agentbricks.services.deployment.names import DeploymentName
from databricks_agentbricks.services.deployment.provisioners import ResourceContext

_DEPLOYMENTS_DIR = "agentbricks_deployments"
APP_DELETE_TIMEOUT_S = 20 * 60


@dataclass(frozen=True)
class AppDeleteWaitResult:
    """How waiting for an App delete ended; neither field set means the App is gone."""

    # The App still existed at the deadline.
    timed_out: bool = False
    # Why the wait stopped early or, after a timeout, the last failed check.
    error: Optional[str] = None


def recorded_source_code_path(app: App) -> Optional[str]:
    """The folder the App's newest deployment came from; a pending rollout is newer than the active."""
    for deployment in (app.pending_deployment, app.active_deployment):
        if deployment is not None and deployment.source_code_path:
            return deployment.source_code_path
    return None


def source_delete_command(path: str) -> str:
    """The shell command that deletes a source folder, quoted for copy-paste."""
    return shlex.join(["databricks", "workspace", "delete", path, "--recursive"])


# What a workspace call can fail with besides a typed API error. Connection failures surface as
# `requests` errors, or TimeoutError once the SDK's retries run out. A failed login or token
# refresh surfaces as whatever the auth type's credential provider raises: OSError for
# `databricks auth login` profiles ("cannot get access token"), ValueError, RuntimeError, or a bare
# Exception. `requests` errors and DatabricksError are OSErrors too, so order the handlers.
_NETWORK_ERRORS = (requests.RequestException, TimeoutError)
# API errors that can clear on their own, so a wait retries them.
_TRANSIENT_API_ERRORS = (InternalError, TemporarilyUnavailable, DeadlineExceeded, TooManyRequests)


def _app_lookup_error(name: DeploymentName, exc: Exception) -> AgentCliError:
    return AgentCliError(
        f"Could not check whether deployment '{name}' exists.",
        error_code=getattr(exc, "error_code", None),
        hint=f"{exc}\nIf the profile isn't signed in, run `agentbricks login`; otherwise check the "
        "network and the profile's host, then rerun the command.",
    )


class AppProvisioner:
    """Create-or-scale the deployed App, wait for its compute, then sync and roll out the source.

    Owns the Databricks App that a deploy creates and rolls source out to. Its operations cover App
    creation, compute readiness, and source rollout; bound stores and tracing have their own
    provisioners.

    Like the rest of the services layer it talks to the terminal only through the injected
    :class:`Reporter`. Non-auth Apps use the injected CLI-backed ``AppsClient``; request-user Apps
    use the SDK-backed scope plan so creation and masked updates retain the required scopes.
    """

    def __init__(
        self,
        apps_client: AppsClient,
        api_client_provider: ApiClientProvider,
        apps_user_auth_client: AppsUserAuthClient,
        reporter: Reporter,
    ) -> None:
        self._apps_client = apps_client
        self._api_client_provider = api_client_provider
        self._apps_user_auth_client = apps_user_auth_client
        self._reporter = reporter

    def prepare_app_auth(
        self,
        name: DeploymentName,
        project: Optional[AgentProject],
        *,
        allow_existing_app_update: bool,
    ) -> AppAuthPlan:
        """Validate request-user requirements and read App scopes before any deployment mutation."""
        return self._apps_user_auth_client.plan_user_auth(
            name, project, allow_existing_app_update=allow_existing_app_update
        )

    @staticmethod
    def _reconcile_scoped_app(plan: AppUserScopeUpdatePlan, instance_count: Optional[int]) -> None:
        """Create or update a request-user App with only the required scopes and explicit scale.

        The SDK path is necessary for scoped creation. Existing Apps receive a masked update that
        preserves unrelated settings; known scope drift fails before that update. Apps has no
        compare-and-swap contract, so owners must still coordinate concurrent changes.
        """
        if not set(plan.existing_scopes or ()).issubset(plan.scopes):
            raise AgentCliError(
                "Automatic removal of Apps user scopes is unsupported.",
                hint="Remove scopes explicitly in Databricks Apps and verify the effective scopes. "
                "The SDK omits empty lists when serializing App; no removal was sent.",
            )
        desired = App(
            name=plan.name,
            user_api_scopes=list(plan.scopes),
            forward_user_access_token=True if plan.existing_scopes is None else None,
        )
        if instance_count is not None:
            desired.compute_min_instances = instance_count
            desired.compute_max_instances = instance_count
        try:
            if plan.existing_scopes is None:
                plan.apps.create(desired)
            else:
                validate_app_user_scope_drift(plan)
                mask = []
                if plan.scopes != plan.existing_scopes:
                    mask.append("user_api_scopes")
                if instance_count is not None:
                    mask.extend(("compute_min_instances", "compute_max_instances"))
                if mask:
                    plan.apps.create_update(
                        plan.name, update_mask=",".join(mask), app=desired
                    ).result(timeout=timedelta(minutes=5))
        except (DatabricksError, TimeoutError) as exc:
            raise AgentCliError(f"Could not reconcile Apps user scopes for '{plan.name}'.") from exc
        wait_for_app_user_scopes(plan)

    def ensure_app_ready(
        self,
        ctx: ResourceContext,
        *,
        auth_plan: AppAuthPlan,
        instance_count: Optional[int],
    ) -> None:
        """Own App creation, optional scale and scope updates, and ACTIVE-compute readiness.

        A request-user App takes the SDK path for scoped creation or masked update; other Apps use
        the CLI. An omitted instance count preserves an existing App's scale on either path.
        """
        name = ctx.project.name
        deployment_exists = ctx.deployment_exists
        # `apps create` blocks for minutes, so show progress while captured output is unavailable.
        if auth_plan.scope_update is not None:
            if auth_plan.scope_update.name != name:
                raise AgentCliError("App auth plan does not match the deployment being prepared.")
            with self._reporter.progress(
                "Configuring the agent's request-user access (this can take a few minutes)…"
            ):
                self._reconcile_scoped_app(auth_plan.scope_update, instance_count)
        elif not deployment_exists:
            with self._reporter.progress(
                "Creating the agent and starting its compute (this can take a few minutes)…"
            ):
                out = self._apps_client.create(name, instance_count)
            self._note_cli_output(out)
        elif instance_count is not None:
            self._note_cli_output(self._apps_client.create_update_instances(name, instance_count))
        # `apps deploy` requires ACTIVE compute, whether the App is new or was previously stopped.
        with self._reporter.progress(
            "Waiting for agent compute to start (this can take a few minutes)…"
        ):
            self._apps_client.wait_for_active(name)

    def _note_cli_output(self, out: str) -> None:
        # stderr, so `--output json` stdout stays parseable.
        old, new = _AGENT_COMPUTE_OUTPUT
        self._reporter.note(out.replace(old, new).rstrip("\n"))

    def resolve_workspace_path(self, ctx: ResourceContext, requested_path: Optional[str]) -> str:
        """Return the explicit destination for source upload, deriving the default lazily."""
        if requested_path:
            return requested_path
        return self.default_workspace_path(ctx.project.name)

    def default_workspace_path(self, name: DeploymentName) -> str:
        """Where deploy uploads the source when no ``--workspace-path`` is given."""
        current_user = self._api_client_provider.get().current_user
        return f"/Workspace/Users/{current_user}/{_DEPLOYMENTS_DIR}/{name}"

    def find(self, name: DeploymentName) -> Optional[App]:
        """The App, or None only when it definitively doesn't exist; any other failure raises."""
        try:
            return self._api_client_provider.get().workspace_client.apps.get(name)
        except NotFound:
            return None
        except Exception as exc:  # noqa: BLE001 - a credential provider may raise anything
            raise _app_lookup_error(name, exc) from exc

    def wait_until_deleted(
        self, name: DeploymentName, timeout_s: int = APP_DELETE_TIMEOUT_S
    ) -> AppDeleteWaitResult:
        """Wait for the App to stop existing, reporting a timeout or the error that stopped it.

        Never raises: the delete is already accepted, so a failure here is reported with the rest of
        the result. Failures that can clear (5xx, throttling, network) are retried until the
        deadline; any other, including a failed login, stops the wait at once.
        """
        deadline = time.monotonic() + timeout_s
        last_error: Optional[Exception] = None
        while True:
            try:
                self._api_client_provider.get().workspace_client.apps.get(name)
                last_error = None
            except NotFound:
                return AppDeleteWaitResult()
            except (*_TRANSIENT_API_ERRORS, *_NETWORK_ERRORS) as exc:
                last_error = exc
            except Exception as exc:  # noqa: BLE001 - the delete is accepted; report, don't raise
                return AppDeleteWaitResult(error=str(exc))
            if time.monotonic() >= deadline:
                return AppDeleteWaitResult(
                    timed_out=True, error=str(last_error) if last_error else None
                )
            time.sleep(10)

    def delete_source(self, path: str) -> bool:
        """Delete an uploaded source folder; False when it was already gone."""
        try:
            self._api_client_provider.get().workspace_client.workspace.delete(path, recursive=True)
        except NotFound:
            return False
        except Exception as exc:  # noqa: BLE001 - the App delete is accepted; report as a warning
            raise AgentCliError(
                f"Could not delete the uploaded source folder '{path}': {exc}",
                error_code=getattr(exc, "error_code", None),
            ) from exc
        return True

    def deploy_source(self, ctx: ResourceContext, workspace_path: str) -> None:
        """Sync source to an already-resolved workspace destination and activate it.

        Distinct from :meth:`AppsClient.deploy`, which is the single ``databricks apps deploy``
        call this drives once the source is synced.
        """
        self._apps_client.sync_source(ctx.project.name, ctx.project.source_dir, workspace_path)
        self._apps_client.deploy(ctx.project.name, workspace_path)
