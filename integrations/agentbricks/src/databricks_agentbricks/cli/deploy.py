"""`agentbricks deploy` and the `agentbricks deployments` group — manage agent deployments.

`agentbricks deploy` is the integrated entry point: it provisions the memory/session stores
bound in `agent.toml`, grants the app's service principal access to them, then rolls out
the deployment. Agent Bricks Runtime deployments receive a persistent Runtime Store; a temporary rollout
switch chooses between the legacy per-app Lakebase project and the service-managed database.
`agent.toml` is the CLI's authoring source, resolved here into the `AGENT_MEMORY_STORE` /
`AGENT_SESSION_STORE` env vars written into `app.yaml` — the runtime reads those, never
`agent.toml`. `agentbricks deployments` covers the lifecycle verbs
(`list`/`get`/`wait`/`logs`/`start`/`stop`/`delete`).

Every verb here is a thin adapter: it builds a `DeployService` from the CLI context, handles any
terminal confirmation, calls one method, and hands the raw result to a presenter. The work itself
lives in the service and its collaborators,
including driving the Databricks Apps runtime via the `databricks apps` CLI (an implementation detail
that is not part of the Agent Bricks CLI surface).
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Optional

import click

from databricks_agentbricks.clients.app_probe import AppProbe
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.apps_user_auth_client import AppsUserAuthClient
from databricks_agentbricks.clients.conversation_store_client import (
    MemoryStoreClient,
    SessionStoreClient,
)
from databricks_agentbricks.clients.databricks_cli import _databricks
from databricks_agentbricks.clients.tracing_client import TracingClient
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.deploy import (
    deploy_result_payload,
    present_background_started,
    present_deleted,
    present_deploy_result,
    present_deployment_detail,
    present_deployments_list,
    present_started,
    present_stopped,
    present_wait_result,
)
from databricks_agentbricks.presentation.reporter import ClickReporter
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.services.deploy_service import DeployRequest, DeployService
from databricks_agentbricks.services.deployment import deploy_state
from databricks_agentbricks.services.deployment.app_provisioner import AppProvisioner
from databricks_agentbricks.services.deployment.config import (
    _DEFAULT_PIP_INDEX_URL,
    _USE_MANAGED_RUNTIME_STORE,
    TRACES_EXPERIMENT_ID_ENV,  # noqa: F401 - compatibility re-export
    TRACES_TRACKING_URI_ENV,  # noqa: F401 - compatibility re-export
    MlflowTracingConfig,  # noqa: F401 - compatibility re-export
    mlflow_tracing_config,  # noqa: F401 - compatibility re-export
)
from databricks_agentbricks.services.deployment.deploy_state import (
    DeployStateReporter,
    DeployStateTracker,
)
from databricks_agentbricks.services.deployment.model_binding_provisioner import (
    ModelBindingProvisioner,
)
from databricks_agentbricks.services.deployment.names import (
    DeploymentName,
    _prefixed_name,  # noqa: F401 - compatibility re-export
)
from databricks_agentbricks.services.deployment.provisioners import (
    MemoryStoreProvisioner,
    RuntimeStoreProvisioner,
    SessionStoreProvisioner,
    TracingProvisioner,
)
from databricks_agentbricks.services.deployment.tool_access_provisioner import ToolAccessProvisioner
from databricks_agentbricks.services.invoke.auth import WorkspaceOAuthAuthenticator
from databricks_agentbricks.services.workspace_auth import preflight_auth

# `deployments wait` returns before a coding agent's usual 120 s command timeout, and exits with
# EX_TEMPFAIL from sysexits.h when the deploy is still running so the caller knows to wait again.
_WAIT_TIMEOUT_S = 100
_EX_TEMPFAIL = 75


def build_deploy_service(obj, *, background: bool = False) -> DeployService:
    """Compose a deployment service for this CLI invocation without opening an API client.

    A deploy records its progress in the project's state file through a reporter that wraps the
    terminal one; ``background`` marks the detached child of `deploy --no-wait`.
    """
    api_client_provider = obj.api_client_provider
    state_tracker = DeployStateTracker(result_payload=deploy_result_payload, background=background)
    reporter = DeployStateReporter(ClickReporter(), state_tracker)
    apps_client = AppsClient(obj.profile, runner=_databricks, on_busy=reporter.note)
    apps_user_auth_client = AppsUserAuthClient(obj.profile)
    return DeployService(
        project_resolver=ProjectResolver(),
        apps_client=apps_client,
        api_client_provider=api_client_provider,
        app_provisioner=AppProvisioner(
            apps_client, api_client_provider, apps_user_auth_client, reporter
        ),
        memory_store_provisioner=MemoryStoreProvisioner(
            MemoryStoreClient(api_client_provider), reporter
        ),
        session_store_provisioner=SessionStoreProvisioner(
            SessionStoreClient(api_client_provider), reporter
        ),
        tracing_provisioner=TracingProvisioner(
            TracingClient(api_client_provider, apps_client, obj.profile), reporter
        ),
        runtime_store_provisioner=RuntimeStoreProvisioner(
            api_client_provider, apps_client, obj.profile, _USE_MANAGED_RUNTIME_STORE, reporter
        ),
        tool_access_provisioner=ToolAccessProvisioner(apps_client, obj.profile, reporter),
        model_binding_provisioner=ModelBindingProvisioner(api_client_provider, reporter),
        reporter=reporter,
        state_tracker=state_tracker,
        app_probe=AppProbe(WorkspaceOAuthAuthenticator(obj.profile).authorization_header),
    )


def _background_argv(obj, request: DeployRequest) -> list[str]:
    """The command line that re-runs this deploy as the detached child of `deploy --no-wait`.

    The same arguments plus the resolved --profile; the child's output is a log, so it uses text.
    """
    argv = [sys.executable, "-m", "databricks_agentbricks.cli"]
    if obj.profile:
        argv += ["--profile", obj.profile]
    argv.append("deploy")
    if request.name is not None:
        argv.append(request.name)
    argv += ["--source", str(pathlib.Path(request.source).resolve())]
    if request.pip_index_url != _DEFAULT_PIP_INDEX_URL:
        argv.append(f"--pip-index-url={request.pip_index_url or ''}")
    if request.workspace_path:
        argv += ["--workspace-path", request.workspace_path]
    if request.instance_count is not None:
        argv += ["--instances", str(request.instance_count)]
    if request.allow_user_scope_update:
        argv.append("--allow-user-scope-update")
    if request.allowed_origins is not None:
        if request.allowed_origins:
            for origin in request.allowed_origins:
                argv += ["--allowed-origin", origin]
        else:
            argv.append("--clear-allowed-origins")
    argv.append("--background-child")
    return argv


def spawn_background_deploy(
    argv: list[str], *, log: pathlib.Path, popen: Optional[Callable[..., Any]] = None
) -> int:
    """Start ``argv`` detached from this terminal, writing its output to ``log``; return its pid.

    The child gets its own session (so closing the shell or the coding agent's command sandbox
    doesn't hang it up), reads from /dev/null, and writes stdout and stderr to the log unbuffered.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kwargs["start_new_session"] = True
    with log.open("wb") as out:
        process = (popen or subprocess.Popen)(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "AGENTBRICKS_DISABLE_AGENT_HINT": "1"},
            close_fds=True,
            **kwargs,
        )
    return process.pid


def _ignore_hangup() -> None:
    """Keep a background deploy running if its original terminal goes away."""
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)


# --- agentbricks deploy -----------------------------------------------------------


@click.command()
@click.argument("name", required=False)
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Local source directory for the deployment (containing app.yaml). Defaults to the "
    "current directory.",
)
@click.option(
    "--pip-index-url",
    default=_DEFAULT_PIP_INDEX_URL,
    show_default=True,
    help="Base URL of the Python Package Index. Defaults to public PyPI.",
)
@click.option(
    "--workspace-path",
    default=None,
    help="Workspace destination for the synced source (defaults to a per-user path).",
)
@click.option(
    "--instances",
    "instance_count",
    type=click.IntRange(min=1, max=5),
    default=None,
    help="Number of deployment instances.",
)
@click.option(
    "--allowed-origin",
    multiple=True,
    metavar="ORIGIN",
    help="Allow this browser origin to call the Agent Bricks server. Repeat for multiple origins; "
    "replaces and records the allowlist in agent.toml. Omit to reuse the recorded origins.",
)
@click.option(
    "--clear-allowed-origins",
    is_flag=True,
    help="Clear the recorded browser origin allowlist and disable runtime CORS on this deploy.",
)
@click.option(
    "--allow-user-scope-update",
    is_flag=True,
    help="Allow Agent Bricks to add missing user API scopes to an existing App from managed-tool "
    "inference and [auth.user] additions. Once added, later deploys do not need this flag.",
)
@click.option(
    "--no-wait",
    is_flag=True,
    help="Run the quick checks, then finish the deploy in a background process and exit. Follow "
    "it with `agentbricks deployments wait`.",
)
# Internal: marks the detached process `--no-wait` starts. Not part of the CLI surface.
@click.option("--background-child", is_flag=True, hidden=True)
@click.pass_obj
def deploy(
    obj,
    name,
    source,
    pip_index_url,
    workspace_path,
    instance_count,
    allow_user_scope_update,
    allowed_origin,
    clear_allowed_origins,
    no_wait,
    background_child,
) -> None:
    """Deploy your agent to Databricks Apps and get back a hosted URL to try it.

    Rolls the agent out to Databricks Apps and prints the URL where you (or anyone you share it with)
    can use it. The deployed agent reaches Databricks model serving through the AI Gateway using the
    app's own identity — no model keys to configure — and `deploy` also reconciles the stores declared
    in agent.toml and wires in any tracing.

    NAME is recorded in agent.toml on the first deploy, so a later `agentbricks deploy` from the project
    directory can omit it (passing NAME again updates the recorded name). New apps are named
    `agent-bricks-<name>`. App names cap at 30 characters, so NAME can be up to 17
    (`agentbricks init` records one that fits). Use the full app name with the
    `agentbricks deployments` commands.

    If a first deploy stops partway, the app it created is kept: re-run `agentbricks deploy` to
    resume, or remove it with `agentbricks deployments delete <app-name>`.

    Any memory/session store declared in agent.toml (for example, by `agentbricks memory/sessions bind`)
    is created if it doesn't exist yet; agent.toml itself is never modified for stores.

    Scaling to multiple instances (--instances) uses best-effort sticky routing, so a browser
    session automatically stays on one instance.

    \b
    To keep a session on one app replica (sticky routing), API clients must resend a stable
    UUID (for example, their session id) in the X-Routing-Key request header on every request.
    The header is a routing hint only - non-blank, no more than 128 UTF-8 bytes, used verbatim -
    not authentication and not the session id itself (send the session id in the request body):
      X-Routing-Key: <uuid>

    A first deploy takes a few minutes (mostly starting app compute). Deploy finishes once the
    agent answers requests. From a coding agent or script with a short command timeout, use
    --no-wait and then `agentbricks deployments wait`.
    """
    if allowed_origin and clear_allowed_origins:
        raise click.UsageError("Use --allowed-origin or --clear-allowed-origins, not both.")
    # Validate auth before any deployment work: an unauthenticated profile would otherwise fail
    # deep into the deploy (or hang on a browser flow), far from anything actionable.
    try:
        validated = preflight_auth(obj.profile, obj.profile_info.source)
    except Exception:
        obj.announce_workspace()
        raise
    obj.api_client_provider.use(validated)
    request = DeployRequest(
        name=name,
        source=source,
        pip_index_url=pip_index_url,
        workspace_path=workspace_path,
        instance_count=instance_count,
        allow_user_scope_update=allow_user_scope_update,
        allowed_origins=() if clear_allowed_origins else (allowed_origin or None),
    )
    if background_child:
        _ignore_hangup()
    service = build_deploy_service(obj, background=background_child)
    if no_wait and not background_child:
        # Fail fast in the foreground on anything quick to check; the child repeats these checks.
        prepared = service.prepare(request)
        service.check_auth()
        source_dir = pathlib.Path(source)
        log = deploy_state.log_path(source_dir, prepared.name)
        pid = spawn_background_deploy(_background_argv(obj, request), log=log)
        state_file = deploy_state.record_background_start(
            source_dir, prepared.name, pid=pid, now=time.time()
        )
        present_background_started(prepared.name, log=log, state_file=state_file, output=obj.output)
        return
    started = time.monotonic()
    result = service.deploy(request)
    present_deploy_result(result, output=obj.output, elapsed_s=time.monotonic() - started)


# --- agentbricks deployments <lifecycle> ------------------------------------------


@click.group()
def deployments() -> None:
    """Inspect and manage deployed agents: list, get, wait, stream logs, start, stop, or delete."""


@deployments.command("list")
@click.pass_obj
def deployments_list(obj) -> None:
    """List Agent Bricks deployments (apps named `agent-bricks-*`)."""
    obj.announce_workspace()
    present_deployments_list(build_deploy_service(obj).list_deployments(), output=obj.output)


@deployments.command("get")
@click.argument("name", type=DeploymentName)
@click.pass_obj
def deployments_get(obj, name) -> None:
    """Get an agent deployment's details."""
    obj.announce_workspace()
    present_deployment_detail(build_deploy_service(obj).get(name), name, output=obj.output)


@deployments.command("wait")
@click.argument("name", required=False)
@click.option(
    "--timeout",
    "timeout_s",
    type=click.IntRange(min=0),
    default=_WAIT_TIMEOUT_S,
    show_default=True,
    help="Seconds to wait before returning exit 75.",
)
@click.option(
    "--source",
    default=".",
    show_default=True,
    type=click.Path(exists=True, file_okay=False),
    help="Project directory the deploy ran from.",
)
@click.pass_context
def deployments_wait(ctx: click.Context, name, timeout_s, source) -> None:
    """Wait for a deployment to finish, for up to --timeout seconds, then report it.

    Follows the deploy started by `agentbricks deploy` (including --no-wait) in this project, or
    otherwise the App's state in the workspace. NAME defaults to the deployment recorded in
    agent.toml.

    Exit status: 0 deployed and answering (or, with a note, deployed but not checkable); 1 failed;
    75 still deploying at the timeout (run the same command again).
    """
    obj = ctx.obj
    result = build_deploy_service(obj).wait(name, pathlib.Path(source), timeout_s)
    present_wait_result(result, output=obj.output)
    if result.status == deploy_state.FAILED:
        ctx.exit(1)
    if result.status == deploy_state.RUNNING:
        ctx.exit(_EX_TEMPFAIL)


@deployments.command("logs")
@click.argument("name", type=DeploymentName)
@click.pass_obj
def deployments_logs(obj, name) -> None:
    """Stream a deployment's logs."""
    obj.announce_workspace()
    build_deploy_service(obj).logs(name)


@deployments.command("start")
@click.argument("name", type=DeploymentName)
@click.pass_obj
def deployments_start(obj, name) -> None:
    """Start a deployment."""
    obj.announce_workspace()
    build_deploy_service(obj).start(name)
    present_started(name, output=obj.output)


@deployments.command("stop")
@click.argument("name", type=DeploymentName)
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_stop(obj, name, yes) -> None:
    """Stop a deployment."""
    if not yes:
        click.confirm(
            f"Stop deployment '{name}'? This cannot be undone.", default=False, abort=True
        )
    obj.announce_workspace()
    build_deploy_service(obj).stop(name)
    present_stopped(name, output=obj.output)


@deployments.command("delete")
@click.argument("name", type=DeploymentName)
@click.option(
    "--delete-source",
    is_flag=True,
    help="Also delete the workspace folder the deployment's source was uploaded to, once the "
    "deployment is gone.",
)
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_delete(obj, name, delete_source, yes) -> None:
    """Delete a deployment, and its Runtime Store when managed provisioning is enabled."""
    service = build_deploy_service(obj)
    # Managed cleanup reaches the shared API provider, whose notice uses the resolved host.
    # A CLI-only delete has no SDK client to consult, so use the offline lookup there.
    if not service.deletes_runtime_store_data():
        obj.announce_workspace()
    try:
        target = service.delete_target(name)
    except Exception:
        obj.announce_workspace()
        raise
    parts = [f"deployment '{name}'"]
    if not yes and service.deletes_runtime_store_data():
        parts.append("its Runtime Store data")
    if delete_source and target.source_path:
        parts.append(f"its source folder {target.source_path}")
    described = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    render.confirm_destroy(described, assume_yes=yes)
    try:
        result = service.delete(target, delete_source=delete_source)
    except Exception:
        obj.announce_workspace()
        raise
    present_deleted(result, output=obj.output)
    if not result.confirmed or result.source_folder_status == "failed":
        raise click.exceptions.Exit(1)
