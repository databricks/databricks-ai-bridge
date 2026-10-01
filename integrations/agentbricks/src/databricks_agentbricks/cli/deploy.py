"""`agentbricks deploy` and the `agentbricks deployments` group — manage agent deployments.

`agentbricks deploy` is the integrated entry point: it provisions the memory/session stores
bound in `agent.toml`, grants the app's service principal access to them, then rolls out
the deployment. Agent Bricks Runtime deployments receive a persistent Runtime Store; a temporary rollout
switch chooses between the legacy per-app Lakebase project and the service-managed database.
`agent.toml` is the CLI's authoring source, resolved here into the `AGENT_MEMORY_STORE` /
`AGENT_SESSION_STORE` env vars written into `app.yaml` — the runtime reads those, never
`agent.toml`. `agentbricks deployments` covers the lifecycle verbs
(`list`/`get`/`logs`/`start`/`stop`/`delete`).

Every verb here is a thin adapter: it builds a `DeployService` from the CLI context, handles any
terminal confirmation, calls one method, and hands the raw result to a presenter. The work itself
lives in the service and its collaborators,
including driving the Databricks Apps runtime via the `databricks apps` CLI (an implementation detail
that is not part of the Agent Bricks CLI surface).
"""

from __future__ import annotations

import click

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.apps_user_auth_client import AppsUserAuthClient
from databricks_agentbricks.clients.conversation_store_client import (
    MemoryStoreClient,
    SessionStoreClient,
)
from databricks_agentbricks.clients.databricks_cli import _databricks
from databricks_agentbricks.clients.tracing_client import TracingClient
from databricks_agentbricks.deployment.config import (
    _DEFAULT_PIP_INDEX_URL,
    _USE_MANAGED_RUNTIME_STORE,
    TRACES_EXPERIMENT_ID_ENV,  # noqa: F401 - compatibility re-export
    TRACES_TRACKING_URI_ENV,  # noqa: F401 - compatibility re-export
    MlflowTracingConfig,  # noqa: F401 - compatibility re-export
    mlflow_tracing_config,  # noqa: F401 - compatibility re-export
)
from databricks_agentbricks.deployment.names import (
    DeploymentName,
    _prefixed_name,  # noqa: F401 - compatibility re-export
)
from databricks_agentbricks.deployment.provisioners import (
    AppProvisioner,
    MemoryStoreProvisioner,
    RuntimeStoreProvisioner,
    SessionStoreProvisioner,
    TracingProvisioner,
)
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.deploy import (
    present_deleted,
    present_deploy_result,
    present_deployment_detail,
    present_deployments_list,
    present_started,
    present_stopped,
)
from databricks_agentbricks.presentation.reporter import ClickReporter
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.services.deploy_service import DeployRequest, DeployService


def build_deploy_service(obj) -> DeployService:
    """Compose a deployment service for this CLI invocation without opening an API client."""
    api_client_provider = obj.api_client_provider
    apps_client = AppsClient(obj.profile, runner=_databricks)
    reporter = ClickReporter()
    return DeployService(
        project_resolver=ProjectResolver(),
        apps_client=apps_client,
        api_client_provider=api_client_provider,
        app_provisioner=AppProvisioner(apps_client, api_client_provider, reporter),
        apps_user_auth_client=AppsUserAuthClient(obj.profile),
        memory_store_provisioner=MemoryStoreProvisioner(
            MemoryStoreClient(api_client_provider, apps_client), reporter
        ),
        session_store_provisioner=SessionStoreProvisioner(
            SessionStoreClient(api_client_provider, apps_client), reporter
        ),
        tracing_provisioner=TracingProvisioner(
            TracingClient(api_client_provider, apps_client, obj.profile), reporter
        ),
        runtime_store_provisioner=RuntimeStoreProvisioner(
            api_client_provider, apps_client, obj.profile, _USE_MANAGED_RUNTIME_STORE, reporter
        ),
        reporter=reporter,
    )


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
    "--allow-user-scope-update",
    is_flag=True,
    help="Allow Agent Bricks to add missing user API scopes to an existing App for tools configured with "
    "auth = 'user'. Once added, later deploys do not need this flag.",
)
@click.pass_obj
def deploy(
    obj,
    name,
    source,
    pip_index_url,
    workspace_path,
    instance_count,
    allow_user_scope_update,
) -> None:
    """Deploy your agent to Databricks Apps and get back a hosted URL to try it.

    Rolls the agent out to Databricks Apps and prints the URL where you (or anyone you share it with)
    can use it. The deployed agent reaches Databricks model serving through the AI Gateway using the
    app's own identity — no model keys to configure — and `deploy` also reconciles the stores declared
    in agent.toml and wires in any tracing.

    NAME is recorded in agent.toml on the first deploy, so a later `agentbricks deploy` from the project
    directory can omit it (passing NAME again updates the recorded name). New apps are named
    `agent-bricks-<name>`. Use the full app name with the `agentbricks deployments` commands.

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
    """
    request = DeployRequest(
        name=name,
        source=source,
        pip_index_url=pip_index_url,
        workspace_path=workspace_path,
        instance_count=instance_count,
        allow_user_scope_update=allow_user_scope_update,
    )
    result = build_deploy_service(obj).deploy(request)
    present_deploy_result(result, output=obj.output)


# --- agentbricks deployments <lifecycle> ------------------------------------------


@click.group()
def deployments() -> None:
    """Inspect and manage deployed agents: list, get, stream logs, start, stop, or delete."""


@deployments.command("list")
@click.pass_obj
def deployments_list(obj) -> None:
    """List Agent Bricks deployments (apps named `agent-bricks-*`)."""
    present_deployments_list(build_deploy_service(obj).list_deployments(), output=obj.output)


@deployments.command("get")
@click.argument("name", type=DeploymentName)
@click.pass_obj
def deployments_get(obj, name) -> None:
    """Get an agent deployment's details."""
    present_deployment_detail(build_deploy_service(obj).get(name), name, output=obj.output)


@deployments.command("logs")
@click.argument("name", type=DeploymentName)
@click.pass_obj
def deployments_logs(obj, name) -> None:
    """Stream a deployment's logs."""
    build_deploy_service(obj).logs(name)


@deployments.command("start")
@click.argument("name", type=DeploymentName)
@click.pass_obj
def deployments_start(obj, name) -> None:
    """Start a deployment."""
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
    build_deploy_service(obj).stop(name)
    present_stopped(name, output=obj.output)


@deployments.command("delete")
@click.argument("name", type=DeploymentName)
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_delete(obj, name, yes) -> None:
    """Delete a deployment and, when managed provisioning is enabled, its Runtime Store."""
    service = build_deploy_service(obj)
    target = f"deployment '{name}'"
    if not yes and service.deletes_runtime_store_data():
        target += " and its Runtime Store data"
    render.confirm_destroy(target, assume_yes=yes)
    service.delete(name)
    present_deleted(name, output=obj.output)
