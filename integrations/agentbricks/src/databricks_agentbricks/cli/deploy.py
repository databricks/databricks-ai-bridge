"""`agentbricks deploy` and the `agentbricks deployments` group — manage agent deployments.

`agentbricks deploy` is the integrated entry point: it provisions the memory/session stores
bound in `agent.toml`, grants the app's service principal access to them, then rolls out
the deployment. Agent Bricks Runtime deployments receive a persistent Runtime Store; a temporary rollout
switch chooses between the legacy per-app Lakebase project and the service-managed database.
`agent.toml` is the CLI's authoring source, resolved here into the `AGENT_MEMORY_STORE` /
`AGENT_SESSION_STORE` env vars written into `app.yaml` — the runtime reads those, never
`agent.toml`. `agentbricks deployments` covers the lifecycle verbs
(`list`/`get`/`logs`/`start`/`stop`/`delete`).

Deployments run on the Databricks Apps runtime, which this module drives via the
`databricks apps` CLI — an implementation detail that is not part of the Agent Bricks CLI surface.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import click

import databricks_agentbricks.lakebase_runtime_store as managed_runtime_store
from databricks_agentbricks import render
from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.cli.composition import build_deploy_service
from databricks_agentbricks.cli.presenter import present_deploy_result
from databricks_agentbricks.databricks_cli import _databricks
from databricks_agentbricks.deployment import (
    _DEFAULT_PIP_INDEX_URL,
    _DEPLOYMENT_PREFIX,
    _USE_MANAGED_RUNTIME_STORE,
    TRACES_EXPERIMENT_ID_ENV,  # noqa: F401 - re-exported for tests referencing deploy_mod.TRACES_EXPERIMENT_ID_ENV
    TRACES_TRACKING_URI_ENV,  # noqa: F401 - re-exported for tests referencing deploy_mod.TRACES_TRACKING_URI_ENV
    MlflowTracingConfig,  # noqa: F401 - re-exported for callers/tests referencing deploy_mod.MlflowTracingConfig
    _prefixed_name,  # noqa: F401 - re-exported for tests referencing deploy_mod._prefixed_name
    _validate_deployment_name,
    mlflow_tracing_config,  # noqa: F401 - re-exported for tests referencing deploy_mod.mlflow_tracing_config
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.render import field
from databricks_agentbricks.services.deploy_service import DeployRequest
from databricks_agentkit import timefmt

# --- databricks CLI plumbing (the deployment runtime) -----------------------


def _confirm_destroy(target: str, *, assume_yes: bool) -> None:
    """Prompt before a destructive deployment op; --yes/-y skips it (for scripts)."""
    if assume_yes:
        return
    if not click.confirm(f"{target}? This cannot be undone.", default=False):
        raise click.Abort()


# --- agent.toml bindings (`agentbricks dev`'s reader) -----------------------
#
# `agentbricks deploy` resolves its bindings through `ProjectResolver` (the service's render-free
# collaborator, which carries the same logic); these two remain for `cli.dev`, whose migration to the
# resolver is a follow-up.


def _load_project(source: pathlib.Path):
    """The AgentProject at `source`, or None when agent.toml is absent."""
    from databricks_agentbricks.agent_project import AgentProject

    if not (source / "agent.toml").is_file():
        return None
    return AgentProject.load(source)


def resource_bindings(
    source: pathlib.Path,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """The (memory store, session store, tracing experiment) bound in agent.toml.

    agent.toml is the single source of truth for an agent's resources, so `agentbricks dev`'s resource
    env/notices and `agentbricks deploy`'s provisioning honor the same bindings. A missing agent.toml
    means nothing is bound; an invalid manifest fails with a clear error.
    """
    project = _load_project(source)
    if project is None:
        return None, None, None
    # str(): agent.toml bindings come back as tomlkit strings, which don't serialize to app.yaml.
    memory = str(project.memory_store) if project.memory_store else None
    session = str(project.session_store) if project.session_store else None
    experiment = str(project.trace_experiment_name) if project.trace_experiment_name else None
    return memory, session, experiment


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
    instances,
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
    API clients that need it must resend a stable UUID in this cookie every request:
      __Host-databricks-app-router=<uuid>
    """
    request = DeployRequest(
        name=name,
        source=source,
        pip_index_url=pip_index_url,
        workspace_path=workspace_path,
        instances=instances,
        allow_user_scope_update=allow_user_scope_update,
    )
    result = build_deploy_service(obj).run(request)
    present_deploy_result(result, output=obj.output)


# --- agentbricks deployments <lifecycle> ------------------------------------------


@click.group()
def deployments() -> None:
    """Inspect and manage deployed agents: list, get, stream logs, start, stop, or delete."""


def _deployment_status(a: dict) -> Optional[str]:
    for key in ("app_status", "compute_status"):
        section = a.get(key)
        if isinstance(section, dict) and field(section, "state"):
            return field(section, "state")
    return field(a, "state")


@deployments.command("list")
@click.pass_obj
def deployments_list(obj) -> None:
    """List Agent Bricks deployments (apps named `agent-bricks-*`)."""
    result = _databricks(
        ["apps", "list", "-o", "json"],
        obj.profile,
        capture=True,
        action="Could not list agent deployments.",
    )
    data = json.loads(result.stdout or "[]")
    items = data.get("apps", data) if isinstance(data, dict) else data
    items = [a for a in items if str(field(a, "name") or "").startswith(_DEPLOYMENT_PREFIX)]
    if obj.output == "json":
        render.emit_json(items)
        return
    rows = [
        [
            render.hyperlink(field(a, "name"), field(a, "url")),
            render.status_pill(_deployment_status(a)),
            timefmt.relative(field(a, "update_time")),
        ]
        for a in items
    ]
    render.resource_table(
        "Agent Deployments",
        [("Name", "left"), ("Status", "left"), ("Updated", "left")],
        rows,
    )


@deployments.command("get")
@click.argument("name")
@click.pass_obj
def deployments_get(obj, name) -> None:
    """Get an agent deployment's details."""
    _validate_deployment_name(name)
    result = _databricks(
        ["apps", "get", name, "-o", "json"],
        obj.profile,
        capture=True,
        action=f"Could not read deployment '{name}'.",
    )
    data = json.loads(result.stdout or "{}")
    if obj.output == "json":
        render.emit_json(data)
        return
    url = field(data, "url")
    render.detail(
        "Agent Deployment",
        field(data, "name") or name,
        {
            "URL": render.hyperlink(url, url) if url else None,
            "Description": field(data, "description"),
            "Created": timefmt.absolute(field(data, "create_time")),
            "Updated": timefmt.absolute(field(data, "update_time")),
        },
        status=_deployment_status(data),
        snippets=[("open", "bash", f"open {url}")] if url else None,
    )


@deployments.command("logs")
@click.argument("name")
@click.pass_obj
def deployments_logs(obj, name) -> None:
    """Stream a deployment's logs."""
    _validate_deployment_name(name)
    _databricks(["apps", "logs", name], obj.profile, action=f"Could not read logs for '{name}'.")


@deployments.command("start")
@click.argument("name")
@click.pass_obj
def deployments_start(obj, name) -> None:
    """Start a deployment."""
    _validate_deployment_name(name)
    _databricks(
        ["apps", "start", name], obj.profile, action=f"Could not start deployment '{name}'."
    )
    if obj.output == "json":
        render.emit_json({"started": name})
        return
    render.success(f"Started deployment '{name}'")


@deployments.command("stop")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_stop(obj, name, yes) -> None:
    """Stop a deployment."""
    _validate_deployment_name(name)
    _confirm_destroy(f"Stop deployment '{name}'", assume_yes=yes)
    _databricks(["apps", "stop", name], obj.profile, action=f"Could not stop deployment '{name}'.")
    if obj.output == "json":
        render.emit_json({"stopped": name})
        return
    render.success(f"Stopped deployment '{name}'")


@deployments.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_delete(obj, name, yes) -> None:
    """Delete a deployment and, when managed provisioning is enabled, its Runtime Store."""
    _validate_deployment_name(name)
    use_managed_runtime_store = _USE_MANAGED_RUNTIME_STORE
    target = (
        f"Delete deployment '{name}' and its Runtime Store data"
        if use_managed_runtime_store
        else f"Delete deployment '{name}'"
    )
    _confirm_destroy(target, assume_yes=yes)
    if use_managed_runtime_store:
        app_service_principal_id = AppsClient(obj.profile, runner=_databricks).service_principal(
            name
        )
        if not app_service_principal_id:
            raise AgentCliError(
                "Could not resolve the app's service principal for Runtime Store cleanup.",
                hint="The deployment was retained. Check access to the app and retry deletion.",
            )
        with render.status("Deleting Runtime Store…"):
            managed_runtime_store.delete(obj.client(), name, app_service_principal_id)
    _databricks(
        ["apps", "delete", name], obj.profile, action=f"Could not delete deployment '{name}'."
    )
    if obj.output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted deployment '{name}'")
