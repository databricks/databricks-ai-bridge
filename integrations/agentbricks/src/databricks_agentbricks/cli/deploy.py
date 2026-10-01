"""`agentbricks deploy` and the `agentbricks deployments` group — manage agent deployments.

`agentbricks deploy` is the integrated entry point: it provisions the memory/session stores
bound in `agent.toml`, grants the app's service principal access to them, then rolls out
the deployment. Agent Bricks Runtime deployments receive a persistent Runtime Store; a temporary rollout
switch chooses between the legacy per-app Lakebase project and the service-managed database.
`agent.toml` is the CLI's authoring source, resolved here into the `AGENT_MEMORY_STORE` /
`AGENT_SESSION_STORE` env vars written into `app.yaml` — the runtime reads those, never
`agent.toml`. `agentbricks deployments` covers the lifecycle verbs
(`list`/`get`/`logs`/`start`/`stop`/`delete`).

Every verb here is a thin adapter: it builds a `DeployService` from the CLI context, calls one method,
and hands the raw result to a presenter. The work itself lives in the service and its collaborators,
including driving the Databricks Apps runtime via the `databricks apps` CLI (an implementation detail
that is not part of the Agent Bricks CLI surface).
"""

from __future__ import annotations

import pathlib
from typing import Any, Callable, Optional

import click

from databricks_agentbricks.cli.composition import build_deploy_service
from databricks_agentbricks.cli.presenter import (
    present_deleted,
    present_deploy_result,
    present_deployment_detail,
    present_deployments_list,
    present_started,
    present_stopped,
)
from databricks_agentbricks.deployment import (
    _DEFAULT_INSTANCE_COUNT,
    _DEFAULT_PIP_INDEX_URL,
    TRACES_EXPERIMENT_ID_ENV,  # noqa: F401 - re-exported for tests referencing deploy_mod.TRACES_EXPERIMENT_ID_ENV
    TRACES_TRACKING_URI_ENV,  # noqa: F401 - re-exported for tests referencing deploy_mod.TRACES_TRACKING_URI_ENV
    DeploymentName,
    MlflowTracingConfig,  # noqa: F401 - re-exported for callers/tests referencing deploy_mod.MlflowTracingConfig
    _prefixed_name,  # noqa: F401 - re-exported for tests referencing deploy_mod._prefixed_name
    _validate_deployment_name,  # noqa: F401 - re-exported for tests referencing deploy_mod._validate_deployment_name
    mlflow_tracing_config,  # noqa: F401 - re-exported for tests referencing deploy_mod.mlflow_tracing_config
)
from databricks_agentbricks.services.deploy_service import DeployRequest, OperationAborted

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
    "instance_count",
    type=click.IntRange(min=1, max=5),
    default=_DEFAULT_INSTANCE_COUNT,
    show_default=True,
    help="Number of deployment instances the agent is pinned to. Every deploy applies this count, so "
    "re-deploying without the flag returns the agent to the default.",
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
    API clients that need it must resend a stable UUID in this cookie every request:
      __Host-databricks-app-router=<uuid>
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


class _DeploymentNameParam(click.ParamType):
    """Parse the lifecycle verbs' NAME argument into a validated ``DeploymentName``.

    Validating as Click parses the argument (rather than inside each command) is what lets
    ``DeployService`` *require* a ``DeploymentName`` and never re-check one - forgetting the check
    becomes a type error, not a name that reaches the workspace. A bad name raises ``AgentCliError``,
    a ``click.ClickException``, which the CLI renders with the same exit code whether it is raised
    here at parse time or from a command body - so this stays behavior-preserving. It deliberately
    does not call ``self.fail()``, which would emit Click's parse-time "Invalid value" usage error
    (a different exit code and message) instead of the CLI's own diagnostic.
    """

    name = "name"

    def convert(self, value: Any, param: Any, ctx: Any) -> DeploymentName:
        """Click's parse hook: promote the raw argument to a ``DeploymentName``, validating it.

        Raises ``AgentCliError`` (not ``self.fail()``) on a bad name, so the CLI renders its own
        diagnostic and exit code rather than Click's "Invalid value" usage error.
        """
        return DeploymentName(value)


_DEPLOYMENT_NAME = _DeploymentNameParam()


def deployment_name_argument(command: Callable[..., Any]) -> Callable[..., Any]:
    """Declare the shared, self-validating NAME argument for the ``deployments`` lifecycle verbs."""
    return click.argument("name", type=_DEPLOYMENT_NAME)(command)


@click.group()
def deployments() -> None:
    """Inspect and manage deployed agents: list, get, stream logs, start, stop, or delete."""


@deployments.command("list")
@click.pass_obj
def deployments_list(obj) -> None:
    """List Agent Bricks deployments (apps named `agent-bricks-*`)."""
    present_deployments_list(build_deploy_service(obj).list_deployments(), output=obj.output)


@deployments.command("get")
@deployment_name_argument
@click.pass_obj
def deployments_get(obj, name) -> None:
    """Get an agent deployment's details."""
    present_deployment_detail(build_deploy_service(obj).get(name), name, output=obj.output)


@deployments.command("logs")
@deployment_name_argument
@click.pass_obj
def deployments_logs(obj, name) -> None:
    """Stream a deployment's logs."""
    build_deploy_service(obj).logs(name)


@deployments.command("start")
@deployment_name_argument
@click.pass_obj
def deployments_start(obj, name) -> None:
    """Start a deployment."""
    build_deploy_service(obj).start(name)
    present_started(name, output=obj.output)


@deployments.command("stop")
@deployment_name_argument
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_stop(obj, name, yes) -> None:
    """Stop a deployment."""
    try:
        build_deploy_service(obj).stop(name, assume_yes=yes)
    except OperationAborted:
        # Click's abort is this layer's business: it prints "Aborted!" and picks the exit code.
        raise click.Abort() from None
    present_stopped(name, output=obj.output)


@deployments.command("delete")
@deployment_name_argument
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
@click.pass_obj
def deployments_delete(obj, name, yes) -> None:
    """Delete a deployment and, when managed provisioning is enabled, its Runtime Store."""
    try:
        build_deploy_service(obj).delete(name, assume_yes=yes)
    except OperationAborted:
        raise click.Abort() from None
    present_deleted(name, output=obj.output)
