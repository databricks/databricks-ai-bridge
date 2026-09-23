"""`mason deploy` and the `mason deployments` group — manage agent deployments.

`mason deploy` is the integrated entry point: it provisions the memory/session stores
bound in `agent.toml`, grants the app's service principal access to them, then rolls out
the deployment. Mason Runtime deployments receive a persistent Runtime Store; a temporary rollout
switch chooses between the legacy per-app Lakebase project and the service-managed database.
`agent.toml` is the CLI's authoring source, resolved here into the `AGENT_MEMORY_STORE` /
`AGENT_SESSION_STORE` env vars written into `app.yaml` — the runtime reads those, never
`agent.toml`. `mason deployments` covers the lifecycle verbs
(`list`/`get`/`logs`/`start`/`stop`/`delete`).

Deployments run on the Databricks Apps runtime, which this module drives via the
`databricks apps` CLI — an implementation detail that is not part of Mason's surface.
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Optional

import click
import yaml

import databricks_mason.lakebase_runtime_store as managed_runtime_store
import databricks_mason.legacy_lakebase_runtime_store as legacy_runtime_store
from databricks_mason import (
    render,
    timefmt,
)
from databricks_mason.app_resources import (
    apply_postgres_resources,
    apply_trace_resources,
)
from databricks_mason.cli.app_auth import (
    apply_app_user_scope_update,
    plan_app_user_scope_update,
    required_user_api_scopes,
    requires_user_auth,
)
from databricks_mason.cli.endpoint_examples import print_agent_invoke_command
from databricks_mason.cli.tracing import (
    TRACES_EXPERIMENT_ID_ENV,
    TRACES_TRACKING_URI_ENV,
    TRACING_BIND_COMMAND,
    MLflowTraceTables,
    create_experiment_idempotent,
    experiment_url,
)
from databricks_mason.databricks_cli import _databricks
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import (
    require_managed_tool_support,
)
from databricks_mason.project_types import AgentServer
from databricks_mason.render import field
from databricks_mason.runtime.store import (
    RUNTIME_STORE_DATABASE_ENV,
    RUNTIME_STORE_LAKEBASE_BRANCH_ENV,
    RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV,
    RUNTIME_STORE_SCHEMA_ENV,
    RUNTIME_STORE_USERNAME_ENV,
)
from databricks_mason.runtime.tool_manifest import MEMORY_STORE_ENV, SESSION_STORE_ENV

# TEMPORARY: the Apps build environment currently can't reach the internal pypi proxy, so builds
# time out installing dependencies. Point the build at public PyPI (sanctioned interim workaround)
# until the proxy is reachable from the build sandbox again, then drop this default. pip reads
# PIP_INDEX_URL; uv reads UV_INDEX_URL / UV_DEFAULT_INDEX — set all three to cover both build paths.
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple/"
_PIP_INDEX_ENVS = ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX")
_AGENT_COMPUTE_OUTPUT = ("App compute", "Agent compute")
# Internal rollout switch. Backend selection is intentionally not part of the user-facing CLI or
# process environment; flip this only in a Mason release after the managed API is fully deployed.
_USE_MANAGED_RUNTIME_STORE = False

# Mason names every deployment `agent-mason-<name>` so `deployments list` can filter to its own apps.
# The `agent-` prefix is what the Databricks agent registry keys on to surface these apps; the
# `-mason-` segment narrows `deployments list` to Mason's own agents, not every `agent-*` app.
_DEPLOYMENT_PREFIX = "agent-mason-"
_MAX_DEPLOYMENT_NAME_LEN = 30  # Databricks Apps name limit


# --- databricks CLI plumbing (the deployment runtime) -----------------------


def _deployment_exists(name: str, profile: Optional[str]) -> bool:
    return _databricks(["apps", "get", name], profile, capture=True, check=False).returncode == 0


def _app_service_principal(name: str, profile: Optional[str]) -> Optional[str]:
    """The app's service principal client id (its Postgres role identity), or None if unavailable."""
    result = _databricks(["apps", "get", name, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("service_principal_client_id")
    except json.JSONDecodeError:
        return None


def _app_url(name: str, profile: Optional[str]) -> Optional[str]:
    """The deployed app's browsable URL, or None if it can't be read."""
    result = _databricks(["apps", "get", name, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("url") or None
    except json.JSONDecodeError:
        return None


def _app_compute_state(name: str, profile: Optional[str]) -> Optional[str]:
    """The app's compute state (e.g. RUNNING), or None if it can't be read."""
    result = _databricks(["apps", "get", name, "-o", "json"], profile, capture=True, check=False)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout).get("compute_status", {}).get("state")
    except json.JSONDecodeError:
        return None


def _validate_deployment_name(name: str) -> str:
    """Reject an empty or unsafe deployment name before it reaches a URL / workspace path."""
    if (
        not (name or "").strip()
        or name != name.strip()
        or any(token in name for token in ("/", "\\", ".."))
        or any(character.isspace() for character in name)
    ):
        raise AgentCliError(
            f"Invalid deployment name {name!r}.",
            hint="Use a non-empty name of letters, digits, and hyphens "
            "(no slashes, spaces, or '..').",
        )
    if len(name) > _MAX_DEPLOYMENT_NAME_LEN:
        raise AgentCliError(
            f"Deployment name {name!r} is too long ({len(name)} > {_MAX_DEPLOYMENT_NAME_LEN}).",
            hint=f"Databricks app names cap at {_MAX_DEPLOYMENT_NAME_LEN} characters, including the "
            f"'{_DEPLOYMENT_PREFIX}' prefix Mason adds on deploy.",
        )
    return name


def _instance_args(instances: Optional[int]) -> list[str]:
    """Build runtime instance arguments from Mason's fixed-count option."""
    if instances is None:
        return []
    return [
        "--compute-min-instances",
        str(instances),
        "--compute-max-instances",
        str(instances),
    ]


def _prefixed_name(name: str) -> str:
    """Mason deployments carry an `agent-mason-` prefix so `deployments list` finds only its own apps."""
    return name if name.startswith(_DEPLOYMENT_PREFIX) else f"{_DEPLOYMENT_PREFIX}{name}"


def _confirm_destroy(target: str, *, assume_yes: bool) -> None:
    """Prompt before a destructive deployment op; --yes/-y skips it (for scripts)."""
    if assume_yes:
        return
    if not click.confirm(f"{target}? This cannot be undone.", default=False):
        raise click.Abort()


def _wait_for_running(name: str, profile: Optional[str], timeout_s: int = 300) -> None:
    """Block until a just-created app's compute is ACTIVE (or raise on timeout).

    `apps create` returns before compute is provisioned, but `apps deploy` requires the app to be
    ACTIVE — so a first deploy races without this wait.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _app_compute_state(name, profile) == "ACTIVE":
            return
        time.sleep(5)
    raise AgentCliError(
        f"App '{name}' did not reach a running state within {timeout_s}s.",
        hint=f"Check `mason deployments get {name}`, then re-run deploy once it's running.",
    )


# --- app.yaml manifest handling ---------------------------------------------


def _upsert_manifest_env(
    source: pathlib.Path,
    updates: dict[str, str],
) -> bool:
    """Reconcile env entries in <source>/app.yaml. Returns True if it scaffolded a new file."""
    app_yaml = source / "app.yaml"
    if app_yaml.exists():
        loaded = yaml.safe_load(app_yaml.read_text())
        doc: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
        scaffolded = False
    else:
        doc = {"command": ["# TODO: set your run command, e.g. ['uvicorn', 'app:app']"], "env": []}
        scaffolded = True

    raw_env = doc.get("env")
    candidates = raw_env if isinstance(raw_env, list) else []
    env: list[dict[str, Any]] = [entry for entry in candidates if isinstance(entry, dict)]
    by_name = {e.get("name"): e for e in env if isinstance(e, dict)}
    for name, value in updates.items():
        if name in by_name:
            by_name[name]["value"] = value
            by_name[name].pop("valueFrom", None)
        else:
            env.append({"name": name, "value": value})
    doc["env"] = env
    app_yaml.write_text(yaml.safe_dump(doc, sort_keys=False))
    return scaffolded


# --- store provisioning -----------------------------------------------------


_MEMORY_STORE_PAGE_SIZE = 100  # the memory-stores list API caps page_size at 100


def _resolve_memory_store(client, display_name: str) -> Optional[dict]:
    """Find a memory store by display name, paging through the list, or None if none matches.

    `get_memory_store` looks up by resource id (`memory-stores/<uuid>`), not the display name users
    pass, so resolving a name means listing and matching on `display_name`. The list API caps
    `page_size` at 100, so page through with the `next_page_token` rather than requesting all at once.
    """
    page_token: Optional[str] = None
    while True:
        listing = client.list_memory_stores(
            page_size=_MEMORY_STORE_PAGE_SIZE, page_token=page_token
        )
        for store in field(listing, "managed_memory_stores") or []:
            if field(store, "display_name") == display_name:
                return store
        page_token = field(listing, "next_page_token")
        if not page_token:
            return None


_LAKEBASE_PERMISSION_DOCS = (
    "https://docs.databricks.com/aws/en/oltp/projects/manage-project-permissions"
)


def _store_create_permission_error(name: str, kind: str, cause: AgentCliError) -> AgentCliError:
    """PERMISSION_DENIED on create: the workspace admin has restricted Lakebase project creation."""
    return AgentCliError(
        f"You don't have permission to create {kind} store '{name}'.",
        error_code=cause.error_code,
        hint=(
            "Creating a managed store provisions a Lakebase project, which your workspace admin "
            "has restricted. Ask your workspace admin to grant you permission to create Lakebase "
            f"projects ({_LAKEBASE_PERMISSION_DOCS}), or bind an existing store you can access "
            "with --no-create-stores."
        ),
    )


def _store_access_error(name: str, kind: str) -> AgentCliError:
    """The store already exists but isn't accessible to the caller."""
    return AgentCliError(
        f"{kind.capitalize()} store '{name}' already exists but you don't have access to it.",
        hint=(
            "Ask the store's owner or your workspace admin to grant you access, or bind a "
            "different store you can access with --no-create-stores."
        ),
    )


def _ensure_memory_store(client, display_name: str) -> tuple[dict, bool]:
    """Create the memory store, or resolve it if it already exists. Returns (store, created)."""
    try:
        return client.create_memory_store(display_name, retry_transient=True), True
    except AgentCliError as exc:
        if exc.error_code == "PERMISSION_DENIED":
            raise _store_create_permission_error(display_name, "memory", exc) from exc
        if exc.error_code != "ALREADY_EXISTS":
            raise
    store = _resolve_memory_store(client, display_name)
    if store is None:
        # ALREADY_EXISTS but not in the caller's listing: the store isn't accessible to them.
        raise _store_access_error(display_name, "memory")
    return store, False


def _ensure_session_store(client, name: str) -> tuple[dict, bool]:
    """Create the session store, or resolve it if it already exists. Returns (store, created)."""
    try:
        return client.create_session_store(name, retry_transient=True), True
    except AgentCliError as exc:
        if exc.error_code == "PERMISSION_DENIED":
            raise _store_create_permission_error(name, "session", exc) from exc
        if exc.error_code != "ALREADY_EXISTS":
            raise
    try:
        return client.get_session_store(name), False
    except AgentCliError as exc:
        if exc.error_code == "PERMISSION_DENIED":
            raise _store_access_error(name, "session") from exc
        raise


def _load_project(source: pathlib.Path):
    """The AgentProject at `source`, or None when agent.toml is absent."""
    from databricks_mason.agent_project import AgentProject

    if not (source / "agent.toml").is_file():
        return None
    return AgentProject.load(source)


def resource_bindings(
    source: pathlib.Path,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """The (memory store, session store, tracing experiment) bound in agent.toml.

    agent.toml is the single source of truth for an agent's resources. Both `mason dev` and `mason
    deploy` resolve through here so the resource env/notices AND the deploy-time provisioning honor the
    same bindings. A missing agent.toml means nothing is bound; an invalid manifest fails with a clear
    error.
    """
    project = _load_project(source)
    if project is None:
        return None, None, None
    # str(): agent.toml bindings come back as tomlkit strings, which don't serialize to app.yaml.
    memory = str(project.memory_store) if project.memory_store else None
    session = str(project.session_store) if project.session_store else None
    experiment = str(project.trace_experiment_name) if project.trace_experiment_name else None
    return memory, session, experiment


def _resolve_deployment_name(project, name: Optional[str]) -> str:
    """The deployment's base name: the NAME arg if given, else agent.toml's [agent].deployment_name.

    Errors when neither is available, pointing the user at the one-time `mason deploy <name>`.
    """
    if name is not None and name.strip():
        return name.strip()
    if project is not None and project.deployment_name:
        return str(project.deployment_name)
    raise AgentCliError(
        "No deployment name given and none recorded in agent.toml.",
        hint="Run `mason deploy <name>` once to name the agent; later `mason deploy` can omit it.",
    )


def _reconcile_declared_stores(
    memory_store: Optional[str], session_store: Optional[str], client
) -> Optional[str]:
    """Create any store DECLARED in agent.toml that doesn't exist yet; return the memory store's id.

    `mason deploy` is the only verb that provisions stores. It reconciles to the names declared in
    agent.toml (by `mason init` or `mason memory/sessions bind`) — never inventing a name and never
    writing bindings back into the manifest. A store created here gets a one-line notice. The memory
    store's bare id is returned so the caller can wire AGENT_MEMORY_STORE (the entries API is keyed
    by id, not display name); session stores resolve by name and need nothing here.
    """
    memory_store_id: Optional[str] = None
    if memory_store:
        with render.status(f"Reconciling memory store '{memory_store}'…"):
            resolved, created = _ensure_memory_store(client, memory_store)
        memory_store_id = (field(resolved, "name") or "").split("/", 1)[-1] or None
        if created:
            render.console().print(f"[green]✓[/] Created memory store {memory_store!r}")
    if session_store:
        with render.status(f"Reconciling session store '{session_store}'…"):
            _, created = _ensure_session_store(client, session_store)
        if created:
            render.console().print(f"[green]✓[/] Created session store {session_store!r}")
    return memory_store_id


@dataclass(frozen=True)
class ResolvedTraceExperiment:
    """The workspace experiment a deploy resolves for tracing: its id and its UC OTEL base tables.

    ``tables`` is empty for a managed experiment; for a UC-backed one it carries the per-kind tables
    the app's service principal must be granted MODIFY on so it can export traces (see
    ``app_resources.apply_trace_resources``).
    """

    experiment_id: str
    tables: MLflowTraceTables


def get_or_create_trace_experiment(
    source: pathlib.Path, client, profile
) -> Optional[ResolvedTraceExperiment]:
    """Get-or-create this project's bound MLflow experiment in the ``profile``'s workspace, or None
    when tracing is unbound (no ``experiment_name`` in agent.toml).

    Resolves by experiment **name**, never a stored id. ``source`` locates agent.toml. Nothing is
    written back to agent.toml. Raises if the experiment can't be created.
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415 - avoid import cycle

    try:
        project = AgentProject.load(source)
    except AgentCliError:
        project = None
    name = project.trace_experiment_name if project is not None else None
    if not name:
        return None
    # Show progress while the experiment is get-or-created (a workspace round-trip), matching the
    # memory/session store reconcile spinners so deploy isn't silent about tracing.
    with render.status(f"Reconciling tracing experiment '{name}'…"):
        experiment_id, tables = create_experiment_idempotent(profile, client, name)
    return ResolvedTraceExperiment(experiment_id=experiment_id, tables=tables)


@dataclass(frozen=True)
class MlflowTracingConfig:
    """The MLflow config that binds a deployed agent to its workspace experiment.

    The agent enables tracing when it sees both a destination (the workspace tracking uri) and an
    experiment id; ``env`` renders them as the two env vars wired into app.yaml. (`mason dev` builds
    its own local tracing env instead - see ``cli.tracing.start_local_tracing_server``.)
    """

    experiment_id: str
    tracking_uri: str = "databricks"

    def env(self) -> dict[str, str]:
        return {
            TRACES_TRACKING_URI_ENV: self.tracking_uri,
            TRACES_EXPERIMENT_ID_ENV: self.experiment_id,
        }


def mlflow_tracing_config(experiment_id: str) -> MlflowTracingConfig:
    """The tracing config binding a deployed agent to ``experiment_id``."""
    return MlflowTracingConfig(experiment_id=experiment_id)


def _grant_store_access(
    client,
    sp: str,
    session_store: Optional[str],
    memory_store: Optional[str],
) -> Optional[str]:
    """Grant the app's service principal read/write on its bound stores, via the managed store API.

    The conversation-store service owns the (service-managed) store Lakebase, so it provisions the
    SP's role and runs the GRANTs itself. Unlike a direct Lakebase grant, this needs neither store
    ownership nor MANAGE on the store's Lakebase project, so it works for non-admin deployers.
    """
    try:
        if session_store:
            client.grant_session_store_permission(session_store, sp)
        if memory_store:
            store = _resolve_memory_store(client, memory_store)
            if store is None:
                return f"memory store {memory_store!r} could not be resolved."
            client.grant_memory_store_permission(field(store, "name"), sp)
    except AgentCliError as exc:
        return exc.hint or str(exc)
    return None


# --- mason deploy -----------------------------------------------------------


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
    help="Allow Mason to add missing user API scopes to an existing App for tools configured with "
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

    NAME is recorded in agent.toml on the first deploy, so a later `mason deploy` from the project
    directory can omit it (passing NAME again updates the recorded name). The deployed app is named
    `agent-mason-<name>` (Mason adds the prefix if absent); use that full name with the `mason
    deployments` commands. `deployments list` shows only apps carrying this prefix.

    Any memory/session store declared in agent.toml (for example, by `mason memory/sessions bind`)
    is created if it doesn't exist yet; agent.toml itself is never modified for stores.

    Scaling to multiple instances (--instances) uses best-effort sticky routing, so a browser
    session automatically stays on one instance.

    \b
    API clients that need it must resend a stable UUID in this cookie every request:
      __Host-databricks-app-router=<uuid>
    """
    source_dir = pathlib.Path(source)
    project = _load_project(source_dir)
    if project is not None and project.tools:
        require_managed_tool_support(source_dir)
    user_auth = requires_user_auth(project)
    base_name = _resolve_deployment_name(project, name)
    name = _prefixed_name(base_name)
    _validate_deployment_name(name)
    if allow_user_scope_update and not user_auth:
        raise AgentCliError(
            "--allow-user-scope-update requires a managed tool with auth = 'user' in agent.toml."
        )
    # A request-user tool cannot use OBO until the App forwards request credentials and grants every
    # required user API scope. New Apps are configured automatically. For an existing App, adding a
    # missing scope requires --allow-user-scope-update; already-configured Apps need no flag.
    user_scope_plan = (
        plan_app_user_scope_update(
            name,
            obj.profile,
            allow_existing_app_update=allow_user_scope_update,
            required_scopes=required_user_api_scopes(project),
        )
        if user_auth
        else None
    )
    if user_scope_plan is not None:
        click.echo(
            "User auth: scope updates are not atomic; coordinate with other App owners. "
            "Users may need to sign out and re-consent after scope changes. "
            "Scopes are never removed automatically when tools change.",
            err=True,
        )
        apply_app_user_scope_update(user_scope_plan, instances=instances)
    # Persist the base name so a later `mason deploy` (no NAME) resolves to the same app.
    if project is not None and project.set_deployment_name(base_name):
        project.write()
    instance_args = _instance_args(instances)
    client = obj.client()
    use_managed_runtime_store = _USE_MANAGED_RUNTIME_STORE

    # 1. Reconcile the stores DECLARED in agent.toml: create any that don't exist yet. `mason deploy`
    #    is the only reconcile-to-cloud verb; agent.toml is the source of truth and is never rewritten.
    memory_store, session_store, _ = resource_bindings(source_dir)
    memory_store_id = _reconcile_declared_stores(memory_store, session_store, client)

    # 2. Provision tracing when bound (`mason init` binds a default experiment): get-or-create the
    #    experiment NAME from agent.toml and wire the two env vars the runtime reads. Resolved by name,
    #    never a stored id, and nothing is written back to agent.toml. (`mason dev` traces to a local
    #    MLflow server instead and never touches this workspace experiment.) The app's SP is granted
    #    write access to it in step 5 (an experiment app resource, plus MODIFY on its UC OTEL tables
    #    when UC-backed). Best-effort: if it can't be set up
    #    (no mlflow, offline, permission), the deploy still proceeds without tracing.
    trace_provision: Optional[ResolvedTraceExperiment] = None
    trace_setup_error: Optional[str] = None
    try:
        trace_provision = get_or_create_trace_experiment(source_dir, client, obj.profile)
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort; never block a deploy
        trace_setup_error = str(exc)
    trace_experiment_id = trace_provision.experiment_id if trace_provision else None
    trace_tables = trace_provision.tables if trace_provision else MLflowTraceTables()
    env_updates: dict[str, str] = {}
    provisioned: dict[str, Any] = {}
    if memory_store:
        provisioned["Memory store"] = memory_store
    if session_store:
        provisioned["Session store"] = session_store
    if trace_experiment_id:
        env_updates.update(mlflow_tracing_config(trace_experiment_id).env())
        provisioned["Traces"] = (
            experiment_url(client.host, trace_experiment_id) or trace_experiment_id
        )
    # Known caveat (pre-existing): `_upsert_manifest_env` is upsert-only, so unbinding tracing and
    # redeploying leaves the previous MLFLOW_EXPERIMENT_ID in app.yaml - deploy adds env but never
    # prunes it. A fresh (never-bound) deploy is clean; pruning stale resource env on redeploy is a
    # separate follow-up.
    if memory_store_id:
        env_updates[MEMORY_STORE_ENV] = memory_store_id
    if session_store:
        env_updates[SESSION_STORE_ENV] = session_store

    legacy_runtime_backend = None
    if (
        project is not None
        and project.server == AgentServer.MASON
        and not use_managed_runtime_store
    ):
        with render.status("Reconciling Runtime Store…"):
            legacy_runtime_backend = legacy_runtime_store.get_or_create_backend(name, obj.profile)
        env_updates[RUNTIME_STORE_LAKEBASE_ENDPOINT_ENV] = legacy_runtime_backend.endpoint_path
        env_updates[RUNTIME_STORE_SCHEMA_ENV] = legacy_runtime_backend.schema
    if pip_index_url:
        for env in _PIP_INDEX_ENVS:
            env_updates[env] = pip_index_url
        provisioned["Package index"] = pip_index_url
    if instances is not None:
        provisioned["Instances"] = str(instances)

    # 3. Patch app.yaml before creating the app. The managed Runtime Store fields are added after
    #    app creation because that API requires the app's service principal.
    scaffolded = False
    if env_updates:
        scaffolded = _upsert_manifest_env(source_dir, env_updates)

    # 4. Ensure the app exists and its compute is active. Create only when new; the compute wait
    #    runs every deploy.
    #
    #    `apps create` itself blocks for minutes (it provisions and waits for compute) and we capture
    #    its output to relabel "App compute" → "Agent compute", so nothing streams meanwhile. Wrap it
    #    in progress (persistent line + spinner) so the CLI isn't silent for the whole provision.
    if user_scope_plan is None and not _deployment_exists(name, obj.profile):
        with render.progress(
            "Creating the agent and starting its compute (this can take a few minutes)…"
        ):
            result = _databricks(
                ["apps", "create", name, *instance_args],
                obj.profile,
                capture=True,
                action=f"Could not create deployment '{name}'.",
            )
        old, new = _AGENT_COMPUTE_OUTPUT
        click.echo((result.stdout or "").replace(old, new), nl=False)
    elif user_scope_plan is None and instance_args:
        update = {
            "app": {
                "compute_min_instances": instances,
                "compute_max_instances": instances,
            },
            "update_mask": "compute_min_instances,compute_max_instances",
        }
        result = _databricks(
            ["apps", "create-update", name, "--json", json.dumps(update)],
            obj.profile,
            capture=True,
            action=f"Could not update deployment '{name}'.",
        )
        old, new = _AGENT_COMPUTE_OUTPUT
        click.echo((result.stdout or "").replace(old, new), nl=False)
    # `apps deploy` requires the app's compute to be ACTIVE — a just-created app may still be
    # starting, and an existing one may be STOPPED — so wait either way. Returns immediately when
    with render.progress("Waiting for agent compute to start (this can take a few minutes)…"):
        _wait_for_running(name, obj.profile)

    if legacy_runtime_backend is not None:
        resource_error = apply_postgres_resources(name, [legacy_runtime_backend], obj.profile)
        if resource_error:
            raise AgentCliError(
                "Could not attach the Lakebase resource required for the Runtime Store.",
                hint=resource_error,
            )
    elif project is not None and project.server == AgentServer.MASON:
        app_service_principal_id = _app_service_principal(name, obj.profile)
        with render.status("Reconciling Runtime Store…"):
            runtime_backend = managed_runtime_store.get_or_create_backend(
                client, name, app_service_principal_id
            )
        managed_env = {
            RUNTIME_STORE_LAKEBASE_BRANCH_ENV: runtime_backend.branch,
            RUNTIME_STORE_DATABASE_ENV: runtime_backend.database_id,
            RUNTIME_STORE_USERNAME_ENV: runtime_backend.username,
        }
        scaffolded = _upsert_manifest_env(source_dir, managed_env) or scaffolded
        env_updates.update(managed_env)

    # 5. Upload the source and roll out the deployment.
    ws_path = workspace_path or f"/Workspace/Users/{client.current_user}/mason_deployments/{name}"
    # Don't ship uv.lock: it pins exact package URLs from whatever index the developer's machine
    # resolved against (often an internal proxy). The Apps build must resolve against its own
    # configured index, so let it lock fresh in-sandbox instead of inheriting the local lock.
    _databricks(
        ["sync", str(source_dir), ws_path, "--exclude", "uv.lock"],
        obj.profile,
        action=f"Could not upload the agent source for '{name}'.",
    )
    _databricks(
        ["apps", "deploy", name, "--source-code-path", ws_path],
        obj.profile,
        action=f"Could not deploy '{name}'.",
    )

    # 6. Grant the app's service principal what it needs to run (best-effort):
    #    - stores: grant the SP read/write via the managed store API (the store service does the
    #      underlying Lakebase grant, so no store ownership / Lakebase MANAGE is required here);
    #    - tracing: bind the experiment as an `experiment` resource (CAN_EDIT) so it can write traces;
    #      a UC-backed experiment also needs MODIFY on its UC OTEL tables (`uc_securable` resources).
    #    The experiment resource is the platform-managed grant — no manual SQL grant needed.
    grants_stores = bool(session_store or memory_store)
    grant_error: Optional[str] = None
    if grants_stores:
        with render.status("Granting the app access to its stores…"):
            sp = _app_service_principal(name, obj.profile)
            if sp is None:
                grant_error = "could not resolve the app's service principal."
            else:
                grant_error = _grant_store_access(client, sp, session_store, memory_store)
    # Reconcile the mason-owned trace resources EVERY deploy, bound or not: with tracing unbound
    # (experiment_id None) the write prunes stale mason-trace-experiment / mason-trace-table-*
    # resources left by an earlier bound deploy. (Whether removing a `uc_securable` resource also
    # revokes the underlying UC MODIFY grant is platform behavior - documented but not yet verified
    # live - so pruning the resource is the right action regardless.)
    trace_grant_error: Optional[str] = None
    with render.status("Granting the agent runtime access to its trace experiment…"):
        trace_grant_error = apply_trace_resources(
            name, trace_experiment_id, trace_tables.otel_tables(), obj.profile
        )

    app_url = _app_url(name, obj.profile)

    if obj.output == "json":
        render.emit_json(
            {
                "deployment": name,
                "url": app_url,
                "workspace_path": ws_path,
                "env": env_updates,
                "trace_experiment_id": trace_experiment_id,
                "uc_trace_tables": [name for _, name in trace_tables.otel_tables()],
                "trace_setup_error": trace_setup_error,
                "trace_grant": None
                if not trace_experiment_id
                else ("granted" if trace_grant_error is None else "failed"),
                "trace_grant_error": trace_grant_error,
                "store_grant": "skipped"
                if not grants_stores
                else ("granted" if grant_error is None else "failed"),
                "store_grant_error": grant_error,
            }
        )
        return

    steps: list[str | tuple[str, str]] = [
        (f"mason deployments get {name}", "Check its status and URL"),
        (f"mason deployments logs {name}", "Tail its logs"),
    ]
    if app_url:
        steps.insert(0, f"Open the deployed agent: {app_url}")
    if scaffolded:
        steps.insert(
            0, f"Set a real `command:` in {source_dir / 'app.yaml'} (a placeholder was written)"
        )
    if grants_stores and grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs read/write on its store tables; that grant couldn't "
            "be applied automatically (it requires store ownership). "
            f"Cause: {grant_error}",
        )
    if trace_experiment_id is None:
        # Deployed without tracing - either unbound, or a bound experiment that couldn't be set up.
        # Tell the developer (in case it wasn't intended) and point at `mason tracing bind`; append the
        # cause when setup actually failed.
        step = (
            "Deployed without tracing. "
            f"Run `{TRACING_BIND_COMMAND}` and redeploy to trace this agent."
        )
        if trace_setup_error is not None:
            step += f" (Tracing setup failed: {trace_setup_error})"
        steps.insert(0, step)
    if trace_experiment_id and trace_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs write access to its trace experiment; that grant "
            f"couldn't be applied automatically. Cause: {trace_grant_error}",
        )
    if grants_stores and grant_error is None:
        provisioned["Store access"] = "granted to app service principal"
    if trace_experiment_id and trace_grant_error is None:
        provisioned["Trace access"] = "granted to agent runtime service principal"
    fields = {"URL": app_url} if app_url else {}
    fields.update({"Workspace path": ws_path, **provisioned})
    render.success(
        f"Deployed agent '{name}'",
        fields=fields,
        next_steps=steps,
    )
    print_agent_invoke_command(
        name, uses_runtime_api=bool(project and project.server == AgentServer.MASON)
    )


# --- mason deployments <lifecycle> ------------------------------------------


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
    """List Mason agent deployments (apps named `agent-mason-*`) in the workspace."""
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
        app_service_principal_id = _app_service_principal(name, obj.profile)
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
