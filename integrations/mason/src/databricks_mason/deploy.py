"""`mason deploy` and the `mason deployments` group — manage agent deployments.

`mason deploy` is the integrated entry point: it provisions the memory/session stores
bound in `agent.toml`, grants the app's service principal access to them, then rolls out
the deployment. The stores are read from `agent.toml` at runtime, so they are not written
into `app.yaml`. `mason deployments` covers the lifecycle verbs
(`list`/`get`/`logs`/`start`/`stop`/`delete`).

Deployments run on the Databricks Apps runtime, which this module drives via the
`databricks apps` CLI — an implementation detail that is not part of Mason's surface.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Optional

import click
import yaml

from databricks_mason import memory_store_access, render, session_store_access, timefmt
from databricks_mason.errors import AgentCliError
from databricks_mason.render import field
from databricks_mason.store_access import _databricks, apply_postgres_resources, grant_tables
from databricks_mason.tracing import TRACES_DEST_ENV, TRACES_EXPERIMENT_ENV, default_experiment

# TEMPORARY: the Apps build environment currently can't reach the internal pypi proxy, so builds
# time out installing dependencies. Point the build at public PyPI (sanctioned interim workaround)
# until the proxy is reachable from the build sandbox again, then drop this default. pip reads
# PIP_INDEX_URL; uv reads UV_INDEX_URL / UV_DEFAULT_INDEX — set all three to cover both build paths.
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple/"
_PIP_INDEX_ENVS = ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX")
_AGENT_COMPUTE_OUTPUT = ("App compute", "Agent compute")

# Mason names every deployment `mason-<name>` so `deployments list` can filter to its own apps.
_DEPLOYMENT_PREFIX = "mason-"
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
    """Mason deployments carry a `mason-` prefix so `deployments list` can find only its own apps."""
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


def _upsert_manifest_env(source: pathlib.Path, updates: dict[str, str]) -> bool:
    """Inject/overwrite env entries in <source>/app.yaml. Returns True if it scaffolded a new file."""
    app_yaml = source / "app.yaml"
    if app_yaml.exists():
        loaded = yaml.safe_load(app_yaml.read_text())
        doc: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
        scaffolded = False
    else:
        doc = {"command": ["# TODO: set your run command, e.g. ['uvicorn', 'app:app']"], "env": []}
        scaffolded = True

    raw_env = doc.get("env")
    env: list[dict[str, Any]] = (
        [entry for entry in raw_env if isinstance(entry, dict)] if isinstance(raw_env, list) else []
    )
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


def _ensure_memory_store(client, display_name: str) -> dict:
    try:
        return client.create_memory_store(display_name, retry_transient=True)
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    store = _resolve_memory_store(client, display_name)
    if store is None:
        raise AgentCliError(f"Memory store '{display_name}' exists but could not be resolved.")
    return store


def _ensure_session_store(client, name: str) -> dict:
    try:
        return client.create_session_store(name, retry_transient=True)
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    return client.get_session_store(name)


def _memory_store_database(client, memory_store: str) -> Optional[str]:
    """Resolve the memory store's per-store Lakebase database name from its storage backend.

    Resolves by display name (what the deploy flag carries), not get_memory_store (which is by id).
    """
    store = _resolve_memory_store(client, memory_store)
    if store is None:
        return None
    backend_id = field(field(store, "storage_backend") or {}, "backend_id")
    return memory_store_access.database_from_backend_id(backend_id) if backend_id else None


def store_bindings(source: pathlib.Path) -> tuple[Optional[str], Optional[str]]:
    """The (memory, session) stores bound in agent.toml via `mason memory/sessions bind`.

    agent.toml is the single source of truth for an agent's stores. Both `mason dev` and `mason
    deploy` resolve through here so the store env AND the deploy-time access grant honor the same
    bindings. Missing/invalid agent.toml is ignored (no stores), so this never blocks a run.
    """
    try:
        from databricks_mason.agent_project import AgentProject

        project = AgentProject.load(source)
    except AgentCliError:
        return None, None
    # str(): agent.toml bindings come back as tomlkit strings, which don't serialize to app.yaml.
    memory = str(project.memory_store) if project.memory_store else None
    session = str(project.session_store) if project.session_store else None
    return memory, session


def validate_stores_and_trace_env(
    client,
    *,
    app: Optional[str],
    memory_store: Optional[str],
    session_store: Optional[str],
    traces_destination: Optional[str],
    traces_experiment: Optional[str],
) -> dict[str, str]:
    """Validate the agent's bound stores exist and build the MLFLOW_* trace env to wire in.

    Shared by `mason deploy` and `mason dev`. Stores are created by `mason memory/sessions bind` and
    read from agent.toml at runtime, so this neither creates them nor writes them to app.yaml — it
    only checks a bound store still exists (a typo or unbound clone fails here, not at runtime) and
    returns the trace env.
    """
    if memory_store and _resolve_memory_store(client, memory_store) is None:
        # Resolve by display name: get_memory_store looks up by resource id, not the bound name.
        raise AgentCliError(
            f"Memory store '{memory_store}' does not exist.",
            hint=f"Run `mason memory bind {memory_store}` to create and bind it.",
        )
    if session_store:
        try:
            client.get_session_store(session_store)
        except AgentCliError as exc:
            raise AgentCliError(
                f"Session store '{session_store}' does not exist.",
                hint=f"Run `mason sessions bind {session_store}` to create and bind it.",
                error_code=exc.error_code,
            ) from exc
    env: dict[str, str] = {}
    if traces_destination:
        env[TRACES_DEST_ENV] = traces_destination
        # The agent enables tracing only when BOTH a destination and an experiment are set, so
        # default the experiment to this agent's per-app path (matching `mason tracing setup --app`),
        # otherwise --with-traces alone would ship a half-config that silently disables tracing.
        env[TRACES_EXPERIMENT_ENV] = traces_experiment or default_experiment(
            client.current_user, app
        )
    elif traces_experiment:
        env[TRACES_EXPERIMENT_ENV] = traces_experiment
    return env


def _grant_store_access(
    app: str,
    sp: str,
    owner: str,
    session_store: Optional[str],
    memory_database: Optional[str],
    profile: Optional[str],
) -> Optional[str]:
    """Give the app's SP access to the deployed stores (best-effort, two steps).

    Binds every store's database as a `postgres` app resource in one update (the update replaces the
    whole resource array, so they must be applied together), then GRANTs the SP read/write on each
    store's tables. Returns None on success or a human-readable reason on the first failure.
    """
    backends = []
    if session_store:
        backends.append(session_store_access.backend(session_store))
    if memory_database:
        backends.append(memory_store_access.backend(memory_database))
    if not backends:
        return None

    error = apply_postgres_resources(app, backends, profile)
    if error:
        return error
    for backend in backends:
        error = grant_tables(backend, sp, owner, profile)
        if error:
            return error
    return None


# --- mason deploy -----------------------------------------------------------


@click.command()
@click.argument("name")
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Local source directory for the deployment (containing app.yaml). Defaults to the "
    "current directory.",
)
@click.option(
    "--with-traces",
    "traces_destination",
    default=None,
    help="UC trace destination 'catalog.schema' to wire in via MLFLOW_TRACING_DESTINATION "
    "(link it first with `mason tracing setup`).",
)
@click.option(
    "--traces-experiment",
    default=None,
    help="MLflow experiment path to wire in via MLFLOW_EXPERIMENT_NAME.",
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
@click.pass_obj
def deploy(
    obj,
    name,
    source,
    traces_destination,
    traces_experiment,
    pip_index_url,
    workspace_path,
    instances,
) -> None:
    """Deploy an agent: validate its bound stores, wire in tracing, and roll out the deployment.

    The app is named `mason-<name>` (Mason adds the prefix if absent); use that full name with the
    other `mason deployments` verbs. `deployments list` shows only apps carrying this prefix.

    Horizontally scaled deployments use best-effort sticky routing (session affinity). Browsers
    preserve the routing cookie automatically.

    \b
    API clients must reuse a stable UUID in this cookie on every request:
      __Host-databricks-app-router=<uuid>
    """
    name = _prefixed_name(name)
    _validate_deployment_name(name)
    instance_args = _instance_args(instances)
    source_dir = pathlib.Path(source)
    client = obj.client()

    # 1. Validate the agent's bound stores (`mason memory/sessions bind` creates them) and build any
    #    trace env. Stores are read from agent.toml at runtime, not wired into app.yaml; the bindings
    #    also drive the store access grant (step 4).
    memory_store, session_store = store_bindings(source_dir)
    with render.status("Checking stores…"):
        env_updates = validate_stores_and_trace_env(
            client,
            app=name,
            memory_store=memory_store,
            session_store=session_store,
            traces_destination=traces_destination,
            traces_experiment=traces_experiment,
        )
    provisioned: dict[str, Any] = {}
    if memory_store:
        provisioned["Memory store"] = memory_store
    if session_store:
        provisioned["Session store"] = session_store
    if traces_destination:
        provisioned["Traces"] = traces_destination
    if pip_index_url:
        for env in _PIP_INDEX_ENVS:
            env_updates[env] = pip_index_url
        provisioned["Package index"] = pip_index_url
    if instances is not None:
        provisioned["Instances"] = str(instances)

    # 2. Patch the app.yaml manifest with any trace/index env (stores are read from agent.toml).
    scaffolded = False
    if env_updates:
        scaffolded = _upsert_manifest_env(source_dir, env_updates)

    # 3. Roll out the deployment (Databricks Apps runtime).
    if not _deployment_exists(name, obj.profile):
        result = _databricks(
            ["apps", "create", name, *instance_args],
            obj.profile,
            capture=True,
            action=f"Could not create deployment '{name}'.",
        )
        old, new = _AGENT_COMPUTE_OUTPUT
        click.echo((result.stdout or "").replace(old, new), nl=False)
        # `apps create` returns before the app's compute is up, but `apps deploy` requires it to be
        # RUNNING — so wait for it, or the first deploy races and fails ("not in RUNNING state").
        # Use progress (persistent line + spinner): this wait runs for minutes, so it must leave
        # visible feedback even where the spinner animation doesn't render.
        with render.progress("Waiting for app compute to start (this can take a few minutes)…"):
            _wait_for_running(name, obj.profile)
    elif instance_args:
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

    # 4. Give the app's SP access to its stores (best-effort, two steps): bind each store's database
    #    as a `postgres` resource (CONNECT), then GRANT the SP read/write on its tables. Without
    #    both, the app runs but the durable store path fails (can't connect, or can't read tables).
    grants_stores = bool(session_store or memory_store)
    grant_error: Optional[str] = None
    if grants_stores:
        with render.status("Granting the app access to its stores…"):
            sp = _app_service_principal(name, obj.profile)
            if sp is None:
                grant_error = "could not resolve the app's service principal."
            else:
                memory_database = (
                    _memory_store_database(client, memory_store) if memory_store else None
                )
                grant_error = _grant_store_access(
                    name, sp, client.current_user, session_store, memory_database, obj.profile
                )

    app_url = _app_url(name, obj.profile)

    if obj.output == "json":
        render.emit_json(
            {
                "deployment": name,
                "url": app_url,
                "workspace_path": ws_path,
                "env": env_updates,
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
        steps.insert(0, (f"open {app_url}", "Open the deployed app"))
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
    if grants_stores and grant_error is None:
        provisioned["Store access"] = "granted to app service principal"
    fields = {"URL": app_url} if app_url else {}
    fields.update({"Workspace path": ws_path, **provisioned})
    render.success(
        f"Deployed agent '{name}'",
        fields=fields,
        next_steps=steps,
    )


# --- mason deployments <lifecycle> ------------------------------------------


@click.group()
def deployments() -> None:
    """Manage agent deployments."""


def _deployment_status(a: dict) -> Optional[str]:
    for key in ("app_status", "compute_status"):
        section = a.get(key)
        if isinstance(section, dict) and field(section, "state"):
            return field(section, "state")
    return field(a, "state")


@deployments.command("list")
@click.pass_obj
def deployments_list(obj) -> None:
    """List Mason agent deployments (apps named `mason-*`) in the workspace."""
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
    """Delete a deployment."""
    _validate_deployment_name(name)
    _confirm_destroy(f"Delete deployment '{name}'", assume_yes=yes)
    _databricks(
        ["apps", "delete", name], obj.profile, action=f"Could not delete deployment '{name}'."
    )
    if obj.output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted deployment '{name}'")
