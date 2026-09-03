"""`mason deploy` and the `mason deployments` group — manage agent deployments.

`mason deploy` is the integrated entry point: it can provision a memory store and a
session store for the agent, inject their identifiers into the deployment's `app.yaml`
env, then roll out the deployment. `mason deployments` covers the lifecycle verbs
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

_MEMORY_ENV = "AGENT_MEMORY_STORE"
_MEMORY_ACTOR_ENV = "AGENT_MEMORY_ACTOR_ID"
_SESSION_ENV = "AGENT_SESSION_STORE"
_SESSION_ACTOR_ENV = "AGENT_SESSION_ACTOR_ID"

# TEMPORARY: the Apps build environment currently can't reach the internal pypi proxy, so builds
# time out installing dependencies. Point the build at public PyPI (sanctioned interim workaround)
# until the proxy is reachable from the build sandbox again, then drop this default. pip reads
# PIP_INDEX_URL; uv reads UV_INDEX_URL / UV_DEFAULT_INDEX — set all three to cover both build paths.
_DEFAULT_PIP_INDEX_URL = "https://pypi.org/simple/"
_PIP_INDEX_ENVS = ("PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX")


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
    return name


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
        return client.create_memory_store(display_name)
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    store = _resolve_memory_store(client, display_name)
    if store is None:
        raise AgentCliError(f"Memory store '{display_name}' exists but could not be resolved.")
    return store


def _ensure_session_store(client, name: str) -> dict:
    try:
        return client.create_session_store(name)
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


def resolve_store_env(
    client,
    *,
    app: Optional[str],
    memory_store: Optional[str],
    session_store: Optional[str],
    traces_destination: Optional[str],
    traces_experiment: Optional[str],
    create_stores: bool,
) -> dict[str, str]:
    """Resolve store/trace references to the AGENT_*/MLFLOW_* env vars that wire them in.

    Shared by `mason deploy` and `mason dev` so both wire an agent's stores into app.yaml the same
    way. With `create_stores` (the default), missing stores are created (idempotent); when it is off
    they must already exist. The memory store resolves to its bare id (the runtime re-adds the
    `memory-stores/` prefix when building the entries URL); the session store and trace destination
    are used verbatim.
    """
    env: dict[str, str] = {}
    if memory_store:
        # Resolve by display name (what users pass) in both cases: get_memory_store looks up by
        # resource id, so it can't resolve a display name on the non-create path.
        if create_stores:
            store = _ensure_memory_store(client, memory_store)
        else:
            store = _resolve_memory_store(client, memory_store)
            if store is None:
                raise AgentCliError(
                    f"Memory store '{memory_store}' does not exist "
                    "(drop --no-create-stores to create it)."
                )
        store_name = field(store, "name") or memory_store
        env[_MEMORY_ENV] = store_name.split("/", 1)[-1]
    if session_store:
        if create_stores:
            _ensure_session_store(client, session_store)
        else:
            # Validate existence up front so a typo fails at deploy time, not at runtime.
            try:
                client.get_session_store(session_store)
            except AgentCliError as exc:
                raise AgentCliError(
                    f"Session store '{session_store}' does not exist "
                    "(drop --no-create-stores to create it).",
                    error_code=exc.error_code,
                ) from exc
        env[_SESSION_ENV] = session_store
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
    "--memory",
    "-m",
    "memory_store",
    default=None,
    help="Memory store display name to wire in via AGENT_MEMORY_STORE.",
)
@click.option(
    "--session",
    "-s",
    "session_store",
    default=None,
    help="Session store name to wire in via AGENT_SESSION_STORE.",
)
@click.option(
    "--actor-id",
    default="agent",
    show_default=True,
    help="Actor id used for managed memory entries and sessions.",
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
    "--no-create-stores",
    is_flag=True,
    help="Require referenced stores to already exist. By default missing stores are created "
    "(idempotent).",
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
@click.pass_obj
def deploy(
    obj,
    name,
    source,
    memory_store,
    session_store,
    actor_id,
    traces_destination,
    traces_experiment,
    no_create_stores,
    pip_index_url,
    workspace_path,
) -> None:
    """Deploy an agent: provision its stores, wire them in, and roll out the deployment."""
    _validate_deployment_name(name)
    source_dir = pathlib.Path(source)
    client = obj.client()

    # 1. Provision / resolve stores and build the env to inject.
    env_updates = resolve_store_env(
        client,
        app=name,
        memory_store=memory_store,
        session_store=session_store,
        traces_destination=traces_destination,
        traces_experiment=traces_experiment,
        create_stores=not no_create_stores,
    )
    provisioned: dict[str, Any] = {}
    if _MEMORY_ENV in env_updates:
        provisioned["Memory store"] = env_updates[_MEMORY_ENV]
        env_updates[_MEMORY_ACTOR_ENV] = actor_id
    if _SESSION_ENV in env_updates:
        provisioned["Session store"] = env_updates[_SESSION_ENV]
        env_updates[_SESSION_ACTOR_ENV] = actor_id
    if traces_destination:
        provisioned["Traces"] = traces_destination
    if pip_index_url:
        for env in _PIP_INDEX_ENVS:
            env_updates[env] = pip_index_url
        provisioned["Package index"] = pip_index_url

    # 2. Patch the app.yaml manifest with the store identifiers.
    scaffolded = False
    if env_updates:
        scaffolded = _upsert_manifest_env(source_dir, env_updates)

    # 3. Roll out the deployment (Databricks Apps runtime).
    if not _deployment_exists(name, obj.profile):
        _databricks(["apps", "create", name], obj.profile)
        # `apps create` returns before the app's compute is up, but `apps deploy` requires it to be
        # RUNNING — so wait for it, or the first deploy races and fails ("not in RUNNING state").
        _wait_for_running(name, obj.profile)
    ws_path = workspace_path or f"/Workspace/Users/{client.current_user}/mason_deployments/{name}"
    # Don't ship uv.lock: it pins exact package URLs from whatever index the developer's machine
    # resolved against (often an internal proxy). The Apps build must resolve against its own
    # configured index, so let it lock fresh in-sandbox instead of inheriting the local lock.
    _databricks(["sync", str(source_dir), ws_path, "--exclude", "uv.lock"], obj.profile)
    _databricks(["apps", "deploy", name, "--source-code-path", ws_path], obj.profile)

    # 4. Give the app's SP access to its stores (best-effort, two steps): bind each store's database
    #    as a `postgres` resource (CONNECT), then GRANT the SP read/write on its tables. Without
    #    both, the app runs but the durable store path fails (can't connect, or can't read tables).
    grants_stores = bool(session_store or memory_store)
    grant_error: Optional[str] = None
    if grants_stores:
        sp = _app_service_principal(name, obj.profile)
        if sp is None:
            grant_error = "could not resolve the app's service principal."
        else:
            memory_database = _memory_store_database(client, memory_store) if memory_store else None
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
    """List agent deployments in the workspace."""
    result = _databricks(["apps", "list", "-o", "json"], obj.profile, capture=True)
    data = json.loads(result.stdout or "[]")
    items = data.get("apps", data) if isinstance(data, dict) else data
    if obj.output == "json":
        render.emit_json(items)
        return
    rows = [
        [
            field(a, "name"),
            render.status_pill(_deployment_status(a)),
            field(a, "url"),
            timefmt.relative(field(a, "update_time")),
        ]
        for a in items
    ]
    render.resource_table(
        "Agent Deployments",
        [("Name", "left"), ("Status", "left"), ("URL", "left"), ("Updated", "left")],
        rows,
    )


@deployments.command("get")
@click.argument("name")
@click.pass_obj
def deployments_get(obj, name) -> None:
    """Get an agent deployment's details."""
    _validate_deployment_name(name)
    result = _databricks(["apps", "get", name, "-o", "json"], obj.profile, capture=True)
    data = json.loads(result.stdout or "{}")
    if obj.output == "json":
        render.emit_json(data)
        return
    url = field(data, "url")
    render.detail(
        "Agent Deployment",
        field(data, "name") or name,
        {
            "URL": url,
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
    _databricks(["apps", "logs", name], obj.profile)


@deployments.command("start")
@click.argument("name")
@click.pass_obj
def deployments_start(obj, name) -> None:
    """Start a deployment."""
    _validate_deployment_name(name)
    _databricks(["apps", "start", name], obj.profile)
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
    _databricks(["apps", "stop", name], obj.profile)
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
    _databricks(["apps", "delete", name], obj.profile)
    if obj.output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted deployment '{name}'")
