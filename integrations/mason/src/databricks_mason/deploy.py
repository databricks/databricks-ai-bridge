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
from typing import Any, Optional

import click
import yaml

from databricks_mason import memory_store_access, render, session_store_access, timefmt
from databricks_mason.errors import AgentCliError
from databricks_mason.render import field
from databricks_mason.store_access import _databricks, apply_postgres_resources, grant_tables
from databricks_mason.tracing import TRACES_DEST_ENV, TRACES_EXPERIMENT_ENV

_MEMORY_ENV = "AGENT_MEMORY_STORE"
_SESSION_ENV = "AGENT_SESSION_STORE"

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


def _ensure_memory_store(client, display_name: str) -> dict:
    try:
        return client.create_memory_store(display_name)
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    listing = client.list_memory_stores(page_size=1000)
    for store in field(listing, "managed_memory_stores") or []:
        if field(store, "display_name") == display_name:
            return store
    raise AgentCliError(f"Memory store '{display_name}' exists but could not be resolved.")


def _ensure_session_store(client, name: str) -> dict:
    try:
        return client.create_session_store(name)
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    return client.get_session_store(name)


def _memory_store_database(client, memory_store: str) -> Optional[str]:
    """Resolve the memory store's per-store Lakebase database name from its storage backend."""
    store = client.get_memory_store(memory_store)
    backend_id = field(field(store, "storage_backend") or {}, "backend_id")
    return memory_store_access.database_from_backend_id(backend_id) if backend_id else None


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
    "--with-memory-store",
    "memory_store",
    default=None,
    help="Memory store display name to wire in via AGENT_MEMORY_STORE.",
)
@click.option(
    "--with-session-store",
    "session_store",
    default=None,
    help="Session store name to wire in via AGENT_SESSION_STORE.",
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
    "--create-stores",
    is_flag=True,
    help="Create the referenced stores if they don't exist (idempotent).",
)
@click.option(
    "--pip-index-url",
    default=_DEFAULT_PIP_INDEX_URL,
    show_default=True,
    help="Package index the Apps build installs from. Defaults to public PyPI as a temporary "
    "workaround: the Apps build environment currently can't reach the internal proxy. Pass an "
    "empty string to use the build's default index.",
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
    traces_destination,
    traces_experiment,
    create_stores,
    pip_index_url,
    workspace_path,
) -> None:
    """Deploy an agent: provision its stores, wire them in, and roll out the deployment."""
    source_dir = pathlib.Path(source)
    client = obj.client()

    # 1. Provision / resolve stores and build the env to inject.
    env_updates: dict[str, str] = {}
    provisioned: dict[str, Any] = {}
    if memory_store:
        store = (
            _ensure_memory_store(client, memory_store)
            if create_stores
            else client.get_memory_store(memory_store)
        )
        # The API's store name is `memory-stores/<id>`; the agent runtime wants the bare <id>
        # (it re-adds the `memory-stores/` prefix when building the entries URL), so strip it.
        store_name = field(store, "name") or memory_store
        env_updates[_MEMORY_ENV] = store_name.split("/", 1)[-1]
        provisioned["Memory store"] = env_updates[_MEMORY_ENV]
    if session_store:
        if create_stores:
            _ensure_session_store(client, session_store)
        env_updates[_SESSION_ENV] = session_store
        provisioned["Session store"] = session_store
    if traces_destination:
        env_updates[TRACES_DEST_ENV] = traces_destination
        provisioned["Traces"] = traces_destination
    if traces_experiment:
        env_updates[TRACES_EXPERIMENT_ENV] = traces_experiment
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

    if obj.output == "json":
        render.emit_json(
            {
                "deployment": name,
                "workspace_path": ws_path,
                "env": env_updates,
                "store_grant": "skipped"
                if not grants_stores
                else ("granted" if grant_error is None else "failed"),
                "store_grant_error": grant_error,
            }
        )
        return

    steps = [f"mason deployments logs {name}", f"mason deployments get {name}"]
    if scaffolded:
        steps.insert(
            0, f"Set a real `command:` in {source_dir / 'app.yaml'} (a placeholder was written)"
        )
    if grants_stores and grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs read/write on its store tables; that grant couldn't "
            "be applied automatically (it requires store ownership and psql). "
            f"Cause: {grant_error}",
        )
    if grants_stores and grant_error is None:
        provisioned["Store access"] = "granted to app service principal"
    render.success(
        f"Deployed agent '{name}'",
        fields={"Workspace path": ws_path, **provisioned},
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
    _databricks(["apps", "logs", name], obj.profile)


@deployments.command("start")
@click.argument("name")
@click.pass_obj
def deployments_start(obj, name) -> None:
    """Start a deployment."""
    _databricks(["apps", "start", name], obj.profile)
    if obj.output == "json":
        render.emit_json({"started": name})
        return
    render.success(f"Started deployment '{name}'")


@deployments.command("stop")
@click.argument("name")
@click.pass_obj
def deployments_stop(obj, name) -> None:
    """Stop a deployment."""
    _databricks(["apps", "stop", name], obj.profile)
    if obj.output == "json":
        render.emit_json({"stopped": name})
        return
    render.success(f"Stopped deployment '{name}'")


@deployments.command("delete")
@click.argument("name")
@click.pass_obj
def deployments_delete(obj, name) -> None:
    """Delete a deployment."""
    _databricks(["apps", "delete", name], obj.profile)
    if obj.output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted deployment '{name}'")
