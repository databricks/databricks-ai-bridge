"""`mason deploy` and the `mason deployments` group — manage agent deployments.

`mason deploy` is the integrated entry point: it provisions the memory/session stores
bound in `agent.toml`, grants the app's service principal access to them, then rolls out
the deployment. Durable agents use a dedicated, app-owned Lakebase project. The managed stores
are read from `agent.toml` at runtime, so they are not written into `app.yaml`. `mason deployments`
covers the lifecycle verbs
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

from databricks_mason import (
    lakebase_durability_store,
    memory_store_access,
    render,
    session_store_access,
    timefmt,
)
from databricks_mason.databricks_cli import _databricks
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import require_managed_tool_support, uses_custom_server
from databricks_mason.render import field
from databricks_mason.store_access import (
    apply_experiment_resource,
    apply_postgres_resources,
    grant_tables,
)
from databricks_mason.tracing import (
    TRACES_EXPERIMENT_ID_ENV,
    TRACES_TRACKING_URI_ENV,
    create_experiment_idempotent,
    default_experiment_name,
    experiment_url,
)

_AGENT_DURABILITY_STORE_ENV = "DATABRICKS_MASON_RUNTIME_ENDPOINT"
_AGENT_DURABILITY_SCHEMA_ENV = "DATABRICKS_MASON_RUNTIME_SCHEMA"
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


def _ensure_memory_store(client, display_name: str) -> tuple[dict, bool]:
    """Create the memory store, or resolve it if it already exists. Returns (store, created)."""
    try:
        return client.create_memory_store(display_name, retry_transient=True), True
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    store = _resolve_memory_store(client, display_name)
    if store is None:
        raise AgentCliError(f"Memory store '{display_name}' exists but could not be resolved.")
    return store, False


def _ensure_session_store(client, name: str) -> tuple[dict, bool]:
    """Create the session store, or resolve it if it already exists. Returns (store, created)."""
    try:
        return client.create_session_store(name, retry_transient=True), True
    except AgentCliError as exc:
        if exc.error_code != "ALREADY_EXISTS":
            raise
    return client.get_session_store(name), False


def _memory_store_database(client, memory_store: str) -> Optional[str]:
    """Resolve the memory store's per-store Lakebase database name from its storage backend.

    Resolves by display name (what the deploy flag carries), not get_memory_store (which is by id).
    """
    store = _resolve_memory_store(client, memory_store)
    if store is None:
        return None
    backend_id = field(field(store, "storage_backend") or {}, "backend_id")
    return memory_store_access.database_from_backend_id(backend_id) if backend_id else None


def _load_project(source: pathlib.Path, *, strict: bool = False):
    """The AgentProject at `source`, or None when there's no readable agent.toml."""
    from databricks_mason.agent_project import AgentProject

    try:
        return AgentProject.load(source)
    except AgentCliError:
        if strict:
            raise
        return None


def store_bindings(source: pathlib.Path) -> tuple[Optional[str], Optional[str]]:
    """The (memory, session) stores bound in agent.toml via `mason memory/sessions bind`.

    agent.toml is the single source of truth for an agent's stores. Both `mason dev` and `mason
    deploy` resolve through here so the store env AND the deploy-time access grant honor the same
    bindings. Missing/invalid agent.toml is ignored (no stores), so this never blocks a run.
    """
    project = _load_project(source)
    if project is None:
        return None, None
    # str(): agent.toml bindings come back as tomlkit strings, which don't serialize to app.yaml.
    memory = str(project.memory_store) if project.memory_store else None
    session = str(project.session_store) if project.session_store else None
    return memory, session


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


def _ensure_default_stores(project, base_name: str, client) -> None:
    """Create + bind default memory/session stores for any slot not already bound in agent.toml.

    Fills only the gaps: a store already bound is left untouched. Defaults are named
    ``<base_name>-memory`` / ``<base_name>-session`` (create-or-reuse), and the new bindings are
    persisted so later deploys reuse them and the validate/grant steps pick them up. There is no way
    to tell from the agent's code whether it uses a store — the binding is the signal — so
    `mason deploy` provisions both by default; pass --no-create-stores to skip.
    """
    changed = False
    if not project.memory_store:
        store = f"{base_name}-memory"
        resolved, _ = _ensure_memory_store(client, store)
        store_id = (field(resolved, "name") or "").split("/", 1)[-1] or None
        project.bind_memory_store(store, store_id)
        changed = True
    if not project.session_store:
        _ensure_session_store(client, f"{base_name}-session")
        project.bind_session_store(f"{base_name}-session")
        changed = True
    if changed:
        project.write()


def validate_stores(client, *, memory_store: Optional[str], session_store: Optional[str]) -> None:
    """Validate the agent's bound stores exist. Shared by `mason deploy` and `mason dev`.

    Stores are created by `mason memory/sessions bind` and read from agent.toml at runtime, so this
    neither creates them nor writes them to app.yaml — it only checks a bound store still exists (a
    typo or unbound clone fails here, not at runtime).
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


def resolve_trace_experiment_id(
    source: pathlib.Path, project_name: str, client, profile
) -> Optional[str]:
    """The MLflow experiment id an agent traces to, or None when tracing is disabled.

    ``project_name`` is the Mason project name (the source directory's basename), not the deployed
    app name. Tracing is on by default. Resolution:

    - `mason tracing disable` was run -> None (tracing off).
    - a pinned experiment id (`mason tracing configure --experiment`) -> that id.
    - otherwise -> create the per-project experiment (`/Users/<you>/mason-traces/<project>`), pin its
      id into agent.toml, and return it. Pinning on first run means later `mason dev` / `mason deploy`
      reuse the same experiment by id rather than re-deriving the default each time — there is no
      separate "default" state once tracing has run once.

    Shared by `mason dev` and `mason deploy` so both trace to the same experiment for a given project.
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415 - avoid import cycle

    try:
        project = AgentProject.load(source)
    except AgentCliError:
        project = None
    if project is not None and project.trace_disabled:
        return None
    pinned = project.trace_experiment_id if project is not None else None
    if pinned:
        return pinned
    experiment_id = create_experiment_idempotent(
        profile, client, default_experiment_name(client.current_user, project_name)
    )
    # Pin the resolved default so subsequent runs reuse it by id (removes the special-cased "recompute
    # the default" path). No agent.toml (raw dir) just means nowhere to pin — still trace this run.
    if project is not None and project.configure_tracing(experiment_id):
        project.write()
    return experiment_id


@dataclass(frozen=True)
class MlflowTracingConfig:
    """The MLflow config that binds a dev/deployed agent to its experiment.

    The agent enables tracing when it sees both a destination (the workspace tracking uri) and an
    experiment id; ``env`` renders them as the two env vars wired into app.yaml.
    """

    experiment_id: str
    tracking_uri: str = "databricks"

    def env(self) -> dict[str, str]:
        return {
            TRACES_TRACKING_URI_ENV: self.tracking_uri,
            TRACES_EXPERIMENT_ID_ENV: self.experiment_id,
        }


def mlflow_tracing_config(experiment_id: str) -> MlflowTracingConfig:
    """The tracing config binding a dev/deployed agent to ``experiment_id``."""
    return MlflowTracingConfig(experiment_id=experiment_id)


def _grant_store_access(
    app: str,
    sp: str,
    owner: str,
    session_store: Optional[str],
    memory_database: Optional[str],
    profile: Optional[str],
) -> Optional[str]:
    """Bind managed-store databases and grant the app service principal table access."""
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
    "--no-create-stores",
    is_flag=True,
    help="Don't auto-create the default memory/session stores for slots unbound in agent.toml.",
)
@click.pass_obj
def deploy(
    obj,
    name,
    source,
    pip_index_url,
    workspace_path,
    instances,
    no_create_stores,
) -> None:
    """Deploy an agent: validate its bound stores, wire in tracing, and roll out the deployment.

    NAME is recorded in agent.toml on first deploy, so later `mason deploy` (from the project dir)
    can omit it; passing NAME again updates the recorded name. The app is named `mason-<name>`
    (Mason adds the prefix if absent); use that full name with the other `mason deployments` verbs.
    `deployments list` shows only apps carrying this prefix.

    By default any memory/session store not yet bound in agent.toml is created and bound as
    `<name>-memory` / `<name>-session`; pass --no-create-stores to skip that.

    Horizontally scaled deployments use best-effort sticky routing (session affinity). Browsers
    preserve the routing cookie automatically.

    \b
    API clients must reuse a stable UUID in this cookie on every request:
      __Host-databricks-app-router=<uuid>
    """
    source_dir = pathlib.Path(source)
    strict_manifest = uses_custom_server(source_dir) and (source_dir / "agent.toml").is_file()
    project = _load_project(source_dir, strict=strict_manifest)
    if project is not None and project.tools:
        require_managed_tool_support(source_dir)
    base_name = _resolve_deployment_name(project, name)
    name = _prefixed_name(base_name)
    _validate_deployment_name(name)
    # Persist the base name so a later `mason deploy` (no NAME) resolves to the same app.
    if project is not None and project.set_deployment_name(base_name):
        project.write()
    instance_args = _instance_args(instances)
    client = obj.client()

    # 0. Create + bind default memory/session stores for any slot unbound in agent.toml (unless
    #    --no-create-stores). Writes the bindings so the store_bindings read below picks them up.
    if (
        project is not None
        and not no_create_stores
        and not (project.memory_store and project.session_store)
    ):
        with render.status("Provisioning default memory/session stores…"):
            _ensure_default_stores(project, base_name, client)

    # 1. Validate the agent's bound stores (`mason memory/sessions bind` creates them). Stores are
    #    read from agent.toml at runtime, not wired into app.yaml; the bindings also drive the store
    #    access grant (step 6).
    memory_store, session_store = store_bindings(source_dir)
    with render.status("Checking stores…"):
        validate_stores(client, memory_store=memory_store, session_store=session_store)

    # 2. Provision tracing (on by default): resolve/create the agent's MLflow experiment and wire the
    #    two env vars the runtime reads. Keyed on the source dir name (NOT the mason-prefixed
    #    deployment name), matching `mason dev`, so dev and deploy trace to the same per-project
    #    experiment. On first run the resolved default experiment id is pinned into agent.toml, so
    #    later runs reuse it. The app's SP is granted write access to it in step 5 (an experiment app
    #    resource). Best-effort: if it can't be set up (no mlflow, offline, permission), the deploy
    #    still proceeds without tracing.
    trace_experiment_id: Optional[str] = None
    trace_setup_error: Optional[str] = None
    try:
        trace_experiment_id = resolve_trace_experiment_id(
            source_dir, source_dir.resolve().name, client, obj.profile
        )
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort; never block a deploy
        trace_setup_error = str(exc)
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

    memory_database = _memory_store_database(client, memory_store) if memory_store else None
    durability_backend = None
    durability_enabled = bool(project and project.durability_enabled)
    if durability_enabled:
        durability_schema = lakebase_durability_store.get_lakebase_schema(name)
        durability_backend = lakebase_durability_store.get_or_create_backend(
            name, obj.profile, create=True
        )
        env_updates[_AGENT_DURABILITY_STORE_ENV] = durability_backend.endpoint_path
        env_updates[_AGENT_DURABILITY_SCHEMA_ENV] = durability_schema
        provisioned["Agent durability store"] = durability_backend.database_path
    if pip_index_url:
        for env in _PIP_INDEX_ENVS:
            env_updates[env] = pip_index_url
        provisioned["Package index"] = pip_index_url
    if instances is not None:
        provisioned["Instances"] = str(instances)

    # 3. Patch the app.yaml manifest with any trace/index env (stores are read from agent.toml).
    scaffolded = False
    if env_updates:
        scaffolded = _upsert_manifest_env(source_dir, env_updates)

    # 4. Ensure the Databricks App exists and its compute is active. Create only when the app is new
    #    (`apps create` errors on an existing app); the compute wait runs every deploy.
    #
    #    `apps create` itself blocks for minutes (it provisions and waits for compute) and we capture
    #    its output to relabel "App compute" → "Agent compute", so nothing streams meanwhile. Wrap it
    #    in progress (persistent line + spinner) so the CLI isn't silent for the whole provision.
    if not _deployment_exists(name, obj.profile):
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
    # `apps deploy` requires the app's compute to be ACTIVE — a just-created app may still be
    # starting, and an existing one may be STOPPED — so wait either way. Returns immediately when
    # compute is already ACTIVE.
    with render.progress("Waiting for agent compute to start (this can take a few minutes)…"):
        _wait_for_running(name, obj.profile)

    if durability_backend is not None:
        resource_error = apply_postgres_resources(name, [durability_backend], obj.profile)
        if resource_error:
            raise AgentCliError(
                "Could not attach the Lakebase resource required for durable execution.",
                hint=resource_error,
            )

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
    #    - stores: bind each store DB as a `postgres` resource (CONNECT) + GRANT read/write on tables;
    #    - tracing: bind the experiment as an `experiment` resource (CAN_EDIT) so it can write traces.
    #    The experiment resource is the platform-managed grant — no manual SQL grant needed.
    grants_stores = bool(session_store or memory_store)
    grant_error: Optional[str] = None
    if grants_stores:
        with render.status("Granting the app access to its stores…"):
            sp = _app_service_principal(name, obj.profile)
            if sp is None:
                grant_error = "could not resolve the app's service principal."
            else:
                grant_error = _grant_store_access(
                    name, sp, client.current_user, session_store, memory_database, obj.profile
                )
    trace_grant_error: Optional[str] = None
    if trace_experiment_id:
        with render.status("Granting the app access to its trace experiment…"):
            trace_grant_error = apply_experiment_resource(name, trace_experiment_id, obj.profile)

    app_url = _app_url(name, obj.profile)

    if obj.output == "json":
        render.emit_json(
            {
                "deployment": name,
                "url": app_url,
                "workspace_path": ws_path,
                "env": env_updates,
                "trace_experiment_id": trace_experiment_id,
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
    if trace_setup_error is not None:
        steps.insert(0, f"Tracing wasn't set up (deployed without it). Cause: {trace_setup_error}")
    if trace_experiment_id and trace_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs write access to its trace experiment; that grant "
            f"couldn't be applied automatically. Cause: {trace_grant_error}",
        )
    if grants_stores and grant_error is None:
        provisioned["Store access"] = "granted to app service principal"
    if trace_experiment_id and trace_grant_error is None:
        provisioned["Trace access"] = "granted to app service principal"
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
