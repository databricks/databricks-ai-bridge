"""`mason dev` — run a scaffolded agent locally, wrapping `databricks apps run-local`.

Runs the app from its ``app.yaml`` exactly as the Databricks Apps runtime would locally: reads the
manifest's command + env, and (with ``--prepare-environment``) builds the venv via uv. This is the
local counterpart to ``mason deploy`` — same source dir, same manifest — so what runs here matches
what ships. Delegating to ``apps run-local`` means mason inherits the Apps team's local-run behavior
rather than re-implementing it.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click
import yaml

from databricks_mason import render
from databricks_mason.agent_project import AgentProject
from databricks_mason.databricks_cli import _databricks
from databricks_mason.deploy import (
    _load_project,
    _upsert_manifest_env,
    mlflow_tracing_config,
    resolve_trace_experiment_id,
    store_bindings,
    validate_stores,
)
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import (
    is_custom_server_template,
    load_project_metadata,
    require_managed_tool_support,
    uses_custom_server,
)

# Default local port; `databricks apps run-local` listens here unless --app-port overrides it.
_DEFAULT_APP_PORT = 8000
_LOCAL_RUNTIME_ENV = "DATABRICKS_MASON_RUNTIME_LOCAL"

# Env vars that pin a package index for the *deployed* Apps build (a cloud-only workaround, see
# `mason deploy`). They point at an index the deploying environment can reach, which is not
# necessarily reachable from the local dev machine — so `mason dev`'s local `uv` build must ignore
# them and use the machine's own configured index instead.
_BUILD_INDEX_ENVS = frozenset({"PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX"})


@click.command()
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Local source directory to run (containing app.yaml). Defaults to the current directory.",
)
@click.option(
    "--prepare-environment/--no-prepare-environment",
    default=None,
    help="Build the app's environment with uv before running. Default: build only if no .venv "
    "exists yet, and reuse it otherwise. Requires uv.",
)
@click.option("--app-port", type=int, default=None, help="Port to run the app on (default 8000).")
@click.pass_obj
def dev(
    obj,
    source: str,
    prepare_environment: Optional[bool],
    app_port: Optional[int],
) -> None:
    """Run a scaffolded agent locally from its app.yaml (wraps `databricks apps run-local`).

    Reads the app's command + env from ``app.yaml`` and runs it the way the Apps runtime does — so
    local behavior matches a deployment. Auth uses the profile (``-p`` / ``mason login``), same as
    ``mason deploy``. The environment is built on first run and reused after; pass
    ``--prepare-environment`` to force a rebuild (e.g. after changing dependencies).

    Tracing is on by default: dev sends the agent's traces to the default mason experiment based on
    the project name (the same one ``mason deploy`` uses), created and pinned into agent.toml on first
    run — configure or turn it off with ``mason tracing configure`` / ``disable``. Stores bound with ``mason memory/sessions
    bind`` are validated here and read from agent.toml at runtime. Locally you already have access, so
    no service-principal grant is needed; that grant happens at ``mason deploy`` time.
    """
    source_dir = pathlib.Path(source)
    app_yaml = source_dir / "app.yaml"
    if not app_yaml.exists():
        raise AgentCliError(
            f"No app.yaml found at {app_yaml}.",
            hint="Run from a scaffolded project, or pass --source <dir> (see `mason init`).",
        )

    strict_manifest = uses_custom_server(source_dir) and (source_dir / "agent.toml").is_file()
    project = _load_project(source_dir, strict=strict_manifest)
    if project is not None and project.tools:
        require_managed_tool_support(source_dir)

    # Validate the agent.toml store bindings and wire tracing into app.yaml. Stores are read from
    # agent.toml at runtime (not written here); tracing is on by default, so resolve/create the
    # per-project experiment and wire its env. Tracing is best-effort locally — if it can't be set up
    # (e.g. no mlflow installed, or offline), dev still runs the agent, just without traces.
    memory_store, session_store = store_bindings(source_dir)
    # `mason dev` never provisions stores (unlike `mason deploy`); warn so the missing durability /
    # long-term memory isn't a silent surprise.
    if not memory_store:
        render.warning(
            "No memory store bound — long-term memory is disabled. Run 'mason memory bind <name>'."
        )
    if not session_store:
        render.warning(
            "No session store bound — conversation history is in-memory (not durable). "
            "Run 'mason sessions bind <name>'."
        )
    env_updates: dict[str, str] = {}
    # Stores legitimately require auth, so build the client eagerly only when stores are bound.
    if memory_store or session_store:
        with render.status("Checking stores…"):
            validate_stores(obj.client(), memory_store=memory_store, session_store=session_store)
    # Tracing is best-effort: build the client and provision inside the try so ANY failure (no auth /
    # offline, no mlflow, permission) degrades to running without traces rather than aborting a purely
    # local run.
    try:
        experiment_id = resolve_trace_experiment_id(
            source_dir, source_dir.resolve().name, obj.client(), obj.profile
        )
        if experiment_id:
            env_updates.update(mlflow_tracing_config(experiment_id).env())
    except Exception as exc:  # noqa: BLE001 - tracing must never block a local run
        render.console().print(
            f"[yellow]⚠[/] Tracing not enabled: {exc}. Proceeding without tracing."
        )
    if env_updates:
        _upsert_manifest_env(source_dir, env_updates)

    # Default: prepare only when there's no venv yet, so repeat runs don't rebuild. Explicit
    # --prepare-environment / --no-prepare-environment overrides the auto-detect.
    if prepare_environment is None:
        prepare_environment = not (source_dir / ".venv").exists()

    # `apps run-local` sets DATABRICKS_APP_NAME just like a deployment. Mark this invocation
    # explicitly so the durability SDK selects its process-local development store instead of
    # requiring the Lakebase resource that `mason deploy` attaches.
    args = [
        "apps",
        "run-local",
        "--env",
        f"{_LOCAL_RUNTIME_ENV}=true",
    ]
    if prepare_environment:
        args.append("--prepare-environment")
    if app_port is not None:
        args += ["--app-port", str(app_port)]

    # If the manifest carries a deploy-only package-index override, run against a filtered copy so
    # the local build uses this machine's index instead of one it may not be able to reach.
    entry_point = _dev_entry_point(app_yaml)
    if entry_point is not None:
        args += ["--entry-point", str(entry_point)]

    # `run-local` prints a generic "go to http://localhost:<port>" line that points at the chat UI —
    # misleading for an API-only project, which serves no page there (404). Print an accurate line up
    # front, keyed on whether this project actually carries the chat-app overlay.
    _announce_local_url(source_dir, app_port or _DEFAULT_APP_PORT)

    # Run in the project dir so run-local finds the app; stream output (no capture).
    _databricks(
        args,
        obj.profile,
        cwd=str(source_dir),
        action="Could not start the agent locally.",
    )


def _announce_local_url(source_dir: pathlib.Path, port: int) -> None:
    """Print how to reach the running app: the chat UI if present, else a sample invoke request."""
    base = f"http://localhost:{port}"
    deploy_name = source_dir.resolve().name
    try:
        template = load_project_metadata(source_dir).template
    except AgentCliError:
        template = None
    tool_step: str | tuple[str, str] = (
        "Edit agent/agent.py to give the agent a tool"
        if is_custom_server_template(template)
        else ("mason tools add mcp <service>", "Give the agent a tool")
    )
    if (source_dir / "runtime" / "ui.py").is_file():
        render.success(
            "Starting agent",
            fields={"Chat UI": base},
            next_steps=[
                f"Open {base} to chat with your agent",
                tool_step,
                ("mason memory bind <store>", "Attach a memory / session store"),
                (f"mason deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )
    else:
        # No page is served at `/`, so give a copy-pasteable request instead of just the URL.
        try:
            durable = AgentProject.load(source_dir).durability_enabled
        except AgentCliError:
            durable = False
        uses_runtime_api = durable or template in {"agent-langgraph", "agent-openai"}
        endpoint = f"{base}/api/invocations" if uses_runtime_api else f"{base}/invocations"
        body = (
            '{"id": "00000000-0000-4000-8000-000000000000", '
            '"input": [{"role": "user", "content": "hi"}]}'
            if uses_runtime_api
            else '{"input": [{"role": "user", "content": "hi"}]}'
        )
        sample = f"curl -X POST {endpoint} -H 'Content-Type: application/json' -d '{body}'"
        render.success(
            "Starting API-only agent (no chat UI — see `mason init --help`)",
            fields={"Invoke": f"POST {endpoint}"},
            next_steps=[
                (sample, "Send a test request"),
                tool_step,
                (f"mason deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )


def _dev_entry_point(app_yaml: pathlib.Path) -> Optional[pathlib.Path]:
    """Return a filtered manifest path when app.yaml pins a build index, else None.

    Strips the deploy-only package-index env vars and writes the result next to app.yaml as
    ``.mason-dev.app.yaml`` (so relative paths still resolve). Returns None when there's nothing to
    strip, so the normal ``app.yaml`` is used unchanged.
    """
    try:
        doc = yaml.safe_load(app_yaml.read_text()) or {}
    except yaml.YAMLError:
        return None
    env = doc.get("env")
    if not isinstance(env, list):
        return None
    filtered = [e for e in env if not (isinstance(e, dict) and e.get("name") in _BUILD_INDEX_ENVS)]
    if len(filtered) == len(env):
        return None  # no index override present — run-local can use app.yaml directly
    doc["env"] = filtered
    dev_yaml = app_yaml.parent / ".mason-dev.app.yaml"
    try:
        dev_yaml.write_text(yaml.safe_dump(doc, sort_keys=False))
    except OSError as exc:
        raise AgentCliError(f"Could not write {dev_yaml}: {exc}") from exc
    return dev_yaml
