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
    _upsert_manifest_env,
    mlflow_tracing_config,
    resolve_trace_experiment_id,
    store_bindings,
    validate_stores,
)
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import load_project_metadata

# Default local port; `databricks apps run-local` listens here unless --app-port overrides it.
_DEFAULT_APP_PORT = 8000
_LOCAL_RUNTIME_ENV = "DATABRICKS_MASON_RUNTIME_LOCAL"
_LOCAL_APP_YAML = "app.masondev.yaml"

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

    # Validate the agent.toml store bindings and wire tracing into app.yaml. Stores are read from
    # agent.toml at runtime (not written here); tracing is on by default, so resolve/create the
    # per-project experiment and wire its env. Tracing is best-effort locally — if it can't be set up
    # (e.g. no mlflow installed, or offline), dev still runs the agent, just without traces.
    memory_store, session_store = store_bindings(source_dir)
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

    args = ["apps", "run-local"]
    if prepare_environment:
        args.append("--prepare-environment")
    if app_port is not None:
        args += ["--app-port", str(app_port)]

    # Run against a local-only manifest that marks durability as in-memory and removes deploy-only
    # package-index overrides.
    entry_point = _dev_entry_point(app_yaml)
    # run-local resolves this relative to cwd and rejects an absolute alternate-manifest path.
    args += ["--entry-point", entry_point.name]

    # `run-local` prints a generic "go to http://localhost:<port>" line that points at the chat UI —
    # misleading for an API-only project, which serves no page there (404). Print an accurate line up
    # front, keyed on whether this project actually carries the chat-app overlay.
    _announce_local_url(source_dir, app_port or _DEFAULT_APP_PORT)

    # Run in the project dir so run-local finds the app; stream output (no capture). Remove the
    # local-only manifest afterward so a later `mason deploy` cannot sync it to the workspace.
    try:
        _databricks(
            args,
            obj.profile,
            cwd=str(source_dir),
            action="Could not start the agent locally.",
        )
    finally:
        entry_point.unlink(missing_ok=True)


def _announce_local_url(source_dir: pathlib.Path, port: int) -> None:
    """Print how to reach the running app: the chat UI if present, else a sample invoke request."""
    base = f"http://localhost:{port}"
    deploy_name = source_dir.resolve().name
    if (source_dir / "runtime" / "ui.py").is_file():
        render.success(
            "Starting agent",
            fields={"Chat UI": base},
            next_steps=[
                f"Open {base} to chat with your agent",
                ("mason tools add mcp <service>", "Give the agent a tool"),
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
        try:
            template = load_project_metadata(source_dir).template
        except AgentCliError:
            template = None
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
                ("mason tools add mcp <service>", "Give the agent a tool"),
                (f"mason deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )


def _dev_entry_point(app_yaml: pathlib.Path) -> pathlib.Path:
    """Write the local-only app manifest consumed by ``apps run-local``.

    The manifest marks the process as local so durability uses its in-memory store. Keeping this in
    the entry point is more reliable than forwarding ``--env`` through the Databricks CLI and does
    not mutate the deployable ``app.yaml``. Deploy-only package-index variables are also removed.
    """
    try:
        doc = yaml.safe_load(app_yaml.read_text()) or {}
    except yaml.YAMLError as exc:
        raise AgentCliError(f"Could not parse {app_yaml}: {exc}") from exc
    if not isinstance(doc, dict):
        raise AgentCliError(f"Invalid {app_yaml}: top level must be an object.")
    env = doc.get("env")
    if env is not None and not isinstance(env, list):
        raise AgentCliError(f"Invalid {app_yaml}: env must be a list.")
    filtered = [
        e for e in (env or []) if not (isinstance(e, dict) and e.get("name") in _BUILD_INDEX_ENVS)
    ]
    filtered = [
        e for e in filtered if not (isinstance(e, dict) and e.get("name") == _LOCAL_RUNTIME_ENV)
    ]
    filtered.append({"name": _LOCAL_RUNTIME_ENV, "value": "true"})
    doc["env"] = filtered
    # The Apps CLI rejects hidden or hyphenated entry-point filenames.
    dev_yaml = app_yaml.parent / _LOCAL_APP_YAML
    try:
        dev_yaml.write_text(yaml.safe_dump(doc, sort_keys=False))
    except OSError as exc:
        raise AgentCliError(f"Could not write {dev_yaml}: {exc}") from exc
    return dev_yaml
