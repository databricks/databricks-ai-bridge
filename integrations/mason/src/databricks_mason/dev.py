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
from databricks_mason.deploy import (
    _TRACING_OFF_HINT,
    provision_trace_experiment,
    store_bindings,
    validate_stores,
)
from databricks_mason.errors import AgentCliError
from databricks_mason.store_access import _databricks
from databricks_mason.tracing import experiment_ui_url, project_trace_location

# Default local port; `databricks apps run-local` listens here unless --app-port overrides it.
_DEFAULT_APP_PORT = 8000

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

    Stores bound with ``mason memory/sessions bind`` and tracing configured with ``mason tracing
    setup`` are both recorded in ``agent.toml`` and picked up here (dev validates the bound stores
    exist and wires the trace experiment, exactly as ``mason deploy`` does). Locally the store owner
    (you) already has access, so no service-principal grant is needed; that grant happens at ``mason
    deploy`` time.
    """
    source_dir = pathlib.Path(source)
    app_yaml = source_dir / "app.yaml"
    if not app_yaml.exists():
        raise AgentCliError(
            f"No app.yaml found at {app_yaml}.",
            hint="Run from a scaffolded project, or pass --source <dir> (see `mason init`).",
        )

    # Stores are bound via `mason memory/sessions bind` (recorded in agent.toml) and read at runtime,
    # so dev only validates that the bound stores still exist - a client/auth call made only when some
    # are bound. Tracing is wired separately below (also read from agent.toml).
    memory_store, session_store = store_bindings(source_dir)
    if memory_store or session_store:
        with render.status("Checking stores…"):
            validate_stores(obj.client(), memory_store=memory_store, session_store=session_store)
    # Tracing is UC-only and opt-in: wire it into app.yaml iff `mason tracing setup` configured a
    # catalog.schema for this project (the exact same path `mason deploy` takes). Check the config
    # first via a cheap agent.toml read, so a plain unconfigured `mason dev` never constructs the
    # workspace client or makes an auth/`me()` call - local iteration stays offline-friendly. When
    # unconfigured, dev doesn't block; the startup panel nudges instead.
    traced = None
    trace_url = None
    trace_schema, _ = project_trace_location(source)
    if trace_schema:
        client = obj.client()
        traced = provision_trace_experiment(
            source_dir, source_dir.resolve().name, client, obj.profile
        )
        if traced:
            trace_url = experiment_ui_url(client.host, traced[0])

    # Default: prepare only when there's no venv yet, so repeat runs don't rebuild. Explicit
    # --prepare-environment / --no-prepare-environment overrides the auto-detect.
    if prepare_environment is None:
        prepare_environment = not (source_dir / ".venv").exists()

    args = ["apps", "run-local"]
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
    _announce_local_url(source_dir, app_port or _DEFAULT_APP_PORT, traced, trace_url)

    # Run in the project dir so run-local finds the app; stream output (no capture).
    _databricks(
        args,
        obj.profile,
        cwd=str(source_dir),
        action="Could not start the agent locally.",
    )


def _announce_local_url(
    source_dir: pathlib.Path,
    port: int,
    traced: Optional[tuple[str, str]],
    trace_url: Optional[str],
) -> None:
    """Print how to reach the running app: the chat UI if present, else a sample invoke request.

    Also states where traces go: the experiment id + a link to its MLflow traces page when tracing
    is configured, or a one-line hint that tracing is off.
    """
    base = f"http://localhost:{port}"
    deploy_name = source_dir.resolve().name
    # traced is (experiment_id, catalog_schema) when `mason tracing setup` ran, else None.
    trace_field: dict[str, str] = {}
    trace_step: list[str | tuple[str, str]] = [_TRACING_OFF_HINT]
    if traced:
        experiment_id, schema = traced
        trace_field["Experiment"] = f"{experiment_id} ({schema})"
        if trace_url:
            trace_field["Traces"] = trace_url
        trace_step = []
    if (source_dir / "runtime" / "ui.py").is_file():
        render.success(
            "Starting agent",
            fields={"Chat UI": base, **trace_field},
            next_steps=[
                f"Open {base} to chat with your agent",
                ("mason tools add mcp <service>", "Give the agent a tool"),
                ("mason memory bind <store>", "Attach a memory / session store"),
                (f"mason deploy {deploy_name}", "Deploy it to Databricks"),
                *trace_step,
            ],
        )
    else:
        # No page is served at `/`, so give a copy-pasteable request instead of just the URL.
        sample = (
            f"curl -X POST {base}/invocations -H 'Content-Type: application/json' "
            '-d \'{"input": [{"role": "user", "content": "hi"}]}\''
        )
        render.success(
            "Starting API-only agent (no chat UI — see `mason init --help`)",
            fields={"Invoke": f"POST {base}/invocations", **trace_field},
            next_steps=[
                (sample, "Send a test request"),
                ("mason tools add mcp <service>", "Give the agent a tool"),
                (f"mason deploy {deploy_name}", "Deploy it to Databricks"),
                *trace_step,
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
