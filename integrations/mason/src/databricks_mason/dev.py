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

from databricks_mason.deploy import _upsert_manifest_env, resolve_store_env
from databricks_mason.errors import AgentCliError
from databricks_mason.store_access import _databricks

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
@click.option(
    "--with-memory-store",
    "memory_store",
    default=None,
    help="Memory store display name to wire in via AGENT_MEMORY_STORE (same as `mason deploy`).",
)
@click.option(
    "--with-session-store",
    "session_store",
    default=None,
    help="Session store name to wire in via AGENT_SESSION_STORE (same as `mason deploy`).",
)
@click.option(
    "--with-traces",
    "traces_destination",
    default=None,
    help="UC trace destination 'catalog.schema' to wire in via MLFLOW_TRACING_DESTINATION.",
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
@click.pass_obj
def dev(
    obj,
    source: str,
    prepare_environment: Optional[bool],
    app_port: Optional[int],
    memory_store: Optional[str],
    session_store: Optional[str],
    traces_destination: Optional[str],
    traces_experiment: Optional[str],
    create_stores: bool,
) -> None:
    """Run a scaffolded agent locally from its app.yaml (wraps `databricks apps run-local`).

    Reads the app's command + env from ``app.yaml`` and runs it the way the Apps runtime does — so
    local behavior matches a deployment. Auth uses the profile (``-p`` / ``mason login``), same as
    ``mason deploy``. The environment is built on first run and reused after; pass
    ``--prepare-environment`` to force a rebuild (e.g. after changing dependencies).

    The ``--with-*`` flags wire an agent's stores/traces into ``app.yaml`` before running, exactly as
    ``mason deploy`` does — so you can iterate locally against a real store without hand-editing env.
    Locally the store owner (you) already has access, so no service-principal grant is needed here;
    that grant happens at ``mason deploy`` time.
    """
    source_dir = pathlib.Path(source)
    app_yaml = source_dir / "app.yaml"
    if not app_yaml.exists():
        raise AgentCliError(
            f"No app.yaml found at {app_yaml}.",
            hint="Run from a scaffolded project, or pass --source <dir> (see `mason init`).",
        )

    # Wire any requested stores/traces into app.yaml first, so run-local reads the updated env.
    if memory_store or session_store or traces_destination or traces_experiment:
        # The agent name defaults to the project dir name, so a per-app trace experiment here matches
        # what `mason deploy <that-name>` derives.
        env_updates = resolve_store_env(
            obj.client(),
            app=source_dir.resolve().name,
            memory_store=memory_store,
            session_store=session_store,
            traces_destination=traces_destination,
            traces_experiment=traces_experiment,
            create_stores=create_stores,
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

    # If the manifest carries a deploy-only package-index override, run against a filtered copy so
    # the local build uses this machine's index instead of one it may not be able to reach.
    entry_point = _dev_entry_point(app_yaml)
    if entry_point is not None:
        args += ["--entry-point", str(entry_point)]

    # Run in the project dir so run-local finds the app; stream output (no capture).
    _databricks(args, obj.profile, cwd=str(source_dir))


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
