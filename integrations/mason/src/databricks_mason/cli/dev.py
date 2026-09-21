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
from databricks_mason.cli.deploy import (
    _load_project,
    _resolve_memory_store,
    store_bindings,
)
from databricks_mason.cli.endpoint_examples import print_agent_invoke_command
from databricks_mason.cli.tracing import start_local_tracing_server, stop_local_tracing_server
from databricks_mason.databricks_cli import _databricks
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import require_managed_tool_support
from databricks_mason.project_types import AgentServer
from databricks_mason.runtime.store import RUNTIME_STORE_LOCAL_ENV
from databricks_mason.runtime.tool_manifest import MEMORY_STORE_ENV, SESSION_STORE_ENV

# Default local port; `databricks apps run-local` listens here unless --app-port overrides it.
_DEFAULT_APP_PORT = 8000
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
    """Run your agent locally so you can try it before deploying.

    Starts the agent on a local server — by default http://localhost:8000 — and prints where to
    reach it: the chat UI if the project has one, otherwise a sample request against the agent's
    API.

    Auth uses your Databricks profile (`-p` / `mason login`), and the agent reaches Databricks model
    serving through the AI Gateway on that profile — so there are no model keys to set up.

    Under the hood this wraps `databricks apps run-local`: it reads the command + env from
    `app.yaml` and runs the app the way the Apps runtime would, so local behavior matches a
    deployment. The environment is built on the first run and reused after; pass
    `--prepare-environment` to force a rebuild (e.g. after changing dependencies).

    Tracing runs locally: dev starts a local MLflow tracking server (sqlite-backed, under `.mason/`)
    and points the agent at it, so traces are recorded on your machine with no workspace experiment
    or setup — open the printed Traces URL to view them. `mason deploy` sends traces to a managed
    workspace experiment instead. Stores bound with `mason memory/sessions bind` are resolved here and
    injected into the dev-only manifest as env, so the runtime picks them up the same way a deployment
    does. Locally you already have access, so no service-principal grant is needed; that grant happens
    at `mason deploy` time.
    """
    source_dir = pathlib.Path(source)
    app_yaml = source_dir / "app.yaml"
    if not app_yaml.exists():
        raise AgentCliError(
            f"No app.yaml found at {app_yaml}.",
            hint="Run from a scaffolded project, or pass --source <dir> (see `mason init`).",
        )

    project = _load_project(source_dir)
    if project is not None and project.tools:
        require_managed_tool_support(source_dir)

    # Read the declared store bindings; they're resolved into the dev-only manifest as env (below),
    # alongside the local tracing env (further below). Both are best-effort — dev still runs the agent
    # if a store isn't created yet or the local tracing server can't start.
    memory_store, session_store = store_bindings(source_dir)
    # `mason dev` never provisions stores (unlike `mason deploy`); warn so missing long-term memory
    # or conversation state is not a silent surprise.
    if not memory_store:
        render.warning(
            "No memory store bound — long-term memory is disabled. Run 'mason memory bind <name>'."
        )
    if not session_store:
        render.warning(
            "No session store bound — conversation history is in-memory (not durable). "
            "Run 'mason sessions bind <name>'."
        )
    local_env: dict[str, str] = {}
    # Declared stores are created by `mason deploy`, not dev — dev never creates them. Check
    # existence for a friendly warning, and wire the resolved store bindings into the local-only
    # manifest (memory store's id — the entries API key; session store's name) so sessions/memory
    # work locally when the store already exists. The runtime reads these from the env.
    # Best-effort: if the client/auth is unavailable (offline, no credentials), degrade to the
    # same "declared but not created yet" warning and keep running, mirroring tracing below.
    if memory_store:
        try:
            with render.status("Checking memory store…"):
                resolved = _resolve_memory_store(obj.client(), memory_store)
            if resolved is None:
                render.warning(
                    f"Memory store '{memory_store}' is declared but not created yet — long-term memory "
                    "is disabled locally. Run `mason deploy` to create it."
                )
            else:
                store_id = (render.field(resolved, "name") or "").split("/", 1)[-1] or None
                if store_id:
                    local_env[MEMORY_STORE_ENV] = store_id
        except Exception:  # noqa: BLE001 - store check must never block a local run
            render.warning(
                f"Memory store '{memory_store}' is declared but not created yet — long-term memory "
                "is disabled locally. Run `mason deploy` to create it."
            )
    if session_store:
        try:
            with render.status("Checking session store…"):
                obj.client().get_session_store(session_store)
            local_env[SESSION_STORE_ENV] = session_store
        except AgentCliError:
            render.warning(
                f"Session store '{session_store}' is declared but not created yet — conversation "
                "history is in-memory (not durable). Run `mason deploy` to create it."
            )
        except Exception:  # noqa: BLE001 - store check must never block a local run
            render.warning(
                f"Session store '{session_store}' is declared but not created yet — conversation "
                "history is in-memory (not durable). Run `mason deploy` to create it."
            )
    # Local tracing: start a local MLflow tracking server backed by sqlite under .mason/ and point the
    # agent at it via the dev-only manifest — for any project, regardless of framework/server. An agent
    # that uses MLflow (autolog or `start_trace`) then traces to it; it's harmless for one that doesn't.
    # Traces stay on the machine — no workspace experiment, no auth, no username needed — and the same
    # server serves the trace UI. Launched via `uvx mlflow` so it needs neither the (skinny) CLI env nor
    # the agent venv, and it owns the sqlite schema (so there's no client/server migration mismatch).
    # Best-effort: any launch failure degrades to running without traces. `mason deploy` handles the
    # managed workspace experiment instead.
    tracing_server, tracing_env = start_local_tracing_server(source_dir)
    trace_url: Optional[str] = None
    if tracing_env:
        local_env.update(tracing_env)
        trace_url = tracing_env["MLFLOW_TRACKING_URI"]

    # Default: prepare only when there's no venv yet, so repeat runs don't rebuild. Explicit
    # --prepare-environment / --no-prepare-environment overrides the auto-detect.
    if prepare_environment is None:
        prepare_environment = not (source_dir / ".venv").exists()

    args = ["apps", "run-local"]
    if prepare_environment:
        args.append("--prepare-environment")
    if app_port is not None:
        args += ["--app-port", str(app_port)]

    # Run against a local-only manifest that forces the Runtime Store in-process, removes deploy-only
    # package-index overrides, and injects any locally resolved store ids and the local tracing env.
    entry_point = _dev_entry_point(app_yaml, local_env or None)
    # run-local resolves this relative to cwd and rejects an absolute alternate-manifest path.
    args += ["--entry-point", entry_point.name]

    # `run-local` prints a generic "go to http://localhost:<port>" line that points at the chat UI —
    # misleading for an API-only project, which serves no page there (404). Print an accurate line up
    # front, keyed on whether this project actually carries the chat-app overlay.
    _announce_local_url(
        source_dir,
        app_port or _DEFAULT_APP_PORT,
        project.server if project else None,
        trace_url,
    )

    # Run in the project dir so run-local finds the app; stream output (no capture). Remove the
    # local-only manifest afterward so a later `mason deploy` cannot sync it to the workspace, and
    # stop the local tracing server.
    try:
        _databricks(
            args,
            obj.profile,
            cwd=str(source_dir),
            action="Could not start the agent locally.",
        )
    finally:
        entry_point.unlink(missing_ok=True)
        if tracing_server is not None:
            stop_local_tracing_server(tracing_server)


def _announce_local_url(
    source_dir: pathlib.Path, port: int, server: AgentServer | None, trace_url: str | None = None
) -> None:
    """Print how to reach the running app: the chat UI if present, else a sample invoke request.

    ``trace_url`` (when tracing is on) is shown alongside so a dev run surfaces where its traces land,
    matching the ``Traces`` line ``mason deploy`` prints.
    """
    base = f"http://localhost:{port}"
    deploy_name = source_dir.resolve().name
    tool_step: str | tuple[str, str] = (
        "Edit agent/agent.py to give the agent a tool"
        if server == AgentServer.CUSTOM
        else ("mason tools add mcp <service>", "Give the agent a tool")
    )
    if (source_dir / "runtime" / "ui.py").is_file():
        fields = {"Chat UI": base}
        if trace_url:
            fields["Traces"] = trace_url
        render.success(
            "Starting agent",
            fields=fields,
            next_steps=[
                f"Open {base} to chat with your agent",
                tool_step,
                ("mason memory bind <store>", "Attach a memory / session store"),
                (f"mason deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )
    else:
        # No page is served at `/`, so give a copy-pasteable request instead of just the URL.
        uses_runtime_api = server == AgentServer.MASON
        endpoint = f"{base}/api/invocations" if uses_runtime_api else f"{base}/invocations"
        body = (
            '{"id": "00000000-0000-4000-8000-000000000000", '
            '"input": [{"role": "user", "content": "hi"}]}'
            if uses_runtime_api
            else '{"input": [{"role": "user", "content": "hi"}]}'
        )
        sample = f"curl -X POST {endpoint} -H 'Content-Type: application/json' -d '{body}'"
        fields = {"Invoke": f"POST {endpoint}"}
        if trace_url:
            fields["Traces"] = trace_url
        render.success(
            "Starting API-only agent (no chat UI — see `mason init --help`)",
            fields=fields,
            next_steps=[
                (sample, "Send a test request"),
                tool_step,
                (f"mason deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )

    print_agent_invoke_command(
        f"--url {base}",
        uses_runtime_api=(source_dir / "runtime" / "ui.py").is_file()
        or server == AgentServer.MASON,
    )


def _dev_entry_point(
    app_yaml: pathlib.Path, extra_env: dict[str, str] | None = None
) -> pathlib.Path:
    """Write the local-only app manifest consumed by ``apps run-local``.

    The manifest marks the process as local so Mason Runtime uses its in-memory store. Keeping this in
    the entry point is more reliable than forwarding ``--env`` through the Databricks CLI and does
    not mutate the deployable ``app.yaml``. Deploy-only package-index variables are also removed.
    ``extra_env`` is merged in (overriding any same-named entries) for dev-only overrides such as
    the resolved memory-store id.
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
        e
        for e in filtered
        if not (isinstance(e, dict) and e.get("name") == RUNTIME_STORE_LOCAL_ENV)
    ]
    for name, value in (extra_env or {}).items():
        filtered = [e for e in filtered if not (isinstance(e, dict) and e.get("name") == name)]
        filtered.append({"name": name, "value": value})
    filtered.append({"name": RUNTIME_STORE_LOCAL_ENV, "value": "true"})
    doc["env"] = filtered
    # The Apps CLI rejects hidden or hyphenated entry-point filenames.
    dev_yaml = app_yaml.parent / _LOCAL_APP_YAML
    try:
        dev_yaml.write_text(yaml.safe_dump(doc, sort_keys=False))
    except OSError as exc:
        raise AgentCliError(f"Could not write {dev_yaml}: {exc}") from exc
    return dev_yaml
