"""`agentbricks dev` — run a scaffolded agent locally, wrapping `databricks apps run-local`.

Runs the app from its ``app.yaml`` exactly as the Databricks Apps runtime would locally: reads the
manifest's command + env, and (with ``--prepare-environment``) builds the venv via uv. This is the
local counterpart to ``agentbricks deploy`` — same source dir, same manifest — so what runs here matches
what ships. Delegating to ``apps run-local`` means agentbricks inherits the Apps team's local-run behavior
rather than re-implementing it.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click
import yaml

from databricks_agentbricks import render
from databricks_agentbricks.cli.deploy import (
    _load_project,
    _resolve_memory_store,
    resource_bindings,
)
from databricks_agentbricks.cli.endpoint_examples import print_agent_invoke_command
from databricks_agentbricks.cli.tracing import start_local_tracing_server, stop_local_tracing_server
from databricks_agentbricks.databricks_cli import _databricks
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_config import require_managed_tool_support
from databricks_agentbricks.project_types import AgentServer
from databricks_agentbricks.render import field
from databricks_agentkit.runtime.store import RUNTIME_STORE_LOCAL_ENV
from databricks_agentkit.runtime.tool_manifest import MEMORY_STORE_ENV, SESSION_STORE_ENV

# Default local port; `databricks apps run-local` listens here unless --app-port overrides it.
_DEFAULT_APP_PORT = 8000
_LOCAL_APP_YAML = "app.agentbricksdev.yaml"

# Env vars that pin a package index for the *deployed* Apps build (a cloud-only workaround, see
# `agentbricks deploy`). They point at an index the deploying environment can reach, which is not
# necessarily reachable from the local dev machine — so `agentbricks dev`'s local `uv` build must ignore
# them and use the machine's own configured index instead.
_BUILD_INDEX_ENVS = frozenset({"PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX"})

# Workspace managed-resource env that `agentbricks deploy` writes into app.yaml but `agentbricks dev` must NOT
# inherit - dev runs every resource locally. These are stripped from the dev manifest:
#   - tracing: dev sets up its OWN local MLflow server, so a stale workspace MLFLOW_EXPERIMENT_ID
#     (which MLflow resolves ahead of MLFLOW_EXPERIMENT_NAME) would point the local agent at an
#     experiment that doesn't exist on the local sqlite server. dev re-adds the local
#     MLFLOW_TRACKING_URI / MLFLOW_EXPERIMENT_NAME itself.
#   - stores: only --workspace-stores opts into the current project's workspace bindings. Strip a
#     prior deploy's ids first, then re-add the ids verified for the selected profile, if requested.
# (The Runtime Store's lakebase env needs no strip: dev already forces RUNTIME_STORE_LOCAL=true.)
_DEPLOY_TRACING_ENVS = frozenset(
    {"MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "MLFLOW_TRACING_DESTINATION"}
)
_DEPLOY_RESOURCE_ENVS = _DEPLOY_TRACING_ENVS | {MEMORY_STORE_ENV, SESSION_STORE_ENV}


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
    "--workspace-stores",
    is_flag=True,
    help="Use existing memory/session stores bound in agent.toml with your selected profile. "
    "Validates access without creating stores; local runs can read and write their data.",
)
@click.pass_obj
def dev(
    obj,
    source: str,
    prepare_environment: Optional[bool],
    app_port: Optional[int],
    workspace_stores: bool,
) -> None:
    """Run your agent locally so you can try it before deploying.

    Starts the agent on a local server — by default http://localhost:8000 — and prints where to
    reach it: the chat UI if the project has one, otherwise a sample request against the agent's
    API.

    Auth uses your Databricks profile (`-p` / `agentbricks login`), and the agent reaches Databricks model
    serving through the AI Gateway on that profile — so there are no model keys to set up.

    Under the hood this wraps `databricks apps run-local`: it reads the command + env from
    `app.yaml` and runs the app the way the Apps runtime would, so local behavior matches a
    deployment. The environment is built on the first run and reused after; pass
    `--prepare-environment` to force a rebuild (e.g. after changing dependencies).

    By default, state is local: tracing goes to a local MLflow tracking server (sqlite-backed, under `.agentbricks/`)
    so traces are recorded on your machine with no workspace experiment or setup - open the printed
    Traces URL to view them (`agentbricks tracing unbind` doesn't affect dev; it only stops the deployed
    agent's tracing). Long-term memory is off and conversation history is in-process, lost on restart.
    Pass --workspace-stores to exercise existing bound stores before deploying, using your selected
    profile's credentials. This reads and writes real workspace data; use development stores.
    Missing or inaccessible bindings stop startup with an actionable error. No stores or grants
    are created by dev. Execution state (background runs and event replay) remains in-process in
    both modes, even when conversation history and memory are durable.
    """
    source_dir = pathlib.Path(source)
    app_yaml = source_dir / "app.yaml"
    if not app_yaml.exists():
        raise AgentCliError(
            f"No app.yaml found at {app_yaml}.",
            hint="Run from a scaffolded project, or pass --source <dir> (see `agentbricks init`).",
        )

    project = _load_project(source_dir)
    if project is not None and project.tools:
        require_managed_tool_support(source_dir)

    # Keep the default local sandbox. Workspace state is an explicit opt-in, validated before
    # starting either process and without modifying app.yaml or provisioning remote resources.
    memory_store, session_store, trace_experiment = resource_bindings(source_dir)
    local_env: dict[str, str] = {}
    if workspace_stores:
        local_env.update(_workspace_store_env(obj, memory_store, session_store))
    elif memory_store:
        render.console().print(
            f"[dim]Memory store '{memory_store}' is bound but `agentbricks dev` runs with "
            "long-term memory off. Use `agentbricks dev --workspace-stores` for an existing store, "
            "or `agentbricks deploy` to provision it.[/]"
        )
    if session_store and not workspace_stores:
        render.console().print(
            f"[dim]Session store '{session_store}' is bound but `agentbricks dev` keeps "
            "conversation history in-process (not durable). Use `agentbricks dev --workspace-stores` "
            "for an existing store, or `agentbricks deploy` to provision it.[/]"
        )
    if trace_experiment:
        render.console().print(
            f"[dim]Tracing experiment '{trace_experiment}' is bound but `agentbricks dev` "
            "traces to a local MLflow server. Run `agentbricks deploy` to trace to the bound experiment.[/]"
        )
    render.console().print(
        "[dim]Execution state is in-process: pending runs and replay events are lost on restart.[/]"
    )
    # Local tracing: start a local MLflow tracking server backed by sqlite under .agentbricks/ and point the
    # agent at it via the dev-only manifest — for any project, regardless of framework/server. An agent
    # that uses MLflow (autolog or `start_trace`) then traces to it; it's harmless for one that doesn't.
    # Traces stay on the machine — no workspace experiment, no auth, no username needed — and the same
    # server serves the trace UI. Launched via `uvx mlflow` so it needs neither the (skinny) CLI env nor
    # the agent venv, and it owns the sqlite schema (so there's no client/server migration mismatch).
    # Best-effort: any launch failure degrades to running without traces. `agentbricks deploy` handles the
    # managed workspace experiment instead.
    tracing_server, tracing_env = start_local_tracing_server(source_dir)
    # Everything after the server starts runs under try/finally, so any failure — e.g. a malformed
    # app.yaml that `_dev_entry_point` rejects — still tears the local server down (and removes the
    # dev-only manifest) instead of orphaning the process.
    entry_point: Optional[pathlib.Path] = None
    try:
        trace_url: Optional[str] = None
        if tracing_env:
            local_env.update(tracing_env)
            uri = tracing_env["MLFLOW_TRACKING_URI"]
            name = tracing_env.get("MLFLOW_EXPERIMENT_NAME")
            # Can't deep-link the experiment (created lazily on the first request; its local id isn't
            # stable), so name it after the local MLflow UI URL so the user knows which one to open.
            trace_url = f"{uri} (experiment name: {name})" if name else uri

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

        # Run in the project dir so run-local finds the app; stream output (no capture).
        _databricks(
            args,
            obj.profile,
            cwd=str(source_dir),
            action="Could not start the agent locally.",
        )
    finally:
        # Remove the local-only manifest so a later `agentbricks deploy` cannot sync it to the workspace, and
        # stop the local tracing server — even if setup above raised before run-local.
        if entry_point is not None:
            entry_point.unlink(missing_ok=True)
        if tracing_server is not None:
            stop_local_tracing_server(tracing_server)


def _announce_local_url(
    source_dir: pathlib.Path, port: int, server: AgentServer | None, trace_url: str | None = None
) -> None:
    """Print how to reach the running app: the chat UI if present, else a sample invoke request.

    ``trace_url`` (when tracing is on) is shown alongside so a dev run surfaces where its traces land,
    matching the ``Traces`` line ``agentbricks deploy`` prints.
    """
    base = f"http://localhost:{port}"
    deploy_name = source_dir.resolve().name
    tool_step: str | tuple[str, str] = (
        "Edit agent/agent.py to give the agent a tool"
        if server == AgentServer.CUSTOM
        else ("agentbricks tools add mcp <service>", "Give the agent a tool")
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
                ("agentbricks memory bind <store>", "Declare a long-term memory store"),
                ("agentbricks sessions bind <store>", "Declare a conversation history store"),
                (f"agentbricks deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )
    else:
        # No page is served at `/`, so give a copy-pasteable request instead of just the URL.
        uses_runtime_api = server == AgentServer.AGENTBRICKS
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
            "Starting API-only agent (no chat UI — see `agentbricks init --help`)",
            fields=fields,
            next_steps=[
                (sample, "Send a test request"),
                tool_step,
                (f"agentbricks deploy {deploy_name}", "Deploy it to Databricks"),
            ],
        )

    print_agent_invoke_command(
        f"--url {base}",
        uses_runtime_api=(source_dir / "runtime" / "ui.py").is_file()
        or server == AgentServer.AGENTBRICKS,
    )


def _workspace_store_env(
    obj, memory_store: str | None, session_store: str | None
) -> dict[str, str]:
    """Resolve existing bindings with the same profile used by run-local; never provision."""
    if not memory_store and not session_store:
        raise AgentCliError(
            "No memory or session store is bound in agent.toml.",
            hint="Run `agentbricks sessions bind <store>` and/or `agentbricks memory bind <store>`, "
            "then create the named stores or select existing ones before using --workspace-stores.",
        )
    try:
        client = obj.client()
    except Exception as exc:
        raise AgentCliError(
            "Could not authenticate for workspace stores.",
            hint="Select your workspace with `agentbricks --profile <profile> dev --workspace-stores`. "
            "Check that profile with `databricks auth describe --profile <profile>`.",
        ) from exc

    env: dict[str, str] = {}
    for kind, name in (("memory", memory_store), ("session", session_store)):
        if not name:
            continue
        try:
            if kind == "memory":
                store = _resolve_memory_store(client, name)
                if store is None:
                    raise AgentCliError("Store not found.", error_code="NOT_FOUND")
                resource_name = field(store, "name") or ""
                store_id = resource_name.removeprefix("memory-stores/")
                if not store_id:
                    raise AgentCliError("Store response has no id.", error_code="INVALID_RESPONSE")
                # Listing resolves a display name; GET verifies access to the resolved resource.
                client.get_memory_store(store_id)
                env[MEMORY_STORE_ENV] = store_id
            else:
                client.get_session_store(name)
                env[SESSION_STORE_ENV] = name
        except AgentCliError as exc:
            code = exc.error_code
            if code in {"NOT_FOUND", "RESOURCE_DOES_NOT_EXIST"}:
                command_group = "sessions" if kind == "session" else "memory"
                hint = (
                    f"Check the selected profile and binding. If the store is new, create it with "
                    f"`agentbricks --profile <profile> {command_group} stores create --name {name}`, "
                    "or run `agentbricks deploy` to provision the declared stores. "
                    "A store absent from your listing may instead require an access grant from its owner."
                )
            elif code == "PERMISSION_DENIED":
                hint = "Ask the store owner for access, or bind a development store you can access."
            elif code in {"UNAUTHENTICATED", "UNAUTHORIZED", "INVALID_ACCESS_TOKEN"}:
                hint = "Refresh credentials with `databricks auth login --profile <profile>`."
            else:
                hint = exc.hint or (
                    "Check the selected profile, workspace connectivity, and store availability, "
                    "then retry. No local server or workspace store was created."
                )
            raise AgentCliError(
                f"Could not verify bound {kind} store '{name}' for local development.",
                error_code=code,
                hint=hint,
            ) from exc

    render.success(
        "Using workspace stores for local development",
        fields={
            "Workspace": client.host,
            "Profile": obj.profile or "configured credentials",
            "Conversation history": session_store or "In-process (lost on restart)",
            "Long-term memory": memory_store or "Off",
        },
    )
    render.console().print(
        "[dim]Store read access verified. Agent requests use your credentials and can write to "
        "these stores; write permissions are checked when used.[/]"
    )
    return env


def _dev_entry_point(
    app_yaml: pathlib.Path, extra_env: dict[str, str] | None = None
) -> pathlib.Path:
    """Write the local-only app manifest consumed by ``apps run-local``.

    The manifest marks the process as local so Agent Bricks Runtime uses its in-memory store. Keeping this in
    the entry point is more reliable than forwarding ``--env`` through the Databricks CLI and does
    not mutate the deployable ``app.yaml``. Deploy-only package-index variables and the deploy-written
    workspace resource env (see ``_DEPLOY_RESOURCE_ENVS`` - tracing + memory/session stores) are
    removed so dev stays fully local. ``extra_env`` is merged in (overriding any same-named entries)
    for dev-only overrides such as the local tracing config.
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
    # Drop the deploy-written workspace resource env (tracing + memory/session stores) so a prior
    # `agentbricks deploy` can't pull dev onto workspace resources: a stale MLFLOW_EXPERIMENT_ID would beat
    # the local MLFLOW_EXPERIMENT_NAME dev re-adds below, and AGENT_MEMORY_STORE / AGENT_SESSION_STORE
    # would silently point the local runtime at the workspace stores instead of its local defaults.
    filtered = [
        e for e in filtered if not (isinstance(e, dict) and e.get("name") in _DEPLOY_RESOURCE_ENVS)
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
