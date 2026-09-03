"""`mason tracing` — route an agent's traces to MLflow / Unity Catalog and inspect them.

Parallel to `mason memory` and `mason sessions`: `setup` provisions the trace destination
(links a UC schema to an MLflow experiment, the analog of creating a store), `list`/`get`
read traces back, and `instrument` prints the wiring snippet (the "Starter code" analog).
`mason deploy --with-traces` injects the destination into a deployment's app.yaml, exactly as
`--memory` / `--session` inject their stores.

MLflow is an optional dependency: `setup`/`list`/`get` need `mlflow[databricks]>=3.9.0`
installed and lazily import it; `instrument` (and the deploy wiring) are pure and need nothing.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Optional

import click

from databricks_mason import render, timefmt
from databricks_mason.errors import AgentCliError

_BREADCRUMB = "Agent Tracing"
# Fallback experiment when there's no agent context (e.g. `tracing setup` with no --app). Prefer a
# per-agent experiment via `default_experiment` so each agent's traces are isolated, not commingled.
_DEFAULT_EXPERIMENT = "/Shared/mason-agent-traces"


def default_experiment(user: str, app: Optional[str]) -> str:
    """The per-agent experiment path for `app`, under the user's workspace home.

    Keeps each agent's traces in their own experiment (and out of other users' view), instead of the
    single shared bucket. `mason tracing setup`, `dev`, and `deploy` all derive the same path for a
    given agent, so the experiment `setup` links is the one the agent logs to. Falls back to the
    shared default only when no app name is available.
    """
    if not app:
        return _DEFAULT_EXPERIMENT
    return f"/Users/{user}/mason-traces/{app}"


# Env vars the deployed agent reads (see deploy.py). MLFLOW_TRACING_DESTINATION is MLflow's own
# "catalog.schema" convention; MLFLOW_EXPERIMENT_NAME is the standard MLflow experiment selector.
TRACES_DEST_ENV = "MLFLOW_TRACING_DESTINATION"
TRACES_EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"


def _mlflow():
    """Import mlflow lazily so the core CLI (and offline wheel) don't depend on it."""
    try:
        import mlflow  # noqa: PLC0415 - intentional lazy import

        return mlflow
    except ImportError as exc:
        raise AgentCliError(
            "MLflow is required for `mason tracing` setup/list/get.",
            hint="Install it: pip install 'mlflow[databricks]>=3.9.0'",
        ) from exc


def _uc_trace_symbols():
    """Import the version-specific UC-tracing symbols, surfacing the same clean error as `_mlflow`.

    `import mlflow` succeeding doesn't guarantee these exist — they were added in the tracing API
    this feature needs. Guard them so an older installed MLflow yields Mason's install hint rather
    than a raw ImportError traceback.
    """
    try:
        from mlflow.entities import UCSchemaLocation  # noqa: PLC0415 - lazy, version-specific
        from mlflow.tracing import (  # noqa: PLC0415
            set_experiment_trace_location,
            unset_experiment_trace_location,
        )

        return UCSchemaLocation, set_experiment_trace_location, unset_experiment_trace_location
    except ImportError as exc:
        raise AgentCliError(
            "This MLflow version is too old for `mason tracing setup` (UC trace destinations).",
            hint="Upgrade it: pip install 'mlflow[databricks]>=3.9.0'",
        ) from exc


def _link_trace_location(set_location, unset_location, location, exp_id: str, relink: bool) -> None:
    """Link the experiment to the UC schema, handling the already-linked case.

    MLflow raises if the experiment is already bound to a storage location. With `relink`, unset the
    existing binding first and re-link; otherwise surface a clean error pointing at `--relink`.
    """
    try:
        set_location(location=location, experiment_id=exp_id)
    except Exception as exc:  # noqa: BLE001 - mlflow raises a generic error when already linked
        if "already" not in str(exc).lower():
            raise
        if not relink:
            raise AgentCliError(
                "This experiment is already linked to a trace storage location.",
                hint="Re-run with --relink to replace the existing link.",
            ) from exc
        unset_location(location=location, experiment_id=exp_id)
        set_location(location=location, experiment_id=exp_id)


def _configure(mlflow, profile: Optional[str], warehouse_id: Optional[str]) -> None:
    """Point MLflow at the workspace (honoring mason's --profile) for UC-backed tracing."""
    mlflow.set_tracking_uri(f"databricks://{profile}" if profile else "databricks")
    if warehouse_id:
        os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = warehouse_id


def _ensure_experiment(mlflow, client, name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        return experiment.experiment_id
    # create_experiment won't make the intermediate workspace folder for a nested path (e.g.
    # /Users/<you>/mason-traces/<app>), so create the parent dir first.
    parent = name.rsplit("/", 1)[0]
    if parent:
        client.ensure_workspace_dir(parent)
    return mlflow.create_experiment(name)


def _attr(obj: Any, *paths: str, default: Any = None) -> Any:
    """Read the first present dotted attribute path (MLflow object shapes vary by version)."""
    for path in paths:
        cur = obj
        for part in path.split("."):
            cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur is not None:
            return cur
    return default


def _status_str(status: Any) -> Optional[str]:
    if status is None:
        return None
    return getattr(status, "name", None) or str(status)


def _split_destination(destination: str) -> tuple[str, str]:
    catalog, _, schema = destination.partition(".")
    if not catalog or not schema:
        raise AgentCliError("--destination must be 'catalog.schema'.")
    return catalog, schema


def _experiment_url(host: str, experiment_id: str) -> str:
    """Workspace URL for an experiment's Traces tab."""
    return f"{host.rstrip('/')}/ml/experiments/{experiment_id}?compareRunsMode=TRACES"


# --- group ------------------------------------------------------------------


@click.group()
def tracing() -> None:
    """Set up and inspect MLflow traces (in Unity Catalog) for your agents."""


# --- setup: provision the UC trace destination ------------------------------


@tracing.command("setup")
@click.option("--catalog", required=True, help="Unity Catalog catalog to store traces in.")
@click.option("--schema", required=True, help="Unity Catalog schema to store traces in.")
@click.option(
    "--app",
    default=None,
    help="Agent name — gives this agent its own experiment (/Users/<you>/mason-traces/<app>) so its "
    "traces aren't commingled with other agents'. Defaults to the current directory name (matching "
    "`mason dev`/`deploy`). Overridden by --experiment.",
)
@click.option(
    "--experiment",
    default=None,
    help="MLflow experiment path (overrides the per-app default derived from --app).",
)
@click.option(
    "--warehouse-id",
    default=None,
    help="SQL warehouse id for trace queries (MLFLOW_TRACING_SQL_WAREHOUSE_ID).",
)
@click.option(
    "--relink",
    is_flag=True,
    help="Replace an existing trace-location link on the experiment (unset, then re-link).",
)
@click.pass_obj
def tracing_setup(obj, catalog, schema, app, experiment, warehouse_id, relink) -> None:
    """Link a UC schema to an MLflow experiment so agent traces land in Unity Catalog."""
    mlflow = _mlflow()
    _configure(mlflow, obj.profile, warehouse_id)
    # Default the agent name to the current directory, matching `mason dev`/`deploy`, so running
    # setup from a project dir needs no --app and still lands in that agent's own experiment.
    app = app or pathlib.Path.cwd().name
    client = obj.client()
    exp_name = experiment or default_experiment(client.current_user, app)
    exp_id = _ensure_experiment(mlflow, client, exp_name)

    UCSchemaLocation, set_location, unset_location = _uc_trace_symbols()
    _link_trace_location(
        set_location,
        unset_location,
        UCSchemaLocation(catalog_name=catalog, schema_name=schema),
        exp_id,
        relink,
    )
    destination = f"{catalog}.{schema}"
    url = _experiment_url(client.host, exp_id)

    if obj.output == "json":
        render.emit_json(
            {
                "experiment": exp_name,
                "experiment_id": exp_id,
                "destination": destination,
                "url": url,
            }
        )
        return
    # dev/deploy derive this same experiment from the agent name (dev: project dir, deploy: <name>),
    # so only spell out --traces-experiment when the experiment was set explicitly (not per-app).
    exp_flag = "" if experiment is None else f" --traces-experiment {exp_name}"
    deploy_name = app
    render.success(
        f"Linked traces for '{exp_name}' to {destination}",
        fields={"Experiment": exp_name, "Destination": destination, "View traces": url},
        next_steps=[
            (f"mason dev --with-traces {destination}{exp_flag}", "Run locally with tracing on"),
            (
                f"mason deploy {deploy_name} --with-traces {destination}{exp_flag}",
                "Deploy with tracing on",
            ),
            (
                f"mason tracing instrument --destination {destination}",
                "Print the code snippet to trace your own agent",
            ),
            (f"mason tracing list --experiment {exp_name}", "List traces once you have some"),
        ],
    )


# --- list / get -------------------------------------------------------------


@tracing.command("list")
@click.option(
    "--experiment", default=None, help=f"MLflow experiment path (default: {_DEFAULT_EXPERIMENT})."
)
@click.option("--limit", type=int, default=20)
@click.pass_obj
def tracing_list(obj, experiment, limit) -> None:
    """List recent agent traces in an experiment."""
    mlflow = _mlflow()
    _configure(mlflow, obj.profile, None)
    exp_name = experiment or _DEFAULT_EXPERIMENT
    traces = mlflow.search_traces(
        experiment_names=[exp_name], max_results=limit, return_type="list"
    )

    if obj.output == "json":
        render.emit_json([_trace_json(t) for t in traces])
        return
    rows = [
        [
            _attr(t, "info.trace_id", "info.request_id"),
            render.status_pill(_status_str(_attr(t, "info.status", "info.state"))),
            _attr(t, "info.execution_time_ms", "info.execution_duration_ms"),
            timefmt.relative(_attr(t, "info.timestamp_ms", "info.request_time")),
        ]
        for t in traces
    ]
    render.resource_table(
        f"Agent Traces · {exp_name}",
        [("Trace ID", "left"), ("Status", "left"), ("Latency (ms)", "left"), ("Created", "left")],
        rows,
    )


@tracing.command("get")
@click.argument("trace_id")
@click.pass_obj
def tracing_get(obj, trace_id) -> None:
    """Get a single trace by id (status, latency, span count, previews)."""
    mlflow = _mlflow()
    _configure(mlflow, obj.profile, None)
    trace = mlflow.get_trace(trace_id)
    if obj.output == "json":
        render.emit_json(_trace_json(trace))
        return
    spans = _attr(trace, "data.spans", default=[]) or []
    render.detail(
        _BREADCRUMB,
        trace_id,
        {
            "Status": _status_str(_attr(trace, "info.status", "info.state")),
            "Latency (ms)": _attr(trace, "info.execution_time_ms", "info.execution_duration_ms"),
            "Spans": len(spans),
            "Request": _attr(trace, "info.request_preview", "data.request"),
            "Response": _attr(trace, "info.response_preview", "data.response"),
            "Created": timefmt.absolute(_attr(trace, "info.timestamp_ms", "info.request_time")),
        },
        status=_status_str(_attr(trace, "info.status", "info.state")),
    )


def _trace_json(trace: Any) -> dict:
    return {
        "trace_id": _attr(trace, "info.trace_id", "info.request_id"),
        "status": _status_str(_attr(trace, "info.status", "info.state")),
        "execution_time_ms": _attr(trace, "info.execution_time_ms", "info.execution_duration_ms"),
        "timestamp_ms": _attr(trace, "info.timestamp_ms", "info.request_time"),
    }


# --- instrument: print the agent wiring snippet (no MLflow needed) ----------


@tracing.command("instrument")
@click.option(
    "--destination",
    default=None,
    help="UC trace destination 'catalog.schema' (from `mason tracing setup`).",
)
@click.option(
    "--experiment", default=None, help=f"MLflow experiment path (default: {_DEFAULT_EXPERIMENT})."
)
@click.pass_obj
def tracing_instrument(obj, destination, experiment) -> None:
    """Print the snippet that routes an OpenAI Agents SDK agent's traces to UC."""
    catalog, schema = _split_destination(destination) if destination else ("<catalog>", "<schema>")
    exp_name = experiment or _DEFAULT_EXPERIMENT
    dest = destination or f"{catalog}.{schema}"
    code = (
        "import mlflow\n"
        "from mlflow.entities import UCSchemaLocation\n\n"
        'mlflow.set_tracking_uri("databricks")\n'
        f'mlflow.set_experiment("{exp_name}")\n'
        f'mlflow.tracing.set_destination(UCSchemaLocation(catalog_name="{catalog}", schema_name="{schema}"))\n'
        "mlflow.openai.autolog()   # OpenAI Agents SDK spans -> Unity Catalog traces\n"
        "# NOTE: do NOT call agents.set_tracing_disabled(True) — that turns tracing off."
    )
    if obj.output == "json":
        render.emit_json({"destination": dest, "experiment": exp_name, "snippet": code})
        return
    render.detail(
        _BREADCRUMB,
        dest,
        {"Experiment": exp_name, "Destination": dest, "Requires": "mlflow[databricks]>=3.9.0"},
        status="ACTIVE",
        snippets=[("python", "python", code)],
    )
