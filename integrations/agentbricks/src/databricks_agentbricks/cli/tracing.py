"""Click commands for configuring and reading Agent Bricks traces."""

from __future__ import annotations

import pathlib

import click

from databricks_agentbricks.clients.local_tracing_client import (
    LocalTracingClient,
    _start_read_server,
    _wait_for_server,
    start_local_tracing_server,
    stop_local_tracing_server,
)
from databricks_agentbricks.clients.mlflow_trace_client import (
    MLflowTraceClient,
    _get_experiment_by_id,
)
from databricks_agentbricks.clients.tracing_client import (
    MLflowTraceTables,
    ResolvedTraceExperiment,
    TraceTable,
    TraceTableKind,
    _mlflow,
    _set_tracking_uri,
    create_experiment_idempotent,
    uc_trace_tables,
)
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.tracing import (
    TRACING_BIND_COMMAND,
    experiment_url,
    show_trace_detail,
    show_trace_list,
)
from databricks_agentbricks.projects.experiment_naming import default_experiment_name
from databricks_agentbricks.services.tracing.targets import TraceTargetResolver
from databricks_agentbricks.services.tracing_service import (
    TraceReadRequest,
    TracingService,
    _status_str,
)

__all__ = [
    "MLflowTraceTables",
    "ResolvedTraceExperiment",
    "TRACES_EXPERIMENT_ID_ENV",
    "TRACES_TRACKING_URI_ENV",
    "TraceTable",
    "TraceTableKind",
    "_get_experiment_by_id",
    "_mlflow",
    "_set_tracking_uri",
    "_start_read_server",
    "_status_str",
    "_wait_for_server",
    "build_tracing_service",
    "create_experiment_idempotent",
    "default_experiment_name",
    "experiment_url",
    "start_local_tracing_server",
    "stop_local_tracing_server",
    "tracing",
    "tracing_bind",
    "tracing_get",
    "tracing_list",
    "tracing_unbind",
    "uc_trace_tables",
]

TRACES_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
TRACES_EXPERIMENT_ID_ENV = "MLFLOW_EXPERIMENT_ID"


def build_tracing_service(obj) -> TracingService:
    mlflow = MLflowTraceClient(
        obj.profile,
        mlflow_loader=_mlflow,
        tracking_uri_setter=_set_tracking_uri,
    )
    return TracingService(mlflow, TraceTargetResolver(mlflow, LocalTracingClient()))


@click.group()
def tracing() -> None:
    """Configure MLflow tracing for your deployed agents, and inspect the traces."""


def _experiment_read_options(command):
    command = click.option(
        "--warehouse",
        "warehouse_id",
        default=None,
        help="SQL warehouse id used to read traces from a UC-backed experiment (required for UC "
        "experiments; ignored for managed). Falls back to the MLFLOW_TRACING_SQL_WAREHOUSE_ID "
        "env var.",
    )(command)
    command = click.option(
        "--experiment-id",
        "experiment_id",
        default=None,
        help="MLflow experiment id to read (e.g. from the experiment URL). Mutually exclusive with "
        "--experiment-name.",
    )(command)
    command = click.option(
        "--experiment-name",
        "experiment_name",
        default=None,
        help="MLflow experiment name to read (an absolute workspace path). Default: this project's "
        "experiment.",
    )(command)
    return command


@tracing.command("bind")
@click.option(
    "--experiment-name",
    "experiment_name",
    default=None,
    help="MLflow experiment name to trace to - an absolute workspace path, e.g. "
    "/Shared/agentbricks_traces/<agent> or /Users/<you>/agentbricks_traces/<agent>. Agent Bricks creates it at "
    "deploy. Mutually exclusive with --experiment-id.",
)
@click.option(
    "--experiment-id",
    "experiment_id",
    default=None,
    help="MLflow experiment id (e.g. copied from the experiment's workspace URL) to trace to. "
    "Resolved to the experiment's name and stored as a name — Agent Bricks stores names, not ids, so the "
    "binding stays valid across workspaces. Mutually exclusive with --experiment-name.",
)
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Project directory containing agent.toml. Defaults to the current directory.",
)
@click.pass_obj
def tracing_bind(obj, experiment_name, experiment_id, source) -> None:
    """Bind tracing to an experiment, by name or id. Requires one of them (like `agentbricks memory/sessions
    bind`); the binding's presence is what turns tracing on.

    The experiment is stored as a NAME, not an id, so the binding stays valid across
    workspaces/profiles — Agent Bricks creates it if absent in the active workspace at deploy. ``--experiment-id``
    (e.g. from the experiment's URL) is a convenience: it's resolved to the experiment's name and
    stored as a name, never as an id.
    """
    name = build_tracing_service(obj).bind(pathlib.Path(source), experiment_name, experiment_id)
    if obj.output == "json":
        render.emit_json({"experiment_name": name})
        return
    render.success(
        f"Tracing on: experiment {name}",
        fields={"Experiment": name},
        next_steps=[
            ("agentbricks dev", "Run locally with tracing on"),
            ("agentbricks tracing list", "List traces once you have some"),
            ("agentbricks tracing unbind", "Turn tracing off"),
        ],
    )


@tracing.command("unbind")
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Project directory containing agent.toml. Defaults to the current directory.",
)
@click.pass_obj
def tracing_unbind(obj, source) -> None:
    """Unbind tracing: remove the experiment binding from agent.toml, turning tracing off for the
    DEPLOYED agent (`agentbricks deploy` then wires no MLflow env).

    Deploy-only: `agentbricks dev` still traces locally to its own MLflow server, so you keep local traces
    while the deployed agent stays untraced.
    """
    build_tracing_service(obj).unbind(pathlib.Path(source))
    if obj.output == "json":
        render.emit_json({"experiment_name": None})
        return
    render.success(
        "Tracing unbound (off for the deployed agent; agentbricks dev still traces locally)",
        next_steps=[(TRACING_BIND_COMMAND, "Turn deployed tracing back on")],
    )


@tracing.command("list")
@_experiment_read_options
@click.option("--limit", type=int, default=20)
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help="Project directory to resolve the default experiment from (default: current dir).",
)
@click.pass_obj
def tracing_list(obj, experiment_name, experiment_id, warehouse_id, limit, source) -> None:
    """List recent agent traces in an experiment.

    An explicit ``--experiment-name`` / ``--experiment-id`` reads that workspace experiment and must
    name one that exists (errors otherwise, so a typo isn't mistaken for an empty experiment). With
    neither, this project's experiment is read: the **workspace** one if it's been provisioned (by
    `agentbricks deploy`), otherwise the local `agentbricks dev` store (``.agentbricks/mlflow.db``), so a not-yet-deployed
    dev run's traces still show up here (tagged "(local dev)"). Nothing traced anywhere yet lists
    nothing. A UC-backed experiment is read through a SQL warehouse (``--warehouse``).
    """
    result = build_tracing_service(obj).list(
        TraceReadRequest(pathlib.Path(source), experiment_name, experiment_id, warehouse_id),
        limit,
    )
    show_trace_list(result, obj.output)


@tracing.command("get")
@click.argument("trace_id")
@_experiment_read_options
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help="Project directory to resolve the experiment from (default: current dir).",
)
@click.pass_obj
def tracing_get(obj, trace_id, experiment_name, experiment_id, warehouse_id, source) -> None:
    """Get a single trace by id (status, latency, span count, previews).

    Reads from the same place as `agentbricks tracing list`: an explicit ``--experiment-name`` /
    ``--experiment-id`` targets that workspace store and must name one that exists (errors otherwise);
    otherwise this project's workspace experiment if provisioned, else its local `agentbricks dev` store.
    A UC-backed experiment is read through a SQL warehouse (``--warehouse``).
    """
    result = build_tracing_service(obj).get(
        TraceReadRequest(pathlib.Path(source), experiment_name, experiment_id, warehouse_id),
        trace_id,
    )
    show_trace_detail(result, obj.output)
