"""`mason tracing` — send an agent's traces to an MLflow experiment and inspect them.

Tracing is managed MLflow tracing (traces are stored in the workspace's MLflow backend) and is **on
by default**: with no configuration, ``mason dev`` and ``mason deploy`` send an agent's traces to a
per-project experiment (``/Users/<you>/mason-traces/<project>``), auto-created and pinned into
agent.toml on first run (so later runs reuse it by id). The experiment is identified everywhere by its
**id** — that single value binds the agent (``MLFLOW_EXPERIMENT_ID``), grants the deployed app (an
experiment app resource), reads traces, and builds the UI link.

``mason tracing configure`` pins a specific experiment by id (or re-enables the per-project default
after
a disable); ``mason tracing disable`` turns tracing off; ``list`` / ``get`` read traces back.

MLflow is an optional dependency: ``configure``/``list``/``get`` and the dev/deploy experiment
provisioning import it lazily and need ``mlflow[databricks]``; nothing else does.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

import click

from databricks_mason import render, timefmt
from databricks_mason.errors import AgentCliError

_BREADCRUMB = "Agent Tracing"
# Per-app experiment folder: each agent's traces stay in their own experiment under the user's home.
_TRACES_DIR = "mason-traces"

# The two env vars the deployed/dev agent reads to enable tracing: a destination (the workspace) and
# an experiment (by id). MLflow turns tracing on only when it has both.
TRACES_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
TRACES_EXPERIMENT_ID_ENV = "MLFLOW_EXPERIMENT_ID"

# Installing the `tracing` extra (rather than a bare mlflow) resolves both the missing- and
# too-old-mlflow cases: the extra carries the version floor `mason tracing` needs.
_INSTALL_HINT = "Install the tracing extra: pip install 'databricks-mason[tracing]'"


def default_experiment_name(user: str, project: Optional[str]) -> str:
    """The per-project experiment path under the user's workspace home (shared by dev and deploy).

    ``project`` is the Mason project name (the source directory's basename), not the deployed app name
    — dev and deploy resolve the same value so they trace to one experiment per project.
    """
    if not project:
        raise AgentCliError("Cannot derive the default tracing experiment without a project name.")
    return f"/Users/{user}/{_TRACES_DIR}/{project}"


def experiment_url(host: Optional[str], experiment_id: str) -> Optional[str]:
    """The workspace MLflow experiment Traces page, or None when the host is unavailable."""
    if not host or host == "unknown":
        return None
    return f"{host.rstrip('/')}/ml/experiments/{experiment_id}?compareRunsMode=TRACES"


def _mlflow():
    """Import mlflow lazily so the core CLI (and offline wheel) don't depend on it."""
    try:
        import mlflow  # noqa: PLC0415 - intentional lazy import

        return mlflow
    except ImportError as exc:
        raise AgentCliError(
            "MLflow is required for `mason tracing` configure/list/get.",
            hint=_INSTALL_HINT,
        ) from exc


def _set_tracking_uri(mlflow, profile: Optional[str]) -> None:
    """Point MLflow at the workspace (honoring mason's --profile)."""
    mlflow.set_tracking_uri(f"databricks://{profile}" if profile else "databricks")


# An experiment linked to a UC schema for trace storage carries this tag (the destination schema);
# managed experiments don't. mason supports managed tracing only (UC support is a follow-up).
_UC_TRACE_TAG = "mlflow.experiment.databricksTraceDestinationPath"


def _is_uc_backed(experiment) -> bool:
    """True if the experiment stores traces in Unity Catalog rather than the managed MLflow backend."""
    return _UC_TRACE_TAG in (getattr(experiment, "tags", None) or {})


def create_experiment_idempotent(profile: Optional[str], client, name: str) -> str:
    """Create the experiment ``name`` if missing and return its id (idempotent).

    ``create_experiment`` won't make the intermediate workspace folder for a nested path (e.g.
    ``/Users/<you>/mason-traces/<project>``), so the parent dir is created first. Used by dev/deploy to
    provision the managed experiment that traces log to.
    """
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, profile)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        return experiment.experiment_id
    parent = name.rsplit("/", 1)[0]
    if parent:
        client.ensure_workspace_dir(parent)
    return mlflow.create_experiment(name)


def _project_experiment_id(obj, source: str, mlflow) -> Optional[str]:
    """The experiment id `list` should read for this project, or None if none exists yet.

    Resolution: the project's pinned ``experiment_id``, else the per-project default experiment (looked up
    by name; None when it hasn't been created yet — i.e. nothing has traced here).
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415 - avoid import cycle

    try:
        project = AgentProject.load(source)
    except AgentCliError:
        project = None
    if project is not None and project.trace_experiment_id:
        return project.trace_experiment_id
    client = obj.client()
    name = default_experiment_name(client.current_user, pathlib.Path(source).resolve().name)
    experiment = mlflow.get_experiment_by_name(name)
    return experiment.experiment_id if experiment else None


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


def _trace_to_json(trace: Any) -> dict:
    return {
        "trace_id": _attr(trace, "info.trace_id", "info.request_id"),
        "status": _status_str(_attr(trace, "info.status", "info.state")),
        "execution_time_ms": _attr(trace, "info.execution_time_ms", "info.execution_duration_ms"),
        "timestamp_ms": _attr(trace, "info.timestamp_ms", "info.request_time"),
    }


# --- group ------------------------------------------------------------------


@click.group()
def tracing() -> None:
    """Configure MLflow tracing for your agents, and inspect the traces."""


# --- configure / disable ----------------------------------------------------


@tracing.command("configure")
@click.option(
    "--experiment",
    "experiment_id",
    default=None,
    help="MLflow experiment id to trace to. Must be an existing experiment. Omit to use (or return "
    "to) the per-project experiment mason creates automatically.",
)
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Project directory containing agent.toml. Defaults to the current directory.",
)
@click.pass_obj
def tracing_configure(obj, experiment_id, source) -> None:
    """Configure tracing via MLflow: pin an experiment, rebind, or re-enable after `disable`.

    Tracing is on by default (a per-project experiment mason creates). Use this to pin a specific
    experiment by id, rebind to a different one, or turn tracing back on after ``mason tracing
    disable``. Omit ``--experiment`` to (re)enable the per-project default.
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415

    if experiment_id:
        # Verify the experiment exists so a wrong id fails here, not silently when the agent runs.
        mlflow = _mlflow()
        _set_tracking_uri(mlflow, obj.profile)
        experiment = mlflow.get_experiment(experiment_id)
        if experiment is None:
            raise AgentCliError(
                f"No MLflow experiment found with id {experiment_id!r}.",
                hint="Pass an existing experiment id, or omit --experiment to use the mason default.",
            )
        if _is_uc_backed(experiment):
            # Traces for a UC-backed experiment land in governed UC tables, which need a SQL warehouse
            # to read and UC grants for a deployed app's SP to write - neither of which mason sets up
            # yet. Reject it up front rather than silently wiring a config that fails at read/deploy.
            raise AgentCliError(
                "UC-backed MLflow tracing is not supported by mason.",
                hint="Pass a managed (non-UC) experiment, or omit --experiment to use the mason "
                "default.",
            )

    project = AgentProject.load(pathlib.Path(source))
    project.configure_tracing(experiment_id)
    project.write()

    target = f"experiment {experiment_id}" if experiment_id else "a per-project experiment"
    if obj.output == "json":
        render.emit_json({"experiment_id": experiment_id, "disabled": False})
        return
    render.success(
        f"Tracing on: {target}",
        fields={"Experiment id": experiment_id} if experiment_id else None,
        next_steps=[
            ("mason dev", "Run locally with tracing on"),
            ("mason tracing list", "List traces once you have some"),
            ("mason tracing disable", "Turn tracing off"),
        ],
    )


@tracing.command("disable")
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Project directory containing agent.toml. Defaults to the current directory.",
)
@click.pass_obj
def tracing_disable(obj, source) -> None:
    """Turn tracing off for this agent (recorded in agent.toml; dev/deploy then wire no MLflow env)."""
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(pathlib.Path(source))
    project.disable_tracing()
    project.write()

    if obj.output == "json":
        render.emit_json({"disabled": True})
        return
    render.success(
        "Tracing off",
        next_steps=[("mason tracing configure", "Turn tracing back on")],
    )


# --- list / get -------------------------------------------------------------


@tracing.command("list")
@click.option(
    "--experiment",
    "experiment_id",
    default=None,
    help="MLflow experiment id to read (default: this project's experiment).",
)
@click.option("--limit", type=int, default=20)
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help="Project directory to resolve the default experiment from (default: current dir).",
)
@click.pass_obj
def tracing_list(obj, experiment_id, limit, source) -> None:
    """List recent agent traces in an experiment.

    Resolution: ``--experiment <id>`` (works standalone), else this project's experiment (the pinned
    one, or its per-project default). A missing experiment just lists nothing (nothing has traced yet).
    """
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, obj.profile)
    exp_id = experiment_id or _project_experiment_id(obj, source, mlflow)
    traces = (
        mlflow.search_traces(locations=[exp_id], max_results=limit, return_type="list")
        if exp_id
        else []
    )

    if obj.output == "json":
        render.emit_json([_trace_to_json(t) for t in traces])
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
        f"Agent Traces · {exp_id or 'no experiment yet'}",
        [("Trace ID", "left"), ("Status", "left"), ("Latency (ms)", "left"), ("Created", "left")],
        rows,
    )


@tracing.command("get")
@click.argument("trace_id")
@click.pass_obj
def tracing_get(obj, trace_id) -> None:
    """Get a single trace by id (status, latency, span count, previews)."""
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, obj.profile)
    trace = mlflow.get_trace(trace_id)
    if trace is None:
        raise AgentCliError(f"No trace found with id {trace_id!r}.")
    if obj.output == "json":
        render.emit_json(_trace_to_json(trace))
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
