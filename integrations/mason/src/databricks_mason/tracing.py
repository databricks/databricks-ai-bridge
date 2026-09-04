"""`mason tracing` - configure where an agent's MLflow traces go, and inspect them.

Tracing is Unity Catalog-only and opt-in. ``mason tracing setup --trace-location <catalog.schema>``
records a UC schema in ``agent.toml``; from then on both ``mason dev`` and ``mason deploy`` send the
agent's traces there (creating a per-app UC-linked experiment that surfaces them in the MLflow UI).
Until setup is run, tracing is simply off - neither ``dev`` nor ``deploy`` blocks on it, so a
developer without a catalog/schema is never stuck.

``list`` / ``get`` read traces back. MLflow is an optional dependency: ``list`` / ``get`` and the
UC experiment provisioning need ``mlflow[databricks]>=3.10.1`` and import it lazily; ``setup`` is
pure and needs nothing.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import Any, Optional

import click

from databricks_mason import render, timefmt
from databricks_mason.errors import AgentCliError

_BREADCRUMB = "Agent Tracing"

# MLflow's own env vars the agent reads at runtime (wired into app.yaml by dev/deploy).
TRACES_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
TRACES_EXPERIMENT_ENV = "MLFLOW_EXPERIMENT_NAME"
TRACES_WAREHOUSE_ENV = "MLFLOW_TRACING_SQL_WAREHOUSE_ID"
# The trace-destination pin: a bare experiment id, or a "catalog.schema" UC location. Wired
# ALONGSIDE the experiment (not instead of it): the experiment var is what the agent runtime's
# enable-gate keys on, while this pin makes MLflow export to exactly this location and ignore any
# ambient OTEL_EXPORTER_OTLP_* exporter in the shell that would otherwise hijack the agent's traces.
TRACES_DESTINATION_ENV = "MLFLOW_TRACING_DESTINATION"

# Per-user workspace folder that holds each app's UC-linked tracing experiment.
_TRACES_DIR = "mason-traces"

# A UC trace location is "catalog.schema" (no dots-in-names, no path separators, exactly one dot).
_UC_SCHEMA = re.compile(r"^[^.\s/]+\.[^.\s/]+$")

_MIGRATE_DOCS = "https://docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc"


def experiment_name(user: str, app: str) -> str:
    """The per-app UC-linked experiment that surfaces its traces (shared by dev and deploy)."""
    return f"/Users/{user}/{_TRACES_DIR}/{app}"


def experiment_ui_url(host: Optional[str], experiment_id: str) -> Optional[str]:
    """The workspace MLflow experiment traces page for ``experiment_id`` (from the profile's host).

    ``host`` is the workspace URL (``MasonClient.host`` / the profile's config host); returns None if
    it's unavailable so callers can just omit the link.
    """
    if not host or host == "unknown":
        return None
    return f"{host.rstrip('/')}/ml/experiments/{experiment_id}/traces"


def validate_uc_schema(location: str) -> str:
    """Accept a Unity Catalog 'catalog.schema'; reject anything else."""
    loc = location.strip()
    if _UC_SCHEMA.match(loc):
        return loc
    raise AgentCliError(
        f"Invalid trace location {location!r}.",
        hint="Use a Unity Catalog schema in 'catalog.schema' form.",
    )


# Installing the `tracing` extra (rather than a bare mlflow) is what actually resolves both the
# missing- and too-old-mlflow cases: the extra carries the version floor `mason tracing` needs, so
# a stale mlflow already in the venv gets upgraded to a compatible one.
_INSTALL_HINT = "Install the tracing extra: pip install 'databricks-mason[tracing]'"


def _mlflow():
    """Import mlflow lazily so the core CLI (and offline wheel) don't depend on it."""
    try:
        import mlflow  # noqa: PLC0415 - intentional lazy import

        return mlflow
    except ImportError as exc:
        raise AgentCliError(
            "MLflow is required for `mason tracing` setup/list/get.",
            hint=_INSTALL_HINT,
        ) from exc


def _uc_trace_symbols():
    """Import the version-specific UC trace-location symbols, with a clean install hint."""
    try:
        from mlflow.entities import UCSchemaLocation  # noqa: PLC0415 - version-specific
        from mlflow.tracing import set_experiment_trace_location  # noqa: PLC0415

        return UCSchemaLocation, set_experiment_trace_location
    except ImportError as exc:
        raise AgentCliError(
            "This MLflow version is too old for `mason tracing setup` (UC trace destinations).",
            hint=_INSTALL_HINT,
        ) from exc


def _configure(mlflow, profile: Optional[str], warehouse_id: Optional[str]) -> None:
    """Point MLflow at the workspace (honoring mason's --profile), with a warehouse for UC ops."""
    mlflow.set_tracking_uri(f"databricks://{profile}" if profile else "databricks")
    if warehouse_id:
        os.environ[TRACES_WAREHOUSE_ENV] = warehouse_id


def ensure_uc_experiment(
    profile: Optional[str], experiment_name: str, catalog_schema: str, warehouse_id: Optional[str]
) -> str:
    """Create ``experiment_name`` if missing and link it to the UC ``catalog.schema`` (idempotent).

    A UC destination can only be linked to an experiment with no existing traces, so this links a
    freshly created experiment; a re-deploy (experiment already linked) is a no-op, and an experiment
    that already holds non-UC traces raises a clear error pointing at the migration docs. Returns the
    experiment **id** (used to build the MLflow experiment UI link shown by dev/deploy).
    """
    mlflow = _mlflow()
    _configure(mlflow, profile, warehouse_id)
    catalog, _, schema = catalog_schema.partition(".")
    uc_schema_location, set_location = _uc_trace_symbols()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment else mlflow.create_experiment(experiment_name)
    )
    try:
        set_location(
            location=uc_schema_location(catalog_name=catalog, schema_name=schema),
            experiment_id=experiment_id,
        )
    except Exception as exc:  # noqa: BLE001 - mlflow raises a generic error; classify by message
        text = str(exc).lower()
        if "contains traces" in text:
            raise AgentCliError(
                f"Experiment {experiment_name!r} already has non-UC traces, so it can't be linked "
                "to Unity Catalog.",
                hint=f"Migrate the existing traces to UC: {_MIGRATE_DOCS}",
            ) from exc
        if "already" in text:
            return experiment_id  # already linked (re-deploy) - idempotent
        raise AgentCliError(
            f"Could not link {experiment_name!r} to {catalog_schema}: {exc}"
        ) from exc
    return experiment_id


def project_trace_location(source: str) -> tuple[Optional[str], Optional[str]]:
    """The (UC schema, warehouse id) bound in the project's agent.toml, or (None, None).

    Cheap (a small TOML read, no workspace call), so callers can decide whether tracing is even
    configured before touching the client - `mason dev` uses it to stay auth-free when it isn't.
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415 - avoid import cycle

    try:
        project = AgentProject.load(source)
    except AgentCliError:
        return None, None
    return project.trace_location, project.trace_warehouse


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


def _trace_json(trace: Any) -> dict:
    return {
        "trace_id": _attr(trace, "info.trace_id", "info.request_id"),
        "status": _status_str(_attr(trace, "info.status", "info.state")),
        "execution_time_ms": _attr(trace, "info.execution_time_ms", "info.execution_duration_ms"),
        "timestamp_ms": _attr(trace, "info.timestamp_ms", "info.request_time"),
    }


# --- group ------------------------------------------------------------------


@click.group()
def tracing() -> None:
    """Configure where your agent's MLflow traces go, and inspect them."""


# --- setup: record the UC trace destination ---------------------------------


@tracing.command("setup")
@click.option(
    "--trace-location",
    "trace_location",
    required=True,
    help="Unity Catalog schema 'catalog.schema' where deployed traces are stored.",
)
@click.option(
    "--warehouse-id",
    default=None,
    help="SQL warehouse id for creating/querying the UC trace tables "
    "(MLFLOW_TRACING_SQL_WAREHOUSE_ID).",
)
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Project directory containing agent.toml. Defaults to the current directory.",
)
@click.pass_obj
def tracing_setup(obj, trace_location, warehouse_id, source) -> None:
    """Turn on Unity Catalog tracing by recording a UC schema in agent.toml.

    Records the ``catalog.schema`` (and optional warehouse). From then on both ``mason dev`` and
    ``mason deploy`` send the agent's traces to that schema, creating a per-app UC-linked experiment
    there. Until this is run, tracing is off (neither command blocks on it).
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415

    location = validate_uc_schema(trace_location)
    project = AgentProject.load(pathlib.Path(source))
    project.bind_trace_location(location, warehouse_id)
    project.write()

    if obj.output == "json":
        render.emit_json({"trace_location": location, "warehouse_id": warehouse_id})
        return
    fields = {"Trace location": location}
    if warehouse_id:
        fields["SQL warehouse"] = warehouse_id
    render.success(
        f"UC tracing configured: {location}",
        fields=fields,
        next_steps=[
            ("mason deploy <name>", "Creates the UC-linked experiment and deploys"),
            (f"mason tracing list --trace-location {location}", "Read traces at this location"),
        ],
    )


# --- list / get -------------------------------------------------------------


@tracing.command("list")
@click.option(
    "--trace-location",
    "trace_location",
    default=None,
    help="Trace location to read: a UC 'catalog.schema', or an experiment id/path. Defaults to the "
    "project's configured UC schema (from `mason tracing setup`).",
)
@click.option(
    "--warehouse-id",
    default=None,
    help="SQL warehouse id for querying UC-backed traces (default: the project's configured "
    "warehouse, MLFLOW_TRACING_SQL_WAREHOUSE_ID, or the workspace default).",
)
@click.option("--limit", type=int, default=20)
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help="Project directory to resolve the default trace location from (default: current dir).",
)
@click.pass_obj
def tracing_list(obj, trace_location, warehouse_id, limit, source) -> None:
    """List recent agent traces at a trace location.

    Resolution order: ``--trace-location`` (works standalone), else the project's configured UC
    schema (``mason tracing setup``). Tracing is UC-only, so a UC schema is queried through a SQL
    warehouse.
    """
    configured_location, configured_warehouse = project_trace_location(source)
    location = trace_location or configured_location
    if not location:
        raise AgentCliError(
            "No trace location configured for this project.",
            hint="Run `mason tracing setup --trace-location <catalog.schema>` first, or pass "
            "--trace-location.",
        )

    warehouse = warehouse_id or configured_warehouse or os.getenv(TRACES_WAREHOUSE_ENV)
    mlflow = _mlflow()
    _configure(mlflow, obj.profile, warehouse)
    traces = mlflow.search_traces(
        locations=[_resolve_location(mlflow, location)], max_results=limit, return_type="list"
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
        f"Agent Traces · {location}",
        [("Trace ID", "left"), ("Status", "left"), ("Latency (ms)", "left"), ("Created", "left")],
        rows,
    )


def _resolve_location(mlflow, location: str) -> str:
    """Turn a location spec into what search_traces wants: a UC schema or an experiment id.

    ``catalog.schema`` and bare numeric ids pass through; an experiment path (``/Users/...``) is
    resolved to its id.
    """
    if _UC_SCHEMA.match(location) or location.isdigit():
        return location
    experiment = mlflow.get_experiment_by_name(location)
    if experiment is None:
        raise AgentCliError(f"No experiment found at {location!r}.")
    return experiment.experiment_id


@tracing.command("get")
@click.argument("trace_id")
@click.pass_obj
def tracing_get(obj, trace_id) -> None:
    """Get a single trace by id (status, latency, span count, previews).

    Needs only the id: a v4 trace id (``trace:/<catalog.schema>/<id>``) is self-locating.
    """
    mlflow = _mlflow()
    _configure(mlflow, obj.profile, os.getenv(TRACES_WAREHOUSE_ENV))
    trace = mlflow.get_trace(trace_id)
    if trace is None:
        raise AgentCliError(f"No trace found with id {trace_id!r}.")
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
