"""`mason tracing` — send an agent's traces to an MLflow experiment and inspect them.

``mason dev`` runs a **local, sqlite-backed MLflow server** (:func:`start_local_tracing_server`) so
traces stay on the machine with no workspace setup, viewable in the local MLflow UI. ``mason deploy``
sends traces to a per-project **workspace** experiment whose **name** lives in agent.toml (a default
``/Shared/mason_traces/<project>`` is written at ``mason init``); deploy get-or-creates it in the active
workspace and wires its ``MLFLOW_EXPERIMENT_ID`` (which binds the agent, grants the deployed app via an
experiment app resource, reads traces, and builds the UI link). Storing the name (not the id) keeps
the binding valid across workspaces and profiles, since an id is workspace-local.

The bound ``experiment_name``'s presence IS the enable switch: a bound name means tracing is on for
deploy, its absence means off. ``mason tracing bind`` binds an experiment (by name or id, one
required); ``mason tracing unbind`` removes the binding; ``list`` / ``get`` read traces back.

A bound experiment may be UC-backed (traces stored in Unity Catalog OTEL tables): deploy supports
exporting to one by granting the app's service principal MODIFY on those tables, and ``list`` /
``get`` read UC traces back through a SQL warehouse (``--warehouse`` or the
``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` env var).

MLflow (``mlflow-skinny``) is a base dependency, but the ``mason tracing`` commands and the deploy
experiment provisioning still import it lazily — ``cli.py`` imports this module at startup, so a
top-level import would pay mlflow's heavy import cost on every ``mason`` command.
"""

from __future__ import annotations

import os
import pathlib
import re
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional

import click

from databricks_mason import render, timefmt
from databricks_mason.errors import AgentCliError

_BREADCRUMB = "Agent Tracing"
# Per-app experiment folder under /Shared: username-free (so `mason init` can name it offline) and
# workspace-independent (so the same name is valid in whatever workspace the active profile targets).
_TRACES_DIR = "mason_traces"

# The two env vars the deployed/dev agent reads to enable tracing: a destination (the workspace) and
# an experiment (by id). MLflow turns tracing on only when it has both.
TRACES_TRACKING_URI_ENV = "MLFLOW_TRACKING_URI"
TRACES_EXPERIMENT_ID_ENV = "MLFLOW_EXPERIMENT_ID"

# The command that binds (enables) tracing. Referenced parameter-free by deploy's "deployed without
# tracing" guidance and by `unbind`'s "turn it back on" step, so those hints can't go stale if the
# flags change; the command's own `--help` documents the flags.
TRACING_BIND_COMMAND = "mason tracing bind"


def default_experiment_name(project: Optional[str], token: Optional[str] = None) -> str:
    """The default MLflow experiment path for a project: ``/Shared/mason_traces/<project>[-<token>]``.

    Under ``/Shared`` (not a user home) so it needs no username — ``mason init`` writes it offline —
    and it's workspace-independent, so the same name is valid in whatever workspace the active profile
    targets (``mason deploy`` get-or-creates it there). ``project`` is the Mason project name (the
    source directory's basename); an optional per-scaffold ``token`` (shared with the default store
    names) keeps like-named projects from colliding in the shared ``/Shared`` namespace.
    """
    if not project:
        raise AgentCliError("Cannot derive the default tracing experiment without a project name.")
    slug = re.sub(r"[^a-z0-9-]+", "-", project.lower()).strip("-") or "agent"
    middle = f"-{token}" if token else ""
    return f"/Shared/{_TRACES_DIR}/{slug}{middle}"


def experiment_url(host: Optional[str], experiment_id: str) -> Optional[str]:
    """The workspace MLflow experiment Traces page, or None when the host is unavailable."""
    if not host or host == "unknown":
        return None
    return f"{host.rstrip('/')}/ml/experiments/{experiment_id}?compareRunsMode=TRACES"


def _mlflow():
    """Import mlflow lazily and return the module.

    mlflow-skinny is a base dependency, so this can't fail on a correct install; the lazy import
    exists only to keep it off the CLI startup path (``cli.py`` imports this module eagerly).
    """
    import mlflow  # noqa: PLC0415 - intentional lazy import (startup cost, not optionality)

    return mlflow


def _workspace_uri(profile: Optional[str]) -> str:
    """The MLflow tracking URI for the workspace (honoring mason's --profile)."""
    return f"databricks://{profile}" if profile else "databricks"


def _set_tracking_uri(mlflow, profile: Optional[str]) -> None:
    """Point MLflow at the workspace (honoring mason's --profile)."""
    mlflow.set_tracking_uri(_workspace_uri(profile))


# An experiment linked to a UC schema for trace storage carries this tag (the destination schema);
# managed experiments don't. UC-backed experiments are supported for trace export at deploy (the
# app's service principal is granted MODIFY on their UC OTEL tables) and for reads via `list`/`get`
# through a SQL warehouse.
_UC_TRACE_TAG = "mlflow.experiment.databricksTraceDestinationPath"


def _is_uc_backed(experiment) -> bool:
    """True if the experiment stores traces in Unity Catalog rather than the managed MLflow backend.

    An MLflow ``Experiment``'s ``tags`` is always a dict (empty when it has none); ``or {}`` just
    guards a missing attribute.
    """
    return _UC_TRACE_TAG in (getattr(experiment, "tags", None) or {})


# The "unified" trace-table tag names a read-side VIEW over the base OTEL tables, not a writable base
# table (MODIFY on a view fails), so it's excluded from the grant set.
_UC_TRACE_UNIFIED_TAG = "mlflow.experiment.databricksTraceStorageTable"

# The per-kind base-table tags: each value is the fully-qualified ``<catalog>.<schema>.<table>`` the
# deployed app must be granted MODIFY on.
_UC_TRACE_SPAN_TAG = "mlflow.experiment.databricksTraceSpanStorageTable"
_UC_TRACE_LOG_TAG = "mlflow.experiment.databricksTraceLogStorageTable"
_UC_TRACE_ANNOTATION_TAG = "mlflow.experiment.databricksTraceAnnotationStorageTable"


@dataclass(frozen=True)
class MLflowTraceTables:
    """The UC OTEL base tables backing a UC experiment's traces (fully-qualified names; None when absent)."""

    spans: Optional[str] = None
    logs: Optional[str] = None
    annotations: Optional[str] = None

    def otel_tables(self) -> tuple[tuple[str, str], ...]:
        """Present (kind, fully_qualified_table) pairs in a stable order, for grants + resource naming."""
        return tuple(
            (kind, name)
            for kind, name in (
                ("spans", self.spans),
                ("logs", self.logs),
                ("annotations", self.annotations),
            )
            if name
        )

    def __bool__(self) -> bool:
        return bool(self.otel_tables())


def uc_trace_tables(experiment) -> MLflowTraceTables:
    """The UC OTEL base tables backing a UC experiment's traces; empty when managed.

    A UC-backed experiment stores traces in Unity Catalog base tables the deployed app must be granted
    MODIFY on (spans, logs, annotations). MLflow records each as a per-kind storage-table tag whose
    value is the fully-qualified ``<catalog>.<schema>.<table>`` - those tags are the source of truth
    (the ``UnityCatalog`` entity only surfaces spans/logs, so reading tags is what catches the
    annotations table). Reading these three specific tags naturally excludes the "unified"
    ``databricksTraceStorageTable`` tag, which names a read-side VIEW, not a writable table; the
    modeling is intentionally explicit per kind, so a new OTEL kind would be a new field, not another
    discovered tag. Falls back to deriving names from the destination path when the per-kind tags are
    absent. Returns an empty ``MLflowTraceTables`` for a managed experiment.
    """
    tags = getattr(experiment, "tags", None) or {}
    if _UC_TRACE_TAG not in tags:
        return MLflowTraceTables()
    tables = MLflowTraceTables(
        spans=tags.get(_UC_TRACE_SPAN_TAG) or None,
        logs=tags.get(_UC_TRACE_LOG_TAG) or None,
        annotations=tags.get(_UC_TRACE_ANNOTATION_TAG) or None,
    )
    if tables:
        return tables
    # Fallback: derive from the destination path (`<catalog>.<schema>[.<table_prefix>]`) when the
    # per-kind storage tags aren't present.
    parts = (tags.get(_UC_TRACE_TAG) or "").split(".")
    if len(parts) == 3:
        prefix = ".".join(parts)
        return MLflowTraceTables(
            spans=f"{prefix}_otel_spans",
            logs=f"{prefix}_otel_logs",
            annotations=f"{prefix}_otel_annotations",
        )
    if len(parts) == 2:
        base = ".".join(parts)
        return MLflowTraceTables(
            spans=f"{base}.mlflow_experiment_trace_otel_spans",
            logs=f"{base}.mlflow_experiment_trace_otel_logs",
            annotations=f"{base}.mlflow_experiment_trace_otel_annotations",
        )
    return MLflowTraceTables()


# MLflow reads the warehouse a UC trace read goes through from this env var - its only supported knob
# (``get_trace`` has no warehouse parameter and ``search_traces``' ``sql_warehouse_id`` kwarg is
# deprecated) - so a UC read exports the resolved id before reading, scoped + restored by the caller
# (``_with_trace_read_target``) rather than left in the process env.
_SQL_WAREHOUSE_ENV = "MLFLOW_TRACING_SQL_WAREHOUSE_ID"


def _resolve_warehouse_id(warehouse_id: Optional[str]) -> Optional[str]:
    """The effective SQL warehouse for a UC trace read: ``--warehouse``, else the env var, else None."""
    return warehouse_id or os.environ.get(_SQL_WAREHOUSE_ENV)


def _restore_warehouse_env(prior: Optional[str]) -> None:
    """Restore ``MLFLOW_TRACING_SQL_WAREHOUSE_ID`` to its pre-read value (removed when there was none)."""
    if prior is None:
        os.environ.pop(_SQL_WAREHOUSE_ENV, None)
    else:
        os.environ[_SQL_WAREHOUSE_ENV] = prior


def _require_warehouse_for_uc_read(experiment, warehouse_id: Optional[str]) -> None:
    """Point MLflow at the SQL warehouse a UC-backed read needs; raise when none is available.

    A managed experiment needs no warehouse (and never gets one set). A UC-backed one is read through
    a warehouse MLflow takes from ``MLFLOW_TRACING_SQL_WAREHOUSE_ID``, so the resolved id is exported
    before the caller runs ``search_traces`` / ``get_trace``; with no warehouse the read would fail
    opaquely, so this raises up front rather than falling through to the local dev store. The export
    is scoped to the read: the enclosing ``_with_trace_read_target`` restores the prior value on exit.
    """
    if not _is_uc_backed(experiment):
        return
    if warehouse_id is None:
        raise AgentCliError(
            "Reading traces from a UC-backed experiment requires a SQL warehouse.",
            hint=f"Pass --warehouse <id> (or set {_SQL_WAREHOUSE_ENV}).",
        )
    os.environ[_SQL_WAREHOUSE_ENV] = warehouse_id


def _get_experiment_by_id(mlflow, experiment_id: str):
    """``mlflow.get_experiment`` but returns ``None`` for an unknown id instead of raising.

    The id lookup raises ``RESOURCE_DOES_NOT_EXIST`` for a missing experiment, whereas the name lookup
    returns ``None`` - normalize so callers treat "not found" the same either way. Other errors (auth,
    network) still propagate.
    """
    from mlflow.exceptions import MlflowException  # noqa: PLC0415 - lazy import (startup cost)

    try:
        return mlflow.get_experiment(experiment_id)
    except MlflowException as exc:
        if getattr(exc, "error_code", "") == "RESOURCE_DOES_NOT_EXIST":
            return None
        raise


def create_experiment_idempotent(
    profile: Optional[str], client, name: str
) -> tuple[str, MLflowTraceTables]:
    """Create the experiment ``name`` if missing; return its id and UC OTEL tables (idempotent).

    ``create_experiment`` won't make the intermediate workspace folder for a nested path (e.g.
    ``/Users/<you>/mason-traces/<project>``), so the parent dir is created first. Used by dev/deploy to
    provision the experiment that traces log to.

    Returns ``(experiment_id, tables)``: for an existing UC-backed experiment, ``tables`` are the UC
    OTEL base tables the deployed app must be granted MODIFY on (see ``uc_trace_tables``); for a
    managed experiment - and for one just created, since mason only creates managed experiments - it
    is an empty ``MLflowTraceTables``.
    """
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, profile)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        return experiment.experiment_id, uc_trace_tables(experiment)
    parent = name.rsplit("/", 1)[0]
    if parent:
        client.ensure_workspace_dir(parent)
    return mlflow.create_experiment(name), MLflowTraceTables()


def _resolve_experiment_name(source: pathlib.Path | str) -> Optional[str]:
    """This project's bound ``experiment_name``, or None when tracing is unbound (off).

    Resolves by name, not a stored id, so it stays correct across workspaces/profiles.
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415 - avoid import cycle

    try:
        project = AgentProject.load(source)
    except AgentCliError:
        project = None
    return project.trace_experiment_name if project is not None else None


@dataclass(frozen=True)
class _TraceReadTarget:
    """Where `list`/`get` read traces from: an MLflow ``tracking_uri`` and the ``experiment_id`` in it.

    ``local`` marks the local ``mason dev`` store (served over a short-lived REST server) so callers can
    label it. A field is None when there's nothing to read (no experiment resolved).
    """

    tracking_uri: Optional[str]
    experiment_id: Optional[str]
    local: bool = False


@contextmanager
def _with_trace_read_target(
    source: pathlib.Path | str,
    profile: Optional[str],
    experiment_name: Optional[str] = None,
    experiment_id: Optional[str] = None,
    warehouse_id: Optional[str] = None,
) -> Iterator[_TraceReadTarget]:
    """Yield the target `list`/`get` should read from, with MLflow's tracking URI already pointed at it.

    Precedence: an explicit ``--experiment-name`` / ``--experiment-id`` (must exist), else this
    project's workspace experiment when it's provisioned, else the local ``mason dev`` store
    (``.mason/mlflow.db``). Both target fields are None when nothing has traced anywhere yet. A
    UC-backed target is read through the resolved SQL warehouse (see
    ``_require_warehouse_for_uc_read``); managed and local reads need none.

    The local store is read over REST from a short-lived MLflow server started here (and torn down on
    exit), not by opening the sqlite file directly: the server that wrote the schema serves it, so the
    CLI's own mlflow-skinny version needn't match the version that wrote the store (the REST protocol is
    compatible across a major version; a direct sqlite open must match the schema head exactly). Callers
    must run their read inside the ``with`` block so the server is still up.
    """
    mlflow = _mlflow()
    # A UC read's warehouse is exported to MLFLOW_TRACING_SQL_WAREHOUSE_ID (see
    # _require_warehouse_for_uc_read); scope that mutation to the read by restoring the prior value
    # on exit, whatever path was taken.
    prior_warehouse = os.environ.get(_SQL_WAREHOUSE_ENV)
    try:
        # An explicit id/name targets the workspace directly (and must exist); it raises here, before
        # any local server is started, so a typo never spins one up.
        explicit = _workspace_experiment_target(
            profile, experiment_name, experiment_id, warehouse_id
        )
        if explicit is not None:
            mlflow.set_tracking_uri(explicit.tracking_uri)
            yield explicit
            return
        # This project's bound experiment, if it's provisioned in the workspace.
        name = _resolve_experiment_name(source)
        if name:
            _set_tracking_uri(mlflow, profile)
            try:
                experiment = mlflow.get_experiment_by_name(name)
            except Exception:  # noqa: BLE001 - workspace unreachable -> try the local dev store
                experiment = None
            if experiment is not None:
                _require_warehouse_for_uc_read(experiment, warehouse_id)
                yield _TraceReadTarget(_workspace_uri(profile), experiment.experiment_id)
                return
        # Unbound, not provisioned, or unreachable -> the local dev store, if any. `mason dev` traces
        # locally regardless of the binding, so its store is worth reading even when tracing is
        # unbound.
        db = pathlib.Path(source).resolve() / _MASON_LOCAL_DIR / "mlflow.db"
        if not db.exists():
            yield _TraceReadTarget(None, None)
            return
        server, base_url = _start_read_server(db)
        if server is None:  # best-effort: couldn't bring the local store up -> nothing to read
            yield _TraceReadTarget(None, None, local=True)
            return
        try:
            mlflow.set_tracking_uri(base_url)
            local = mlflow.get_experiment_by_name(pathlib.Path(source).resolve().name)
            yield _TraceReadTarget(base_url, local.experiment_id if local else None, local=True)
        finally:
            stop_local_tracing_server(server)
    finally:
        _restore_warehouse_env(prior_warehouse)


def _workspace_experiment_target(
    profile: Optional[str],
    experiment_name: Optional[str],
    experiment_id: Optional[str],
    warehouse_id: Optional[str] = None,
) -> Optional[_TraceReadTarget]:
    """The workspace target for an explicit ``--experiment-id`` / ``--experiment-name``, or None when
    neither is given (the caller falls back to the project default).

    An explicit identifier must exist: an unknown id or name raises rather than resolving to nothing,
    so a typo isn't mistaken for an empty experiment. A UC-backed target needs the resolved SQL
    warehouse (``_require_warehouse_for_uc_read``).
    """
    if not (experiment_id or experiment_name):
        return None
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, profile)
    if experiment_id:
        experiment = _get_experiment_by_id(mlflow, experiment_id)
        if experiment is None:
            raise AgentCliError(
                f"No MLflow experiment found with id {experiment_id!r} in this workspace.",
                hint="Check the id, or omit it to use this project's experiment.",
            )
        _require_warehouse_for_uc_read(experiment, warehouse_id)
        return _TraceReadTarget(_workspace_uri(profile), experiment_id)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise AgentCliError(
            f"No MLflow experiment named {experiment_name!r} in this workspace.",
            hint="Check the name, or omit it to use this project's experiment.",
        )
    _require_warehouse_for_uc_read(experiment, warehouse_id)
    return _TraceReadTarget(_workspace_uri(profile), experiment.experiment_id)


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


# --- local dev tracing (`mason dev`) ----------------------------------------

# Local-only scratch dir (gitignored) under a dev project: holds the sqlite tracing store + artifacts.
_MASON_LOCAL_DIR = ".mason"
# The local tracing server's MLflow: a broad 3.x range (the runtime's floor). `mason dev` writes the
# sqlite store with it, and `list`/`get` read that store back over REST from a short-lived server (see
# _with_trace_read_target) - never by opening the file with the CLI's own mlflow-skinny. So this needn't
# match the CLI's version, and uvx reuses ONE cached environment across mason releases (fast startup
# after the first install) instead of cold-installing a new exact version on every mlflow bump.
_MLFLOW_SPEC = "mlflow>=3.10,<4"
# Pin the uvx server's interpreter: on Python 3.13 an older mlflow drags in a pyarrow that builds from
# source (slow, and fails without a C toolchain); 3.12 resolves to prebuilt wheels. uv fetches it if the
# machine lacks it.
_SERVER_PYTHON = "3.12"


def _free_port() -> int:
    """Ask the OS for a free localhost port for the local MLflow tracking server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _mlflow_server_argv(db: pathlib.Path, artifacts: pathlib.Path, port: int) -> list[str]:
    """The ``uvx`` argv for a local, sqlite-backed MLflow tracking server on ``port``.

    Shared by the `mason dev` server and the short-lived server `list`/`get` use to read the local store
    (see _MLFLOW_SPEC / _SERVER_PYTHON for the version + interpreter pins).
    """
    return [
        "uvx",
        "--python",
        _SERVER_PYTHON,
        "--from",
        _MLFLOW_SPEC,
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{db}",
        "--default-artifact-root",
        str(artifacts),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def _wait_for_server(base_url: str, server: subprocess.Popen, timeout: float = 60.0) -> bool:
    """Poll the local MLflow server's health endpoint until it answers, its process exits, or timeout.

    `mason dev` doesn't wait (the agent traces to the server as it comes up), but a read command queries
    immediately and then tears the server down, so it must block until the server is live. Returns True
    once it responds, False if the process died (e.g. install/bind failure) or it never came up.
    """
    import urllib.error  # noqa: PLC0415 - only needed when reading the local store
    import urllib.request  # noqa: PLC0415

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:  # process exited before serving -> it won't come up
            return False
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.3)  # not up yet (connection refused / starting) -> retry
    return False


def _start_read_server(db: pathlib.Path) -> tuple[subprocess.Popen | None, Optional[str]]:
    """Start a short-lived MLflow server over the existing ``mason dev`` store and wait until it's ready.

    Lets `list`/`get` read local traces over REST (the server owns the sqlite schema). Returns
    ``(server, base_url)``, or ``(None, None)`` when it can't start - best-effort, like `mason dev`, so a
    read degrades to showing nothing rather than erroring. The caller stops the server when done.
    """
    artifacts = (db.parent / "mlartifacts").resolve()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    try:
        # Dup the log fd to the child and close our copy; keeps the server's output for debugging.
        with (db.parent / "mlflow-read.log").open("w") as log:
            server = subprocess.Popen(
                _mlflow_server_argv(db, artifacts, port), stdout=log, stderr=subprocess.STDOUT
            )
    except OSError as exc:
        render.diagnostic(
            "warning", f"could not read local traces - {exc}", help="is `uv` installed?"
        )
        return None, None
    if not _wait_for_server(base_url, server):
        stop_local_tracing_server(server)
        render.diagnostic(
            "warning", "local trace store did not come up", help="see .mason/mlflow-read.log"
        )
        return None, None
    return server, base_url


def start_local_tracing_server(
    source_dir: pathlib.Path,
) -> tuple[subprocess.Popen | None, dict[str, str]]:
    """Start a local, sqlite-backed MLflow tracking server for `mason dev` tracing.

    Returns ``(server_process, env)`` — ``env`` carries the ``MLFLOW_*`` vars for the dev-only manifest
    so the agent traces locally — or ``(None, {})`` when the server can't be started (dev then runs
    without traces). Launched via ``uvx mlflow`` so it depends on neither the skinny CLI env nor the
    agent venv; it stores traces in ``<source>/.mason/mlflow.db`` and serves the trace UI + REST API on
    a free localhost port. Because the server owns the sqlite schema, there is no client/server
    migration mismatch. The agent picks up the env through the same runtime gate a deployment uses (a
    destination + an experiment), so no agent code differs between dev and deploy.
    """
    mason_dir = source_dir / _MASON_LOCAL_DIR
    try:
        mason_dir.mkdir(exist_ok=True)
        db = (mason_dir / "mlflow.db").resolve()
        artifacts = (mason_dir / "mlartifacts").resolve()
        port = _free_port()
        # Passing the log file into Popen dups its fd to the child; closing our copy here is safe and
        # keeps the server's output for debugging a failed local-tracing run.
        with (mason_dir / "mlflow-server.log").open("w") as log:
            server = subprocess.Popen(
                _mlflow_server_argv(db, artifacts, port), stdout=log, stderr=subprocess.STDOUT
            )
    except OSError as exc:
        render.diagnostic(
            "warning",
            f"local tracing not started — {exc}",
            help="running without traces (is `uv` installed?)",
        )
        return None, {}
    return server, {
        "MLFLOW_TRACKING_URI": f"http://127.0.0.1:{port}",
        # A bare experiment name is fine for local MLflow (no workspace path / username needed).
        "MLFLOW_EXPERIMENT_NAME": source_dir.resolve().name,
    }


def stop_local_tracing_server(server: subprocess.Popen) -> None:
    """Stop the local MLflow tracking server started for `mason dev`."""
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()


# --- group ------------------------------------------------------------------


@click.group()
def tracing() -> None:
    """Configure MLflow tracing for your deployed agents, and inspect the traces."""


def _check_experiment_flags(
    experiment_name: Optional[str], experiment_id: Optional[str], *, require_one: bool = False
) -> None:
    """Validate the mutually-exclusive ``--experiment-name`` / ``--experiment-id`` selectors.

    Both given is always an error. ``require_one`` additionally rejects *neither* - `bind` needs one to
    set the binding, while `list`/`get` may omit both and fall back to this project's experiment.
    """
    if experiment_name and experiment_id:
        raise AgentCliError(
            "Pass --experiment-name or --experiment-id, not both.",
            hint="They select the same experiment; use whichever identifier you have.",
        )
    if require_one and not (experiment_name or experiment_id):
        raise AgentCliError(
            "Pass --experiment-name or --experiment-id to bind tracing to an experiment.",
            hint="Absence of a bound experiment means tracing is off; `mason tracing unbind` clears it.",
        )


def _experiment_read_options(command):
    """Add the shared ``--experiment-name`` / ``--experiment-id`` / ``--warehouse`` read options."""
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


# --- bind / unbind ----------------------------------------------------------


@tracing.command("bind")
@click.option(
    "--experiment-name",
    "experiment_name",
    default=None,
    help="MLflow experiment name to trace to - an absolute workspace path, e.g. "
    "/Shared/mason_traces/<agent> or /Users/<you>/mason_traces/<agent>. mason get-or-creates it at "
    "deploy. Mutually exclusive with --experiment-id.",
)
@click.option(
    "--experiment-id",
    "experiment_id",
    default=None,
    help="MLflow experiment id (e.g. copied from the experiment's workspace URL) to trace to. "
    "Resolved to the experiment's name and stored as a name — mason persists names, not ids, so the "
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
    """Bind tracing to an experiment, by name or id. Requires one of them (like `mason memory/sessions
    bind`); the binding's presence is what turns tracing on.

    The experiment is stored as a NAME, not an id, so the binding stays valid across
    workspaces/profiles — mason get-or-creates it in the active workspace at deploy. ``--experiment-id``
    (e.g. from the experiment's URL) is a convenience: it's resolved to the experiment's name and
    stored as a name, never as an id.
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415

    _check_experiment_flags(experiment_name, experiment_id, require_one=True)

    # The name to store. --experiment-id is resolved to the experiment's name (mason persists names,
    # not ids). A UC-backed experiment is supported for trace export: deploy grants the app's service
    # principal MODIFY on its UC OTEL tables (see app_resources.apply_trace_resources).
    name = experiment_name
    if experiment_id:
        mlflow = _mlflow()
        _set_tracking_uri(mlflow, obj.profile)
        experiment = _get_experiment_by_id(mlflow, experiment_id)
        if experiment is None:
            raise AgentCliError(
                f"No MLflow experiment found with id {experiment_id!r}.",
                hint="Pass an existing experiment id, or use --experiment-name.",
            )
        name = experiment.name
    elif experiment_name:
        if not experiment_name.startswith("/"):
            raise AgentCliError(
                f"Experiment name must be an absolute workspace path, got {experiment_name!r}.",
                hint="Use a path like /Shared/mason_traces/<agent> or "
                "/Users/<you>/mason_traces/<agent>.",
            )
        # A not-yet-created name is fine (deploy creates it), so no workspace lookup happens here.

    project = AgentProject.load(pathlib.Path(source))
    project.bind_tracing(name)
    project.write()

    if obj.output == "json":
        render.emit_json({"experiment_name": name})
        return
    render.success(
        f"Tracing on: experiment {name}",
        fields={"Experiment": name},
        next_steps=[
            ("mason dev", "Run locally with tracing on"),
            ("mason tracing list", "List traces once you have some"),
            ("mason tracing unbind", "Turn tracing off"),
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
    DEPLOYED agent (`mason deploy` then wires no MLflow env).

    Deploy-only: `mason dev` still traces locally to its own MLflow server, so you keep local traces
    while the deployed agent stays untraced.
    """
    from databricks_mason.agent_project import AgentProject  # noqa: PLC0415

    project = AgentProject.load(pathlib.Path(source))
    project.unbind_tracing()
    project.write()

    if obj.output == "json":
        render.emit_json({"experiment_name": None})
        return
    render.success(
        "Tracing unbound (off for the deployed agent; mason dev still traces locally)",
        next_steps=[(TRACING_BIND_COMMAND, "Turn deployed tracing back on")],
    )


# --- list / get -------------------------------------------------------------


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
    `mason deploy`), otherwise the local `mason dev` store (``.mason/mlflow.db``), so a not-yet-deployed
    dev run's traces still show up here (tagged "(local dev)"). Nothing traced anywhere yet lists
    nothing. A UC-backed experiment is read through a SQL warehouse (``--warehouse``).
    """
    _check_experiment_flags(experiment_name, experiment_id)
    mlflow = _mlflow()
    # Read inside the context manager: for a local dev store it keeps the short-lived MLflow server up
    # for the duration of the search (return_type="list" materializes the rows before it's torn down).
    with _with_trace_read_target(
        source, obj.profile, experiment_name, experiment_id, _resolve_warehouse_id(warehouse_id)
    ) as target:
        traces = []
        if target.experiment_id:
            traces = mlflow.search_traces(
                locations=[target.experiment_id], max_results=limit, return_type="list"
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
        where = " (local dev)" if target.local else ""
        render.resource_table(
            f"Agent Traces · {str(target.experiment_id) + where if target.experiment_id else 'no experiment yet'}",
            [
                ("Trace ID", "left"),
                ("Status", "left"),
                ("Latency (ms)", "left"),
                ("Created", "left"),
            ],
            rows,
        )


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

    Reads from the same place as `mason tracing list`: an explicit ``--experiment-name`` /
    ``--experiment-id`` targets that workspace store and must name one that exists (errors otherwise);
    otherwise this project's workspace experiment if provisioned, else its local `mason dev` store.
    A UC-backed experiment is read through a SQL warehouse (``--warehouse``).
    """
    _check_experiment_flags(experiment_name, experiment_id)
    mlflow = _mlflow()
    warehouse = _resolve_warehouse_id(warehouse_id)
    # Read inside the context manager so a local dev store's short-lived server stays up for the fetch.
    with _with_trace_read_target(
        source, obj.profile, experiment_name, experiment_id, warehouse
    ) as target:
        # get_trace resolves by id and needs only the tracking URI; when nothing resolved, fall back to
        # the workspace so an id still looks there.
        if target.tracking_uri is None:
            mlflow.set_tracking_uri(_workspace_uri(obj.profile))
            if warehouse:
                # UC-ness can't be known in this fallthrough, so a provided warehouse is exported for
                # the fetch (harmless for a managed experiment) and restored right after: the env var
                # is MLflow's only supported knob (get_trace takes no warehouse parameter), and the
                # read must not leave the process env mutated.
                prior = os.environ.get(_SQL_WAREHOUSE_ENV)
                os.environ[_SQL_WAREHOUSE_ENV] = warehouse
                try:
                    trace = mlflow.get_trace(trace_id)
                finally:
                    _restore_warehouse_env(prior)
            else:
                trace = mlflow.get_trace(trace_id)
        else:
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
                "Latency (ms)": _attr(
                    trace, "info.execution_time_ms", "info.execution_duration_ms"
                ),
                "Spans": len(spans),
                "Request": _attr(trace, "info.request_preview", "data.request"),
                "Response": _attr(trace, "info.response_preview", "data.response"),
                "Created": timefmt.absolute(_attr(trace, "info.timestamp_ms", "info.request_time")),
            },
            status=_status_str(_attr(trace, "info.status", "info.state")),
        )
