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

MLflow (``mlflow-skinny``) is a base dependency, but the ``mason tracing`` commands and the deploy
experiment provisioning still import it lazily — ``cli.py`` imports this module at startup, so a
top-level import would pay mlflow's heavy import cost on every ``mason`` command.
"""

from __future__ import annotations

import pathlib
import re
import socket
import subprocess
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
# managed experiments don't. mason supports managed tracing only (UC support is a follow-up).
_UC_TRACE_TAG = "mlflow.experiment.databricksTraceDestinationPath"


def _is_uc_backed(experiment) -> bool:
    """True if the experiment stores traces in Unity Catalog rather than the managed MLflow backend.

    An MLflow ``Experiment``'s ``tags`` is always a dict (empty when it has none); ``or {}`` just
    guards a missing attribute.
    """
    return _UC_TRACE_TAG in (getattr(experiment, "tags", None) or {})


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


def create_experiment_idempotent(profile: Optional[str], client, name: str) -> str:
    """Create the experiment ``name`` if missing and return its id (idempotent).

    ``create_experiment`` won't make the intermediate workspace folder for a nested path (e.g.
    ``/Users/<you>/mason-traces/<project>``), so the parent dir is created first. Used by dev/deploy to
    provision the managed experiment that traces log to.

    Rejects a name that already resolves to a UC-backed experiment: binding one is blocked up front,
    but a hand-edited ``agent.toml`` can point at one directly, so deploy re-checks here - mason
    supports managed (non-UC) tracing only.
    """
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, profile)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        if _is_uc_backed(experiment):
            raise AgentCliError(
                "UC-backed MLflow tracing is not supported by mason.",
                hint="Point this project's tracing at a managed (non-UC) experiment.",
            )
        return experiment.experiment_id
    parent = name.rsplit("/", 1)[0]
    if parent:
        client.ensure_workspace_dir(parent)
    return mlflow.create_experiment(name)


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

    A field is None when there's nothing to read (no experiment resolved).
    """

    tracking_uri: Optional[str]
    experiment_id: Optional[str]


def _trace_read_target(source: pathlib.Path | str, profile: Optional[str]) -> _TraceReadTarget:
    """The target `list`/`get` read this project's traces from: its workspace experiment when that
    exists and is managed, else the local ``mason dev`` store (``.mason/mlflow.db``).

    Both fields are None when nothing has traced anywhere yet.
    """
    mlflow = _mlflow()
    name = _resolve_experiment_name(source)
    # A bound experiment that's provisioned + managed in the workspace wins.
    if name:
        _set_tracking_uri(mlflow, profile)
        try:
            experiment = mlflow.get_experiment_by_name(name)
        except Exception:  # noqa: BLE001 - workspace unreachable -> try the local dev store
            experiment = None
        if experiment is not None and not _is_uc_backed(experiment):
            return _TraceReadTarget(_workspace_uri(profile), experiment.experiment_id)
    # Unbound, not provisioned, or unreachable -> the local dev store, if any. `mason dev` traces
    # locally regardless of the binding, so its store is worth reading even when tracing is unbound.
    db = pathlib.Path(source).resolve() / _MASON_LOCAL_DIR / "mlflow.db"
    if db.exists():
        uri = f"sqlite:///{db}"
        mlflow.set_tracking_uri(uri)
        local = mlflow.get_experiment_by_name(pathlib.Path(source).resolve().name)
        if local is not None:
            return _TraceReadTarget(uri, local.experiment_id)
    return _TraceReadTarget(None, None)


def _workspace_experiment_target(
    profile: Optional[str], experiment_name: Optional[str], experiment_id: Optional[str]
) -> Optional[_TraceReadTarget]:
    """The workspace target for an explicit ``--experiment-id`` / ``--experiment-name``, or None when
    neither is given (the caller falls back to the project default).

    An explicit identifier must exist: an unknown id or name raises rather than resolving to nothing,
    so a typo isn't mistaken for an empty experiment.
    """
    if not (experiment_id or experiment_name):
        return None
    mlflow = _mlflow()
    _set_tracking_uri(mlflow, profile)
    if experiment_id:
        if _get_experiment_by_id(mlflow, experiment_id) is None:
            raise AgentCliError(
                f"No MLflow experiment found with id {experiment_id!r} in this workspace.",
                hint="Check the id, or omit it to use this project's experiment.",
            )
        return _TraceReadTarget(_workspace_uri(profile), experiment_id)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise AgentCliError(
            f"No MLflow experiment named {experiment_name!r} in this workspace.",
            hint="Check the name, or omit it to use this project's experiment.",
        )
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
# Fallback MLflow spec for the local tracing server: the 3.x range (the runtime's floor). Normally the
# server is pinned to the CLI's own mlflow version (see _server_mlflow_spec) so the sqlite schema it
# writes matches what `mason tracing list`/`get` read back with.
_MLFLOW_SPEC = "mlflow>=3.10,<4"


def _server_mlflow_spec() -> str:
    """The ``uvx --from`` spec for the local server's MLflow.

    Pin to the CLI's installed mlflow(-skinny) release so the server writes a sqlite schema the CLI can
    read back (same version = same schema head, so `mason tracing list`/`get` open the local store
    without a migration mismatch). Fall back to the 3.x range for a dev/local build whose version isn't
    a plain ``X.Y.Z`` release on PyPI.
    """
    try:
        import mlflow  # noqa: PLC0415 - lazy, only when launching the local server

        version = mlflow.__version__
        if re.fullmatch(r"\d+\.\d+\.\d+", version):
            return f"mlflow=={version}"
    except Exception:  # noqa: BLE001 - fall back to the range
        pass
    return _MLFLOW_SPEC


def _free_port() -> int:
    """Ask the OS for a free localhost port for the local MLflow tracking server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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
                [
                    "uvx",
                    "--from",
                    _server_mlflow_spec(),
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
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
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
    """Configure MLflow tracing for your agents, and inspect the traces."""


# --- configure / disable ----------------------------------------------------


@tracing.command("bind")
@click.option(
    "--experiment-name",
    "experiment_name",
    default=None,
    help="MLflow experiment name to trace to — an absolute workspace path, e.g. "
    "/Shared/mason_traces/<agent> or /Users/<you>/mason_traces/<agent>. mason get-or-creates it at "
    "deploy. Omit to (re)enable the default /Shared experiment.",
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

    if experiment_name and experiment_id:
        raise AgentCliError(
            "Pass --experiment-name or --experiment-id, not both.",
            hint="They set the same binding; use whichever identifier you have.",
        )
    if not (experiment_name or experiment_id):
        raise AgentCliError(
            "Pass --experiment-name or --experiment-id to bind tracing to an experiment.",
            hint="Absence of a bound experiment means tracing is off; `mason tracing unbind` clears it.",
        )

    # The name to store. --experiment-id is resolved to the experiment's name (mason persists names,
    # not ids). Either way, a UC-backed experiment is rejected up front — mason supports managed
    # tracing only (UC traces need a SQL warehouse to read and UC grants for the app's SP).
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
        if _is_uc_backed(experiment):
            raise AgentCliError(
                "UC-backed MLflow tracing is not supported by mason.",
                hint="Pass a managed (non-UC) experiment.",
            )
        name = experiment.name
    elif experiment_name:
        if not experiment_name.startswith("/"):
            raise AgentCliError(
                f"Experiment name must be an absolute workspace path, got {experiment_name!r}.",
                hint="Use a path like /Shared/mason_traces/<agent> or "
                "/Users/<you>/mason_traces/<agent>.",
            )
        # A not-yet-created name is fine (deploy creates it); only reject a name that already resolves
        # to a UC-backed experiment.
        mlflow = _mlflow()
        _set_tracking_uri(mlflow, obj.profile)
        existing = mlflow.get_experiment_by_name(experiment_name)
        if existing is not None and _is_uc_backed(existing):
            raise AgentCliError(
                "UC-backed MLflow tracing is not supported by mason.",
                hint="Pass a managed (non-UC) experiment.",
            )

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
        next_steps=[
            ("mason tracing bind --experiment-name <path>", "Turn deployed tracing back on")
        ],
    )


# --- list / get -------------------------------------------------------------


@tracing.command("list")
@click.option(
    "--experiment-name",
    "experiment_name",
    default=None,
    help="MLflow experiment name to read (an absolute workspace path). Default: this project's "
    "experiment.",
)
@click.option(
    "--experiment-id",
    "experiment_id",
    default=None,
    help="MLflow experiment id to read (e.g. from the experiment URL). Mutually exclusive with "
    "--experiment-name.",
)
@click.option("--limit", type=int, default=20)
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help="Project directory to resolve the default experiment from (default: current dir).",
)
@click.pass_obj
def tracing_list(obj, experiment_name, experiment_id, limit, source) -> None:
    """List recent agent traces in an experiment.

    An explicit ``--experiment-name`` / ``--experiment-id`` reads that workspace experiment and must
    name one that exists (errors otherwise, so a typo isn't mistaken for an empty experiment). With
    neither, this project's experiment is read: the **workspace** one if it's been provisioned (by
    `mason deploy`), otherwise the local `mason dev` store (``.mason/mlflow.db``), so a not-yet-deployed
    dev run's traces still show up here (tagged "(local dev)"). Nothing traced anywhere yet lists
    nothing.
    """
    if experiment_name and experiment_id:
        raise AgentCliError(
            "Pass --experiment-name or --experiment-id, not both.",
            hint="They select the same experiment; use whichever identifier you have.",
        )
    mlflow = _mlflow()
    target = _workspace_experiment_target(
        obj.profile, experiment_name, experiment_id
    ) or _trace_read_target(source, obj.profile)
    traces = []
    if target.experiment_id:
        mlflow.set_tracking_uri(target.tracking_uri)
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
    where = " (local dev)" if (target.tracking_uri or "").startswith("sqlite:") else ""
    render.resource_table(
        f"Agent Traces · {str(target.experiment_id) + where if target.experiment_id else 'no experiment yet'}",
        [("Trace ID", "left"), ("Status", "left"), ("Latency (ms)", "left"), ("Created", "left")],
        rows,
    )


@tracing.command("get")
@click.argument("trace_id")
@click.option(
    "--experiment-name",
    "experiment_name",
    default=None,
    help="MLflow experiment name whose store holds the trace (an absolute workspace path). Default: "
    "this project's experiment.",
)
@click.option(
    "--experiment-id",
    "experiment_id",
    default=None,
    help="MLflow experiment id whose store holds the trace. Mutually exclusive with "
    "--experiment-name.",
)
@click.option(
    "--source",
    default=".",
    type=click.Path(file_okay=False),
    help="Project directory to resolve the experiment from (default: current dir).",
)
@click.pass_obj
def tracing_get(obj, trace_id, experiment_name, experiment_id, source) -> None:
    """Get a single trace by id (status, latency, span count, previews).

    Reads from the same place as `mason tracing list`: an explicit ``--experiment-name`` /
    ``--experiment-id`` targets that workspace store and must name one that exists (errors otherwise);
    otherwise this project's workspace experiment if provisioned, else its local `mason dev` store.
    """
    if experiment_name and experiment_id:
        raise AgentCliError(
            "Pass --experiment-name or --experiment-id, not both.",
            hint="They select the same experiment; use whichever identifier you have.",
        )
    mlflow = _mlflow()
    target = _workspace_experiment_target(
        obj.profile, experiment_name, experiment_id
    ) or _trace_read_target(source, obj.profile)
    mlflow.set_tracking_uri(target.tracking_uri or _workspace_uri(obj.profile))
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
