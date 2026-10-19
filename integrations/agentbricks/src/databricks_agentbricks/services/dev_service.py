"""The framework-agnostic workflow behind ``agentbricks dev``.

``DevService`` prepares a temporary, local-only Apps manifest and hands the resulting plan to
the CLI for presentation and execution.  The service deliberately has no Click or presentation
dependencies: local tracing and the Apps command are ports supplied by the command composition.
In particular, preparing a project never resolves a workspace client or provisions a resource.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.app_manifest import AppManifest
from databricks_agentbricks.projects.config import require_managed_tool_support
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentkit.runtime.cors import CORS_ALLOWED_ORIGINS_ENV
from databricks_agentkit.runtime.store import RUNTIME_STORE_LOCAL_ENV
from databricks_agentkit.runtime.tool_manifest import MEMORY_STORE_ENV, SESSION_STORE_ENV

# Default local port; ``databricks apps run-local`` listens here unless ``--app-port`` overrides it.
_DEFAULT_APP_PORT = 8000
_LOCAL_APP_YAML = "app.agentbricksdev.yaml"

# Env vars that pin a package index for the *deployed* Apps build (a cloud-only workaround, see
# ``agentbricks deploy``).  They point at an index the deploying environment can reach, which is
# not necessarily reachable from the local dev machine, so local ``uv`` builds must ignore them.
_BUILD_INDEX_ENVS = frozenset({"PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX"})

# Workspace-managed resource env written by ``agentbricks deploy``.  A dev run creates no workspace
# resources: tracing is local, memory is off, and sessions are in-process.  Remove these entries so
# a previous deployment cannot silently pull a local run onto workspace resources.
_DEPLOY_TRACING_ENVS = frozenset(
    {"MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "MLFLOW_TRACING_DESTINATION"}
)
_DEPLOY_RESOURCE_ENVS = _DEPLOY_TRACING_ENVS | {MEMORY_STORE_ENV, SESSION_STORE_ENV}


@dataclass(frozen=True)
class DevRequest:
    """Inputs for one ``agentbricks dev`` invocation."""

    source: str
    prepare_environment: bool | None
    app_port: int | None


@dataclass(frozen=True)
class DevPreview:
    """Facts resolved while preparing a local run, for presentation by the caller."""

    source_dir: pathlib.Path
    port: int
    server: AgentServer | None
    tracing_uri: str | None
    local_experiment_name: str | None
    memory_store: str | None
    session_store: str | None
    trace_experiment: str | None
    has_chat_ui: bool


@dataclass(frozen=True)
class DevPlan:
    """The prepared local manifest and execution options for one run."""

    preview: DevPreview
    entry_point: pathlib.Path
    prepare_environment: bool
    requested_port: int | None


class LocalTracing(Protocol):
    """Port for the best-effort local tracing process used by a dev run."""

    def start(self, source_dir: pathlib.Path) -> tuple[subprocess.Popen | None, dict[str, str]]:
        """Start tracing for ``source_dir`` and return its process and manifest env."""

    def stop(self, server: subprocess.Popen) -> None:
        """Stop a process returned by :meth:`start`."""


class DevService:
    """Prepare and run an Agent Bricks project locally.

    The service owns validation, project metadata, the local-only manifest, and lifecycle cleanup.
    It does not own output or a workspace client; those concerns are supplied by the CLI through
    ``LocalTracing`` and ``AppsClient``. ``port_in_use`` reports whether a local port is taken, so
    tests can simulate a busy port without binding a real socket.
    """

    def __init__(
        self,
        project_resolver: ProjectResolver,
        apps_client: AppsClient,
        local_tracing: LocalTracing,
        port_in_use: Callable[[int], bool],
    ) -> None:
        self._project_resolver = project_resolver
        self._apps_client = apps_client
        self._local_tracing = local_tracing
        self._port_in_use = port_in_use

    @contextmanager
    def prepare(self, request: DevRequest) -> Iterator[DevPlan]:
        """Prepare a local run and clean up its temporary resources on context exit.

        The yielded plan is valid while the context is open.  In particular, the temporary
        ``app.agentbricksdev.yaml`` remains in place while presentation and ``run`` execute, then
        is removed along with the local tracing process even when either of those operations fails.
        """
        source_dir = pathlib.Path(request.source)
        app_yaml = source_dir / "app.yaml"
        if not app_yaml.exists():
            raise AgentCliError(
                f"No app.yaml found at {app_yaml}.",
                hint="Run from a scaffolded project, or pass --source <dir> (see `agentbricks init`).",
            )

        project = self._project_resolver.load(source_dir)
        if project is not None and project.tools:
            require_managed_tool_support(source_dir)

        # Fail before starting tracing or writing the local manifest: `apps run-local` does not
        # report a busy port clearly, and a stopped `agentbricks dev` can leave its server running.
        port = request.app_port or _DEFAULT_APP_PORT
        if self._port_in_use(port):
            raise AgentCliError(
                f"Port {port} is already in use (possibly a previous agentbricks dev run).",
                hint="Stop that process, or run `agentbricks dev --app-port <port>` with a free port.",
            )

        # Resource bindings are read only for the preview/presentation.  Dev is fully local and
        # deliberately does not provision, fetch, or otherwise contact any workspace resource.
        memory_store, session_store, trace_experiment = self._project_resolver.resource_bindings(
            source_dir
        )

        # The adapter owns best-effort launch behavior (the CLI adapter delegates to the existing
        # helper, which degrades to ``(None, {})`` when local tracing cannot start).
        tracing_server, tracing_env = self._local_tracing.start(source_dir)
        entry_point: pathlib.Path | None = None
        try:
            # Repeat runs reuse an existing environment; an explicit request overrides that
            # auto-detection exactly as the old Click command did.
            prepare_environment = request.prepare_environment
            if prepare_environment is None:
                prepare_environment = not (source_dir / ".venv").exists()

            entry_point = _dev_entry_point(
                app_yaml,
                tracing_env or None,
                allowed_origins=project.allowed_origins if project is not None else None,
            )
            preview = DevPreview(
                source_dir=source_dir,
                # Preserve the CLI's historical display/default behavior while retaining the
                # caller's explicit value separately in ``requested_port``.
                port=port,
                server=project.server if project else None,
                tracing_uri=tracing_env.get("MLFLOW_TRACKING_URI"),
                local_experiment_name=tracing_env.get("MLFLOW_EXPERIMENT_NAME"),
                memory_store=memory_store,
                session_store=session_store,
                trace_experiment=trace_experiment,
                has_chat_ui=(source_dir / "runtime" / "ui.py").is_file(),
            )
            yield DevPlan(
                preview=preview,
                entry_point=entry_point,
                prepare_environment=prepare_environment,
                requested_port=request.app_port,
            )
        finally:
            # Keep cleanup independent: a failure removing the manifest must not orphan the tracing
            # process, and a setup/run/presentation exception must not leave either resource behind.
            try:
                if entry_point is not None:
                    entry_point.unlink(missing_ok=True)
            finally:
                if tracing_server is not None:
                    self._local_tracing.stop(tracing_server)

    def run(self, plan: DevPlan) -> None:
        """Run a prepared plan through the injected Apps local-run client."""
        self._apps_client.run_local(
            plan.preview.source_dir,
            plan.entry_point.name,
            prepare_environment=plan.prepare_environment,
            app_port=plan.requested_port,
        )


def _dev_entry_point(
    app_yaml: pathlib.Path,
    extra_env: dict[str, str] | None = None,
    *,
    allowed_origins: tuple[str, ...] | None = None,
) -> pathlib.Path:
    """Write the local-only manifest consumed by ``apps run-local``.

    The deployable ``app.yaml`` is never mutated.  This copy forces the Runtime Store into local
    mode, removes deploy-only package-index overrides and workspace resource env, and merges local
    tracing settings supplied by the adapter.
    """
    manifest = AppManifest.parse(app_yaml.read_text(), source=app_yaml)
    env = manifest.raw_env()
    filtered = [e for e in env if not (isinstance(e, dict) and e.get("name") in _BUILD_INDEX_ENVS)]
    filtered = [
        e
        for e in filtered
        if not (isinstance(e, dict) and e.get("name") == RUNTIME_STORE_LOCAL_ENV)
    ]
    filtered = [
        e for e in filtered if not (isinstance(e, dict) and e.get("name") in _DEPLOY_RESOURCE_ENVS)
    ]
    for name, value in (extra_env or {}).items():
        filtered = [e for e in filtered if not (isinstance(e, dict) and e.get("name") == name)]
        filtered.append({"name": name, "value": value})
    if allowed_origins is not None:
        filtered = [
            e
            for e in filtered
            if not (isinstance(e, dict) and e.get("name") == CORS_ALLOWED_ORIGINS_ENV)
        ]
        if allowed_origins:
            filtered.append(
                {"name": CORS_ALLOWED_ORIGINS_ENV, "value": json.dumps(allowed_origins)}
            )
    filtered.append({"name": RUNTIME_STORE_LOCAL_ENV, "value": "true"})
    manifest.set_env(filtered)

    # The Apps CLI rejects hidden or hyphenated entry-point filenames.
    dev_yaml = app_yaml.parent / _LOCAL_APP_YAML
    try:
        # Overwritten on purpose: the name is dev's own, and a run stopped by a signal (an IDE stop
        # button, `kill`) exits without its cleanup and leaves the file behind.
        with dev_yaml.open("w") as output:
            output.write(manifest.to_yaml())
    except OSError as exc:
        # Don't leave a partial manifest for `apps run-local` to pick up.
        dev_yaml.unlink(missing_ok=True)
        raise AgentCliError(f"Could not write {dev_yaml}: {exc}") from exc
    return dev_yaml
