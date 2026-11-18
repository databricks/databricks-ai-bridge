"""Prepare a local agent process and local tracing without Apps or workspace provisioning."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from databricks_agentbricks.clients.local_process_client import LocalProcessClient
from databricks_agentbricks.clients.local_tracing_client import LocalTracingClient
from databricks_agentbricks.diagnostics import CommandDiagnostics, FailureStage
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.config import require_managed_tool_support
from databricks_agentbricks.projects.local_run import load_local_run
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentkit.runtime.cors import CORS_ALLOWED_ORIGINS_ENV
from databricks_agentkit.runtime.store import RUNTIME_STORE_LOCAL_ENV
from databricks_agentkit.runtime.tool_manifest import (
    DEV_BOUND_MEMORY_STORE_ENV,
    MEMORY_STORE_ENV,
    SESSION_STORE_ENV,
)

_DEFAULT_APP_PORT = 8000
_BUILD_INDEX_ENVS = frozenset({"PIP_INDEX_URL", "UV_INDEX_URL", "UV_DEFAULT_INDEX"})
_DEPLOY_RESOURCE_ENVS = frozenset(
    {
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_ID",
        "MLFLOW_EXPERIMENT_NAME",
        "MLFLOW_TRACING_DESTINATION",
        MEMORY_STORE_ENV,
        SESSION_STORE_ENV,
    }
)


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
    tracing_warning: str | None
    tracing_help: str | None
    memory_store: str | None
    session_store: str | None
    trace_experiment: str | None
    has_chat_ui: bool


@dataclass(frozen=True)
class DevPlan:
    preview: DevPreview
    command: tuple[str, ...]
    environment: dict[str, str | None]
    prepare_environment: bool


class DevService:
    def __init__(
        self,
        project_resolver: ProjectResolver,
        local_process: LocalProcessClient,
        local_tracing: LocalTracingClient,
        port_in_use: Callable[[int], bool],
        diagnostics: CommandDiagnostics | None = None,
    ) -> None:
        self._project_resolver = project_resolver
        self._local_process = local_process
        self._local_tracing = local_tracing
        self._port_in_use = port_in_use
        self._diagnostics = diagnostics or CommandDiagnostics()

    @contextmanager
    def prepare(self, request: DevRequest) -> Iterator[DevPlan]:
        with self._diagnostics.phase(FailureStage.PROJECT):
            source_dir = pathlib.Path(request.source)
            project = self._project_resolver.load(source_dir)
            local = load_local_run(source_dir)
            if project is not None and project.tools:
                require_managed_tool_support(source_dir)
            port = _DEFAULT_APP_PORT if request.app_port is None else request.app_port
            if not 1 <= port <= 65535:
                raise AgentCliError(
                    "Port must be between 1 and 65535.",
                    error_category="ERROR_CATEGORY_CONFIGURATION",
                )
            if self._port_in_use(port):
                raise AgentCliError(
                    f"Port {port} is already in use (possibly a previous agentbricks dev run).",
                    hint="Stop that process, or run `agentbricks dev --app-port <port>` with a free port.",
                    error_category="ERROR_CATEGORY_CONFIGURATION",
                )
            memory_store, session_store, trace_experiment = (
                self._project_resolver.resource_bindings(source_dir)
            )
            prepare_environment = request.prepare_environment
            if prepare_environment is None:
                prepare_environment = not (source_dir / ".venv").exists()
            self._diagnostics.prepare_environment = prepare_environment

        with self._diagnostics.phase(FailureStage.TRACING):
            tracing = self._local_tracing.start_dev(source_dir)
        try:
            environment = {
                key: value
                for key, value in local.environment.items()
                if key not in _BUILD_INDEX_ENVS | _DEPLOY_RESOURCE_ENVS
            }
            # Empty overrides prevent a template's load_dotenv from restoring deployed bindings.
            environment.update(
                {key: "" for key in _DEPLOY_RESOURCE_ENVS - {MEMORY_STORE_ENV, SESSION_STORE_ENV}}
            )
            environment.update(tracing.environment)
            environment[RUNTIME_STORE_LOCAL_ENV] = "true"
            if memory_store:
                environment[DEV_BOUND_MEMORY_STORE_ENV] = memory_store
            if project is not None:
                environment[CORS_ALLOWED_ORIGINS_ENV] = json.dumps(list(project.allowed_origins))
            preview = DevPreview(
                source_dir=source_dir,
                port=port,
                server=project.server if project else None,
                tracing_uri=tracing.environment.get("MLFLOW_TRACKING_URI"),
                local_experiment_name=tracing.environment.get("MLFLOW_EXPERIMENT_NAME"),
                tracing_warning=tracing.warning,
                tracing_help=tracing.help,
                memory_store=memory_store,
                session_store=session_store,
                trace_experiment=trace_experiment,
                has_chat_ui=(source_dir / "runtime" / "ui.py").is_file(),
            )
            yield DevPlan(preview, local.command, environment, prepare_environment)
        finally:
            with self._diagnostics.phase(FailureStage.CLEANUP):
                if tracing.server is not None:
                    self._local_tracing.stop(tracing.server)

    def run(
        self,
        plan: DevPlan,
        on_ready: Callable[[], None] | None = None,
        quiet_status: Callable[[float], str] | None = None,
    ) -> None:
        with self._diagnostics.phase(FailureStage.RUN_LOCAL):
            self._local_process.run(
                plan.preview.source_dir,
                plan.command,
                plan.environment,
                prepare_environment=plan.prepare_environment,
                port=plan.preview.port,
                on_ready=on_ready,
                quiet_status=quiet_status,
            )
