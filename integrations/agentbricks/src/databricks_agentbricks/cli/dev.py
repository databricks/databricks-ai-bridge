"""Run the local agent loop and chat UI independently of the deployed execution host."""

from __future__ import annotations

from typing import Optional

import click

from databricks_agentbricks.clients.local_ports import port_in_use
from databricks_agentbricks.clients.local_process_client import LocalProcessClient
from databricks_agentbricks.clients.local_tracing_client import LocalTracingClient
from databricks_agentbricks.clients.process_runner import run_process as _run_process
from databricks_agentbricks.diagnostics import CommandDiagnostics
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.dev import (
    announce_bound_resources,
    announce_starting,
    local_trace_label,
    starting_status,
)
from databricks_agentbricks.presentation.dev import (
    announce_local_url as _announce_local_url,
)
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.services.dev_service import (
    DevRequest,
    DevService,
)


def build_dev_service(obj) -> DevService:
    """Compose the local workflow without opening a workspace API client."""
    diagnostics = getattr(obj, "diagnostics", None) or CommandDiagnostics()
    obj.diagnostics = diagnostics
    diagnostics.app_listened = False
    return DevService(
        project_resolver=ProjectResolver(),
        local_process=LocalProcessClient(obj.profile, runner=_run_process, diagnostics=diagnostics),
        local_tracing=LocalTracingClient(),
        port_in_use=port_in_use,
        diagnostics=diagnostics,
    )


@click.command()
@click.option(
    "--source",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Local source directory to run (containing agent.toml or app.yaml). Defaults to the current directory.",
)
@click.option(
    "--prepare-environment/--no-prepare-environment",
    default=None,
    help="Build the app's environment with uv before running. Default: build only if no .venv "
    "exists yet, and reuse it otherwise. Requires uv.",
)
@click.option("--app-port", type=int, default=None, help="Port to run the agent on (default 8000).")
@click.pass_obj
def dev(
    obj,
    source: str,
    prepare_environment: Optional[bool],
    app_port: Optional[int],
) -> None:
    """Run your agent locally so you can try it before deploying.

    Starts the agent on a local server — by default http://localhost:8000 — and, once it answers,
    prints where to reach it: the chat UI if the project has one, otherwise a sample request against
    the agent's API. If the agent fails to start, you get an error instead.

    Auth uses your Databricks profile (`-p` / `agentbricks login`), and the agent reaches Databricks model
    serving through the AI Gateway on that profile — so there are no model keys to set up.

    Runs [dev].command from agent.toml directly; existing projects can keep their app.yaml
    command and env. No Databricks Apps CLI or deployed runtime is needed. Python dependencies
    are installed into the project's .venv on the first run and reused after; pass
    --prepare-environment to update them after dependency changes. Custom commands must honor
    HOST and PORT to use the selected local address and port.

    Everything runs locally: `agentbricks dev` is a local deployment that does not depend on a Databricks
    workspace for its resources. Tracing goes to a local MLflow tracking server (sqlite-backed, under `.agentbricks/`)
    so traces are recorded on your machine with no workspace experiment or setup - open the printed
    Traces URL to view them (`agentbricks tracing unbind` doesn't affect dev; it only stops the deployed
    agent's tracing). Long-term memory is off and conversation history is in-process (not durable):
    the memory/session stores bound with `agentbricks memory/sessions bind` are created and used only when you
    `agentbricks deploy`, not here. So there's nothing to provision and no service-principal grant to make;
    that all happens at `agentbricks deploy` time.
    """
    service = build_dev_service(obj)
    with service.prepare(
        DevRequest(source=source, prepare_environment=prepare_environment, app_port=app_port)
    ) as plan:
        if plan.preview.tracing_warning is not None:
            render.diagnostic(
                "warning", plan.preview.tracing_warning, help=plan.preview.tracing_help
            )
        announce_bound_resources(plan.preview)
        announce_starting()
        # The green panel waits until the agent answers on its port, so a failed start (bad auth, a
        # build error, a crash on import) never looks like a successful one.
        try:
            service.run(
                plan,
                on_ready=lambda: _announce_local_url(
                    plan.preview.source_dir,
                    plan.preview.port,
                    plan.preview.server,
                    local_trace_label(plan.preview),
                    has_chat_ui=plan.preview.has_chat_ui,
                ),
                # A spinner in the quiet stretches before the agent answers (e.g. while it imports
                # its libraries), so a long wait doesn't look like a hang.
                quiet_status=starting_status,
            )
        except AgentCliError:
            # Separate the error from the agent logs it follows.
            render.console().print()
            raise
