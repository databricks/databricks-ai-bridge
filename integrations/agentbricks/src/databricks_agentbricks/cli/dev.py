"""`agentbricks dev` — run a scaffolded agent locally, wrapping `databricks apps run-local`.

Runs the app from its ``app.yaml`` exactly as the Databricks Apps runtime would locally: reads the
manifest's command + env, and (with ``--prepare-environment``) builds the venv via uv. This is the
local counterpart to ``agentbricks deploy`` — same source dir, same manifest — so what runs here matches
what ships. Delegating to ``apps run-local`` means agentbricks inherits the Apps team's local-run behavior
rather than re-implementing it.
"""

from __future__ import annotations

from typing import Optional

import click

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.databricks_cli import _databricks
from databricks_agentbricks.clients.local_ports import free_port, port_in_use
from databricks_agentbricks.clients.local_tracing_client import LocalTracingClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.dev import (
    announce_bound_resources,
    announce_starting,
    local_trace_label,
)
from databricks_agentbricks.presentation.dev import (
    announce_local_url as _announce_local_url,
)
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.services.dev_service import (
    DevRequest,
    DevService,
    _dev_entry_point,  # noqa: F401 - compatibility re-export for existing callers
)


def build_dev_service(obj) -> DevService:
    """Compose the local workflow without opening a workspace API client."""
    return DevService(
        project_resolver=ProjectResolver(),
        apps_client=AppsClient(obj.profile, runner=_databricks),
        local_tracing=LocalTracingClient(),
        port_in_use=port_in_use,
        free_port=free_port,
    )


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

    Under the hood this wraps `databricks apps run-local`: it reads the command + env from
    `app.yaml` and runs the app the way the Apps runtime would, so local behavior matches a
    deployment. The environment is built on the first run and reused after; pass
    `--prepare-environment` to force a rebuild (e.g. after changing dependencies).

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
            )
        except AgentCliError:
            # Separate the error from the run-local logs it follows.
            render.console().print()
            raise
