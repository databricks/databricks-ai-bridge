"""`agentbricks dev` — run a scaffolded agent locally, wrapping `databricks apps run-local`.

Runs the app from its ``app.yaml`` exactly as the Databricks Apps runtime would locally: reads the
manifest's command + env, and (with ``--prepare-environment``) builds the venv via uv. This is the
local counterpart to ``agentbricks deploy`` — same source dir, same manifest — so what runs here matches
what ships. Delegating to ``apps run-local`` means agentbricks inherits the Apps team's local-run behavior
rather than re-implementing it.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.databricks_cli import _databricks
from databricks_agentbricks.clients.local_ports import port_in_use
from databricks_agentbricks.clients.local_tracing_client import LocalTracingClient
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.dev import (
    announce_agent_profile,
    announce_bound_resources,
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
from databricks_agentbricks.services.workspace_auth import preflight_auth

# `.env` credential sets that authenticate the agent on their own, without a profile.
_ENV_CREDENTIALS = (
    ("DATABRICKS_TOKEN",),
    ("DATABRICKS_CLIENT_ID", "DATABRICKS_CLIENT_SECRET"),
)


def _env_credential(env_file: dict[str, str]) -> Optional[str]:
    """The explicit credential `.env` sets (e.g. ``DATABRICKS_TOKEN``), or None."""
    for names in _ENV_CREDENTIALS:
        if all(env_file.get(name) for name in names):
            return "/".join(names)
    return None


def build_dev_service(obj) -> DevService:
    """Compose the local workflow without opening a workspace API client."""
    return DevService(
        project_resolver=ProjectResolver(),
        apps_client=AppsClient(obj.profile, runner=_databricks),
        local_tracing=LocalTracingClient(),
        port_in_use=port_in_use,
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
@click.option("--app-port", type=int, default=None, help="Port to run the app on (default 8000).")
@click.pass_obj
def dev(
    obj,
    source: str,
    prepare_environment: Optional[bool],
    app_port: Optional[int],
) -> None:
    """Run your agent locally so you can try it before deploying.

    Starts the agent on a local server — by default http://localhost:8000 — and prints where to
    reach it: the chat UI if the project has one, otherwise a sample request against the agent's
    API.

    Auth uses your Databricks profile, resolved in this order: `-p`, the project's `.env`,
    then `DATABRICKS_CONFIG_PROFILE`. The agent reaches Databricks model serving through the AI
    Gateway on that profile — so there are no model keys to set up.

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
    # The project's `.env` profile is already selected (see `CliContext.select_profile`); the
    # locally running agent reads the same file, which dev also consults for explicit credentials.
    env_file = obj.project_env(pathlib.Path(source))
    env_credential = _env_credential(env_file)
    # Explicit `.env` credentials authenticate the agent directly, so the profile needs neither
    # validation nor a place in the dev manifest (a bare DATABRICKS_HOST is not a credential).
    uses_env_credentials = bool(obj.profile and env_credential)
    local_env: dict[str, str] = {}
    if obj.profile and not uses_env_credentials:
        # Landing the profile in the dev manifest puts it in the agent's process env, where it
        # beats `.env` because the template loads dotenv with override=False.
        local_env["DATABRICKS_CONFIG_PROFILE"] = obj.profile

    service = build_dev_service(obj)
    service.check_project(source)
    # Auth is checked before anything is started: an unauthenticated external-browser profile
    # would otherwise hang the local agent on a browser flow with no error.
    if not env_credential:
        try:
            validated = preflight_auth(obj.profile, obj.profile_info.source)
        except Exception:
            obj.announce_workspace()
            raise
        obj.announce_workspace(validated.host)
    elif not uses_env_credentials:
        obj.announce_workspace()
    with service.prepare(
        DevRequest(
            source=source,
            prepare_environment=prepare_environment,
            app_port=app_port,
            local_env=local_env,
        )
    ) as plan:
        if plan.preview.tracing_warning is not None:
            render.diagnostic(
                "warning", plan.preview.tracing_warning, help=plan.preview.tracing_help
            )
        announce_bound_resources(plan.preview)
        announce_agent_profile(
            obj.profile,
            obj.profile_info.source,
            env_file.get("DATABRICKS_CONFIG_PROFILE"),
            env_credential=env_credential if uses_env_credentials else None,
        )
        _announce_local_url(
            plan.preview.source_dir,
            plan.preview.port,
            plan.preview.server,
            local_trace_label(plan.preview),
            has_chat_ui=plan.preview.has_chat_ui,
        )
        service.run(plan)
