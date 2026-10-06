"""`agentbricks dev` — run a scaffolded agent locally, wrapping `databricks apps run-local`.

Runs the app from its ``app.yaml`` exactly as the Databricks Apps runtime would locally: reads the
manifest's command + env, and (with ``--prepare-environment``) builds the venv via uv. This is the
local counterpart to ``agentbricks deploy`` — same source dir, same manifest — so what runs here matches
what ships. Delegating to ``apps run-local`` means agentbricks inherits the Apps team's local-run behavior
rather than re-implementing it.
"""

from __future__ import annotations

import pathlib
import subprocess
from typing import Optional

import click

from databricks_agentbricks.cli.auth import _parse_env_file, preflight_auth, profile_host
from databricks_agentbricks.cli.tracing import start_local_tracing_server, stop_local_tracing_server
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.databricks_cli import _databricks
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


class _CliLocalTracing:
    """Adapt the existing local tracing process helpers to the service port."""

    def start(self, source_dir: pathlib.Path) -> tuple[subprocess.Popen | None, dict[str, str]]:
        return start_local_tracing_server(source_dir)

    def stop(self, server: subprocess.Popen) -> None:
        stop_local_tracing_server(server)


def build_dev_service(obj, preflight=None) -> DevService:
    """Compose the local workflow without opening a workspace API client."""
    return DevService(
        project_resolver=ProjectResolver(),
        apps_client=AppsClient(obj.profile, runner=_databricks),
        local_tracing=_CliLocalTracing(),
        preflight=preflight,
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
    source_dir = pathlib.Path(source)
    # Fold the project's `.env` into the profile resolution before the profile is used, so the CLI
    # and the locally running agent (which reads that same `.env`) agree on one profile. Parse the
    # file once here — dev also consults it below for explicit credentials.
    env_file = _parse_env_file(source_dir / ".env")
    obj.use_project_profile(source_dir, env_file)
    # An explicit DATABRICKS_TOKEN in `.env` authenticates the agent directly, so there is no
    # profile to validate or to land in the dev manifest (a bare DATABRICKS_HOST is not a
    # credential — the profile still flows to the agent, which carries its own host).
    uses_env_credentials = bool(obj.profile and env_file.get("DATABRICKS_TOKEN"))
    local_env: dict[str, str] = {}
    if obj.profile and not uses_env_credentials:
        # Landing the profile in the dev manifest puts it in the agent's process env, where it
        # beats `.env` because the template loads dotenv with override=False.
        local_env["DATABRICKS_CONFIG_PROFILE"] = obj.profile

    def _preflight() -> None:
        if not env_file.get("DATABRICKS_TOKEN"):
            preflight_auth(obj.profile, obj.profile_info.source, action="dev")

    service = build_dev_service(obj, _preflight)
    with service.prepare(
        DevRequest(
            source=source,
            prepare_environment=prepare_environment,
            app_port=app_port,
            local_env=local_env,
        )
    ) as plan:
        announce_bound_resources(plan.preview)
        announce_agent_profile(
            obj.profile,
            obj.profile_info.source,
            env_file.get("DATABRICKS_CONFIG_PROFILE"),
            uses_env_credentials=uses_env_credentials,
        )
        # When `.env` credentials win, the resolved profile/host would point at a different
        # workspace than the agent actually uses, so they are left out of the announcement.
        announce_profile = None if uses_env_credentials else obj.profile
        _announce_local_url(
            plan.preview.source_dir,
            plan.preview.port,
            plan.preview.server,
            local_trace_label(plan.preview),
            has_chat_ui=plan.preview.has_chat_ui,
            profile=announce_profile,
            profile_source=obj.profile_info.source,
            host=profile_host(announce_profile),
        )
        service.run(plan)
