"""`agentbricks` — the Databricks CLI for agent deployment, memory, and sessions.

Root Click group. Global `--profile` and `--output` flow to every subcommand via
`CliContext` on `ctx.obj`; subcommands build an authenticated API client on demand.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import click

from databricks_agentbricks import errors
from databricks_agentbricks.cli.auth import ProfileInfo, resolve_profile
from databricks_agentbricks.cli.deploy import deploy, deployments
from databricks_agentbricks.cli.dev import dev
from databricks_agentbricks.cli.doctor import doctor
from databricks_agentbricks.cli.endpoint import endpoint
from databricks_agentbricks.cli.init import init
from databricks_agentbricks.cli.memory import memory
from databricks_agentbricks.cli.profile import profile as profile_group
from databricks_agentbricks.cli.sessions import sessions
from databricks_agentbricks.cli.tools import tools
from databricks_agentbricks.cli.tracing import tracing
from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.presentation.help import configure_help
from databricks_agentbricks.projects.config import is_agentbricks_project
from databricks_agentkit._api_client import _AgentBricksApiClient


class CliContext:
    """Shared per-invocation state: selected profile, output mode, lazily-built client."""

    def __init__(self, profile: Optional[str], output: str):
        self.profile_info = resolve_profile(profile)
        self.output = output
        self.api_client_provider = ApiClientProvider(self.profile)

    @property
    def profile(self) -> Optional[str]:
        return self.profile_info.name

    def use_project_profile(
        self, project_dir: pathlib.Path, env_values: Optional[dict[str, str]] = None
    ) -> None:
        """Fold in the project's `.env` profile (see `resolve_profile`) unless `-p` was given.

        The root group calls this for the current directory when it's an Agent Bricks project;
        `dev` and `deploy` call it again with their source dir, whose `.env` then replaces the
        current directory's. Only an explicit `-p` outranks the `.env`.
        ``env_values`` passes an already-parsed ``<project_dir>/.env`` so it isn't read twice.
        """
        if self.profile_info.source == "--profile":
            return
        resolved = resolve_profile(None, project_dir, env_values)
        if resolved.name != self.profile_info.name:
            self.api_client_provider = ApiClientProvider(resolved.name)
        self.profile_info = resolved

    def use_flag_profile(self, profile: str) -> None:
        """Apply a `-p/--profile` given after the subcommand; like the global one, it beats `.env`."""
        if profile != self.profile:
            self.api_client_provider = ApiClientProvider(profile)
        self.profile_info = ProfileInfo(profile, "--profile")

    def client(self) -> _AgentBricksApiClient:
        """Compatibility accessor for commands not yet migrated to ``api_client_provider``."""
        return self.api_client_provider.get()


@click.group(name="agentbricks", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--profile", "-p", default=None, help="~/.databrickscfg profile to authenticate with."
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format (default: text).",
)
@click.version_option(package_name="databricks-agentbricks")
@click.pass_context
def agentbricks(ctx: click.Context, profile: Optional[str], output: str) -> None:
    """Agent Bricks is a CLI for building and deploying custom AI agents on Databricks.

    The Agent Bricks CLI is experimental: its commands and the underlying agent APIs are all in
    preview, may need to be enabled for your workspace, and are likely to change in
    backward-incompatible ways.

    Scaffold an agent project from a template, run it locally with a chat UI, and deploy it to
    Databricks Apps — then manage the tools, memory, sessions, and tracing behind it, all from one
    authenticated command.

    New here? The examples below take you from an empty directory to a deployed agent. Agent Bricks
    authenticates with a Databricks profile: sign in with `agentbricks profile login <profile>`,
    then run `agentbricks init <dir> --profile <profile>` to record it in the project's .env.
    Inside a project, its .env profile is used (change it with `agentbricks profile set`);
    --profile / -p overrides it, and outside a project DATABRICKS_CONFIG_PROFILE applies (without
    any of these, the Databricks SDK's default authentication is used).

    Agents built with Agent Bricks combine the platform's capabilities:

    \b
      Models       Call Databricks model serving out of the box, routed through
                   the AI Gateway for capacity on your existing Databricks auth.
      Tools        Data sandboxes, managed MCP services, Unity Catalog
                   functions, and local Python tools the agent can call.
      Memory       Long-term memory the agent recalls across conversations.
      Sessions     The transcript, history, and state of a single conversation.
      Tracing      MLflow traces in Unity Catalog to debug and evaluate runs.
      Deployment   Hosting on Databricks Apps, with scaling and sticky routing.

    `agentbricks deploy` provisions and wires these into a single agent hosted on Databricks Apps.
    """
    # Let errors render to match the selected output mode (JSON errors for -o json).
    errors.set_output_mode(output)
    ctx.obj = CliContext(profile=profile, output=output)
    cwd = pathlib.Path.cwd()
    if is_agentbricks_project(cwd):
        # Every command run inside a project uses its `.env` profile, so e.g. `endpoint invoke`
        # targets the same workspace that `deploy` just used.
        ctx.obj.use_project_profile(cwd)


def _apply_subcommand_profile(
    ctx: click.Context, _param: click.Parameter, value: Optional[str]
) -> None:
    obj = ctx.find_object(CliContext)
    if value and obj is not None:
        obj.use_flag_profile(value)


def _accept_profile_anywhere(command: click.Command) -> None:
    """Let `-p/--profile` follow the subcommand too (`agentbricks dev -p X`), not only precede it.

    Commands that declare their own `--profile` (`init`) keep it.
    """
    if isinstance(command, click.Group):
        for subcommand in command.commands.values():
            _accept_profile_anywhere(subcommand)
        return
    if any("--profile" in param.opts for param in command.params):
        return
    command.params.append(
        click.Option(
            ["--profile", "-p"],
            default=None,
            expose_value=False,
            callback=_apply_subcommand_profile,
            help="~/.databrickscfg profile to authenticate with (same as the global -p).",
        )
    )


agentbricks.add_command(init)
agentbricks.add_command(profile_group)
agentbricks.add_command(doctor)
agentbricks.add_command(dev)
agentbricks.add_command(memory)
agentbricks.add_command(sessions)
agentbricks.add_command(tracing)
agentbricks.add_command(deploy)
agentbricks.add_command(deployments)
agentbricks.add_command(endpoint)
agentbricks.add_command(tools)
for _command in agentbricks.commands.values():
    _accept_profile_anywhere(_command)
configure_help(agentbricks)


def main() -> None:
    # Click derives the display name from argv[0] so the `agentbricks` script is shown in help output.
    agentbricks()


if __name__ == "__main__":
    main()
