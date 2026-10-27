"""`agentbricks` — the Databricks CLI for agent deployment, memory, and sessions.

Root Click group. Global `--profile` and `--output` flow to every subcommand via
`CliContext` on `ctx.obj`; subcommands build an authenticated API client on demand.
"""

from __future__ import annotations

import sys
from typing import Optional

import click

from databricks_agentbricks import errors
from databricks_agentbricks.cli.auth import load_default_profile, login, logout
from databricks_agentbricks.cli.deploy import deploy, deployments
from databricks_agentbricks.cli.dev import dev
from databricks_agentbricks.cli.doctor import doctor
from databricks_agentbricks.cli.endpoint import endpoint
from databricks_agentbricks.cli.experimental import experimental
from databricks_agentbricks.cli.init import init
from databricks_agentbricks.cli.memory import memory
from databricks_agentbricks.cli.sessions import sessions
from databricks_agentbricks.cli.tools import tools
from databricks_agentbricks.cli.tracing import tracing
from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.presentation.help import configure_help
from databricks_agentkit._api_client import _AgentBricksApiClient


class CliContext:
    """Shared per-invocation state: selected profile, output mode, lazily-built client."""

    def __init__(self, profile: Optional[str], output: str):
        self.profile = profile
        self.output = output
        self.api_client_provider = ApiClientProvider(profile)

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
    authenticates with a Databricks profile: run `agentbricks login` once to save a default, or pass
    --profile / -p (without one, the Databricks SDK's default authentication is used).

    Agents built with Agent Bricks combine the platform's capabilities:

    \b
      Models       Call Databricks model serving out of the box, routed through
                   the AI Gateway for capacity on your existing Databricks auth;
                   bind each LLM call to a model service, then search models and
                   prompts for every call on your own eval set.
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
    ctx.obj = CliContext(profile=profile or load_default_profile(), output=output)


agentbricks.add_command(login)
agentbricks.add_command(logout)
agentbricks.add_command(init)
agentbricks.add_command(doctor)
agentbricks.add_command(dev)
agentbricks.add_command(memory)
agentbricks.add_command(sessions)
agentbricks.add_command(tracing)
agentbricks.add_command(deploy)
agentbricks.add_command(deployments)
agentbricks.add_command(endpoint)
agentbricks.add_command(tools)
agentbricks.add_command(experimental)
configure_help(agentbricks)


def _emit_trailing_spacer() -> None:
    """Print one blank line after a command finishes, so its output is visually separated from the
    next shell prompt.

    Applies to every invocation — success or error — because it runs from the entry point that all
    commands pass through, rather than from any single render path. Emitted only for human-facing
    text output on an interactive terminal: skipped under `-o json` and when stdout is piped or
    redirected, so captured output stays byte-clean and no stray blank line appears when a command's
    own output went somewhere other than the terminal.
    """
    try:
        # Suppress under `-o json` even on a (pseudo-)TTY — a machine may be parsing stdout under
        # `docker run -t` / `script` / a CI PTY, where isatty() alone would wrongly allow the
        # spacer. `errors._OUTPUT_MODE` is a process global (set by the root group, read the same
        # way by render.py), so it's still available here after the Click context is torn down.
        if errors._OUTPUT_MODE == "json" or not sys.stdout.isatty():
            return
        click.echo()
    except Exception:
        # This runs in main()'s `finally`, so nothing here may raise and mask the command's real
        # exit path — swallow anything (a closed/detached stdout, a broken pipe, etc.).
        pass


def main() -> None:
    # A `finally` keeps Click's `main()` untouched (exit codes, --help, broken-pipe/abort handling)
    # while guaranteeing the spacer prints on every path, including when a command exits non-zero.
    try:
        # Click derives the display name from argv[0] so the `agentbricks` script is shown in help
        # output.
        agentbricks()
    finally:
        _emit_trailing_spacer()


if __name__ == "__main__":
    main()
