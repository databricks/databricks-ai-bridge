"""`mason` — the Databricks CLI for agent deployment, memory, and sessions.

Root Click group. Global `--profile` and `--output` flow to every subcommand via
`CliContext` on `ctx.obj`; subcommands build an authenticated API client on demand.
"""

from __future__ import annotations

from typing import Optional

import click

from databricks_mason import errors
from databricks_mason._api_client import _MasonApiClient
from databricks_mason.auth import load_default_profile, login, logout
from databricks_mason.deploy import deploy, deployments
from databricks_mason.dev import dev
from databricks_mason.help import configure_help
from databricks_mason.init import init
from databricks_mason.mcp import mcp
from databricks_mason.memory import memory
from databricks_mason.sessions import sessions
from databricks_mason.tools import tools
from databricks_mason.tracing import tracing


class CliContext:
    """Shared per-invocation state: selected profile, output mode, lazily-built client."""

    def __init__(self, profile: Optional[str], output: str):
        self.profile = profile
        self.output = output
        self._client: Optional[_MasonApiClient] = None

    def client(self) -> _MasonApiClient:
        if self._client is None:
            self._client = _MasonApiClient(self.profile)
        return self._client


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
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
@click.version_option(package_name="databricks-mason", prog_name="mason")
@click.pass_context
def mason(ctx: click.Context, profile: Optional[str], output: str) -> None:
    """Mason is a CLI for building and deploying custom AI agents on Databricks.

    Scaffold an agent project from a template, run it locally with a chat UI, and deploy it to
    Databricks Apps — then manage the tools, memory, sessions, and tracing behind it, all from one
    authenticated command.

    New here? The examples below take you from an empty directory to a deployed agent. Mason
    authenticates with a Databricks profile: run `mason login` once to save a default, or pass
    --profile / -p (without one, the Databricks SDK's default authentication is used).

    Mason is experimental: the CLI, its commands, and the underlying agent APIs are all in preview,
    may need to be enabled for your workspace, and are likely to change in backward-incompatible
    ways.

    An agent you build with Mason can combine the platform's capabilities:

    \b
      Tools      Give the agent actions: data sandboxes, managed MCP services,
                 Unity Catalog functions, and local Python tools.
      Memory     Long-term memory the agent recalls across conversations.
      Sessions   The transcript, history, and state of a single conversation.
      Tracing    MLflow traces in Unity Catalog to debug and evaluate runs.

    `mason deploy` provisions and wires these into a single agent hosted on Databricks Apps.
    """
    # Let errors render to match the selected output mode (JSON errors for -o json).
    errors.set_output_mode(output)
    ctx.obj = CliContext(profile=profile or load_default_profile(), output=output)


mason.add_command(login)
mason.add_command(logout)
mason.add_command(init)
mason.add_command(dev)
mason.add_command(memory)
mason.add_command(mcp)
mason.add_command(sessions)
mason.add_command(tracing)
mason.add_command(deploy)
mason.add_command(deployments)
mason.add_command(tools)
configure_help(mason)


def main() -> None:
    mason(prog_name="mason")


if __name__ == "__main__":
    main()
