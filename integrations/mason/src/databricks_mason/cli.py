"""`mason` — the Databricks CLI for agent deployment, memory, and sessions.

Root Click group. Global `--profile` and `--output` flow to every subcommand via
`CliContext` on `ctx.obj`; subcommands build a `MasonClient` from it on demand.
"""

from __future__ import annotations

from typing import Optional

import click

from databricks_mason.auth import load_default_profile, login, logout
from databricks_mason.client import MasonClient
from databricks_mason.deploy import deploy, deployments
from databricks_mason.dev import dev
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
        self._client: Optional[MasonClient] = None

    def client(self) -> MasonClient:
        if self._client is None:
            self._client = MasonClient(self.profile)
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
    """Mason: deploy agents and manage their memory and sessions.

    Targets the agents/v1 preview APIs served on a workspace; auth comes from a
    .databrickscfg profile (pass --profile / -p, run `mason login` to save a default,
    or rely on the SDK's default resolution).
    """
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


def main() -> None:
    mason(prog_name="mason")


if __name__ == "__main__":
    main()
