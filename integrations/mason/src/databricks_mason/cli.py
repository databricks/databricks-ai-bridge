"""`mason` — the Databricks CLI for agent deployment, memory, and sessions.

Root Click group. Global `--profile` and `--output` flow to every subcommand via
`CliContext` on `ctx.obj`; subcommands build an `AgentApiClient` from it on demand.
"""

from __future__ import annotations

from typing import Optional

import click

from databricks_mason.auth import load_default_profile, login, logout
from databricks_mason.client import AgentApiClient
from databricks_mason.deploy import deploy, deployments
from databricks_mason.memory import memory
from databricks_mason.sessions import sessions
from databricks_mason.tracing import tracing


class CliContext:
    """Shared per-invocation state: selected profile, output mode, lazily-built client."""

    def __init__(self, profile: Optional[str], output: str):
        self.profile = profile
        self.output = output
        self._client: Optional[AgentApiClient] = None

    def client(self) -> AgentApiClient:
        if self._client is None:
            self._client = AgentApiClient(self.profile)
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
mason.add_command(memory)
mason.add_command(sessions)
mason.add_command(tracing)
mason.add_command(deploy)
mason.add_command(deployments)


def main() -> None:
    mason(prog_name="mason")


if __name__ == "__main__":
    main()
