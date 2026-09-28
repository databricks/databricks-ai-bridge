"""Agent Bricks CLI: the ``agentbricks`` command and its subcommands.

The console entrypoint (``project.scripts``) is ``databricks_agentbricks.cli:main``, re-exported here from
``databricks_agentbricks.cli.app`` (the command tree) so that path stays stable. The entry function is named
``app`` there rather than ``main`` so it does not shadow this package's re-exported ``main``. These
modules are the command surface; the SDK and deployed-agent runtime live under
``databricks_agentkit``.
"""

from databricks_agentbricks.cli.app import main

__all__ = ["main"]
