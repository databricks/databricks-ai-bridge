"""Mason CLI: the ``mason`` command and its subcommands.

The console entrypoint (``project.scripts``) is ``databricks_mason.cli:main``, re-exported here from
``databricks_mason.cli.app`` (the command tree) so that path stays stable. The entry function is named
``app`` there rather than ``main`` so it does not shadow this package's re-exported ``main``. These
modules are the command surface only; the SDK (``MasonClient``) and the deployed-agent runtime live at
the package top and under ``databricks_mason.runtime`` respectively, and never import from here.
"""

from databricks_mason.cli.app import main

__all__ = ["main"]
