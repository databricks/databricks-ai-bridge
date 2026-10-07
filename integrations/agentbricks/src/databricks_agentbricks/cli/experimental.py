"""`agentbricks experimental` — commands whose interface is still being designed.

Commands here work and are tested, but their names, options, and behavior may change between
releases without a deprecation period.
"""

from __future__ import annotations

import click

from databricks_agentbricks.cli.models import models


@click.group()
def experimental() -> None:
    """Commands still being designed; their interface may change between releases."""


experimental.add_command(models)
