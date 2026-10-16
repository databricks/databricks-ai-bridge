"""``python -m databricks_agentbricks.cli``: the same entry point as the ``agentbricks`` script.

`agentbricks deploy --no-wait` re-runs the CLI this way in its background process, so the child uses
the interpreter and package of the parent rather than whichever ``agentbricks`` is first on PATH.
"""

from databricks_agentbricks.cli import main

main()
