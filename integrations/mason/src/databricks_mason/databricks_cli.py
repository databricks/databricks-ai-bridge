"""Thin wrapper around the ``databricks`` CLI.

Mason shells out to the ``databricks`` CLI for the operations it doesn't call over the REST client
(``apps`` lifecycle, ``sync``, ``postgres`` endpoint/credential lookups). This is the single place
that builds the argv, threads the ``--profile`` through, and normalizes a non-zero exit into an
``AgentCliError``. Kept separate from any one caller so ``deploy``, ``dev``, and ``store_access``
share it without importing each other.
"""

from __future__ import annotations

import subprocess
from typing import Optional

from databricks_mason.errors import AgentCliError


def _databricks(
    args: list[str],
    profile: Optional[str],
    *,
    capture: bool = False,
    check: bool = True,
    cwd: Optional[str] = None,
    action: Optional[str] = None,
) -> subprocess.CompletedProcess:
    cmd = ["databricks", *args]
    if profile:
        cmd += ["--profile", profile]
    result = subprocess.run(cmd, text=True, capture_output=capture, cwd=cwd)
    if check and result.returncode != 0:
        # Mason drives the `databricks apps` CLI as an implementation detail; surface a failure in
        # Mason's own terms (`action`) rather than echoing the raw subcommand and --profile, which
        # leaks the underlying tool at the customer. The captured stderr still rides along as the
        # hint so debugging isn't lost.
        detail = (result.stderr or result.stdout or "").strip() if capture else None
        raise AgentCliError(action or "A Databricks CLI command failed.", hint=detail)
    return result
