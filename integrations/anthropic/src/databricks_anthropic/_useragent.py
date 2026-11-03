"""Partner attribution: the User-Agent this connector sends on every Databricks call.

Fixed by the connector (not user-configurable), per Databricks partner telemetry guidance:
https://databrickslabs.github.io/partner-architecture/isv-partners/telemetry-attribution

- SDK clients the connector builds get ``product=PRODUCT_NAME, product_version=PRODUCT_VERSION``.
- SDK clients passed in by callers get the same token appended (``tag``).
- Databricks CLI processes (ours, and the SDK's ``databricks auth token`` refreshes) and the agent's own
  SDK/CLI calls inside the sandbox carry ``DATABRICKS_{SDK,CLI}_UPSTREAM[_VERSION]``.

``sandbox/worker.py`` runs without this package, so it repeats the two constants; keep them equal.
"""

from __future__ import annotations

import os
from typing import Any

PRODUCT_NAME = "ClaudeManagedAgents-DatabricksSandbox"
PRODUCT_VERSION = "0.1.0"

UPSTREAM_ENV = {
    "DATABRICKS_SDK_UPSTREAM": PRODUCT_NAME,
    "DATABRICKS_SDK_UPSTREAM_VERSION": PRODUCT_VERSION,
    "DATABRICKS_CLI_UPSTREAM": PRODUCT_NAME,
    "DATABRICKS_CLI_UPSTREAM_VERSION": PRODUCT_VERSION,
}


def client_kwargs() -> dict[str, str]:
    """``WorkspaceClient(...)`` / ``Config(...)`` kwargs that set this connector as the product."""
    return {"product": PRODUCT_NAME, "product_version": PRODUCT_VERSION}


def tag(workspace_client: Any) -> Any:
    """Append the connector token to a caller-supplied client (idempotent; keeps the caller's product)."""
    config = workspace_client.config
    # Match the whole token: `upstream/<PRODUCT_NAME>` (from UPSTREAM_ENV) is not our product token.
    if f"{PRODUCT_NAME}/{PRODUCT_VERSION}" not in config.user_agent.split():
        config.with_user_agent_extra(PRODUCT_NAME, PRODUCT_VERSION)
    return workspace_client


def apply_process_env() -> None:
    """Set the upstream variables on this process so every CLI subprocess it spawns is attributed."""
    os.environ.update(UPSTREAM_ENV)


def subprocess_env() -> dict[str, str]:
    return {**os.environ, **UPSTREAM_ENV}
