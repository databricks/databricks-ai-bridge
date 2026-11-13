"""Partner attribution: the User-Agent this connector sends on every Databricks call.

Fixed by the connector (not user-configurable), per Databricks partner telemetry guidance:
https://databrickslabs.github.io/partner-architecture/isv-partners/telemetry-attribution

- ``apply()`` (run when the package is imported) sets the SDK's global product with
  ``databricks.sdk.useragent.with_product``, per the partner guide for the Python SDK
  (https://databrickslabs.github.io/partner-architecture/isv-partners/telemetry-attribution/sdks). Clients
  built as ``WorkspaceClient(config=Config(...))`` afterwards send it. ``WorkspaceClient(host=..., ...)``
  without a config defaults its product to ``unknown/0.0.0`` and so needs ``tag``.
- SDK clients the connector builds also pass ``product=PRODUCT_NAME, product_version=PRODUCT_VERSION``.
- SDK clients created *before* the import, and passed in by callers, get the token via ``tag``. The SDK caches the
  User-Agent when a client is built, so ``tag`` refreshes that cache too.
- Databricks CLI processes (ours, and the SDK's ``databricks auth token`` refreshes) and the agent's own
  SDK/CLI calls inside the sandbox carry ``DATABRICKS_{SDK,CLI}_UPSTREAM[_VERSION]``.

``sandbox/worker.py`` runs without this package, so it repeats the two constants; keep them equal.
"""

from __future__ import annotations

import os
from typing import Any

from databricks.sdk import useragent

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


def apply() -> None:
    """Attribute this process's Databricks SDK and CLI calls to the connector (idempotent)."""
    useragent.with_product(PRODUCT_NAME, PRODUCT_VERSION)
    apply_process_env()


def tag(workspace_client: Any) -> Any:
    """Make a caller-supplied client send the connector token (idempotent; keeps the caller's product)."""
    config = workspace_client.config
    # Match the whole token: `upstream/<PRODUCT_NAME>` (from UPSTREAM_ENV) is not our product token.
    if f"{PRODUCT_NAME}/{PRODUCT_VERSION}" not in config.user_agent.split():
        config.with_user_agent_extra(PRODUCT_NAME, PRODUCT_VERSION)
    # The SDK copies config.user_agent into its HTTP client when the WorkspaceClient is built
    # (databricks.sdk.core.ApiClient), so a later change to the config alone never reaches requests.
    base_client = getattr(getattr(workspace_client, "api_client", None), "_api_client", None)
    if base_client is not None and hasattr(base_client, "_user_agent_base"):
        base_client._user_agent_base = config.user_agent
    return workspace_client


def apply_process_env() -> None:
    """Set the upstream variables on this process so every CLI subprocess it spawns is attributed."""
    os.environ.update(UPSTREAM_ENV)


def subprocess_env() -> dict[str, str]:
    return {**os.environ, **UPSTREAM_ENV}
