"""Partner attribution: the User-Agent this connector sends on every Databricks call.

Fixed by the connector (not user-configurable), per Databricks partner telemetry guidance:
https://databrickslabs.github.io/partner-architecture/isv-partners/telemetry-attribution

- ``apply()`` (run when the package is imported) only sets the SDK's global product with
  ``databricks.sdk.useragent.with_product``, per the partner guide for the Python SDK
  (https://databrickslabs.github.io/partner-architecture/isv-partners/telemetry-attribution/sdks). Clients
  built as ``WorkspaceClient(config=Config(...))`` afterwards send it. ``WorkspaceClient(host=..., ...)``
  without a config defaults its product to ``unknown/0.0.0`` and so needs ``tag``.
- SDK clients the connector builds also pass ``product=PRODUCT_NAME, product_version=PRODUCT_VERSION``.
- SDK clients created *before* the import, and passed in by callers, get the token via ``tag``. The SDK caches the
  User-Agent when a client is built, so ``tag`` refreshes that cache too.
- Importing the package never changes ``os.environ``. The CLI subprocesses the connector starts get
  ``DATABRICKS_{SDK,CLI}_UPSTREAM[_VERSION]`` through ``subprocess_env()``; the in-sandbox worker (its own process)
  sets them for the agent's SDK and CLI calls; the App sets them in its ``config.env``.

The version is read from the installed package, so it never needs editing per release. ``sandbox/worker.py`` runs
without this package installed, so the launcher passes the version to it as ``CMA_PRODUCT_VERSION``.
"""

from __future__ import annotations

import os
import re
from importlib import metadata
from typing import Any

from databricks.sdk import useragent

PRODUCT_NAME = "ClaudeManagedAgents-DatabricksSandbox"


UNKNOWN_VERSION = "0.0.0"  # the SDK only accepts semantic versions, so "unknown" is spelled 0.0.0


def semver(version: str) -> str:
    """A PEP 440 version as the semantic version the SDK's User-Agent requires (``0.1.0.dev0`` -> ``0.1.0-dev0``)."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?[.-]?([A-Za-z0-9.]*)", version or "")
    if not match:
        return UNKNOWN_VERSION
    major, minor, patch, rest = match.groups()
    base = f"{major}.{minor}.{patch or 0}"
    rest = re.sub(r"[^A-Za-z0-9]", "", rest)
    return f"{base}-{rest}" if rest else base


def _installed_version() -> str:
    try:
        return semver(metadata.version("databricks-anthropic"))
    except metadata.PackageNotFoundError:  # e.g. run from a source tree that isn't installed
        return UNKNOWN_VERSION


PRODUCT_VERSION = _installed_version()

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
    """Attribute this process's Databricks SDK calls to the connector (idempotent). Leaves ``os.environ`` alone."""
    useragent.with_product(PRODUCT_NAME, PRODUCT_VERSION)


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


def subprocess_env() -> dict[str, str]:
    """This process's environment plus the upstream variables, for a CLI subprocess (``os.environ`` unchanged)."""
    return {**os.environ, **UPSTREAM_ENV}
