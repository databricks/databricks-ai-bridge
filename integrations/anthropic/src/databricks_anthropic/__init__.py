"""Databricks integrations for Anthropic's Claude platform.

Run Claude Managed Agents self-hosted sessions in Databricks Sandbox. The hosted dispatcher (a Databricks App) is in
``app/`` next to this package; the library below is what it runs and can host elsewhere.
"""

from databricks_anthropic import _useragent
from databricks_anthropic.sandbox import (
    CleanupAction,
    DatabricksSandboxDispatcher,
    DatabricksSandboxLauncher,
    LaunchResult,
    SandboxMcpServer,
    sandbox_id_for_session,
)

# Partner attribution before any WorkspaceClient is created (the SDK reads the product when a client is built).
_useragent.apply()

__all__ = [
    "CleanupAction",
    "DatabricksSandboxDispatcher",
    "DatabricksSandboxLauncher",
    "LaunchResult",
    "SandboxMcpServer",
    "sandbox_id_for_session",
]
