"""Run Claude Managed Agents self-hosted sessions in Databricks Sandbox.

- :class:`DatabricksSandboxDispatcher` polls a self-hosted environment's work queue and
  hands each claimed session to the launcher.
- :class:`DatabricksSandboxLauncher` gets or creates the session's sandbox, bootstraps the
  worker's virtualenv, and starts the Anthropic ``EnvironmentWorker`` inside it.

The worker (``databricks_anthropic.sandbox.worker``) runs inside the sandbox and uses the
sandbox's built-in Databricks credential, so the agent's tools act as whoever created the
sandbox (the dispatcher's identity).
"""

from databricks_anthropic.sandbox.dispatcher import CleanupAction, DatabricksSandboxDispatcher
from databricks_anthropic.sandbox.launcher import (
    DatabricksSandboxLauncher,
    LaunchResult,
    SandboxMcpServer,
    sandbox_id_for_session,
)
from databricks_anthropic.sandbox.memory import MemoryConfig, memory_custom_tools

__all__ = [
    "CleanupAction",
    "DatabricksSandboxDispatcher",
    "DatabricksSandboxLauncher",
    "LaunchResult",
    "MemoryConfig",
    "SandboxMcpServer",
    "memory_custom_tools",
    "sandbox_id_for_session",
]
