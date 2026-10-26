"""A LangChain tool that runs shell commands inside a Databricks sandbox.

``DatabricksSandboxTool`` is a ``langchain_core`` :class:`~langchain_core.tools.BaseTool`
that executes a shell command in an existing, RUNNING Databricks sandbox via the
``databricks-sdk`` ``execute_command_sync`` unary call and returns the command's
combined output. Unlike :class:`~databricks_langchain.sandbox.DatabricksSandbox`
(the Deep Agents backend), this is a plain tool: it works with any LangChain
tool-calling agent or LangGraph graph and does not require ``deepagents``.

Requires the ``sandbox`` extra::

    pip install 'databricks-langchain[sandbox]'
"""

from __future__ import annotations

from databricks.sdk import WorkspaceClient
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

try:
    from databricks.sdk.service.sandbox import ExecuteCommandStatus
    from google.protobuf.duration_pb2 import Duration  # ty: ignore[unresolved-import]

    _sandbox_imports_available = True
except ImportError:
    _sandbox_imports_available = False

# execute_command_sync is direct-exec (no shell, no login profile), so route the
# command through a login shell to get a PATH and the usual tools (python3, grep).
_SHELL = "/bin/bash"
_SHELL_ARGS = ["-lc"]

_IMPORT_ERROR = (
    "DatabricksSandboxTool requires databricks-langchain[sandbox]. "
    "Install with: pip install 'databricks-langchain[sandbox]'"
)


class DatabricksSandboxToolInput(BaseModel):
    """Input schema for :class:`DatabricksSandboxTool`."""

    command: str = Field(description="The shell command to run in the Databricks sandbox.")


class DatabricksSandboxTool(BaseTool):
    """Run a shell command inside an existing, RUNNING Databricks sandbox.

    The sandbox lifecycle (create/start/delete) is the caller's responsibility;
    this tool only runs commands in a sandbox that already exists.
    """

    name: str = "databricks_sandbox"
    description: str = (
        "Run a shell command inside an isolated Databricks sandbox and return its "
        "combined stdout and stderr. Use this to execute code, scripts, or shell "
        "commands (for example `python3 script.py`) in a fresh environment."
    )
    args_schema: type[BaseModel] = DatabricksSandboxToolInput

    workspace_client: WorkspaceClient = Field(
        description="Authenticated Databricks WorkspaceClient."
    )
    sandbox_name: str = Field(
        description="Resource name of a running sandbox, e.g. 'sandboxes/{id}'."
    )
    timeout: int | None = Field(
        default=None, description="Optional per-command timeout, in seconds."
    )

    @model_validator(mode="after")
    def _require_sandbox_extra(self) -> DatabricksSandboxTool:
        if not _sandbox_imports_available:
            raise ImportError(_IMPORT_ERROR)
        return self

    def _run(self, command: str, **_: object) -> str:
        """Execute ``command`` in the sandbox and return its combined output."""
        execution_timeout = Duration(seconds=self.timeout) if self.timeout is not None else None
        response = self.workspace_client.sandbox.execute_command_sync(
            self.sandbox_name,
            _SHELL,
            args=[*_SHELL_ARGS, command],
            execution_timeout=execution_timeout,
        )
        output = (response.stdout or "") + (response.stderr or "")
        if response.truncated:
            output += "\n[output truncated]"
        if response.status != ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED:
            output += f"\n[status: {response.status}]"
        if response.exit_code:
            output += f"\n[exit code: {response.exit_code}]"
        return output
