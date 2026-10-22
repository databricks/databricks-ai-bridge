"""A context manager that provisions a Databricks sandbox for DatabricksSandboxTool.

``DatabricksSandboxSession`` creates (or reuses) a sandbox, waits for it to reach
RUNNING, yields a ready :class:`DatabricksSandboxTool`, and deletes the sandbox on
exit. Lifecycle stays in application code: the tool and the agent only execute
commands, never create or delete sandboxes.

Requires the ``sandbox`` extra::

    pip install 'databricks-langchain[sandbox]'
"""

from __future__ import annotations

import time
import uuid
from types import TracebackType

from databricks.sdk import WorkspaceClient

from databricks_langchain.sandbox_tool import DatabricksSandboxTool

try:
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.sandbox import (
        ComputeSpec,
        Sandbox,
        SandboxSpec,
        SandboxState,
    )
    from google.protobuf.duration_pb2 import Duration  # ty: ignore[unresolved-import]

    _sandbox_imports_available = True
except ImportError:
    _sandbox_imports_available = False

_IMPORT_ERROR = (
    "DatabricksSandboxSession requires databricks-langchain[sandbox]. "
    "Install with: pip install 'databricks-langchain[sandbox]'"
)


class DatabricksSandboxSession:
    """Provision a Databricks sandbox and hand back a ready DatabricksSandboxTool.

    The sandbox is created (or reused) and started on entry, and deleted on exit.
    The agent only ever receives the tool, so sandbox lifecycle stays with the
    application rather than the model.

    Use as a context manager (recommended)::

        with DatabricksSandboxSession(WorkspaceClient()) as tool:
            agent = create_agent(model, tools=[tool])
            agent.invoke(...)
        # sandbox deleted here

    or explicitly::

        session = DatabricksSandboxSession(WorkspaceClient(), sandbox_id="my-sb")
        tool = session.start()
        try:
            ...
        finally:
            session.close()

    By default only sandboxes this session created are deleted on exit; a reused,
    pre-existing sandbox is left alone. Pass ``delete_on_exit`` to override.
    """

    def __init__(
        self,
        workspace_client: WorkspaceClient,
        *,
        sandbox_id: str | None = None,
        inactivity_timeout: int = 900,
        command_timeout: int | None = None,
        create_if_missing: bool = True,
        delete_on_exit: bool | None = None,
        ready_timeout: int = 180,
        poll_interval: int = 3,
    ) -> None:
        if not _sandbox_imports_available:
            raise ImportError(_IMPORT_ERROR)
        self._client = workspace_client
        self._sandbox_id = sandbox_id or f"lc-{uuid.uuid4().hex[:12]}"
        self._inactivity_timeout = inactivity_timeout
        self._command_timeout = command_timeout
        self._create_if_missing = create_if_missing
        self._delete_on_exit = delete_on_exit
        self._ready_timeout = ready_timeout
        self._poll_interval = poll_interval
        self._name = f"sandboxes/{self._sandbox_id}"
        self._created = False

    @property
    def name(self) -> str:
        """Resource name of the managed sandbox, ``sandboxes/{id}``."""
        return self._name

    def start(self) -> DatabricksSandboxTool:
        """Create/reuse the sandbox, wait for RUNNING, and return a ready tool."""
        try:
            sandbox = self._client.sandbox.get_sandbox(self._name)
        except NotFound:
            if not self._create_if_missing:
                raise
            sandbox = self._client.sandbox.create_sandbox(
                Sandbox(
                    spec=SandboxSpec(
                        compute=ComputeSpec(
                            inactivity_timeout=Duration(seconds=self._inactivity_timeout)
                        )
                    )
                ),
                sandbox_id=self._sandbox_id,
            )
            self._created = True
        try:
            if self._is_stopped(sandbox):
                self._client.sandbox.start_sandbox(self._name)
            self._wait_for_running()
        except BaseException:
            # Never leak a sandbox we just created if bring-up fails.
            if self._created:
                self._safe_delete()
            raise
        return DatabricksSandboxTool(
            workspace_client=self._client,
            sandbox_name=self._name,
            timeout=self._command_timeout,
        )

    def close(self) -> None:
        """Delete the sandbox when this session owns its lifecycle."""
        delete = self._created if self._delete_on_exit is None else self._delete_on_exit
        if delete:
            self._safe_delete()

    def __enter__(self) -> DatabricksSandboxTool:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _wait_for_running(self) -> None:
        deadline = time.monotonic() + self._ready_timeout
        while time.monotonic() < deadline:
            if self._is_running(self._client.sandbox.get_sandbox(self._name)):
                return
            time.sleep(self._poll_interval)
        msg = f"sandbox {self._name} did not reach RUNNING within {self._ready_timeout}s"
        raise TimeoutError(msg)

    @staticmethod
    def _is_stopped(sandbox: Sandbox) -> bool:
        status = sandbox.status
        return status is not None and status.state in (
            SandboxState.SANDBOX_STATE_STOPPED,
            SandboxState.SANDBOX_STATE_STOPPING,
        )

    @staticmethod
    def _is_running(sandbox: Sandbox) -> bool:
        status = sandbox.status
        return status is not None and status.state == SandboxState.SANDBOX_STATE_RUNNING

    def _safe_delete(self) -> None:
        try:
            self._client.sandbox.delete_sandbox(self._name)
        except NotFound:
            pass
