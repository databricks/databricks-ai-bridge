"""Sandbox lifecycle helpers for the sandbox integration tests.

Create (or reuse) a Databricks sandbox, wait for it to reach RUNNING, and delete
it, using only released ``databricks-sdk`` CRUD APIs. The backend under test only
runs commands in a live sandbox; provisioning stays here so it never leaks into
the backend.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from databricks.sdk.errors import NotFound
from databricks.sdk.service.sandbox import (
    ComputeSpec,
    Sandbox,
    SandboxSpec,
    SandboxState,
)
from google.protobuf.duration_pb2 import Duration

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient


def provision_sandbox(
    client: WorkspaceClient, sandbox_id: str, *, inactivity_timeout_s: int = 900
) -> str:
    """Create or reuse the sandbox, start it if stopped, and wait for RUNNING.

    Returns the sandbox resource name, ``sandboxes/{sandbox_id}``.
    """
    name = f"sandboxes/{sandbox_id}"
    try:
        sandbox = client.sandbox.get_sandbox(name)
    except NotFound:
        sandbox = client.sandbox.create_sandbox(
            Sandbox(
                spec=SandboxSpec(
                    compute=ComputeSpec(inactivity_timeout=Duration(seconds=inactivity_timeout_s))
                )
            ),
            sandbox_id=sandbox_id,
        )
    if sandbox.status.state in (
        SandboxState.SANDBOX_STATE_STOPPED,
        SandboxState.SANDBOX_STATE_STOPPING,
    ):
        client.sandbox.start_sandbox(name)
    _wait_for_state(client, name, SandboxState.SANDBOX_STATE_RUNNING)
    return name


def delete_sandbox(client: WorkspaceClient, name: str) -> None:
    client.sandbox.delete_sandbox(name)


def _wait_for_state(
    client: WorkspaceClient,
    name: str,
    target: SandboxState,
    *,
    timeout_s: int = 180,
    poll_s: int = 3,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if client.sandbox.get_sandbox(name).status.state == target:
            return
        time.sleep(poll_s)
    msg = f"sandbox {name} did not reach {target} within {timeout_s}s"
    raise TimeoutError(msg)
