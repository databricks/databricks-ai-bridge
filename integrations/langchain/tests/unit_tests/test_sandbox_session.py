"""Unit tests for DatabricksSandboxSession provisioning and teardown.

These mock the SDK's sandbox CRUD (``get_sandbox`` / ``create_sandbox`` /
``start_sandbox`` / ``delete_sandbox``) so the create-or-reuse, start, wait, and
delete-on-exit logic is tested without a live workspace.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

pytest.importorskip("databricks.sdk.service.sandbox")

from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.errors import NotFound  # noqa: E402
from databricks.sdk.service.sandbox import SandboxState  # noqa: E402

from databricks_langchain import DatabricksSandboxSession, DatabricksSandboxTool  # noqa: E402

RUNNING = SandboxState.SANDBOX_STATE_RUNNING
STOPPED = SandboxState.SANDBOX_STATE_STOPPED
STOPPING = SandboxState.SANDBOX_STATE_STOPPING


def _sandbox(state: SandboxState) -> MagicMock:
    sandbox = MagicMock()
    sandbox.status.state = state
    return sandbox


def _client() -> MagicMock:
    return MagicMock(spec=WorkspaceClient)


def _session(
    client: MagicMock,
    *,
    create_if_missing: bool = True,
    delete_on_exit: bool | None = None,
    ready_timeout: int = 180,
) -> DatabricksSandboxSession:
    return DatabricksSandboxSession(
        cast(WorkspaceClient, client),
        sandbox_id="sb1",
        poll_interval=0,
        create_if_missing=create_if_missing,
        delete_on_exit=delete_on_exit,
        ready_timeout=ready_timeout,
    )


def test_name_from_sandbox_id() -> None:
    assert _session(_client()).name == "sandboxes/sb1"


def test_auto_generates_sandbox_id() -> None:
    session = DatabricksSandboxSession(cast(WorkspaceClient, _client()))
    assert session.name.startswith("sandboxes/lc-")


def test_creates_when_missing_and_returns_tool() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = [NotFound("missing"), _sandbox(RUNNING)]
    client.sandbox.create_sandbox.return_value = _sandbox(RUNNING)
    tool = _session(client).start()
    assert isinstance(tool, DatabricksSandboxTool)
    assert tool.sandbox_name == "sandboxes/sb1"
    client.sandbox.create_sandbox.assert_called_once()


def test_reuses_existing_without_creating() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = [_sandbox(RUNNING), _sandbox(RUNNING)]
    _session(client).start()
    client.sandbox.create_sandbox.assert_not_called()


def test_starts_when_stopped() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = [_sandbox(STOPPED), _sandbox(RUNNING)]
    _session(client).start()
    client.sandbox.start_sandbox.assert_called_once_with("sandboxes/sb1")


def test_waits_out_stopping_before_starting() -> None:
    # ESM rejects StartSandbox on a STOPPING sandbox (ABORTED); the session must
    # wait for it to settle to STOPPED before calling start_sandbox.
    client = _client()
    client.sandbox.get_sandbox.side_effect = [
        _sandbox(STOPPING),
        _sandbox(STOPPED),
        _sandbox(RUNNING),
    ]
    _session(client).start()
    client.sandbox.start_sandbox.assert_called_once_with("sandboxes/sb1")


def test_deletes_on_exit_when_created() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = [NotFound("missing"), _sandbox(RUNNING)]
    client.sandbox.create_sandbox.return_value = _sandbox(RUNNING)
    with _session(client):
        pass
    client.sandbox.delete_sandbox.assert_called_once_with("sandboxes/sb1")


def test_reused_sandbox_not_deleted_by_default() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = [_sandbox(RUNNING), _sandbox(RUNNING)]
    with _session(client):
        pass
    client.sandbox.delete_sandbox.assert_not_called()


def test_delete_on_exit_true_forces_delete_of_reused() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = [_sandbox(RUNNING), _sandbox(RUNNING)]
    with _session(client, delete_on_exit=True):
        pass
    client.sandbox.delete_sandbox.assert_called_once_with("sandboxes/sb1")


def test_created_sandbox_deleted_if_bringup_fails() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = [NotFound("missing")]
    client.sandbox.create_sandbox.return_value = _sandbox(STOPPED)
    client.sandbox.start_sandbox.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        _session(client).start()
    client.sandbox.delete_sandbox.assert_called_once_with("sandboxes/sb1")


def test_timeout_deletes_created_sandbox() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = [NotFound("missing")]
    client.sandbox.create_sandbox.return_value = _sandbox(RUNNING)
    with pytest.raises(TimeoutError):
        _session(client, ready_timeout=0).start()
    client.sandbox.delete_sandbox.assert_called_once_with("sandboxes/sb1")


def test_missing_sandbox_raises_when_create_disabled() -> None:
    client = _client()
    client.sandbox.get_sandbox.side_effect = NotFound("missing")
    with pytest.raises(NotFound):
        _session(client, create_if_missing=False).start()
    client.sandbox.create_sandbox.assert_not_called()
