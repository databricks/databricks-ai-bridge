"""Unit tests for the DatabricksSandboxTool -> execute_command_sync mapping.

These mock the SDK's unary ``execute_command_sync`` so the tool's mapping (shell
wrapping, combined output, timeout, exit/status annotations) is tested without a
live sandbox or deepagents.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

pytest.importorskip("databricks.sdk.service.sandbox")

from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.service.sandbox import ExecuteCommandStatus  # noqa: E402
from google.protobuf.duration_pb2 import Duration  # noqa: E402  # ty: ignore[unresolved-import]

from databricks_langchain import DatabricksSandboxTool  # noqa: E402

COMPLETED = ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED
FAILED = ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_FAILED
EXPECTED_EXIT = 2


class _Response:
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = 0,
        status: ExecuteCommandStatus = COMPLETED,
        truncated: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.status = status
        self.truncated = truncated


def _tool(
    response: _Response, *, timeout: int | None = None
) -> tuple[DatabricksSandboxTool, MagicMock]:
    client = MagicMock(spec=WorkspaceClient)
    client.sandbox.execute_command_sync.return_value = response
    tool = DatabricksSandboxTool(
        workspace_client=cast(WorkspaceClient, client),
        sandbox_name="sandboxes/sb_1",
        timeout=timeout,
    )
    return tool, client.sandbox.execute_command_sync


def test_tool_metadata() -> None:
    tool, _ = _tool(_Response())
    assert tool.name == "databricks_sandbox"
    assert "sandbox" in tool.description.lower()
    assert tool.args_schema is not None


def test_run_wraps_in_login_shell_and_combines_output() -> None:
    tool, exec_mock = _tool(_Response(stdout="out\n", stderr="err\n"))
    result = tool._run(command="echo hi")
    assert result == "out\nerr\n"
    assert exec_mock.call_args.args[0] == "sandboxes/sb_1"
    assert exec_mock.call_args.args[1] == "/bin/bash"
    assert exec_mock.call_args.kwargs["args"] == ["-lc", "echo hi"]


def test_timeout_maps_to_duration() -> None:
    tool, exec_mock = _tool(_Response(), timeout=5)
    tool._run(command="sleep 1")
    assert exec_mock.call_args.kwargs["execution_timeout"] == Duration(seconds=5)


def test_no_timeout_passes_none() -> None:
    tool, exec_mock = _tool(_Response())
    tool._run(command="echo hi")
    assert exec_mock.call_args.kwargs["execution_timeout"] is None


def test_nonzero_exit_is_annotated() -> None:
    tool, _ = _tool(_Response(stdout="boom\n", exit_code=EXPECTED_EXIT))
    result = tool._run(command="false")
    assert "boom" in result
    assert f"[exit code: {EXPECTED_EXIT}]" in result


def test_failed_status_is_annotated() -> None:
    tool, _ = _tool(_Response(stderr="killed", exit_code=None, status=FAILED))
    result = tool._run(command="x")
    assert "killed" in result
    assert "[status:" in result


def test_truncated_output_is_annotated() -> None:
    tool, _ = _tool(_Response(stdout="partial", truncated=True))
    result = tool._run(command="cat big")
    assert "partial" in result
    assert "[output truncated]" in result
