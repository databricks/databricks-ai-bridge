"""Unit tests for the DatabricksSandbox -> execute_command_sync mapping.

These mock the SDK's unary ``execute_command_sync`` so the adapter's mapping
logic (shell wrapping, combined output, base64 transfer, error codes) is tested
without a live sandbox.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("databricks.sdk.service.sandbox")

from databricks.sdk.service.sandbox import ExecuteCommandStatus  # noqa: E402
from deepagents.backends.protocol import (  # noqa: E402
    FileDownloadResponse,
    FileUploadResponse,
)
from google.protobuf.duration_pb2 import Duration  # noqa: E402  # ty: ignore[unresolved-import]

from databricks_langchain import DatabricksSandbox  # noqa: E402

COMPLETED = ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED
FAILED = ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_FAILED
EXPECTED_EXIT = 3


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


class _SandboxApi:
    """Records the last execute_command_sync call and returns a canned response."""

    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def execute_command_sync(
        self,
        name: str,
        cmd: str,
        *,
        args: list[str] | None = None,
        envs: dict[str, str] | None = None,
        execution_timeout: Duration | None = None,
    ) -> _Response:
        self.calls.append(
            {
                "name": name,
                "cmd": cmd,
                "args": args,
                "envs": envs,
                "execution_timeout": execution_timeout,
            }
        )
        return self._response


class _Client:
    def __init__(self, response: _Response) -> None:
        self.sandbox = _SandboxApi(response)


def _backend(response: _Response) -> tuple[DatabricksSandbox, _SandboxApi]:
    client = _Client(response)
    return DatabricksSandbox(client, "sandboxes/sb_1"), client.sandbox  # ty: ignore[invalid-argument-type]


def test_id_returns_resource_name() -> None:
    backend, _ = _backend(_Response())
    assert backend.id == "sandboxes/sb_1"


def test_execute_combines_output_and_wraps_in_login_shell() -> None:
    backend, api = _backend(
        _Response(stdout="out\n", stderr="err\n", exit_code=EXPECTED_EXIT, truncated=True)
    )
    result = backend.execute("echo hi")
    assert result.output == "out\nerr\n"
    assert result.exit_code == EXPECTED_EXIT
    assert result.truncated is True
    assert api.calls[0]["cmd"] == "/bin/bash"
    assert api.calls[0]["args"] == ["-lc", "echo hi"]


def test_execute_timeout_maps_to_duration() -> None:
    backend, api = _backend(_Response())
    backend.execute("sleep 1", timeout=5)
    assert api.calls[0]["execution_timeout"] == Duration(seconds=5)


def test_upload_sends_base64_and_creates_parent_dir() -> None:
    content = bytes(range(8))
    backend, api = _backend(_Response())
    responses = backend.upload_files([("/w/f.bin", content)])
    assert responses[0].path == "/w/f.bin"
    assert responses[0].error is None
    envs = api.calls[0]["envs"]
    assert envs["DBX_PATH"] == "/w/f.bin"
    assert envs["DBX_B64"] == base64.b64encode(content).decode("ascii")
    assert "mkdir -p" in api.calls[0]["args"][1]
    assert "base64 -d" in api.calls[0]["args"][1]


def test_upload_reports_error_on_nonzero_exit() -> None:
    backend, _ = _backend(_Response(stderr="disk full", exit_code=1))
    responses = backend.upload_files([("/w/f.bin", b"x")])
    assert responses[0].error == "disk full"


def test_download_decodes_base64_stdout() -> None:
    payload = b"hello world"
    backend, _ = _backend(_Response(stdout=base64.b64encode(payload).decode("ascii")))
    responses = backend.download_files(["/w/f.bin"])
    assert responses[0].content == payload
    assert responses[0].error is None


def test_download_missing_maps_to_file_not_found() -> None:
    backend, _ = _backend(
        _Response(stderr="base64: /w/missing: No such file or directory", exit_code=1)
    )
    responses = backend.download_files(["/w/missing"])
    assert responses[0].content is None
    assert responses[0].error == "file_not_found"


def test_execute_failed_status_passes_through() -> None:
    backend, _ = _backend(_Response(stderr="killed", exit_code=None, status=FAILED))
    result = backend.execute("false")
    assert result.exit_code is None
    assert "killed" in result.output


def test_upload_relative_path_rejected_without_exec() -> None:
    backend, api = _backend(_Response())
    responses = backend.upload_files([("relative.txt", b"x")])
    assert responses == [FileUploadResponse(path="relative.txt", error="invalid_path")]
    assert api.calls == []


def test_download_relative_path_rejected_without_exec() -> None:
    backend, api = _backend(_Response())
    responses = backend.download_files(["relative.txt"])
    assert responses == [
        FileDownloadResponse(path="relative.txt", content=None, error="invalid_path")
    ]
    assert api.calls == []
