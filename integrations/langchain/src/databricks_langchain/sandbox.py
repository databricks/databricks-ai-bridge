"""A Deep Agents sandbox backend backed by the Databricks Sandbox SDK.

``DatabricksSandbox`` subclasses :class:`deepagents.backends.sandbox.BaseSandbox`,
so a Deep Agents agent can run its shell and filesystem tools inside a Databricks
sandbox. ``BaseSandbox`` derives ``ls``/``read``/``write``/``edit``/``grep``/
``glob``/``delete`` on top of three primitives the subclass supplies: ``execute``,
``upload_files`` and ``download_files`` (plus an ``id``).

All three primitives are built from a single ``databricks-sdk`` unary call,
``w.sandbox.execute_command_sync`` -- the backend needs no streaming RPC. The SDK
exposes no file-transfer API, so ``upload_files``/``download_files`` move bytes as
base64 over that exec call. Because the sandbox daemon caps a unary response at
512 KiB and a gRPC request near 4 MiB, large files are transferred in chunks
(base64 piped in for upload, byte ranges read out for download), so the backend
round-trips files far larger than a single response.

Requires the ``deepagents`` extra::

    pip install 'databricks-langchain[deepagents]'
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

try:
    from databricks.sdk.service.sandbox import ExecuteCommandStatus
    from deepagents.backends.protocol import (
        ExecuteResponse,
        FileDownloadResponse,
        FileUploadResponse,
    )
    from deepagents.backends.sandbox import BaseSandbox
    from google.protobuf.duration_pb2 import Duration  # ty: ignore[unresolved-import]

    _sandbox_imports_available = True
except ImportError:
    BaseSandbox = object  # type: ignore
    _sandbox_imports_available = False

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sandbox import ExecuteCommandSyncResponse

# execute_command_sync is direct-exec (no shell, no login profile), so route every
# command through a login shell to get a PATH and the tools the derived filesystem
# helpers rely on (python3, grep).
_SHELL = "/bin/bash"
_SHELL_ARGS = ["-lc"]

# Upload sends base64 in a command env var, so each chunk must stay under the
# kernel's per-argument/env limit (MAX_ARG_STRLEN, 128 KiB), not just the gRPC
# request size. A multiple of 4 keeps every chunk valid base64 that decodes
# independently (padding only ever lands in the final chunk).
_UPLOAD_B64_CHUNK = 120 * 1024
# Download reads raw byte ranges; keep each base64 response under the 512 KiB
# unary cap. A multiple of 3 avoids base64 padding except on the final chunk.
_DOWNLOAD_RAW_CHUNK = 300 * 1024

_IMPORT_ERROR = (
    "DatabricksSandbox requires databricks-langchain[deepagents]. "
    "Install with: pip install 'databricks-langchain[deepagents]'"
)


class _SandboxFileError(Exception):
    """Internal signal carrying a protocol error code for a failed file op."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DatabricksSandbox(BaseSandbox):
    """Deep Agents sandbox backend over an existing, RUNNING Databricks sandbox.

    Lifecycle (create/start/delete) is the application's responsibility, as the
    Deep Agents protocol prescribes; this backend only runs commands and moves
    files in a sandbox that already exists.
    """

    def __init__(self, client: WorkspaceClient, name: str) -> None:
        """Bind to a running sandbox by workspace client and resource name."""
        if not _sandbox_imports_available:
            raise ImportError(_IMPORT_ERROR)
        self._client = client
        self._name = name

    @property
    def id(self) -> str:
        """Return the sandbox resource name."""
        return self._name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run ``command`` in the sandbox and return its combined output."""
        response = self._exec(command, timeout=timeout)
        # Contract: ExecuteResponse.output is combined output. The derived
        # filesystem helpers already fold their own stderr into stdout at the
        # command level (every template ends in `2>&1` or `2>/dev/null`), so for
        # those calls `stderr` is empty; concatenating is correct and only ever
        # adds output for the agent's own shell commands.
        output = (response.stdout or "") + (response.stderr or "")
        return ExecuteResponse(
            output=output,
            exit_code=response.exit_code,
            truncated=bool(response.truncated),
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Write each ``(path, content)`` pair into the sandbox."""
        return [self._upload_one(path, content) for path, content in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Read each path from the sandbox as bytes."""
        return [self._download_one(path) for path in paths]

    def _upload_one(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error="invalid_path")
        encoded = base64.b64encode(content).decode("ascii")
        pieces = [
            encoded[i : i + _UPLOAD_B64_CHUNK] for i in range(0, len(encoded), _UPLOAD_B64_CHUNK)
        ] or [""]
        try:
            for index, piece in enumerate(pieces):
                # First chunk creates parent dirs and truncates; later ones append.
                redirect = ">" if index == 0 else ">>"
                mkdir = 'mkdir -p "$(dirname "$DBX_PATH")"; ' if index == 0 else ""
                script = (
                    f"set -e -o pipefail; {mkdir}"
                    f'printf %s "$DBX_B64" | base64 -d {redirect} "$DBX_PATH"'
                )
                response = self._exec(script, envs={"DBX_PATH": path, "DBX_B64": piece})
                if not self._completed(response):
                    return FileUploadResponse(path=path, error=self._file_error(response))
        except Exception as exc:
            # Surface any SDK/transport error on this file without failing the batch.
            return FileUploadResponse(path=path, error=str(exc))
        return FileUploadResponse(path=path)

    def _download_one(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, error="invalid_path")
        kind = self._classify(path)
        if kind != "OK":
            return FileDownloadResponse(path=path, error=kind)
        try:
            content = self._read_chunks(path)
        except _SandboxFileError as err:
            return FileDownloadResponse(path=path, error=err.code)
        except Exception as exc:
            # Surface any SDK/transport error on this file without failing the batch.
            return FileDownloadResponse(path=path, error=str(exc))
        return FileDownloadResponse(path=path, content=content)

    def _read_chunks(self, path: str) -> bytes:
        buf = bytearray()
        offset = 0
        while True:
            # tail -c is 1-based; read a raw byte range and base64 it so each
            # response stays under the unary cap regardless of file size.
            script = 'tail -c +$((DBX_OFF + 1)) "$DBX_PATH" | head -c "$DBX_LEN" | base64'
            response = self._exec(
                script,
                envs={
                    "DBX_PATH": path,
                    "DBX_OFF": str(offset),
                    "DBX_LEN": str(_DOWNLOAD_RAW_CHUNK),
                },
            )
            if not self._completed(response):
                raise _SandboxFileError(self._file_error(response))
            # b64decode(validate=False) ignores the newlines `base64` wraps with.
            chunk = base64.b64decode(response.stdout or "")
            buf.extend(chunk)
            if len(chunk) < _DOWNLOAD_RAW_CHUNK:
                return bytes(buf)
            offset += _DOWNLOAD_RAW_CHUNK

    def _classify(self, path: str) -> str:
        """Return a protocol error code for a download, or ``"OK"`` to proceed."""
        script = (
            'if [ ! -e "$DBX_PATH" ]; then echo MISSING; '
            'elif [ -d "$DBX_PATH" ]; then echo ISDIR; '
            'elif [ ! -r "$DBX_PATH" ]; then echo NOREAD; '
            "else echo OK; fi"
        )
        response = self._exec(script, envs={"DBX_PATH": path})
        token = (response.stdout or "").strip()
        return {
            "MISSING": "file_not_found",
            "ISDIR": "is_directory",
            "NOREAD": "permission_denied",
        }.get(token, "OK")

    def _exec(
        self,
        command: str,
        *,
        timeout: int | None = None,
        envs: dict[str, str] | None = None,
    ) -> ExecuteCommandSyncResponse:
        execution_timeout = Duration(seconds=timeout) if timeout is not None else None
        return self._client.sandbox.execute_command_sync(
            self._name,
            _SHELL,
            args=[*_SHELL_ARGS, command],
            envs=envs,
            execution_timeout=execution_timeout,
        )

    @staticmethod
    def _completed(response: ExecuteCommandSyncResponse) -> bool:
        return (
            response.status == ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED
            and response.exit_code == 0
        )

    @staticmethod
    def _file_error(response: ExecuteCommandSyncResponse) -> str:
        detail = ((response.stderr or "") + (response.stdout or "")).lower()
        if "no such file" in detail:
            return "file_not_found"
        if "permission denied" in detail:
            return "permission_denied"
        if "is a directory" in detail or "directory" in detail:
            return "is_directory"
        return (response.stderr or response.stdout or "command failed").strip() or "command failed"
