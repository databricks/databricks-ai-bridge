"""A Databricks sandbox backend for the OpenAI Agents SDK.

``DatabricksSandboxClient`` / ``DatabricksSandboxSession`` implement the Agents
SDK sandbox-provider contract (``agents.sandbox.session.sandbox_client`` and
``agents.sandbox.session.base_sandbox_session``) on top of the released
``databricks-sdk`` ``w.sandbox`` surface, so an OpenAI ``Runner`` can execute an
agent's shell and filesystem tools inside a Databricks sandbox::

    from agents.run import RunConfig
    from agents.run_config import SandboxRunConfig
    from databricks_openai import DatabricksSandboxClient

    RunConfig(sandbox=SandboxRunConfig(client=DatabricksSandboxClient(profile=...)))

Unlike LangChain Deep Agents (where the application owns the sandbox lifecycle),
the Agents SDK client owns it: ``create`` provisions a sandbox and waits for it
to reach RUNNING, ``delete`` removes it, and the session ``shutdown`` stops it so
``resume`` can reattach. This mirrors the first-party remote backends under
``agents.extensions.sandbox`` (Vercel, E2B, ...).

Every session primitive rides the single unary ``execute_command_sync`` RPC.
Commands run through a login shell so bare programs resolve on ``PATH``; file
read/write and workspace persist/hydrate move bytes as chunked base64 over that
same call, because the SDK exposes no native file-transfer API. The daemon caps a
unary response near 512 KiB and a single command arg/env near 128 KiB, so
transfers are chunked to round-trip files far larger than one response.

Requires the ``sandbox`` extra (``openai-agents>=0.20,<0.21``,
``databricks-sdk[sandbox]>=0.153.0``)::

    pip install 'databricks-openai[sandbox]'
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Literal

try:
    from agents.sandbox.errors import (
        ConfigurationError,
        ErrorCode,
        ExecTimeoutError,
        ExecTransportError,
        WorkspaceArchiveReadError,
        WorkspaceArchiveWriteError,
        WorkspaceReadNotFoundError,
        WorkspaceStartError,
        WorkspaceWriteTypeError,
    )
    from agents.sandbox.manifest import Manifest
    from agents.sandbox.session import SandboxSession, SandboxSessionState
    from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
    from agents.sandbox.session.dependencies import Dependencies
    from agents.sandbox.session.manager import Instrumentation
    from agents.sandbox.session.sandbox_client import (
        BaseSandboxClient,
        BaseSandboxClientOptions,
    )
    from agents.sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
    from agents.sandbox.types import ExecResult, User
    from agents.sandbox.workspace_paths import sandbox_path_str
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound
    from databricks.sdk.service.sandbox import (
        ComputeSpec,
        ExecuteCommandStatus,
        Sandbox,
        SandboxSpec,
        SandboxState,
    )
    from google.protobuf.duration_pb2 import Duration  # ty: ignore[unresolved-import]

    _COMPLETED = ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED
    _TIMED_OUT = ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_TIMED_OUT
    _RUNNING = SandboxState.SANDBOX_STATE_RUNNING
    _STOPPED_STATES = (
        SandboxState.SANDBOX_STATE_STOPPED,
        SandboxState.SANDBOX_STATE_STOPPING,
    )
    _DEFAULT_MANIFEST_ROOT = str(Manifest.model_fields["root"].default)

    _sandbox_imports_available = True
except ImportError:
    BaseSandboxClient = object  # type: ignore[assignment,misc]
    BaseSandboxClientOptions = object  # type: ignore[assignment,misc]
    BaseSandboxSession = object  # type: ignore[assignment,misc]
    SandboxSessionState = object  # type: ignore[assignment,misc]
    _sandbox_imports_available = False

if TYPE_CHECKING:
    from databricks.sdk.service.sandbox import ExecuteCommandSyncResponse

# /home/sandbox-agent is a dedicated, writable volume owned by the sandbox user
# that survives stop/restart; "/" and "/tmp" are an ephemeral overlay and $HOME
# is not writable. Workspaces that leave Manifest.root at the SDK default land here.
DEFAULT_DATABRICKS_WORKSPACE_ROOT = "/home/sandbox-agent/agent-workspace"

DEFAULT_INACTIVITY_TIMEOUT_S = 900
DEFAULT_WAIT_FOR_RUNNING_TIMEOUT_S = 180
_WAIT_POLL_INTERVAL_S = 3

_SHELL = "/bin/bash"
# execute_command_sync is direct-exec (no shell, no PATH/login profile), but the
# base session issues bare argv (ls, mkdir, python3, test) that need PATH resolution
# and relative paths resolved from the workspace root. Run everything through a login
# shell that cds into the root (best-effort; it may not exist yet at mkdir time) and
# exec's the argv as $1.. so PATH lookup and the workspace cwd both apply.
_CD_EXEC_SCRIPT = 'cd "$DBX_CWD" 2>/dev/null; exec "$@"'

# base64 of a chunk sent in a command env var must stay under the per-arg/env limit
# (MAX_ARG_STRLEN, 128 KiB); a multiple of 4 keeps every chunk independently decodable.
_UPLOAD_B64_CHUNK = 120 * 1024
# A downloaded raw byte range, once base64-encoded, must stay under the unary response
# cap (512 KiB); a multiple of 3 avoids base64 padding except on the final chunk.
_DOWNLOAD_RAW_CHUNK = 300 * 1024

_IMPORT_ERROR = (
    "DatabricksSandboxClient requires databricks-openai[sandbox]. "
    "Install with: pip install 'databricks-openai[sandbox]'"
)


def _duration_or_none(timeout: float | None) -> Duration | None:
    if timeout is None:
        return None
    seconds = int(timeout)
    return Duration(seconds=max(seconds, 1))


def _completed(response: ExecuteCommandSyncResponse) -> bool:
    return bool(response.status == _COMPLETED and response.exit_code == 0)


def _resolve_manifest_root(manifest: Manifest | None) -> Manifest:
    if manifest is None:
        return Manifest(root=DEFAULT_DATABRICKS_WORKSPACE_ROOT)
    if manifest.root == _DEFAULT_MANIFEST_ROOT:
        return manifest.model_copy(update={"root": DEFAULT_DATABRICKS_WORKSPACE_ROOT})
    return manifest


class DatabricksSandboxClientOptions(BaseSandboxClientOptions):
    """Client options for the Databricks sandbox backend."""

    type: Literal["databricks"] = "databricks"
    sandbox_id: str | None = None
    inactivity_timeout_s: int = DEFAULT_INACTIVITY_TIMEOUT_S
    wait_for_running_timeout_s: int = DEFAULT_WAIT_FOR_RUNNING_TIMEOUT_S

    def __init__(
        self,
        sandbox_id: str | None = None,
        inactivity_timeout_s: int = DEFAULT_INACTIVITY_TIMEOUT_S,
        wait_for_running_timeout_s: int = DEFAULT_WAIT_FOR_RUNNING_TIMEOUT_S,
        *,
        type: Literal["databricks"] = "databricks",
    ) -> None:
        if not _sandbox_imports_available:
            raise ImportError(_IMPORT_ERROR)
        super().__init__(
            type=type,
            sandbox_id=sandbox_id,
            inactivity_timeout_s=inactivity_timeout_s,
            wait_for_running_timeout_s=wait_for_running_timeout_s,
        )


class DatabricksSandboxSessionState(SandboxSessionState):
    """Serializable state for a Databricks-backed session.

    Only the sandbox identity and timeouts persist; the ``WorkspaceClient`` (which
    holds credentials) is supplied by the client at ``resume`` time and never
    serialized.
    """

    type: Literal["databricks"] = "databricks"
    sandbox_id: str
    inactivity_timeout_s: int = DEFAULT_INACTIVITY_TIMEOUT_S
    wait_for_running_timeout_s: int = DEFAULT_WAIT_FOR_RUNNING_TIMEOUT_S


class DatabricksSandboxSession(BaseSandboxSession):
    """SandboxSession backed by a Databricks sandbox over ``execute_command_sync``."""

    state: DatabricksSandboxSessionState
    _client: WorkspaceClient
    _provisioned: bool

    def __init__(self, *, state: DatabricksSandboxSessionState, client: WorkspaceClient) -> None:
        if not _sandbox_imports_available:
            raise ImportError(_IMPORT_ERROR)
        self.state = state
        self._client = client
        self._provisioned = False

    @classmethod
    def from_state(
        cls, state: DatabricksSandboxSessionState, *, client: WorkspaceClient
    ) -> DatabricksSandboxSession:
        return cls(state=state, client=client)

    @property
    def _name(self) -> str:
        return f"sandboxes/{self.state.sandbox_id}"

    # -- lifecycle -----------------------------------------------------------

    async def _ensure_sandbox(self) -> None:
        if self._provisioned:
            return
        await asyncio.to_thread(self._provision_blocking)
        self._provisioned = True

    def _provision_blocking(self) -> None:
        name = self._name
        try:
            sandbox = self._client.sandbox.get_sandbox(name)
        except NotFound:
            sandbox = self._client.sandbox.create_sandbox(
                Sandbox(
                    spec=SandboxSpec(
                        compute=ComputeSpec(
                            inactivity_timeout=Duration(seconds=self.state.inactivity_timeout_s)
                        )
                    )
                ),
                sandbox_id=self.state.sandbox_id,
            )
        if sandbox.status is not None and sandbox.status.state in _STOPPED_STATES:
            self._client.sandbox.start_sandbox(name)
        self._wait_for_running()

    def _wait_for_running(self) -> None:
        deadline = time.monotonic() + self.state.wait_for_running_timeout_s
        while time.monotonic() < deadline:
            status = self._client.sandbox.get_sandbox(self._name).status
            if status is not None and status.state == _RUNNING:
                return
            time.sleep(_WAIT_POLL_INTERVAL_S)
        raise WorkspaceStartError(
            path=Path(self.state.manifest.root),
            context={
                "backend": "databricks",
                "sandbox_id": self.state.sandbox_id,
                "reason": "sandbox_did_not_reach_running",
            },
        )

    async def _reconnect_if_running(self) -> bool:
        """Mark the session provisioned if its sandbox is already RUNNING, for resume."""

        def _probe() -> bool:
            try:
                status = self._client.sandbox.get_sandbox(self._name).status
            except NotFound:
                return False
            return status is not None and status.state == _RUNNING

        running = await asyncio.to_thread(_probe)
        if running:
            self._provisioned = True
        return running

    async def _ensure_backend_started(self) -> None:
        await self._ensure_sandbox()

    async def _prepare_backend_workspace(self) -> None:
        await self._ensure_sandbox()
        result = await self.exec("mkdir", "-p", "--", self.state.manifest.root, shell=False)
        if not result.ok():
            raise WorkspaceStartError(
                path=Path(self.state.manifest.root),
                context={
                    "backend": "databricks",
                    "stdout": result.stdout.decode("utf-8", errors="replace"),
                    "stderr": result.stderr.decode("utf-8", errors="replace"),
                },
            )

    async def running(self) -> bool:
        if not self._provisioned:
            return False

        def _check() -> bool:
            try:
                status = self._client.sandbox.get_sandbox(self._name).status
            except NotFound:
                return False
            return status is not None and status.state == _RUNNING

        return await asyncio.to_thread(_check)

    async def _shutdown_backend(self) -> None:
        if not self._provisioned:
            return

        def _stop() -> None:
            with contextlib.suppress(NotFound):
                self._client.sandbox.stop_sandbox(self._name)

        await asyncio.to_thread(_stop)

    # -- command execution ---------------------------------------------------

    async def _exec_sync(
        self,
        program: str,
        args: list[str],
        *,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> ExecuteCommandSyncResponse:
        await self._ensure_sandbox()

        def _run() -> ExecuteCommandSyncResponse:
            return self._client.sandbox.execute_command_sync(
                self._name,
                program,
                args=args,
                envs=envs,
                execution_timeout=_duration_or_none(timeout),
            )

        return await asyncio.to_thread(_run)

    async def _exec_internal(
        self, *command: str | Path, timeout: float | None = None
    ) -> ExecResult:
        argv = [str(part) for part in command]
        if not argv:
            return ExecResult(stdout=b"", stderr=b"", exit_code=0)

        args = ["-lc", _CD_EXEC_SCRIPT, _SHELL, *argv]
        try:
            response = await self._exec_sync(
                _SHELL, args, envs={"DBX_CWD": self.state.manifest.root}, timeout=timeout
            )
        except Exception as exc:
            raise ExecTransportError(
                command=argv,
                context={"backend": "databricks", "sandbox_id": self.state.sandbox_id},
                cause=exc,
            ) from exc

        if response.status == _TIMED_OUT:
            raise ExecTimeoutError(command=argv, timeout_s=timeout)

        return ExecResult(
            stdout=(response.stdout or "").encode("utf-8"),
            stderr=(response.stderr or "").encode("utf-8"),
            exit_code=response.exit_code if response.exit_code is not None else 1,
        )

    # -- file transfer (chunked base64 over exec) ----------------------------

    async def _upload_bytes(self, path: str, content: bytes) -> None:
        encoded = base64.b64encode(content).decode("ascii")
        pieces = [
            encoded[i : i + _UPLOAD_B64_CHUNK] for i in range(0, len(encoded), _UPLOAD_B64_CHUNK)
        ] or [""]
        for index, piece in enumerate(pieces):
            # First chunk creates parent dirs and truncates; later chunks append.
            redirect = ">" if index == 0 else ">>"
            mkdir = 'mkdir -p "$(dirname "$DBX_PATH")"; ' if index == 0 else ""
            script = (
                f'set -e -o pipefail; {mkdir}printf %s "$DBX_B64" '
                f'| base64 -d {redirect} "$DBX_PATH"'
            )
            response = await self._exec_sync(
                _SHELL, ["-lc", script], envs={"DBX_PATH": path, "DBX_B64": piece}
            )
            if not _completed(response):
                raise WorkspaceArchiveWriteError(
                    path=Path(path),
                    context={
                        "backend": "databricks",
                        "stderr": (response.stderr or "")[:4096],
                        "stdout": (response.stdout or "")[:4096],
                    },
                )

    async def _download_bytes(self, path: str) -> bytes:
        buf = bytearray()
        offset = 0
        while True:
            # tail -c is 1-based; read a raw byte range and base64 it so each response
            # stays under the unary cap regardless of file size.
            script = 'tail -c +$((DBX_OFF + 1)) "$DBX_PATH" | head -c "$DBX_LEN" | base64'
            response = await self._exec_sync(
                _SHELL,
                ["-lc", script],
                envs={
                    "DBX_PATH": path,
                    "DBX_OFF": str(offset),
                    "DBX_LEN": str(_DOWNLOAD_RAW_CHUNK),
                },
            )
            if not _completed(response):
                raise WorkspaceArchiveReadError(
                    path=Path(path),
                    context={
                        "backend": "databricks",
                        "stderr": (response.stderr or "")[:4096],
                    },
                )
            chunk = base64.b64decode(response.stdout or "")
            buf.extend(chunk)
            if len(chunk) < _DOWNLOAD_RAW_CHUNK:
                break
            offset += _DOWNLOAD_RAW_CHUNK
        return bytes(buf)

    async def _classify_path(self, path: str) -> str:
        script = (
            'if [ ! -e "$DBX_PATH" ]; then echo MISSING; '
            'elif [ -d "$DBX_PATH" ]; then echo ISDIR; '
            'elif [ ! -r "$DBX_PATH" ]; then echo NOREAD; '
            "else echo OK; fi"
        )
        response = await self._exec_sync(_SHELL, ["-lc", script], envs={"DBX_PATH": path})
        token = (response.stdout or "").strip()
        return {
            "MISSING": "file_not_found",
            "ISDIR": "is_directory",
            "NOREAD": "permission_denied",
        }.get(token, "OK")

    def _reject_user_arg(self, op: Literal["exec", "read", "write"], user: str | User) -> None:
        name = user.name if isinstance(user, User) else user
        raise ConfigurationError(
            message=(
                "DatabricksSandboxSession does not support sandbox-local users; "
                f"`{op}` must be called without `user`"
            ),
            error_code=ErrorCode.SANDBOX_CONFIG_INVALID,
            op=op,
            context={"backend": "databricks", "user": name},
        )

    async def read(self, path: Path, *, user: str | User | None = None) -> io.IOBase:
        if user is not None:
            self._reject_user_arg("read", user)
        normalized = self.normalize_path(path)
        target = sandbox_path_str(normalized)
        kind = await self._classify_path(target)
        if kind == "file_not_found":
            raise WorkspaceReadNotFoundError(path=normalized)
        if kind != "OK":
            raise WorkspaceArchiveReadError(
                path=normalized, context={"backend": "databricks", "reason": kind}
            )
        return io.BytesIO(await self._download_bytes(target))

    async def write(self, path: Path, data: io.IOBase, *, user: str | User | None = None) -> None:
        if user is not None:
            self._reject_user_arg("write", user)
        normalized = self.normalize_path(path, for_write=True)
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, (bytes, bytearray)):
            raise WorkspaceWriteTypeError(path=normalized, actual_type=type(payload).__name__)
        await self._upload_bytes(sandbox_path_str(normalized), bytes(payload))

    # -- workspace persistence (tar + chunked base64) ------------------------

    def _archive_path(self) -> str:
        return f"/tmp/openai-agents-{self.state.session_id.hex}.tar"

    async def persist_workspace(self) -> io.IOBase:
        root = self.state.manifest.root
        archive = self._archive_path()
        excludes = [
            f"--exclude=./{rel.as_posix()}"
            for rel in sorted(self._persist_workspace_skip_relpaths(), key=lambda p: p.as_posix())
        ]
        tar_command = ("tar", "-C", root, "-cf", archive, "--no-wildcards", *excludes, ".")
        result = await self.exec(*tar_command, shell=False)
        if not result.ok():
            raise WorkspaceArchiveReadError(
                path=Path(root),
                context={
                    "backend": "databricks",
                    "stderr": result.stderr.decode("utf-8", errors="replace"),
                },
            )
        try:
            data = await self._download_bytes(archive)
        finally:
            await self._remove_archive(archive)
        return io.BytesIO(data)

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        raw = data.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if not isinstance(raw, (bytes, bytearray)):
            raise WorkspaceWriteTypeError(
                path=Path(self.state.manifest.root), actual_type=type(raw).__name__
            )
        root = self.state.manifest.root
        archive = self._archive_path()
        await self._upload_bytes(archive, bytes(raw))
        try:
            result = await self.exec("tar", "-C", root, "-xf", archive, shell=False)
            if not result.ok():
                raise WorkspaceArchiveWriteError(
                    path=Path(root),
                    context={
                        "backend": "databricks",
                        "stderr": result.stderr.decode("utf-8", errors="replace"),
                    },
                )
        finally:
            await self._remove_archive(archive)

    async def _remove_archive(self, archive: str) -> None:
        with contextlib.suppress(Exception):
            await self._exec_sync(_SHELL, ["-lc", 'rm -f "$DBX_PATH"'], envs={"DBX_PATH": archive})


# ``BaseSandboxClient`` is generic in its options type; subscripting binds the
# option class for type checkers. Fall back to the bare base (``object`` when the
# contract is unavailable) so importing this module never requires the extra.
if _sandbox_imports_available:
    _SandboxClientBase = BaseSandboxClient[DatabricksSandboxClientOptions]
else:
    _SandboxClientBase = BaseSandboxClient


class DatabricksSandboxClient(_SandboxClientBase):  # type: ignore[valid-type,misc]
    """Databricks-backed sandbox client for the OpenAI Agents SDK.

    Provisioning uses released ``databricks-sdk`` CRUD APIs. Pass a configured
    ``WorkspaceClient``, or a CLI ``profile`` to build one.
    """

    backend_id = "databricks"
    supports_default_options = True
    _instrumentation: Instrumentation

    def __init__(
        self,
        client: WorkspaceClient | None = None,
        *,
        profile: str | None = None,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        if not _sandbox_imports_available:
            raise ImportError(_IMPORT_ERROR)
        super().__init__()
        if client is None:
            client = WorkspaceClient(profile=profile) if profile else WorkspaceClient()
        self._client = client
        self._instrumentation = (
            instrumentation if instrumentation is not None else Instrumentation()
        )
        self._dependencies = dependencies

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: DatabricksSandboxClientOptions | None = None,
    ) -> SandboxSession:
        resolved_options = options if options is not None else DatabricksSandboxClientOptions()
        resolved_manifest = _resolve_manifest_root(manifest)
        self._validate_manifest_for_create(resolved_manifest)
        session_id = uuid.uuid4()
        sandbox_id = resolved_options.sandbox_id or f"oai-agents-{session_id.hex[:12]}"
        state = DatabricksSandboxSessionState(
            session_id=session_id,
            manifest=resolved_manifest,
            snapshot=resolve_snapshot(snapshot, str(session_id)),
            sandbox_id=sandbox_id,
            inactivity_timeout_s=resolved_options.inactivity_timeout_s,
            wait_for_running_timeout_s=resolved_options.wait_for_running_timeout_s,
        )
        inner = DatabricksSandboxSession.from_state(state, client=self._client)
        await inner._ensure_sandbox()
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def delete(self, session: SandboxSession) -> SandboxSession:
        """Delete the backing Databricks sandbox."""
        inner = session._inner
        if not isinstance(inner, DatabricksSandboxSession):
            msg = "DatabricksSandboxClient.delete expects a DatabricksSandboxSession"
            raise TypeError(msg)

        def _delete() -> None:
            with contextlib.suppress(NotFound):
                self._client.sandbox.delete_sandbox(inner._name)

        await asyncio.to_thread(_delete)
        return session

    async def resume(self, state: SandboxSessionState) -> SandboxSession:
        if not isinstance(state, DatabricksSandboxSessionState):
            msg = "DatabricksSandboxClient.resume expects a DatabricksSandboxSessionState"
            raise TypeError(msg)
        state.assert_path_grants_rebound()
        inner = DatabricksSandboxSession.from_state(state, client=self._client)
        reconnected = await inner._reconnect_if_running()
        inner._set_start_state_preserved(reconnected)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return self._deserialize_session_state_payload(payload, DatabricksSandboxSessionState)
