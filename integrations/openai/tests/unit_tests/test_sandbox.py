"""Unit tests for the DatabricksSandbox -> execute_command_sync mapping.

The ``databricks-sdk[sandbox]`` dependency is optional, so these tests inject a
fake ``databricks`` SDK module tree into ``sys.modules`` (the same pattern the
Daytona and Vercel backend tests upstream use) and import the backend module
fresh against it. They then patch the adapter's single ``execute_command_sync``
wrapper (``_exec_sync``) so the mapping logic -- login-shell wrapping, output
mapping, chunked base64 transfer, path classification, and session-state
registration -- is exercised without a live sandbox or the real SDK installed.
"""

from __future__ import annotations

import base64
import importlib
import io
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("agents.sandbox.session.sandbox_client")

from agents.sandbox.errors import (  # noqa: E402
    ExecTimeoutError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
)
from agents.sandbox.manifest import Manifest  # noqa: E402
from agents.sandbox.snapshot import resolve_snapshot  # noqa: E402

ROOT = "/home/sandbox-agent/agent-workspace"


class _ExecuteCommandStatus:
    EXECUTE_COMMAND_STATUS_COMPLETED = "completed"
    EXECUTE_COMMAND_STATUS_TIMED_OUT = "timed_out"


class _SandboxState:
    SANDBOX_STATE_RUNNING = "running"
    SANDBOX_STATE_STOPPED = "stopped"
    SANDBOX_STATE_STOPPING = "stopping"


class _ExecuteCommandSyncResponse:
    def __init__(
        self,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = 0,
        status: str = _ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED,
        truncated: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.status = status
        self.truncated = truncated


class _NotFound(Exception):
    pass


class _Placeholder:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self._args = args
        self._kwargs = kwargs


COMPLETED = _ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED
TIMED_OUT = _ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_TIMED_OUT


def _response(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = 0,
    status: str = COMPLETED,
    truncated: bool = False,
) -> _ExecuteCommandSyncResponse:
    return _ExecuteCommandSyncResponse(
        stdout=stdout, stderr=stderr, exit_code=exit_code, status=status, truncated=truncated
    )


@pytest.fixture
def mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Inject a fake ``databricks`` SDK and import the backend module against it."""
    # Load the package (and its real dependencies) once before swapping in fakes,
    # so popping and reimporting the submodule below does not rerun the heavy
    # ``databricks_openai`` package __init__ under the fake SDK.
    import databricks_openai  # noqa: F401

    fake_databricks: Any = types.ModuleType("databricks")
    fake_sdk: Any = types.ModuleType("databricks.sdk")
    fake_sdk.WorkspaceClient = _Placeholder
    fake_errors: Any = types.ModuleType("databricks.sdk.errors")
    fake_errors.NotFound = _NotFound
    fake_service: Any = types.ModuleType("databricks.sdk.service")
    fake_sandbox: Any = types.ModuleType("databricks.sdk.service.sandbox")
    fake_sandbox.ComputeSpec = _Placeholder
    fake_sandbox.ExecuteCommandStatus = _ExecuteCommandStatus
    fake_sandbox.ExecuteCommandSyncResponse = _ExecuteCommandSyncResponse
    fake_sandbox.Sandbox = _Placeholder
    fake_sandbox.SandboxSpec = _Placeholder
    fake_sandbox.SandboxState = _SandboxState

    monkeypatch.setitem(sys.modules, "databricks", fake_databricks)
    monkeypatch.setitem(sys.modules, "databricks.sdk", fake_sdk)
    monkeypatch.setitem(sys.modules, "databricks.sdk.errors", fake_errors)
    monkeypatch.setitem(sys.modules, "databricks.sdk.service", fake_service)
    monkeypatch.setitem(sys.modules, "databricks.sdk.service.sandbox", fake_sandbox)
    sys.modules.pop("databricks_openai.sandbox", None)
    return importlib.import_module("databricks_openai.sandbox")


def _session(mod: Any, responses: list[_ExecuteCommandSyncResponse]) -> tuple[Any, list[dict]]:
    """Build a session whose ``_exec_sync`` is replaced by a recording stub.

    ``responses`` is a list consumed in call order (the last entry repeats).
    """
    state = mod.DatabricksSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root=ROOT),
        snapshot=resolve_snapshot(None, "sid"),
        sandbox_id="sb1",
    )
    session = mod.DatabricksSandboxSession.from_state(state, client=object())
    session._provisioned = True
    calls: list[dict] = []

    async def fake_exec_sync(program, args, *, envs=None, timeout=None):
        calls.append({"program": program, "args": args, "envs": envs, "timeout": timeout})
        return responses[min(len(calls) - 1, len(responses) - 1)]

    session._exec_sync = fake_exec_sync  # type: ignore[method-assign]
    return session, calls


def test_exports_right_base_classes(mod: Any) -> None:
    from agents.sandbox.session.base_sandbox_session import BaseSandboxSession
    from agents.sandbox.session.sandbox_client import BaseSandboxClient

    assert issubclass(mod.DatabricksSandboxClient, BaseSandboxClient)
    assert issubclass(mod.DatabricksSandboxSession, BaseSandboxSession)


def test_no_pty_support(mod: Any) -> None:
    # The buffered backend leaves the inherited default: no interactive PTY, so
    # the Shell capability does not expose its write_stdin tool.
    session, _ = _session(mod, [_response()])
    assert session.supports_pty() is False


@pytest.mark.asyncio
async def test_exec_internal_wraps_login_shell_and_maps_output(mod: Any) -> None:
    session, calls = _session(
        mod, [_response(stdout="out", stderr="err", exit_code=3, truncated=True)]
    )
    result = await session._exec_internal("ls", "-la")
    assert result.stdout == b"out"
    assert result.stderr == b"err"
    assert result.exit_code == 3
    assert calls[0]["program"] == "/bin/bash"
    assert calls[0]["args"] == ["-lc", mod._CD_EXEC_SCRIPT, "/bin/bash", "ls", "-la"]
    assert calls[0]["envs"] == {"DBX_CWD": ROOT}


@pytest.mark.asyncio
async def test_exec_internal_timed_out_raises(mod: Any) -> None:
    session, _ = _session(mod, [_response(status=TIMED_OUT, exit_code=None)])
    with pytest.raises(ExecTimeoutError):
        await session._exec_internal("sleep", "100", timeout=1)


@pytest.mark.asyncio
async def test_exec_internal_null_exit_code_maps_to_nonzero(mod: Any) -> None:
    session, _ = _session(mod, [_response(stderr="killed", exit_code=None, status=COMPLETED)])
    result = await session._exec_internal("false")
    assert result.exit_code == 1
    assert not result.ok()


@pytest.mark.asyncio
async def test_upload_single_chunk_sends_base64_and_creates_parent_dir(mod: Any) -> None:
    content = bytes(range(16))
    session, calls = _session(mod, [_response()])
    await session._upload_bytes("/w/f.bin", content)
    assert len(calls) == 1
    assert calls[0]["envs"]["DBX_PATH"] == "/w/f.bin"
    assert calls[0]["envs"]["DBX_B64"] == base64.b64encode(content).decode("ascii")
    script = calls[0]["args"][1]
    assert "mkdir -p" in script
    assert "base64 -d >" in script


@pytest.mark.asyncio
async def test_upload_multi_chunk_splits_and_appends(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "_UPLOAD_B64_CHUNK", 8)
    content = b"abcdefghijklmnopqrstuvwxyz"
    encoded = base64.b64encode(content).decode("ascii")
    session, calls = _session(mod, [_response()])
    await session._upload_bytes("/w/f.bin", content)
    assert len(calls) > 1
    assert "".join(call["envs"]["DBX_B64"] for call in calls) == encoded
    redirects = ["base64 -d >> " not in call["args"][1] for call in calls]
    assert redirects[0] is True  # first chunk truncates
    assert all(not r for r in redirects[1:])  # later chunks append


@pytest.mark.asyncio
async def test_upload_reports_error_on_nonzero_exit(mod: Any) -> None:
    session, _ = _session(mod, [_response(stderr="disk full", exit_code=1)])
    with pytest.raises(WorkspaceArchiveWriteError):
        await session._upload_bytes("/w/f.bin", b"x")


@pytest.mark.asyncio
async def test_download_reassembles_chunks(mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_DOWNLOAD_RAW_CHUNK", 4)
    payload = b"hello world"  # 11 bytes -> chunks of 4, 4, 3 (last < 4 stops the loop)
    responses = [
        _response(stdout=base64.b64encode(payload[0:4]).decode("ascii")),
        _response(stdout=base64.b64encode(payload[4:8]).decode("ascii")),
        _response(stdout=base64.b64encode(payload[8:]).decode("ascii")),
    ]
    session, calls = _session(mod, responses)
    data = await session._download_bytes("/w/f.bin")
    assert data == payload
    assert calls[0]["envs"] == {"DBX_PATH": "/w/f.bin", "DBX_OFF": "0", "DBX_LEN": "4"}
    assert calls[1]["envs"]["DBX_OFF"] == "4"


@pytest.mark.asyncio
async def test_classify_maps_tokens(mod: Any) -> None:
    for token, expected in [
        ("MISSING", "file_not_found"),
        ("ISDIR", "is_directory"),
        ("NOREAD", "permission_denied"),
        ("OK", "OK"),
    ]:
        session, _ = _session(mod, [_response(stdout=f"{token}\n")])
        assert await session._classify_path("/w/x") == expected


@pytest.mark.asyncio
async def test_read_missing_raises_not_found(mod: Any) -> None:
    session, _ = _session(mod, [_response(stdout="MISSING\n")])
    with pytest.raises(WorkspaceReadNotFoundError):
        await session.read(Path(f"{ROOT}/missing.txt"))


@pytest.mark.asyncio
async def test_read_ok_decodes_file(mod: Any) -> None:
    payload = b"hello world"
    responses = [
        _response(stdout="OK\n"),  # classify
        _response(stdout=base64.b64encode(payload).decode("ascii")),  # download
    ]
    session, _ = _session(mod, responses)
    stream = await session.read(Path(f"{ROOT}/f.txt"))
    assert stream.read() == payload


@pytest.mark.asyncio
async def test_write_uploads_payload(mod: Any) -> None:
    session, calls = _session(mod, [_response()])
    await session.write(Path(f"{ROOT}/f.txt"), io.BytesIO(b"data"))
    assert calls[0]["envs"]["DBX_PATH"] == f"{ROOT}/f.txt"
    assert calls[0]["envs"]["DBX_B64"] == base64.b64encode(b"data").decode("ascii")


def test_options_accepts_positional_and_defaults(mod: Any) -> None:
    options = mod.DatabricksSandboxClientOptions("my-sandbox", 600)
    assert options.sandbox_id == "my-sandbox"
    assert options.inactivity_timeout_s == 600
    assert options.type == "databricks"
    defaults = mod.DatabricksSandboxClientOptions()
    assert defaults.inactivity_timeout_s == mod.DEFAULT_INACTIVITY_TIMEOUT_S


def test_session_state_round_trips_through_client(mod: Any) -> None:
    client = mod.DatabricksSandboxClient(client=object())
    state = mod.DatabricksSandboxSessionState(
        session_id=uuid.uuid4(),
        manifest=Manifest(root=ROOT),
        snapshot=resolve_snapshot(None, "sid"),
        sandbox_id="sb-42",
    )
    restored = client.deserialize_session_state(state.model_dump(mode="json"))
    assert isinstance(restored, mod.DatabricksSandboxSessionState)
    assert restored.sandbox_id == "sb-42"
    assert restored.manifest.root == ROOT
