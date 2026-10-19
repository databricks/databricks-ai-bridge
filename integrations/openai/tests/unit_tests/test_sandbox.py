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
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
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


def _fake_module(name: str, **attrs: Any) -> Any:
    """Build a stand-in module and populate it with the given attributes."""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


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

    fake_databricks = _fake_module("databricks")
    fake_sdk = _fake_module("databricks.sdk", WorkspaceClient=_Placeholder)
    fake_errors = _fake_module("databricks.sdk.errors", NotFound=_NotFound)
    fake_service = _fake_module("databricks.sdk.service")
    fake_sandbox = _fake_module(
        "databricks.sdk.service.sandbox",
        ComputeSpec=_Placeholder,
        ExecuteCommandStatus=_ExecuteCommandStatus,
        ExecuteCommandSyncResponse=_ExecuteCommandSyncResponse,
        Sandbox=_Placeholder,
        SandboxSpec=_Placeholder,
        SandboxState=_SandboxState,
    )

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

    session._exec_sync = fake_exec_sync
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


def test_missing_extra_raises_import_error(mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_sandbox_imports_available", False)
    for construct in (
        lambda: mod.DatabricksSandboxClientOptions(),
        lambda: mod.DatabricksSandboxClient(),
        lambda: mod.DatabricksSandboxSession(state=None, client=None),
    ):
        with pytest.raises(ImportError, match=r"databricks-openai\[sandbox\]"):
            construct()


def test_resolve_manifest_root_defaults_to_persistent_volume(mod: Any) -> None:
    assert mod._resolve_manifest_root(None).root == mod.DEFAULT_DATABRICKS_WORKSPACE_ROOT
    # The SDK default root is rewritten to the writable, restart-surviving volume.
    assert mod._resolve_manifest_root(Manifest()).root == mod.DEFAULT_DATABRICKS_WORKSPACE_ROOT
    # An explicit custom root is preserved untouched.
    assert (
        mod._resolve_manifest_root(Manifest(root="/custom/workspace")).root == "/custom/workspace"
    )


# -- lifecycle fakes (provisioning / resume / shutdown branches) -------------

_RAISE_NOTFOUND = object()


def _async_return(value: Any) -> Any:
    async def _f(*args: Any, **kwargs: Any) -> Any:
        return value

    return _f


class _Status:
    def __init__(self, state: str) -> None:
        self.state = state


class _SandboxObj:
    def __init__(self, state: str | None) -> None:
        self.status = _Status(state) if state is not None else None


class _FakeLifecycleApi:
    """Recording fake of ``client.sandbox`` for CRUD/provisioning branches.

    ``get_results`` drives successive ``get_sandbox`` calls; each item is a state
    string or ``_RAISE_NOTFOUND`` (raise _NotFound), and the last item repeats.
    ``create_state`` is the state of the sandbox returned by ``create_sandbox``;
    ``stop_raises`` makes ``stop_sandbox`` raise _NotFound.
    """

    def __init__(
        self,
        *,
        get_results: list[object] | None = None,
        create_state: str = _SandboxState.SANDBOX_STATE_RUNNING,
        stop_raises: bool = False,
    ) -> None:
        self._get_results = list(get_results or [_SandboxState.SANDBOX_STATE_RUNNING])
        self._get_i = 0
        self._create_state = create_state
        self._stop_raises = stop_raises
        self.calls: list[tuple[str, object]] = []

    def _next_get(self) -> object:
        result = self._get_results[min(self._get_i, len(self._get_results) - 1)]
        self._get_i += 1
        return result

    def get_sandbox(self, name: str) -> _SandboxObj:
        self.calls.append(("get", name))
        result = self._next_get()
        if result is _RAISE_NOTFOUND:
            raise _NotFound
        return _SandboxObj(result)  # type: ignore[arg-type]

    def create_sandbox(self, sandbox: object, sandbox_id: str | None = None) -> _SandboxObj:
        self.calls.append(("create", sandbox_id))
        return _SandboxObj(self._create_state)

    def start_sandbox(self, name: str) -> None:
        self.calls.append(("start", name))

    def stop_sandbox(self, name: str) -> None:
        self.calls.append(("stop", name))
        if self._stop_raises:
            raise _NotFound

    def delete_sandbox(self, name: str) -> None:
        self.calls.append(("delete", name))


class _FakeLifecycleClient:
    def __init__(self, api: _FakeLifecycleApi) -> None:
        self.sandbox = api


def _lifecycle_session(mod: Any, api: _FakeLifecycleApi, **state_kwargs: Any) -> Any:
    kwargs = {
        "session_id": uuid.uuid4(),
        "manifest": Manifest(root=ROOT),
        "snapshot": resolve_snapshot(None, "sid"),
        "sandbox_id": "sb1",
        **state_kwargs,
    }
    state = mod.DatabricksSandboxSessionState(**kwargs)
    return mod.DatabricksSandboxSession.from_state(state, client=_FakeLifecycleClient(api))


def _kinds(api: _FakeLifecycleApi) -> list[str]:
    return [call[0] for call in api.calls]


RUNNING = _SandboxState.SANDBOX_STATE_RUNNING
STOPPED = _SandboxState.SANDBOX_STATE_STOPPED


@pytest.mark.asyncio
async def test_provision_creates_new_sandbox_when_absent(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "_WAIT_POLL_INTERVAL_S", 0)
    api = _FakeLifecycleApi(get_results=[_RAISE_NOTFOUND, RUNNING])
    session = _lifecycle_session(mod, api)
    await session._ensure_sandbox()
    assert ("create", "sb1") in api.calls
    assert "start" not in _kinds(api)
    assert session._provisioned is True


@pytest.mark.asyncio
async def test_provision_starts_stopped_sandbox(mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_WAIT_POLL_INTERVAL_S", 0)
    api = _FakeLifecycleApi(get_results=[STOPPED, RUNNING])
    session = _lifecycle_session(mod, api)
    await session._ensure_sandbox()
    assert "create" not in _kinds(api)
    assert ("start", "sandboxes/sb1") in api.calls


@pytest.mark.asyncio
async def test_provision_running_sandbox_is_noop(mod: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "_WAIT_POLL_INTERVAL_S", 0)
    api = _FakeLifecycleApi(get_results=[RUNNING])
    session = _lifecycle_session(mod, api)
    await session._ensure_sandbox()
    assert "create" not in _kinds(api)
    assert "start" not in _kinds(api)
    assert session._provisioned is True


def test_wait_for_running_timeout_raises(mod: Any) -> None:
    session = _lifecycle_session(mod, _FakeLifecycleApi(get_results=[STOPPED]))
    session.state.wait_for_running_timeout_s = 0
    with pytest.raises(WorkspaceStartError) as exc:
        session._wait_for_running()
    assert exc.value.context["reason"] == "sandbox_did_not_reach_running"


@pytest.mark.asyncio
async def test_reconnect_if_running_true_sets_provisioned(mod: Any) -> None:
    session = _lifecycle_session(mod, _FakeLifecycleApi(get_results=[RUNNING]))
    assert await session._reconnect_if_running() is True
    assert session._provisioned is True


@pytest.mark.asyncio
async def test_reconnect_if_running_false_when_absent(mod: Any) -> None:
    session = _lifecycle_session(mod, _FakeLifecycleApi(get_results=[_RAISE_NOTFOUND]))
    assert await session._reconnect_if_running() is False
    assert session._provisioned is False


@pytest.mark.asyncio
async def test_resume_rejects_non_databricks_state(mod: Any) -> None:
    client = mod.DatabricksSandboxClient(client=object())
    with pytest.raises(TypeError):
        await client.resume(object())


@pytest.mark.asyncio
async def test_create_rewrites_root_and_generates_sandbox_id(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _noop(self: Any) -> None:
        self._provisioned = True

    monkeypatch.setattr(mod.DatabricksSandboxSession, "_ensure_sandbox", _noop)
    client = mod.DatabricksSandboxClient(client=object())
    session = await client.create()
    inner = session._inner
    assert inner.state.manifest.root == mod.DEFAULT_DATABRICKS_WORKSPACE_ROOT
    assert inner.state.sandbox_id.startswith("oai-agents-")


@pytest.mark.asyncio
async def test_shutdown_stops_and_clears_provisioned(mod: Any) -> None:
    api = _FakeLifecycleApi()
    session = _lifecycle_session(mod, api)
    session._provisioned = True
    await session._shutdown_backend()
    assert ("stop", "sandboxes/sb1") in api.calls
    assert session._provisioned is False


@pytest.mark.asyncio
async def test_shutdown_noop_when_not_provisioned(mod: Any) -> None:
    api = _FakeLifecycleApi()
    session = _lifecycle_session(mod, api)
    session._provisioned = False
    await session._shutdown_backend()
    assert "stop" not in _kinds(api)


@pytest.mark.asyncio
async def test_shutdown_suppresses_not_found(mod: Any) -> None:
    api = _FakeLifecycleApi(stop_raises=True)
    session = _lifecycle_session(mod, api)
    session._provisioned = True
    await session._shutdown_backend()  # must not raise
    assert session._provisioned is False


@pytest.mark.asyncio
async def test_create_deletes_self_created_sandbox_on_provision_failure(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(self: Any) -> None:
        raise RuntimeError("provision failed")

    monkeypatch.setattr(mod.DatabricksSandboxSession, "_ensure_sandbox", _boom)
    api = _FakeLifecycleApi()
    client = mod.DatabricksSandboxClient(client=_FakeLifecycleClient(api))
    with pytest.raises(RuntimeError):
        await client.create()
    assert "delete" in _kinds(api)


@pytest.mark.asyncio
async def test_create_keeps_user_supplied_sandbox_on_failure(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _boom(self: Any) -> None:
        raise RuntimeError("provision failed")

    monkeypatch.setattr(mod.DatabricksSandboxSession, "_ensure_sandbox", _boom)
    api = _FakeLifecycleApi()
    client = mod.DatabricksSandboxClient(client=_FakeLifecycleClient(api))
    with pytest.raises(RuntimeError):
        await client.create(options=mod.DatabricksSandboxClientOptions(sandbox_id="user-sb"))
    assert "delete" not in _kinds(api)


@pytest.mark.asyncio
async def test_classify_failed_response_maps_to_error(mod: Any) -> None:
    session, _ = _session(mod, [_response(exit_code=1, status=COMPLETED)])
    assert await session._classify_path("/w/x") == "classify_failed"


@pytest.mark.asyncio
async def test_classify_unexpected_output_maps_to_error(mod: Any) -> None:
    session, _ = _session(mod, [_response(stdout="login banner noise\n")])
    assert await session._classify_path("/w/x") == "unexpected_classify_output"


@pytest.mark.asyncio
async def test_read_unexpected_classify_raises_archive_error(mod: Any) -> None:
    session, _ = _session(mod, [_response(stdout="garbage\n")])
    with pytest.raises(WorkspaceArchiveReadError):
        await session.read(Path(f"{ROOT}/f.txt"))


@pytest.mark.asyncio
async def test_download_script_uses_pipefail(mod: Any) -> None:
    session, calls = _session(mod, [_response(stdout=base64.b64encode(b"x").decode("ascii"))])
    await session._download_bytes("/w/f.bin")
    assert "set -o pipefail" in calls[0]["args"][1]


@pytest.mark.asyncio
async def test_persist_workspace_builds_excludes_and_cleans_up(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _lifecycle_session(mod, _FakeLifecycleApi())
    session._provisioned = True
    exec_calls: list[tuple[Any, ...]] = []

    async def fake_exec(*command: Any, **kwargs: Any) -> Any:
        exec_calls.append(command)
        return mod.ExecResult(stdout=b"", stderr=b"", exit_code=0)

    removed: list[str] = []

    async def fake_remove(archive: str) -> None:
        removed.append(archive)

    monkeypatch.setattr(session, "exec", fake_exec)
    monkeypatch.setattr(session, "_download_bytes", _async_return(b"archive-bytes"))
    monkeypatch.setattr(session, "_remove_archive", fake_remove)
    monkeypatch.setattr(
        session, "_persist_workspace_skip_relpaths", lambda: {Path("a"), Path("b/c")}
    )
    stream = await session.persist_workspace()
    assert stream.read() == b"archive-bytes"
    tar = exec_calls[0]
    assert "--no-wildcards" in tar
    assert "--exclude=./a" in tar
    assert "--exclude=./b/c" in tar
    assert removed  # archive cleaned up


@pytest.mark.asyncio
async def test_persist_workspace_raises_on_tar_failure(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _lifecycle_session(mod, _FakeLifecycleApi())
    session._provisioned = True

    async def fake_exec(*command: Any, **kwargs: Any) -> Any:
        return mod.ExecResult(stdout=b"", stderr=b"boom", exit_code=1)

    monkeypatch.setattr(session, "exec", fake_exec)
    monkeypatch.setattr(session, "_persist_workspace_skip_relpaths", set)
    with pytest.raises(WorkspaceArchiveReadError):
        await session.persist_workspace()


@pytest.mark.asyncio
async def test_persist_workspace_cleans_up_on_download_failure(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _lifecycle_session(mod, _FakeLifecycleApi())
    session._provisioned = True
    removed: list[str] = []

    async def fake_exec(*command: Any, **kwargs: Any) -> Any:
        return mod.ExecResult(stdout=b"", stderr=b"", exit_code=0)

    async def failing_download(archive: str) -> bytes:
        raise RuntimeError("network down")

    async def fake_remove(archive: str) -> None:
        removed.append(archive)

    monkeypatch.setattr(session, "exec", fake_exec)
    monkeypatch.setattr(session, "_download_bytes", failing_download)
    monkeypatch.setattr(session, "_remove_archive", fake_remove)
    monkeypatch.setattr(session, "_persist_workspace_skip_relpaths", set)
    with pytest.raises(RuntimeError):
        await session.persist_workspace()
    assert removed  # finally-block cleanup still ran
