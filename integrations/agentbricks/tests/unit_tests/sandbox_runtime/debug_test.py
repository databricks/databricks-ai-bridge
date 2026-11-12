"""Offline developer-terminal authorization, session ownership and SSH lifecycle checks."""

import asyncio
import base64
import hashlib
import shlex
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from databricks.sdk.service.apps import (
    AppAccessControlResponse,
    AppPermission,
    AppPermissionLevel,
    AppPermissions,
)
from databricks.sdk.service.iam import ComplexValue, User
from fastapi import FastAPI, HTTPException, WebSocketDisconnect
from fastapi.testclient import TestClient
from paramiko.common import cMSG_CHANNEL_REQUEST
from paramiko.message import Message

from databricks_agentkit.sandbox import debug
from databricks_agentkit.sandbox.control import scoped_session
from databricks_agentkit.sandbox.environments import EnvironmentStore
from databricks_agentkit.sandbox.lakebox import Lakebox, RemoteError
from databricks_agentkit.sandbox.store import ExecutionIdentityConflict, SessionBusy


@pytest.fixture(autouse=True)
def debug_environment(monkeypatch):
    monkeypatch.delenv("DATABRICKS_APP_NAME", raising=False)
    monkeypatch.delenv("SANDBOX_ALLOW_LOCAL_DEBUG", raising=False)


def acl(*, user=None, group=None, manage=True, inherited=False):
    return AppAccessControlResponse(
        user_name=user,
        group_name=group,
        all_permissions=[
            AppPermission(
                permission_level=AppPermissionLevel.CAN_MANAGE
                if manage
                else AppPermissionLevel.CAN_USE,
                inherited=inherited,
            )
        ],
    )


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setenv("DATABRICKS_APP_NAME", "debug-app")
    developer = MagicMock()
    developer.current_user.me.return_value = User(
        user_name="dev@example.com", groups=[ComplexValue(display="developers", value="group-id")]
    )
    developer.apps.get_permissions.return_value = AppPermissions(access_control_list=[])
    factory = MagicMock(return_value=developer)
    monkeypatch.setattr(debug, "WorkspaceClient", factory)
    workspace = SimpleNamespace(config=SimpleNamespace(host="https://workspace.example"))
    return developer, factory, workspace


@pytest.mark.parametrize(
    "entry",
    [
        acl(user="dev@example.com"),
        acl(user="dev@example.com", inherited=True),
        acl(group="developers"),
        acl(group="developers", inherited=True),
    ],
)
def test_authorizer_requires_verified_user_or_group_manage(auth, entry):
    developer, factory, workspace = auth
    developer.apps.get_permissions.return_value = AppPermissions(access_control_list=[entry])
    assert (
        debug.authorize(
            {"x-agentbricks-debug-authorization": "Bearer developer-token"}, "127.0.0.1", workspace
        )
        == "dev@example.com"
    )
    factory.assert_called_once_with(
        host="https://workspace.example", token="developer-token", auth_type="pat"
    )
    developer.current_user.me.assert_called_once_with()
    developer.apps.get_permissions.assert_called_once_with("debug-app")


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [acl(user="dev@example.com", manage=False)],
        [acl(group="developers", manage=False)],
        [acl(user="someone-else@example.com")],
        [acl(group="other-group")],
    ],
)
def test_authorizer_rejects_use_or_unmatched_manage(auth, entries):
    developer, _, workspace = auth
    developer.apps.get_permissions.return_value = AppPermissions(access_control_list=entries)
    with pytest.raises(HTTPException) as failure:
        debug.authorize(
            {"x-agentbricks-debug-authorization": "Bearer developer-token"}, "127.0.0.1", workspace
        )
    assert failure.value.status_code == 403


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"x-agentbricks-debug-authorization": "Bearer "},
        {"x-agentbricks-debug-authorization": "Basic token"},
        {
            "origin": "https://browser.example",
            "x-agentbricks-debug-authorization": "Bearer developer-token",
        },
        {"origin": "null"},
    ],
)
def test_authorizer_rejects_missing_tokens_and_browser_origins_before_sdk(auth, headers):
    _, factory, workspace = auth
    with pytest.raises(HTTPException) as failure:
        debug.authorize(headers, "127.0.0.1", workspace)
    assert failure.value.status_code == 403
    factory.assert_not_called()


@pytest.mark.parametrize(
    "enabled,peer,allowed",
    [
        ("true", "127.0.0.1", True),
        ("true", "::1", True),
        ("false", "127.0.0.1", False),
        ("true", "10.0.0.1", False),
        ("true", "localhost", False),
        ("TRUE", "127.0.0.1", False),
    ],
)
def test_local_debug_is_explicit_and_loopback_only(monkeypatch, enabled, peer, allowed):
    monkeypatch.setenv("SANDBOX_ALLOW_LOCAL_DEBUG", enabled)
    if allowed:
        assert debug.authorize({}, peer, None) == "local-developer"
        with pytest.raises(HTTPException):
            debug.authorize({"origin": "http://localhost"}, peer, None)
    else:
        with pytest.raises(HTTPException):
            debug.authorize({}, peer, None)


from databricks_agentkit.sandbox.session_client import SessionSandboxClient


class Registry:
    def __init__(self):
        self.row = {"sandbox_id": "sandbox-one", "source_hash": "a" * 64, "status": "ready"}
        self.owner = "app-principal"
        self.locked = False
        self.active_keys = set()
        self.busy = False
        self.keys = []
        self.saved = []

    @contextmanager
    def session(self, key):
        self.keys.append(key)
        if self.busy:
            raise SessionBusy("Session is running")
        assert key not in self.active_keys
        self.active_keys.add(key)
        self.locked = True
        try:
            yield self
        finally:
            self.active_keys.remove(key)
            self.locked = bool(self.active_keys)

    def get(self):
        return self.row

    def bind_owner(self, principal):
        if self.owner and self.owner != principal:
            raise ExecutionIdentityConflict("Another execution identity owns the sandbox")
        self.owner = principal

    def runtime(self, key):
        return {"sandbox": self.row, "execution_principal": self.owner}

    def save(self, digest, sandbox_id, status, metrics=None):
        self.saved.append((digest, sandbox_id, status))


@pytest.fixture
def runtime():
    registry = Registry()
    lakebox, ssh, channel, client = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    lakebox.get.return_value = {"sandboxId": "sandbox-one", "status": "Running"}
    lakebox.start.return_value = lakebox.get.return_value
    ssh.open_terminal.return_value = (client, channel)
    channel.recv_ready.return_value = False
    channel.recv_stderr_ready.return_value = False
    channel.exit_status_ready.return_value = True
    channel.recv_exit_status.return_value = 0
    channel.closed = False
    orchestrator = SimpleNamespace(
        registry=registry,
        lakebox=lakebox,
        namespace="environment-namespace",
        execution_session_id=lambda key: key,
        execution_principal=lambda: "app-principal",
    )
    orchestrator.sandboxes = SessionSandboxClient(registry, lakebox, orchestrator.namespace)
    orchestrator.sandboxes._principal = "app-principal"
    state = SimpleNamespace(
        orchestrator=orchestrator, debug_lakebox=ssh, environments=MagicMock(), control=MagicMock()
    )
    state.environments.session_agent.return_value = "pinned-agent"
    state.control.agent.return_value = {"id": "pinned-agent", "execution_artifact": None}
    return state, registry, lakebox, ssh, client, channel


def test_open_uses_pinned_session_and_releases_lock_without_create_or_install(runtime):
    state, registry, lakebox, ssh, client, channel = runtime

    def open_terminal(*args, **kwargs):
        assert registry.locked
        return client, channel

    ssh.open_terminal.side_effect = open_terminal
    result = debug.open_session(
        state, "environment", debug.TerminalRequest(session_id="conversation")
    )
    assert result[:2] == (client, channel)
    assert result[2]["agent_id"] == "pinned-agent"
    assert result[2]["sandbox_id"] == "sandbox-one"
    state.environments.session_agent.assert_called_once_with("environment", "conversation")
    key = scoped_session("pinned-agent", "conversation")
    assert registry.keys == ["invocation:" + key, key, key]
    assert not registry.locked
    assert registry.saved == [("a" * 64, "sandbox-one", "ready")]
    deployment = hashlib.sha256(b"environment-namespace").hexdigest()[:16]
    assert f"{deployment}/{'a' * 64}/source" in result[2]["source_directory"]
    assert b". .venv/bin/activate" in channel.sendall.call_args.args[0]
    lakebox.create.assert_not_called()
    lakebox.connect.assert_not_called()
    lakebox.start.assert_not_called()
    state.control.create_run.assert_not_called()
    client.close.assert_not_called()


@pytest.mark.parametrize("command", [None, "python -V"])
def test_missing_installation_files_refuse_terminal_and_release_lock(runtime, command):
    state, registry, lakebox, ssh, _, _ = runtime
    connection = ssh.connect.return_value
    remote = connection.__enter__.return_value

    def missing_files(probe, *, timeout):
        assert registry.locked
        assert timeout == 30
        assert "ready.json" in probe and ".venv/bin/activate" in probe
        raise RemoteError("Installed files are missing")

    remote.run.side_effect = missing_files
    with pytest.raises(HTTPException) as failure:
        debug.open_session(
            state, "environment", debug.TerminalRequest(session_id="conversation", command=command)
        )
    assert failure.value.status_code == 409
    assert "Installed agent files are missing" in failure.value.detail
    assert not registry.locked
    assert registry.saved == []
    connection.__exit__.assert_called_once()
    ssh.open_terminal.assert_not_called()
    lakebox.create.assert_not_called()
    lakebox.start.assert_not_called()
    lakebox.connect.assert_not_called()
    state.control.create_run.assert_not_called()


def test_command_runs_inside_installed_virtualenv_and_preserves_shell_quoting(runtime):
    state, _, _, ssh, _, channel = runtime
    command = "python -c 'print(42)'"
    debug.open_session(
        state, "environment", debug.TerminalRequest(session_id="conversation", command=command)
    )
    wrapped = ssh.open_terminal.call_args.args[1]
    shell = shlex.split(wrapped)
    assert shell[:2] == ["bash", "-c"]
    assert ". .venv/bin/activate && exec bash -c " + shlex.quote(command) in shell[2]
    channel.sendall.assert_not_called()


@pytest.mark.parametrize("command", [None, "python -V"])
def test_https_session_opens_grpc_adapter_without_ssh_or_duplicate_setup(
    runtime, monkeypatch, command
):
    state, registry, lakebox, connection, _, _ = runtime
    state.debug_transport = "https"
    state.debug_lakebox = lakebox
    adapter = MagicMock()

    def open_grpc(workspace, sandbox_id, setup, request):
        assert registry.locked
        assert workspace is lakebox.workspace
        assert sandbox_id == "sandbox-one"
        assert "cd --" in setup and ". .venv/bin/activate" in setup
        assert request.command == command
        return adapter

    factory = MagicMock(side_effect=open_grpc)
    monkeypatch.setattr(debug, "GrpcTerminal", factory)
    result = debug.open_session(
        state, "environment", debug.TerminalRequest(session_id="conversation", command=command)
    )
    assert result[:2] == (adapter, adapter)
    assert result[2]["transport"] == "https"
    assert not registry.locked
    factory.assert_called_once()
    connection.open_terminal.assert_not_called()
    connection.connect.assert_not_called()
    adapter.sendall.assert_not_called()
    lakebox.connect.return_value.__enter__.return_value.run.assert_called_once()
    lakebox.create.assert_not_called()
    lakebox.start.assert_not_called()


@pytest.mark.parametrize(
    "row,status",
    [
        (None, 404),
        ({"sandbox_id": None}, 404),
        ({"sandbox_id": "sandbox-one", "status": "failed"}, 409),
        ({"sandbox_id": "sandbox-one", "status": "bootstrapping"}, 409),
    ],
)
def test_uncreated_or_uninstalled_session_never_opens_ssh(runtime, row, status):
    state, registry, lakebox, ssh, _, _ = runtime
    registry.row = row
    with pytest.raises(HTTPException) as failure:
        debug.open_session(state, "environment", debug.TerminalRequest(session_id="conversation"))
    assert failure.value.status_code == status
    assert not registry.locked
    lakebox.get.assert_not_called()
    lakebox.create.assert_not_called()
    ssh.open_terminal.assert_not_called()


@pytest.mark.parametrize("reason", ["owner", "busy", "missing"])
def test_ownership_busy_and_missing_do_not_replace_or_open_sandbox(runtime, reason):
    state, registry, lakebox, ssh, _, _ = runtime
    expected = HTTPException
    if reason == "owner":
        registry.owner = "other-principal"
        expected = ExecutionIdentityConflict
    elif reason == "busy":
        registry.busy = True
        expected = SessionBusy
    else:
        lakebox.get.return_value = None
    with pytest.raises(expected):
        debug.open_session(state, "environment", debug.TerminalRequest(session_id="conversation"))
    assert not registry.locked
    lakebox.create.assert_not_called()
    lakebox.start.assert_not_called()
    ssh.open_terminal.assert_not_called()


def test_stopped_session_resumes_exact_existing_sandbox(runtime):
    state, registry, lakebox, ssh, _, _ = runtime
    registry.row["status"] = "stopped"
    lakebox.get.return_value = {"sandboxId": "sandbox-one", "status": "Stopped"}
    debug.open_session(state, "environment", debug.TerminalRequest(session_id="conversation"))
    lakebox.start.assert_called_once_with("sandbox-one")
    lakebox.create.assert_not_called()
    assert ssh.open_terminal.call_args.args[0]["sandboxId"] == "sandbox-one"
    assert not registry.locked


def test_failed_setup_closes_client_and_releases_lock(runtime):
    state, registry, _, _, client, channel = runtime
    channel.sendall.side_effect = OSError("socket closed")
    with pytest.raises(OSError):
        debug.open_session(state, "environment", debug.TerminalRequest(session_id="conversation"))
    client.close.assert_called_once_with()
    assert not registry.locked
    assert registry.saved == []


def test_environment_resolution_uses_frozen_publication_without_database_access():
    control = SimpleNamespace(registry=MagicMock(), catalog=MagicMock(), namespace="namespace")
    store = EnvironmentStore(control)
    store.get = MagicMock(return_value={"default_agent_id": "frozen-agent"})
    assert store.session_agent("environment", "conversation") == "frozen-agent"
    control.registry.connect.assert_not_called()


@pytest.mark.parametrize("command", [None, "printf hello"])
def test_gateway_terminal_requests_do_not_wait_for_acknowledgements(command):
    client, channel, transport = MagicMock(), MagicMock(), MagicMock()
    channel.remote_chanid = 37
    client.get_transport.return_value = transport
    transport.open_session.return_value = channel
    lakebox = object.__new__(Lakebox)
    lakebox._open_client = MagicMock(return_value=client)
    sandbox = {"sandboxId": "sandbox-one"}
    assert lakebox.open_terminal(sandbox, command, term="vt100", columns=112, rows=45) == (
        client,
        channel,
    )
    transport.set_keepalive.assert_called_once_with(30)
    messages = [
        Message(call.args[0].asbytes()) for call in transport._send_user_message.call_args_list
    ]
    assert len(messages) == (2 if command is None else 1)
    request_types = []
    for message in messages:
        assert message.get_byte() == cMSG_CHANNEL_REQUEST
        assert message.get_int() == 37
        request_type = message.get_text()
        request_types.append(request_type)
        assert message.get_boolean() is False
        if request_type == "pty-req":
            assert message.get_text() == "vt100"
            assert [message.get_int() for _ in range(4)] == [112, 45, 0, 0]
            assert message.get_string() == b""
        else:
            assert request_type == ("shell" if command is None else "exec")
            if command is not None:
                assert message.get_text() == command
    assert request_types == (["pty-req", "shell"] if command is None else ["exec"])
    client.close.assert_not_called()


def test_gateway_channel_setup_error_closes_ssh_client():
    client = MagicMock()
    client.get_transport.return_value.open_session.side_effect = OSError("connection lost")
    lakebox = object.__new__(Lakebox)
    lakebox._open_client = MagicMock(return_value=client)
    with pytest.raises(OSError):
        lakebox.open_terminal({"sandboxId": "sandbox-one"})
    client.close.assert_called_once_with()


def test_websocket_denial_never_accepts_or_opens_ssh(runtime, monkeypatch, caplog):
    state, _, _, ssh, _, _ = runtime
    app = FastAPI()
    app.state.orchestrator = state.orchestrator
    app.add_api_websocket_route("/debug/{environment}", debug.terminal)
    monkeypatch.setattr(
        debug, "authorize", MagicMock(side_effect=RuntimeError("secret-developer-token"))
    )
    with pytest.raises(WebSocketDisconnect) as failure:
        with TestClient(app).websocket_connect("/debug/environment"):
            pytest.fail("Denied developer terminal must not be accepted")
    assert failure.value.code == 1008
    ssh.open_terminal.assert_not_called()
    assert "secret-developer-token" not in caplog.text


def test_websocket_ready_output_exit_closes_client_and_releases_lock(runtime, monkeypatch):
    state, registry, _, _, client, channel = runtime
    chunks = [b"hello\xff"]
    channel.recv_ready.side_effect = lambda: bool(chunks)
    channel.recv.side_effect = lambda _: chunks.pop(0)
    app = FastAPI()
    for key, value in vars(state).items():
        setattr(app.state, key, value)
    app.add_api_websocket_route("/debug/{environment}", debug.terminal)
    monkeypatch.setattr(debug, "authorize", lambda *_: "developer")
    with TestClient(app).websocket_connect("/debug/environment") as websocket:
        websocket.send_json({"session_id": "conversation"})
        assert websocket.receive_json()["type"] == "ready"
        assert not registry.locked
        output = websocket.receive_json()
        assert output == {"type": "stdout", "data": base64.b64encode(b"hello\xff").decode()}
        assert websocket.receive_json() == {"type": "exit", "code": 0}
    client.close.assert_called_once_with()
    channel.close.assert_called_once_with()


def test_relay_stdin_resize_eof_and_cleanup():
    async def exercise():
        websocket = SimpleNamespace(send_json=AsyncMock())
        inputs = asyncio.Queue()
        for message in (
            {"type": "stdin", "data": base64.b64encode(b"hello").decode()},
            {"type": "resize", "columns": 120, "rows": 40},
            {"type": "eof"},
        ):
            inputs.put_nowait(message)
        websocket.receive_json = inputs.get
        channel = MagicMock()
        channel.closed = False
        channel.recv_ready.return_value = channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.side_effect = lambda: channel.shutdown_write.called
        channel.recv_exit_status.return_value = 7
        await asyncio.wait_for(debug.relay(websocket, channel), 2)
        channel.sendall.assert_called_once_with(b"hello")
        channel.resize_pty.assert_called_once_with(width=120, height=40)
        channel.shutdown_write.assert_called_once_with()
        channel.close.assert_called_once_with()
        websocket.send_json.assert_awaited_once_with({"type": "exit", "code": 7})

    asyncio.run(exercise())


def test_failed_resize_emits_error_without_cleanup_manufacturing_exit(runtime, monkeypatch):
    state, _, _, _, _, channel = runtime
    channel.exit_status_ready.side_effect = lambda: channel.closed
    channel.recv_exit_status.return_value = 255
    channel.resize_pty.side_effect = RemoteError("Permission denied: secret-token")

    def close():
        channel.closed = True

    channel.close.side_effect = close
    metadata = {"type": "ready", "transport": "https", "sandbox_id": "sandbox-one"}
    monkeypatch.setattr(debug, "authorize", lambda *_: "developer")
    monkeypatch.setattr(debug, "open_session", lambda *_: (channel, channel, metadata))

    async def threadpool(function, *args, **kwargs):
        result = function(*args, **kwargs)
        if function == channel.close:
            # Model close waiting for the reader after it marks the channel finished.
            # An uncancelled output task gets a polling turn during this wait.
            await asyncio.sleep(0.04)
        return result

    monkeypatch.setattr(debug, "run_in_threadpool", threadpool)

    async def exercise():
        websocket = SimpleNamespace(
            app=SimpleNamespace(state=state),
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
            accept=AsyncMock(),
            close=AsyncMock(),
            send_json=AsyncMock(),
            receive_json=AsyncMock(
                side_effect=[
                    {"session_id": "conversation"},
                    {"type": "resize", "columns": 120, "rows": 40},
                ]
            ),
        )
        await asyncio.wait_for(debug.terminal(websocket, "environment"), 2)
        assert [call.args[0] for call in websocket.send_json.await_args_list] == [
            metadata,
            {"type": "error", "message": "Terminal connection failed (RemoteError)"},
        ]
        channel.resize_pty.assert_called_once_with(width=120, height=40)
        channel.recv_exit_status.assert_not_called()
        assert channel.closed
        websocket.close.assert_awaited_once_with()
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "message",
    [
        {"type": "stdin", "data": "not valid base64!"},
        {"type": "resize", "columns": 0, "rows": 24},
        {"type": "unknown"},
    ],
)
def test_relay_invalid_input_closes_channel(message):
    async def exercise():
        websocket = SimpleNamespace(
            receive_json=AsyncMock(return_value=message), send_json=AsyncMock()
        )
        channel = MagicMock()
        channel.closed = False
        channel.recv_ready.return_value = channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = False
        with pytest.raises(ValueError):
            await asyncio.wait_for(debug.relay(websocket, channel), 2)
        channel.close.assert_called_once_with()

    asyncio.run(exercise())


def test_resume_timeout_releases_lock_and_never_creates(runtime, monkeypatch):
    state, registry, lakebox, ssh, _, _ = runtime
    lakebox.get.return_value = {"sandboxId": "sandbox-one", "status": "Stopped"}
    lakebox.start.return_value = {"sandboxId": "sandbox-one", "status": "Starting"}
    clock = iter([0, 0, 241])
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.session_client.time",
        SimpleNamespace(perf_counter=lambda: 0.0, monotonic=lambda: next(clock)),
    )
    with pytest.raises(HTTPException) as failure:
        debug.open_session(state, "environment", debug.TerminalRequest(session_id="conversation"))
    assert failure.value.status_code == 504
    assert not registry.locked
    lakebox.start.assert_called_once_with("sandbox-one")
    lakebox.create.assert_not_called()
    ssh.open_terminal.assert_not_called()


def test_disappearing_sandbox_is_not_replaced(runtime):
    state, registry, lakebox, ssh, _, _ = runtime
    lakebox.get.return_value = {"sandboxId": "sandbox-one", "status": "Stopped"}
    lakebox.start.return_value = None
    with pytest.raises(HTTPException) as failure:
        debug.open_session(state, "environment", debug.TerminalRequest(session_id="conversation"))
    assert failure.value.status_code == 404
    assert not registry.locked
    lakebox.create.assert_not_called()
    ssh.open_terminal.assert_not_called()


def test_relay_disconnect_closes_channel_and_cancels_output():
    async def exercise():
        websocket = SimpleNamespace(
            receive_json=AsyncMock(side_effect=WebSocketDisconnect()), send_json=AsyncMock()
        )
        channel = MagicMock()
        channel.closed = False
        channel.recv_ready.return_value = channel.recv_stderr_ready.return_value = False
        channel.exit_status_ready.return_value = False
        with pytest.raises(WebSocketDisconnect):
            await asyncio.wait_for(debug.relay(websocket, channel), 2)
        channel.close.assert_called_once_with()
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": ""},
        {"session_id": "conversation", "columns": 0},
        {"session_id": "conversation", "term": "xterm;command"},
    ],
)
def test_invalid_handshake_does_not_open_or_resume(runtime, monkeypatch, payload):
    state, _, lakebox, ssh, _, _ = runtime
    app = FastAPI()
    for key, value in vars(state).items():
        setattr(app.state, key, value)
    app.add_api_websocket_route("/debug/{environment}", debug.terminal)
    monkeypatch.setattr(debug, "authorize", lambda *_: "developer")
    with TestClient(app).websocket_connect("/debug/environment") as websocket:
        websocket.send_json(payload)
        assert websocket.receive_json() == {"type": "error", "message": "Invalid terminal request"}
    ssh.open_terminal.assert_not_called()
    lakebox.start.assert_not_called()


def test_missing_environment_revision_does_not_create_pin():
    control = SimpleNamespace(registry=MagicMock(), catalog=MagicMock(), namespace="namespace")
    store = EnvironmentStore(control)
    store.get = MagicMock(return_value={"default_agent_id": None})
    db = control.registry.connect.return_value.__enter__.return_value
    db.execute.return_value.fetchone.return_value = None
    with pytest.raises(ValueError, match="publish it again"):
        store.session_agent("environment", "conversation")
    control.registry.connect.assert_not_called()
    db.execute.assert_not_called()
