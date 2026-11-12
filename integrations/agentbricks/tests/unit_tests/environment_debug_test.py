"""Session terminal CLI contract and WebSocket transport, without live compute."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli.app import agentbricks
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services import environment_debug as debug


def test_cli_passes_the_session_and_command_to_debug_service(monkeypatch):
    launcher = MagicMock()
    constructor = MagicMock(return_value=launcher)
    connect = MagicMock(return_value=7)
    monkeypatch.setattr("databricks_agentbricks.cli.environment.LauncherClient", constructor)
    monkeypatch.setattr(debug, "connect", connect)
    result = CliRunner().invoke(
        agentbricks,
        [
            "--profile",
            "test",
            "environment",
            "ssh",
            "my-env",
            "--session-id",
            "chat-one",
            "--app",
            "exact-app",
            "--command",
            "exit 7",
        ],
    )
    assert result.exit_code == 7, result.output
    constructor.assert_called_once_with("test", app="exact-app", url=None, auth=None)
    connect.assert_called_once_with(launcher, "my-env", "chat-one", command="exit 7")


@pytest.mark.parametrize("session", [" ", "x" * 129, "é" * 65])
def test_invalid_session_is_rejected_before_workspace_access(session, monkeypatch):
    constructor = MagicMock()
    monkeypatch.setattr("databricks_agentbricks.cli.environment.LauncherClient", constructor)
    result = CliRunner().invoke(
        agentbricks, ["environment", "ssh", "env", "--session-id", session, "--app", "app"]
    )
    assert result.exit_code != 0
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "base,expected",
    [
        ("https://app.example/base", "wss://app.example/base/api/environments/my-env/ssh"),
        ("http://localhost:8082", "ws://localhost:8082/api/environments/my-env/ssh"),
    ],
)
def test_socket_url(base, expected):
    assert debug.websocket_url(SimpleNamespace(url=base), "my-env") == expected


def test_remote_cleartext_is_rejected():
    with pytest.raises(AgentCliError, match="require HTTPS"):
        debug.websocket_url(SimpleNamespace(url="http://remote.example"), "env")


@pytest.fixture
def socket(monkeypatch):
    socket = MagicMock()
    socket.recv.return_value = json.dumps(
        {"type": "ready", "sandbox_id": "sandbox-one", "source_directory": "$HOME/agent/source"}
    )
    socket.__iter__.return_value = iter(
        [
            json.dumps({"type": "stdout", "data": base64.b64encode(b"output\n").decode()}),
            json.dumps({"type": "stderr", "data": base64.b64encode(b"diagnostic\n").decode()}),
            json.dumps({"type": "exit", "code": 9}),
        ]
    )
    connect = MagicMock()
    connect.return_value.__enter__.return_value = socket
    monkeypatch.setattr("websockets.sync.client.connect", connect)
    monkeypatch.setattr(
        debug.WorkspaceOAuthAuthenticator, "authorization_header", lambda _: "Bearer hidden"
    )
    return socket, connect


def test_command_preserves_output_exit_status_and_sends_credentials_only_in_headers(socket, capsys):
    socket, connect = socket
    launcher = SimpleNamespace(url="https://app.example", profile="test", authenticate=True)
    assert debug.connect(launcher, "env", "session-one", command="printf output") == 9
    captured = capsys.readouterr()
    assert captured.out == "output\n"
    assert "diagnostic\n" in captured.err
    assert "Connected to sandbox-one" in captured.err
    assert "hidden" not in captured.out + captured.err
    assert connect.call_args.args == ("wss://app.example/api/environments/env/ssh",)
    assert connect.call_args.kwargs["additional_headers"] == {
        "Authorization": "Bearer hidden",
        "X-AgentBricks-Debug-Authorization": "Bearer hidden",
        "X-Routing-Key": "session-one",
    }
    sent = [json.loads(call.args[0]) for call in socket.send.call_args_list]
    assert sent[0]["session_id"] == "session-one"
    assert sent[0]["command"] == "printf output"
    assert sent[1] == {"type": "eof"}
    assert "hidden" not in json.dumps(sent)


def test_hosted_debug_requires_auth_before_socket_opens(socket):
    _, connect = socket
    launcher = SimpleNamespace(url="https://app.example", profile="test", authenticate=False)
    with pytest.raises(AgentCliError, match="requires --app"):
        debug.connect(launcher, "env", "s", command="pwd")
    connect.assert_not_called()


def test_interactive_shell_requires_a_terminal(socket, monkeypatch):
    _, connect = socket
    launcher = SimpleNamespace(url="http://localhost:8082", profile=None, authenticate=False)
    monkeypatch.setattr(debug.sys.stdin, "isatty", lambda: False)
    with pytest.raises(AgentCliError, match="interactive terminal"):
        debug.connect(launcher, "env", "s")
    connect.assert_not_called()


def test_missing_sandbox_error_is_actionable(socket):
    socket, _ = socket
    socket.recv.return_value = json.dumps({"type": "error", "message": "Session has no sandbox"})
    launcher = SimpleNamespace(url="http://localhost:8082", profile=None, authenticate=False)
    with pytest.raises(AgentCliError, match="Session has no sandbox"):
        debug.connect(launcher, "env", "s", command="pwd")


def test_disconnect_is_not_reported_as_success(socket):
    socket, _ = socket
    socket.__iter__.return_value = iter([])
    launcher = SimpleNamespace(url="http://localhost:8082", profile=None, authenticate=False)
    with pytest.raises(AgentCliError, match="without an exit status"):
        debug.connect(launcher, "env", "s", command="pwd")
