"""Offline checks of the terminal's actual protobuf boundary and channel lifecycle."""

import queue
import shlex
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from databricks_agentkit.sandbox import _command_pb2 as wire
from databricks_agentkit.sandbox import terminal

EOF = object()


def start(command_id="debug-command", pid=1234):
    return wire.ExecuteCommandResponse(
        process_event=wire.ProcessEvent(start=wire.StartEvent(command_id=command_id, pid=pid))
    )


def data(stream, value):
    return wire.ExecuteCommandResponse(
        process_event=wire.ProcessEvent(data=wire.DataEvent(**{stream: value}))
    )


def end(code=0):
    return wire.ExecuteCommandResponse(
        process_event=wire.ProcessEvent(
            end=wire.EndEvent(**({"exit_code": code} if code is not None else {}))
        )
    )


class ResponseStream:
    def __init__(self):
        self.messages = queue.Queue()
        self.messages.put(start())
        self.cancel = MagicMock(side_effect=lambda: self.messages.put(EOF))

    def __iter__(self):
        return self

    def __next__(self):
        message = self.messages.get(timeout=3)
        if message is EOF:
            raise StopIteration
        if isinstance(message, Exception):
            raise message
        return message

    def feed(self, *messages):
        for message in messages:
            self.messages.put(message)


class Channel:
    def __init__(self):
        self.stream = ResponseStream()
        self.calls = []
        self.input = None
        self.input_future = MagicMock()
        self.input_future.done.return_value = False
        self.close = MagicMock()
        self.terminate_error = None

    def unary_stream(self, path, *, request_serializer, response_deserializer):
        def execute(message, **kwargs):
            self.calls.append((path, request_serializer(message), kwargs))
            assert response_deserializer(start().SerializeToString()) == start()
            return self.stream

        return execute

    def stream_unary(self, path, *, request_serializer, response_deserializer):
        def future(messages, **kwargs):
            self.input = messages
            self.input_serializer = request_serializer
            self.calls.append((path, None, kwargs))
            assert response_deserializer(b"") == wire.StreamInputResponse()
            return self.input_future

        return SimpleNamespace(future=future)

    def unary_unary(self, path, *, request_serializer, response_deserializer):
        def terminate(message, **kwargs):
            self.calls.append((path, request_serializer(message), kwargs))
            if self.terminate_error:
                raise self.terminate_error
            return response_deserializer(b"")

        return terminate


@pytest.fixture
def gateway(monkeypatch):
    channel = Channel()
    factory = MagicMock(return_value=channel)
    credentials = object()
    monkeypatch.setattr(terminal.grpc, "secure_channel", factory)
    monkeypatch.setattr(terminal.grpc, "ssl_channel_credentials", lambda: credentials)
    workspace = SimpleNamespace(
        config=SimpleNamespace(
            host="https://workspace.example:443",
            authenticate=MagicMock(
                return_value={"Authorization": "Bearer app-token", "X-Test": "app-metadata"}
            ),
        )
    )
    opened = []

    def open_terminal(**options):
        request = SimpleNamespace(
            **{"command": None, "term": "xterm-256color", "columns": 101, "rows": 35, **options}
        )
        result = terminal.GrpcTerminal(
            workspace, "sandbox-one", "cd /source && . .venv/bin/activate", request
        )
        opened.append(result)
        return result

    yield SimpleNamespace(
        channel=channel,
        workspace=workspace,
        factory=factory,
        credentials=credentials,
        open=open_terminal,
    )
    for result in opened:
        result.close()


def finish(channel, result, *messages):
    channel.stream.feed(*messages, EOF)
    assert result._finished.wait(2), "Reader did not finish"


def drain(result, *, stderr=False):
    ready = result.recv_stderr_ready if stderr else result.recv_ready
    receive = result.recv_stderr if stderr else result.recv
    chunks = []
    while ready():
        chunks.append(receive(32768))
    return b"".join(chunks)


def test_tls_auth_and_pty_request_have_real_sdk_wire_layout(gateway):
    result = gateway.open()
    gateway.factory.assert_called_once_with("workspace.example:443", gateway.credentials)
    path, payload, kwargs = gateway.channel.calls[0]
    assert path == "/databricks.sandbox.SandboxService/ExecuteCommand"
    message = wire.ExecuteCommandRequest.FromString(payload)
    assert (message.pty.cols, message.pty.rows) == (101, 35)
    assert message.cmd == "/bin/bash" and message.args[0] == "-lc"
    assert "cd /source && . .venv/bin/activate" in message.args[1]
    assert "export TERM=xterm-256color" in message.args[1]
    assert "exec /bin/bash --noprofile --norc -i" in message.args[1]
    assert kwargs == {
        "metadata": [
            ("authorization", "Bearer app-token"),
            ("x-test", "app-metadata"),
            ("x-databricks-sandbox-name", "sandbox-one"),
        ],
        "timeout": 3600,
    }
    assert result.command_id == "debug-command" and result.pid == 1234
    sdk = pytest.importorskip("databricks.sandbox.sandbox_service_pb2")
    parsed = sdk.ExecuteCommandRequest.FromString(payload)
    assert parsed.cmd == message.cmd and list(parsed.args) == list(message.args)
    assert parsed.pty == sdk.PtyConfig(cols=101, rows=35)
    assert parsed.SerializeToString() == payload


def test_pipe_command_preserves_quoting_has_no_pty_or_input_rpc(gateway):
    command = "python -c 'print(42)'"
    result = gateway.open(command=command)
    message = wire.ExecuteCommandRequest.FromString(gateway.channel.calls[0][1])
    assert not message.HasField("pty")
    assert "exec /bin/bash -c " + shlex.quote(command) + " < /dev/null" in message.args[1]
    assert gateway.channel.input is None
    result.shutdown_write()
    finish(gateway.channel, result, data("stdout", b"42\n"), end())
    assert drain(result) == b"42\n"
    assert result.recv_exit_status() == 0


@pytest.mark.parametrize("stream", ["pty", "stdout", "stderr"])
def test_end_before_last_data_does_not_drop_trailing_bytes(gateway, stream):
    result = gateway.open()
    gateway.channel.stream.feed(end(7), data(stream, b"trailing\xff\n"))
    assert not result.exit_status_ready()
    finish(gateway.channel, result)
    assert result.exit_status_ready()
    assert drain(result, stderr=stream == "stderr") == b"trailing\xff\n"
    assert result.recv_exit_status() == 7
    result.close()
    assert not [call for call in gateway.channel.calls if call[0].endswith("TerminateCommand")]


def test_large_output_keeps_every_byte_in_bounded_chunks(gateway):
    result = gateway.open()
    payload = bytes(range(256)) * 512
    finish(gateway.channel, result, data("pty", payload), end())
    chunks = []
    while result.recv_ready():
        chunks.append(result.recv(32768))
    assert all(len(chunk) <= 32768 for chunk in chunks)
    assert b"".join(chunks) == payload


def test_stream_input_targets_command_preserves_binary_and_pty_eof(gateway):
    result = gateway.open()
    channel = gateway.channel
    initial = next(channel.input)
    assert initial.WhichOneof("event") == "start"
    assert initial.start.command_id == "debug-command"
    result.sendall(b"hello\x00\xff\n\x03")
    message = next(channel.input)
    assert message.data.data == b"hello\x00\xff\n\x03"
    result.shutdown_write()
    assert next(channel.input).data.data == b"\x04"
    assert channel.calls[1][0] == "/databricks.sandbox.SandboxService/StreamInput"
    assert channel.calls[1][2] == channel.calls[0][2]
    sdk = pytest.importorskip("databricks.sandbox.sandbox_service_pb2")
    assert (
        sdk.StreamInputRequest.FromString(channel.input_serializer(initial)).start.command_id
        == "debug-command"
    )
    assert (
        sdk.StreamInputRequest.FromString(channel.input_serializer(message)).data.data
        == message.data.data
    )


def test_resize_changes_exact_process_pty_over_https_then_stops_after_exit(gateway, monkeypatch):
    remote = MagicMock()
    factory = MagicMock(return_value=remote)
    monkeypatch.setattr(terminal, "HttpsRemote", factory)
    result = gateway.open()
    result.resize_pty(width=120, height=45)
    factory.assert_called_once_with(gateway.workspace, "sandbox-one")
    remote._exec.assert_called_once_with(
        "/bin/bash",
        [
            "-c",
            'terminal=$(/bin/ps -p "$1" -o tty=) && '
            '[[ "$terminal" =~ ^pts/[0-9]+$ ]] && '
            'exec /usr/bin/sudo -n /bin/stty -F "/dev/$terminal" rows "$2" cols "$3"',
            "agentbricks-resize",
            "1234",
            "45",
            "120",
        ],
        10,
    )
    finish(gateway.channel, result, end())
    result.resize_pty(width=80, height=24)
    assert remote._exec.call_count == 1


def test_pipe_commands_ignore_resize(gateway, monkeypatch):
    remote = MagicMock()
    monkeypatch.setattr(terminal, "HttpsRemote", remote)
    result = gateway.open(command="true")
    result.resize_pty(width=100, height=40)
    remote.assert_not_called()


def test_disconnect_terminates_only_debug_command_and_cancels_both_calls(gateway):
    result = gateway.open()
    result.close()
    result.close()
    termination = [call for call in gateway.channel.calls if call[0].endswith("TerminateCommand")]
    assert len(termination) == 1
    path, payload, kwargs = termination[0]
    assert path == "/databricks.sandbox.SandboxService/TerminateCommand"
    assert wire.TerminateCommandRequest.FromString(payload).command_id == "debug-command"
    assert kwargs["timeout"] == 10
    assert kwargs["metadata"] == gateway.channel.calls[0][2]["metadata"]
    gateway.channel.stream.cancel.assert_called_once()
    gateway.channel.input_future.cancel.assert_called_once()
    gateway.channel.close.assert_called_once()
    assert not result._reader.is_alive()
    with pytest.raises(OSError, match="closed"):
        result.sendall(b"late input")


def test_cleanup_survives_termination_failure_without_logging_credentials(gateway, caplog):
    result = gateway.open()
    gateway.channel.terminate_error = RuntimeError("Bearer secret-app-token")
    result.close()
    gateway.channel.stream.cancel.assert_called_once()
    gateway.channel.input_future.cancel.assert_called_once()
    gateway.channel.close.assert_called_once()
    assert "RuntimeError" in caplog.text and "secret-app-token" not in caplog.text


@pytest.mark.parametrize("last", [None, end(None)])
def test_missing_exit_status_is_error_and_still_terminates_command(gateway, last):
    result = gateway.open()
    messages = [data("pty", b"partial output")]
    if last is not None:
        messages.append(last)
    finish(gateway.channel, result, *messages)
    assert drain(result) == b"partial output"
    with pytest.raises(RuntimeError, match="output failed"):
        result.recv_exit_status()
    result.close()
    assert (
        len([call for call in gateway.channel.calls if call[0].endswith("TerminateCommand")]) == 1
    )


def test_output_transport_failure_preserves_partial_text_and_sanitizes_error(gateway):
    result = gateway.open()
    finish(gateway.channel, result, data("pty", b"partial"), RuntimeError("Bearer secret"))
    assert drain(result) == b"partial"
    with pytest.raises(RuntimeError, match=r"output failed \(RuntimeError\)") as failure:
        result.recv_exit_status()
    assert "secret" not in str(failure.value)
    result.close()
    assert any(call[0].endswith("TerminateCommand") for call in gateway.channel.calls)


def test_input_transport_failure_is_reported_while_output_is_active(gateway):
    result = gateway.open()
    gateway.channel.input_future.done.return_value = True
    gateway.channel.input_future.exception.return_value = RuntimeError("Bearer secret")
    with pytest.raises(RuntimeError, match="input stream failed") as failure:
        result.exit_status_ready()
    assert "secret" not in str(failure.value)


@pytest.mark.parametrize("first", [end(), start(command_id=""), start(pid=0)])
def test_invalid_start_cancels_stream_and_closes_channel(gateway, first):
    gateway.channel.stream.messages.get_nowait()
    gateway.channel.stream.feed(first)
    with pytest.raises(RuntimeError, match="process (start|identity)"):
        gateway.open()
    gateway.channel.stream.cancel.assert_called_once()
    gateway.channel.close.assert_called_once()


@pytest.mark.parametrize("host", ["http://workspace.example", "workspace.example", "https://"])
def test_workspace_requires_tls_before_connecting(gateway, host):
    gateway.workspace.config.host = host
    with pytest.raises(ValueError, match="HTTPS workspace"):
        gateway.open()
    gateway.factory.assert_not_called()


def test_authentication_failure_cancels_stream_resources(gateway):
    gateway.workspace.config.authenticate.side_effect = RuntimeError("auth unavailable")
    with pytest.raises(RuntimeError, match="auth unavailable"):
        gateway.open()
    assert gateway.channel.calls == []
    gateway.channel.close.assert_called_once()
