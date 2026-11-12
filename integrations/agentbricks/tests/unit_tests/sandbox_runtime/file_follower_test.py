import json
import os
import select
import subprocess
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import pytest

from databricks_agentkit.sandbox import _command_pb2 as wire
from databricks_agentkit.sandbox.streaming import RPC, ExecuteCommand, FileFollower


class LocalCall:
    """Cancellation detaches, without killing the command (same as sandbox gRPC)."""

    def __init__(self, cmd, args):
        self.process = subprocess.Popen(
            [cmd, *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.canceled = threading.Event()

    def cancel(self):
        self.canceled.set()

    def __iter__(self):
        yield wire.ExecuteCommandResponse(
            process_event=wire.ProcessEvent(start=wire.StartEvent(pid=self.process.pid))
        )
        while not self.canceled.is_set():
            if select.select([self.process.stdout], [], [], 0.02)[0]:
                chunk = os.read(self.process.stdout.fileno(), 100)
                if chunk:
                    yield wire.ExecuteCommandResponse(
                        process_event=wire.ProcessEvent(data=wire.DataEvent(stdout=chunk))
                    )
                else:
                    return


@pytest.mark.parametrize("failed", [False, True])
def test_tail_delivers_before_completion_and_cleanup_terminates_reader(
    tmp_path, monkeypatch, failed
):
    calls = []

    class LocalClient:
        def __init__(self, *args):
            pass

        def execute(self, cmd, args, timeout):
            call = LocalCall(cmd, args)
            calls.append(call)
            assert "tail -c +1 -F" in args[1]
            assert "timeout" in args[1]
            return call

        def close(self):
            pass

    monkeypatch.setattr("databricks_agentkit.sandbox.streaming.ExecuteCommand", LocalClient)
    remote = SimpleNamespace(
        workspace=None,
        sandbox_id="local",
        run=lambda cmd, **kw: subprocess.check_output(cmd, shell=True).decode(),
    )
    path = tmp_path / "attempt.jsonl"
    path.touch()
    got_delta = threading.Event()
    with FileFollower(remote, str(path), lambda _: got_delta.set()) as follower:
        with path.open("a") as file:
            file.write(json.dumps({"type": "delta", "delta": "Hello 🌍"}) + "\n")
            file.flush()
            assert got_delta.wait(timeout=3)
            assert calls[0].process.poll() is None
            file.write(
                json.dumps({"type": "end", "status": "failed" if failed else "completed"}) + "\n"
            )
        follower.finish()
    assert calls[0].process.wait(timeout=3) != 0
    assert not os.path.exists(follower.pid_path)
    assert path.exists()  # Attempt files survive the reader.


def test_stream_disconnect_catches_up_once_without_duplicate_deltas(tmp_path, monkeypatch):
    class Disconnected:
        def __iter__(self):
            data = (
                b'{"type":"delta","delta":"early"}\n' + '{"type":"delta","delta":"🌍'.encode()[:-2]
            )
            yield wire.ExecuteCommandResponse(
                process_event=wire.ProcessEvent(data=wire.DataEvent(stdout=data))
            )
            raise grpc.RpcError("lost stream")

        def cancel(self):
            pass

    client = MagicMock()
    client.execute.return_value = Disconnected()
    monkeypatch.setattr(
        "databricks_agentkit.sandbox.streaming.ExecuteCommand", lambda *args: client
    )
    path = tmp_path / "attempt.jsonl"
    path.write_text(
        '{"type":"delta","delta":"early"}\n{"type":"delta","delta":"🌍late"}\n{"type":"end","status":"completed"}\n'
    )
    commands, chunks = [], []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.check_output(command, shell=True).decode()

    remote = SimpleNamespace(workspace=None, sandbox_id="local", run=run)
    with FileFollower(remote, str(path), chunks.append) as follower:
        follower.thread.join(timeout=2)
        follower.finish()
    assert chunks == ["early", "🌍late"]
    assert len([cmd for cmd in commands if "base64.b64encode(f.read())" in cmd]) == 1


def test_execute_command_uses_preview_route_and_refreshes_auth_per_call(monkeypatch):
    channel = MagicMock()
    monkeypatch.setattr(grpc, "secure_channel", lambda *args, **kwargs: channel)
    workspace = MagicMock()
    workspace.config.host = "https://workspace.example"
    workspace.config.authenticate.side_effect = [
        {"Authorization": "Bearer one"},
        {"Authorization": "Bearer two"},
    ]
    client = ExecuteCommand(workspace, "sandbox-one")
    for _ in range(2):
        client.execute("/bin/tail", ["-F", "file"], 30)
    assert channel.unary_stream.call_args.args[0] == RPC
    rpc = channel.unary_stream.return_value
    assert ("authorization", "Bearer one") in rpc.call_args_list[0].kwargs["metadata"]
    assert ("authorization", "Bearer two") in rpc.call_args_list[1].kwargs["metadata"]
    assert ("x-databricks-sandbox-name", "sandbox-one") in rpc.call_args.kwargs["metadata"]
    request = rpc.call_args.args[0]
    assert request.cmd == "/bin/tail" and list(request.args) == ["-F", "file"]
    client.close()
    channel.close.assert_called_once()
