"""Real child-process coverage for local server supervision."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from databricks_agentbricks.clients.process_runner import run_process

_CHILD = """
import pathlib, signal, socket, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
def interrupt(signum, frame):
    (root/'interrupted').touch()
    sys.exit(130)
signal.signal(signal.SIGINT, interrupt)
grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
(root/'grandchild').write_text(str(grandchild.pid))
sock = socket.socket()
sock.bind(('127.0.0.1', 0)); sock.listen()
(root/'port').write_text(str(sock.getsockname()[1]))
print('server started', flush=True)
while not (root/'exit').exists(): time.sleep(0.02)
"""

_PARENT = """
import pathlib, socket, sys
from databricks_agentbricks.clients import terminal_relay
from databricks_agentbricks.clients.process_runner import run_process
root = pathlib.Path(sys.argv[1])
terminal_relay.supported = lambda: sys.argv[2] == 'pty'
def ready():
    path = root/'port'
    if not path.exists(): return False
    try:
        with socket.create_connection(('127.0.0.1', int(path.read_text())), timeout=0.1): return True
    except OSError: return False
def on_ready():
    (root/'ready').touch()
    if sys.argv[3] == 'callback-error': raise RuntimeError('callback failed')
run_process([sys.executable, '-c', sys.argv[4], str(root)], ready_probe=ready,
            on_ready=on_ready, quiet_status=lambda elapsed: 'starting')
"""


def _await(path, process):
    deadline = time.monotonic() + 15
    while not path.exists():
        assert process.poll() is None, "supervisor exited before readiness"
        assert time.monotonic() < deadline, f"timed out waiting for {path}"
        time.sleep(0.02)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if sys.platform.startswith("linux"):
        from pathlib import Path

        status = Path(f"/proc/{pid}/stat")
        try:
            return status.read_text().split(") ", 1)[1][0] != "Z"
        except FileNotFoundError:
            return False
    return True


@pytest.mark.skipif(os.name != "posix", reason="process-group signals require POSIX")
@pytest.mark.parametrize("mode", ["stream", "pty"])
@pytest.mark.parametrize("ending", ["interrupt", "terminate", "normal", "callback-error"])
def test_supervisor_reaps_server_and_descendants(tmp_path, mode, ending):
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT, str(tmp_path), mode, ending, _CHILD],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    grandchild = None
    try:
        _await(tmp_path / "ready", parent)
        grandchild = int((tmp_path / "grandchild").read_text())
        if ending == "normal":
            (tmp_path / "exit").touch()
        elif ending != "callback-error":
            parent.send_signal(signal.SIGINT if ending == "interrupt" else signal.SIGTERM)
        stdout, stderr = parent.communicate(timeout=15)
        assert b"server started" in stdout + stderr
        if ending == "normal":
            assert parent.returncode == 0, stderr
        elif ending == "terminate":
            assert parent.returncode == 143, stderr
        else:
            assert parent.returncode != 0
        if ending == "interrupt":
            assert (tmp_path / "interrupted").exists(), "server did not receive Ctrl-C"
        deadline = time.monotonic() + 3
        while _alive(grandchild) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _alive(grandchild), "tool process survived its local server"
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout=5)
        if grandchild and _alive(grandchild):
            os.kill(grandchild, signal.SIGKILL)


def test_failed_start_does_not_announce_readiness(capfd):
    ready = []
    result = run_process(
        [sys.executable, "-c", 'import sys; print("bad configuration"); sys.exit(17)'],
        ready_probe=lambda: False,
        on_ready=lambda: ready.append(True),
    )
    assert result.returncode == 17
    assert not ready
    assert "bad configuration" in capfd.readouterr().out
    assert "bad configuration" in result.stdout


def test_explicit_environment_and_working_directory(tmp_path):
    (tmp_path / "inspect.py").write_text(
        'import json, os; print(json.dumps({"cwd":os.getcwd(),"value":os.getenv("LOCAL_VALUE")}))'
    )
    result = run_process(
        [sys.executable, "inspect.py"],
        cwd=str(tmp_path),
        env={**os.environ, "LOCAL_VALUE": "set"},
    )
    assert json.loads(result.stdout) == {"cwd": str(tmp_path), "value": "set"}
