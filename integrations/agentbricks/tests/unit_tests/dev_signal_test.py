"""Exercise local-run telemetry with real processes and POSIX terminal-group signals."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest

_CHILD = """
import os, pathlib, socket, sys, time
root = pathlib.Path(sys.argv[1])
def publish(name, value):
    temporary = root / (name + '.tmp')
    temporary.write_text(str(value))
    temporary.replace(root / name)
if sys.argv[2] == 'listening':
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    sock.listen()
    publish('port', sock.getsockname()[1])
print('startup diagnostic token=synthetic-secret', file=sys.stderr, flush=True)
publish('child', os.getpid())
deadline = time.monotonic() + 30
while not (root / 'exit').exists() and time.monotonic() < deadline:
    time.sleep(0.02)
sys.exit(130)
"""

_PARENT = """
import json, os, pathlib, signal, sys
import click
from types import SimpleNamespace
from databricks_agentbricks.cli import telemetry
from databricks_agentbricks.clients import databricks_cli, terminal_relay
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.local_ports import accepts_connections
from databricks_agentbricks.diagnostics import CommandDiagnostics, FailureStage
root = pathlib.Path(sys.argv[1])
facts = CommandDiagnostics()
if sys.argv[3] == 'cleanup':
    def interrupted_finish(self):
        os.kill(os.getpid(), signal.SIGINT)
    databricks_cli.StreamRelay.finish = interrupted_finish
def runner(args, profile, **kwargs):
    def probe():
        port = root / 'port'
        return port.exists() and accepts_connections(int(port.read_text()))
    run = terminal_relay.run if sys.argv[3] == 'relay' else databricks_cli._run_until_ready
    extra = {'quiet_status': lambda elapsed: 'starting'} if sys.argv[3] == 'relay' else {}
    return run([sys.executable, '-c', sys.argv[4], str(root), sys.argv[2]],
               cwd=None, env=None, probe=probe, on_ready=kwargs['on_ready'],
               on_exit=kwargs['on_exit'], **extra)
exc = None
try:
    with facts.phase(FailureStage.RUN_LOCAL):
        AppsClient(None, runner=runner, diagnostics=facts).run_local(
            root, 'app.yaml', prepare_environment=False, port=8000, app_port=8001,
            on_ready=lambda: (root / 'listened').touch())
except BaseException as error:
    exc = error
root_ctx = click.Context(click.Group('agentbricks'), obj=SimpleNamespace(diagnostics=facts))
ctx = click.Context(click.Command('dev'), parent=root_ctx)
log = telemetry._build_log(ctx, exit_code=telemetry._exit_code(exc) if exc else 0,
    error_category=telemetry._error_category(exc) if exc else None, exception=exc)
(root / 'result.json').write_text(json.dumps({'log': log.to_dict(), 'facts': {
    'app_listened': facts.app_listened, 'parent_interrupted': facts.parent_interrupted,
    'subprocess_exit_code': facts.subprocess_exit_code}}))
"""


def _await_file(path, process):
    deadline = time.monotonic() + 20
    while not path.exists():
        assert process.poll() is None, (
            (path.parent / "result.json").read_text()
            if (path.parent / "result.json").exists()
            else "harness exited before recording the expected state"
        )
        assert time.monotonic() < deadline, f"timed out waiting for {path.name}"
        time.sleep(0.02)


@pytest.mark.parametrize("listening", [False, True])
@pytest.mark.parametrize(
    "relay,target",
    [
        (False, "group"),
        (True, "group"),
        (False, "child"),
        (True, "child"),
        (False, "exit130"),
        (True, "exit130"),
        (False, "cleanup"),
    ],
)
def test_local_run_preserves_parent_and_child_interrupt_facts(tmp_path, listening, relay, target):
    if sys.platform == "win32" and (relay or target != "exit130"):
        pytest.skip(
            "PTY/process-group SIGINT requires POSIX; native Windows Ctrl+C needs a console"
        )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _PARENT,
            str(tmp_path),
            "listening" if listening else "starting",
            "cleanup" if target == "cleanup" else "relay" if relay else "stream",
            _CHILD,
        ],
        start_new_session=sys.platform != "win32",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _await_file(tmp_path / "child", process)
        if listening:
            _await_file(tmp_path / "listened", process)
        if target == "group":
            os.killpg(process.pid, signal.SIGINT)
        elif target == "child":
            os.kill(int((tmp_path / "child").read_text()), signal.SIGINT)
        else:
            (tmp_path / "exit").touch()
        process.wait(timeout=15)
        assert process.returncode == 0
        result = json.loads((tmp_path / "result.json").read_text())
        log = result["log"]
        facts = result["facts"]
        assert "diagnostics" not in log
        assert facts["app_listened"] is listening
        assert facts["parent_interrupted"] is (target in {"group", "cleanup"})
        assert log["exit_code"] == 1
        if target in {"group", "cleanup"}:
            assert log["error_category"] == "ERROR_CATEGORY_INTERRUPTED"
            assert "failure_message" not in log
        else:
            assert log["error_category"] == "ERROR_CATEGORY_RUNTIME"
            assert "startup diagnostic" in log["failure_message"]
            assert "synthetic-secret" not in log["failure_message"]
        if target in {"exit130", "cleanup"}:
            assert facts["subprocess_exit_code"] == 130
        else:
            assert facts["subprocess_exit_code"] in (-signal.SIGINT, 130)
    finally:
        if sys.platform != "win32":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            process.kill()
        process.wait(timeout=5)
