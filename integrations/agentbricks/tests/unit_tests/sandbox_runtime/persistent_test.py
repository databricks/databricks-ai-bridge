"""Real local processes exercise the same stdlib scripts shipped to sandboxes."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import ANY

import pytest

from databricks_agentkit.sandbox.persistent import CLIENT_SCRIPT, DAEMON_SCRIPT, WORKER_VERSION


@pytest.fixture
def worker(tmp_path):
    (tmp_path / "test_agent.py").write_text(
        "import asyncio, contextvars, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "from uuid import uuid4\n"
        "import_id = uuid4().hex\n"
        "context = contextvars.ContextVar('request', default=None)\n"
        "calls = 0\n"
        "async def invoke(request, state_dir, *, emit):\n"
        "    global calls\n"
        "    calls += 1\n"
        "    previous = context.get()\n"
        "    context.set(request['invocation_id'])\n"
        "    disk = state_dir / 'count'\n"
        "    count = int(disk.read_text()) + 1 if disk.exists() else 1\n"
        "    disk.write_text(str(count))\n"
        "    emit(request['invocation_id'])\n"
        "    action = request.get('action')\n"
        "    if action == 'crash': os._exit(19)\n"
        "    if action == 'fail': raise ValueError('agent failed')\n"
        "    if action == 'child_hang':\n"
        "        subprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(.6); Path('escaped-child').write_text('bad')\"])\n"
        "        time.sleep(10)\n"
        "    if action in ('background', 'stubborn'):\n"
        "        async def background():\n"
        "            try: await asyncio.sleep(.3)\n"
        "            except asyncio.CancelledError:\n"
        "                if action != 'stubborn': raise\n"
        "                await asyncio.sleep(5)\n"
        "            (state_dir / 'escaped-task').write_text('bad')\n"
        "        asyncio.create_task(background())\n"
        "        await asyncio.sleep(0)\n"
        "    if action == 'sleep': await asyncio.sleep(.8)\n"
        "    return {'calls':calls, 'disk':count, 'import_id':import_id,\n"
        "        'loop':id(asyncio.get_running_loop()), 'previous':previous}\n"
    )
    states = set()

    def invoke(
        state="session",
        invocation="first",
        *,
        action=None,
        script=CLIENT_SCRIPT,
        wait=True,
        prepare=False,
        probe=False,
    ):
        directory = tmp_path / state
        states.add(directory)
        stream = directory / "streams" / (invocation + ".jsonl")
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(directory),
                "test_agent:invoke",
                sys.executable,
                "--stream-file",
                str(stream),
                *(["--prepare"] if prepare else []),
                *(["--probe"] if probe else []),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp_path,
        )
        request = json.dumps({"invocation_id": invocation, "action": action}).encode()
        if not wait:
            process.stdin.write(request)
            process.stdin.close()
            return process, stream
        stdout, stderr = process.communicate(request, timeout=20)
        return process.returncode, json.loads(stdout) if stdout else None, stderr.decode(), stream

    yield invoke
    for state in states:
        key = hashlib.sha256(str(state.resolve()).encode()).hexdigest()[:32]
        address = Path("/tmp") / ("ab-worker-" + key) / "rpc.sock"
        try:
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(2)
                client.connect(str(address))
                client.sendall(b'{"op":"shutdown"}\n')
                client.recv(4096)
        except OSError:
            pass


def test_reuses_process_imports_and_loop_but_isolates_streams_and_context(worker):
    first = worker(invocation="first")
    second = worker(invocation="second")
    assert first[0] == second[0] == 0, (first[2], second[2])
    a, b = first[1], second[1]
    assert a["worker"]["id"] == b["worker"]["id"]
    assert a["worker"]["pid"] == b["worker"]["pid"]
    assert not a["worker"]["reused"] and b["worker"]["reused"]
    assert a["result"]["import_id"] == b["result"]["import_id"]
    assert a["result"]["loop"] == b["result"]["loop"]
    assert a["result"]["calls"] == a["result"]["disk"] == 1
    assert b["result"]["calls"] == b["result"]["disk"] == 2
    assert a["result"]["previous"] is b["result"]["previous"] is None
    for invocation, result in [("first", first), ("second", second)]:
        records = [json.loads(line) for line in result[3].read_text().splitlines()]
        assert records == [
            {"type": "delta", "delta": invocation, "at": ANY},
            {"type": "end", "status": "completed"},
        ]


def test_sessions_have_independent_workers_even_with_long_state_paths(worker):
    a = worker(state="a" * 150)[1]
    b = worker(state="other")[1]
    assert a["worker"]["id"] != b["worker"]["id"]
    assert a["worker"]["pid"] != b["worker"]["pid"]
    assert a["result"]["disk"] == b["result"]["disk"] == 1


def test_idle_session_keeps_worker_and_imports_until_sandbox_shutdown(worker):
    # Accelerate any finite listener deadline so an accidental idle timeout
    # fails this real-process test without waiting ten minutes.
    daemon = DAEMON_SCRIPT.replace(
        "\nimport socket\n",
        "\nimport socket\n"
        "class AcceleratedIdleSocket(socket.socket):\n"
        "    def accept(self):\n"
        "        if self.gettimeout() is not None:\n"
        "            self.settimeout(.05)\n"
        "        return super().accept()\n"
        "socket.socket = AcceleratedIdleSocket\n",
    )
    script = CLIENT_SCRIPT.replace(repr(DAEMON_SCRIPT), repr(daemon))
    first = worker(invocation="first", script=script)
    assert first[0] == 0, first[2]
    time.sleep(0.2)
    following = worker(invocation="after-idle", script=script)
    assert following[0] == 0, following[2]
    assert first[1]["worker"]["id"] == following[1]["worker"]["id"]
    assert first[1]["result"]["import_id"] == following[1]["result"]["import_id"]
    assert following[1]["worker"]["reused"]
    assert following[1]["worker"]["invocation_count"] == 2


def test_crashed_worker_restarts_on_next_turn_and_preserves_disk_state(worker):
    first = worker()[1]
    os.kill(first["worker"]["pid"], signal.SIGKILL)
    restarted = worker(invocation="second")
    assert restarted[0] == 0, restarted[2]
    assert restarted[1]["worker"]["id"] != first["worker"]["id"]
    assert restarted[1]["result"]["calls"] == 1
    assert restarted[1]["result"]["disk"] == 2


@pytest.mark.parametrize("action", ["fail", "crash"])
def test_agent_failure_does_not_replay_and_next_turn_uses_fresh_worker(worker, action):
    failed = worker(action=action)
    assert failed[0] != 0
    assert (
        "agent failed" in failed[2] if action == "fail" else "outcome may be unknown" in failed[2]
    )
    if action == "fail":
        assert json.loads(failed[3].read_text().splitlines()[-1])["status"] == "failed"
    next_turn = worker(invocation="next")
    assert next_turn[0] == 0, next_turn[2]
    assert next_turn[1]["result"]["calls"] == 1
    assert next_turn[1]["result"]["disk"] == 2


def test_client_loss_does_not_replay_or_cancel_accepted_agent_work(worker):
    process, stream = worker(action="sleep", wait=False)
    deadline = time.monotonic() + 5
    while not stream.exists() or not stream.read_bytes():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    process.kill()
    process.wait(timeout=5)
    deadline = time.monotonic() + 5
    while len(stream.read_text().splitlines()) < 2:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert json.loads(stream.read_text().splitlines()[-1])["status"] == "completed"
    following = worker(invocation="next")
    assert following[0] == 0, following[2]
    assert following[1]["result"]["disk"] == 2


def test_worker_version_upgrade_restarts_without_losing_state(worker):
    first = worker()[1]
    upgraded = worker(
        invocation="upgraded", script=CLIENT_SCRIPT.replace(WORKER_VERSION, "new-version")
    )
    assert upgraded[0] == 0, upgraded[2]
    assert upgraded[1]["worker"]["id"] != first["worker"]["id"]
    assert upgraded[1]["result"]["disk"] == 2


def test_watchdog_kills_blocked_worker_without_replaying(worker):
    script = CLIENT_SCRIPT.replace('"210", "100"', '".2", "100"')
    failed = worker(action="sleep", script=script)
    assert failed[0] != 0
    assert "outcome may be unknown" in failed[2]
    following = worker(invocation="next")
    assert following[0] == 0, following[2]
    assert following[1]["result"]["disk"] == 2


def test_timeout_terminates_tool_subprocesses_before_replacement(worker, tmp_path):
    script = CLIENT_SCRIPT.replace('"210", "100"', '".2", "100"')
    failed = worker(action="child_hang", script=script)
    assert failed[0] != 0
    following = worker(invocation="next", action="sleep")
    assert following[0] == 0, following[2]
    assert not (tmp_path / "escaped-child").exists()


def test_invocation_background_tasks_are_cancelled_before_next_turn(worker):
    first = worker(action="background")[1]
    following = worker(invocation="next", action="sleep")
    assert following[0] == 0, following[2]
    assert following[1]["worker"]["id"] == first["worker"]["id"]
    assert not (following[3].parent.parent / "escaped-task").exists()


def test_failed_task_cleanup_preserves_reply_and_retires_worker(worker):
    first = worker(action="stubborn")
    assert first[0] == 0, first[2]
    assert first[1]["worker"]["retiring"]
    following = worker(invocation="next")
    assert following[0] == 0, following[2]
    assert following[1]["worker"]["id"] != first[1]["worker"]["id"]
    assert following[1]["result"]["disk"] == 2
    assert not (following[3].parent.parent / "escaped-task").exists()


def test_recycles_worker_after_configured_invocation_count(worker):
    script = CLIENT_SCRIPT.replace('"210", "100"', '"210", "2"')
    a = worker(invocation="a", script=script)[1]
    b = worker(invocation="b", script=script)[1]
    c = worker(invocation="c", script=script)[1]
    assert a["worker"]["id"] == b["worker"]["id"] != c["worker"]["id"]
    assert c["result"]["calls"] == 1 and c["result"]["disk"] == 3


def test_prepare_imports_handler_without_invoking_and_reuses_worker(worker, tmp_path):
    first = worker(prepare=True)
    assert first[0] == 0, first[2]
    assert first[1]["result"]["level"] == "handler-imported"
    assert first[1]["worker"]["invocation_count"] == 0
    assert not (tmp_path / "session/count").exists()
    again = worker(prepare=True)
    invoked = worker(invocation="real")
    assert first[1]["worker"]["id"] == again[1]["worker"]["id"] == invoked[1]["worker"]["id"]
    assert invoked[1]["worker"]["reused"]
    assert invoked[1]["result"]["calls"] == 1


def test_prepare_hook_runs_once_per_resident_worker(worker, tmp_path):
    with (tmp_path / "test_agent.py").open("a") as file:
        file.write(
            "\nasync def initialize(state_dir):\n"
            "    marker = state_dir / 'initialized'\n"
            "    marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else '1')\n"
            "    return {'loop': id(asyncio.get_running_loop())}\n"
            "invoke.prepare_session = initialize\n"
        )
    first = worker(prepare=True)
    again = worker(prepare=True)
    assert first[0] == again[0] == 0, (first[2], again[2])
    assert first[1]["result"]["level"] == "initialized"
    assert (tmp_path / "session/initialized").read_text() == "1"
    invoked = worker(invocation="real")
    assert first[1]["result"]["details"]["loop"] == invoked[1]["result"]["loop"]


def test_pool_probe_never_starts_worker_and_reports_unused_prepared_process(worker, tmp_path):
    code, result, error, _ = worker(probe=True)
    assert code == 0 and result is None, error
    assert not (tmp_path / "session/worker.log").exists()
    prepared = worker(prepare=True)[1]
    probe = worker(probe=True)[1]
    assert probe["prepared"] and probe["invocation_count"] == 0
    assert probe["id"] == prepared["worker"]["id"]
    invoked = worker(invocation="real-session-first-turn")[1]
    assert invoked["worker"]["id"] == probe["id"]
    assert worker(probe=True)[1]["invocation_count"] == 1
    os.kill(probe["pid"], signal.SIGKILL)
    time.sleep(0.05)
    assert worker(probe=True)[1] is None


def test_shutdown_hook_runs_after_turns_and_on_termination(worker, tmp_path):
    with (tmp_path / "test_agent.py").open("a") as file:
        file.write(
            "\nasync def shutdown(state_dir):\n"
            "    await asyncio.sleep(.05)\n"
            "    (state_dir / 'drained').write_text(str(calls))\n"
            "invoke.shutdown_session = shutdown\n"
        )
    first = worker()[1]
    second = worker(invocation="second")[1]
    assert first["worker"]["id"] == second["worker"]["id"]
    marker = tmp_path / "session/drained"
    assert not marker.exists()
    os.kill(second["worker"]["pid"], signal.SIGTERM)
    deadline = time.monotonic() + 5
    while not marker.exists():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert marker.read_text() == "2"
    third = worker(invocation="after-shutdown")
    assert third[0] == 0, third[2]
    assert third[1]["worker"]["id"] != second["worker"]["id"]
    assert third[1]["result"]["disk"] == 3


def test_blocking_shutdown_is_bounded_and_preserves_successful_reply(worker, tmp_path):
    with (tmp_path / "test_agent.py").open("a") as file:
        file.write(
            "\ndef shutdown(state_dir): time.sleep(30)\ninvoke.shutdown_session = shutdown\n"
        )
    daemon = DAEMON_SCRIPT.replace(
        "threading.Timer(10, terminate_worker)", "threading.Timer(.2, terminate_worker)"
    )
    script = CLIENT_SCRIPT.replace(repr(DAEMON_SCRIPT), repr(daemon)).replace(
        '"210", "100"', '"210", "1"'
    )
    first = worker(script=script)
    assert first[0] == 0, first[2]
    second = worker(invocation="second", script=script)
    assert second[0] == 0, second[2]
    assert first[1]["worker"]["id"] != second[1]["worker"]["id"]
    assert second[1]["result"]["disk"] == 2
