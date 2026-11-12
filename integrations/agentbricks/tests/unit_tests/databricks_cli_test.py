"""Unit tests for the databricks CLI wrapper (`_databricks`)."""

from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import sys
import time
import types

import pytest

from databricks_agentbricks.clients import databricks_cli as dc


def test_databricks_failure_hides_the_apps_subcommand(monkeypatch):
    # AgentBricks drives `databricks apps` as an implementation detail; a failure must be reported in
    # AgentBricks's terms (the `action`) and must NOT echo the raw subcommand or --profile at the customer.
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="boom from the CLI")

    monkeypatch.setattr(dc.subprocess, "run", fake_run)
    try:
        dc._databricks(
            ["apps", "run-local", "--prepare-environment"],
            "e2-dogfood",
            capture=True,
            action="Could not start the agent locally.",
        )
        raise AssertionError("expected AgentCliError")
    except dc.AgentCliError as exc:
        assert exc.message == "Could not start the agent locally."
        assert "apps" not in exc.message and "e2-dogfood" not in exc.message  # no leak
        assert exc.failure_detail == "boom from the CLI"
        assert exc.hint == "boom from the CLI"  # underlying stderr preserved for debugging


def test_databricks_failure_without_action_is_generic(monkeypatch):
    # No action label: still no raw subcommand/profile echo — just a generic message.
    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="err")

    monkeypatch.setattr(dc.subprocess, "run", fake_run)
    try:
        dc._databricks(["apps", "deploy", "x"], "prof", capture=True)
        raise AssertionError("expected AgentCliError")
    except dc.AgentCliError as exc:
        assert "apps" not in exc.message and "prof" not in exc.message


def test_to_stderr_keeps_stdout_for_the_agentbricks_payload(monkeypatch):
    # `--output json` prints Agent Bricks' payload on stdout; the CLI's own output must not land there.
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return types.SimpleNamespace(returncode=0, stdout=None, stderr=None)

    monkeypatch.setattr(dc.subprocess, "run", fake_run)
    dc._databricks(["sync", "src", "/Workspace/dst"], "prof", to_stderr=True)
    assert seen["stdout"] == 2
    dc._databricks(["apps", "logs", "x"], "prof")
    assert seen["stdout"] is None


def test_tee_relays_output_to_stderr_and_returns_its_tail(monkeypatch, capsys):
    calls = []

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            calls.append((cmd, kwargs))
            self.stdout = iter(["Uploading…\n", "Error: Another operation is in progress.\n"])
            self.returncode = 1

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(dc.subprocess, "Popen", FakePopen)

    result = dc._databricks(["apps", "deploy", "x"], "prof", tee=True, check=False)

    captured = capsys.readouterr()
    assert captured.out == ""  # stdout stays clean for -o json
    assert "Another operation is in progress" in captured.err
    assert result.returncode == 1
    assert "Another operation is in progress" in result.stdout
    cmd, kwargs = calls[0]
    assert cmd == ["databricks", "apps", "deploy", "x", "--profile", "prof"]
    assert kwargs["stderr"] is dc.subprocess.STDOUT


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_tee_omits_pem_diagnostics_without_changing_displayed_output(capsys):
    result = dc._run_tee(
        _python(
            "print('-----BEGIN PRIVATE KEY-----'); "
            "print('synthetic-key-body\\n' * 400, end=''); "
            "print('-----END PRIVATE KEY-----'); "
            "raise SystemExit(1)"
        ),
        cwd=None,
        env=None,
    )
    assert result.returncode == 1
    assert result.stdout == "[REDACTED]"
    assert capsys.readouterr().err.count("synthetic-key-body") == 400


def test_run_until_ready_calls_on_ready_once_the_probe_succeeds(monkeypatch):
    monkeypatch.setattr(dc, "_READY_POLL_S", 0.01)
    probes = iter([False, False, True])
    ready: list[bool] = []
    result = dc._run_until_ready(
        _python("import time; time.sleep(0.3)"),
        cwd=None,
        env=None,
        probe=lambda: next(probes),
        on_ready=lambda: ready.append(True),
    )
    assert ready == [True]
    assert result.returncode == 0


def test_run_until_ready_skips_on_ready_when_the_command_exits_first(monkeypatch):
    # A server that fails during startup never becomes ready: no success callback, exit code kept.
    monkeypatch.setattr(dc, "_READY_POLL_S", 0.01)
    ready: list[bool] = []
    result = dc._run_until_ready(
        _python("import sys; sys.exit(3)"),
        cwd=None,
        env=None,
        probe=lambda: False,
        on_ready=lambda: ready.append(True),
    )
    assert ready == []
    assert result.returncode == 3


def test_databricks_with_ready_probe_uses_the_readiness_runner(monkeypatch):
    seen: dict = {}

    def fake_until_ready(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(dc, "_run_until_ready", fake_until_ready)
    probe, on_ready = (lambda: True), (lambda: None)
    dc._databricks(["apps", "run-local"], "prof", ready_probe=probe, on_ready=on_ready)
    assert seen["cmd"] == ["databricks", "apps", "run-local", "--profile", "prof"]
    assert seen["probe"] is probe and seen["on_ready"] is on_ready


@pytest.mark.parametrize(
    "failure_output, retained",
    [
        ("x" * 20000 + "\nMissing dependency\n", "Missing dependency\n"),
        (
            "credentials = {\n"
            + " \n" * 2000
            + '"value": "synthetic-opaque-value"\n}\nMissing dependency\n',
            "[REDACTED]",
        ),
        (
            "password:\n" + "\n" * 4000 + "  synthetic-opaque-value\nMissing dependency\n",
            "[REDACTED]",
        ),
    ],
)
def test_streamed_failure_keeps_bounded_output_and_stream_destinations(
    capfd, failure_output, retained
):
    result = dc._run_until_ready(
        _python(f"import sys; print('out'); sys.stderr.write({failure_output!r}); sys.exit(3)"),
        cwd=None,
        env=None,
        probe=lambda: False,
        on_ready=None,
    )
    output = capfd.readouterr()
    assert output.out == "out\n"
    assert output.err == failure_output
    assert result.stdout == "out\n"
    assert result.stderr == retained
    assert result.returncode == 3


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX descriptor identity")
@pytest.mark.parametrize("destination", ["file", "pipe"])
def test_merged_output_preserves_child_write_order(tmp_path, destination):
    child = (
        "import os, sys\n"
        "for i in range(512):\n"
        "    os.write(1, f'out {i}\\n'.encode())\n"
        "    os.write(2, f'err {i}\\n'.encode())\n"
        "sys.exit(3)"
    )
    result_path = tmp_path / "result.json"
    runner = (
        "import json, pathlib, sys; "
        "from databricks_agentbricks.clients.databricks_cli import _run_until_ready; "
        f"result = _run_until_ready([sys.executable, '-c', {child!r}], "
        "cwd=None, env=None, probe=lambda: False, on_ready=None); "
        f"pathlib.Path({str(result_path)!r}).write_text("
        "json.dumps([result.returncode, result.stdout, result.stderr]))"
    )
    output_path = tmp_path / "combined.log"
    with output_path.open("wb") as output:
        completed = subprocess.run(
            _python(runner),
            stdout=output if destination == "file" else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=5,
            check=True,
        )
    observed = output_path.read_bytes() if destination == "file" else completed.stdout
    expected = "".join(f"out {i}\nerr {i}\n" for i in range(512)).encode()
    assert observed == expected
    returncode, stdout, stderr = json.loads(result_path.read_text())
    assert returncode == 3
    assert stdout.endswith("out 511\nerr 511\n")
    assert len(stdout.encode()) <= 3072
    assert stderr == ""


def test_descendant_holding_pipe_open_does_not_delay_completion():
    import time

    start = time.monotonic()
    result = dc._run_until_ready(
        _python(
            "import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)']); print('parent done', flush=True)"
        ),
        cwd=None,
        env=None,
        probe=lambda: False,
        on_ready=None,
    )
    assert time.monotonic() - start < 2.5
    assert "parent done" in result.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX descendant signal cleanup")
@pytest.mark.parametrize("continuous", [False, True])
def test_writing_descendant_does_not_extend_post_exit_drain(tmp_path, monkeypatch, continuous):
    pid_file = tmp_path / "descendant.pid"
    descendant = (
        "import time; [(print('descendant', flush=True), time.sleep(0.03)) for _ in range(200)]"
    )
    if continuous:
        descendant = (
            "import os, time\n"
            "deadline = time.monotonic() + 3\n"
            "while time.monotonic() < deadline:\n"
            "    os.write(1, b'descendant\\n' * 8192)\n"
        )
    parent = (
        "import pathlib, subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant!r}]); "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid)); "
        "print('direct child failed', flush=True); sys.exit(3)"
    )
    relay = dc.StreamRelay
    with open(os.devnull, "wb") as sink:
        monkeypatch.setattr(
            dc, "StreamRelay", lambda stream, destination: relay(stream, sink.fileno())
        )
        start = time.monotonic()
        try:
            result = dc._run_until_ready(
                _python(parent), cwd=None, env=None, probe=lambda: False, on_ready=None
            )
            assert time.monotonic() - start < 2.5
            assert result.returncode == 3
            if not continuous:
                assert "direct child failed" in result.stdout
        finally:
            if pid_file.exists():
                try:
                    os.kill(int(pid_file.read_text()), signal.SIGTERM)
                except ProcessLookupError:
                    pass


def test_closed_output_destination_keeps_draining_child():
    from databricks_agentbricks.clients.output_tail import StreamRelay

    with dc.subprocess.Popen(
        _python("print('x' * 1000000)"), stdout=dc.subprocess.PIPE, bufsize=0
    ) as proc:
        assert proc.stdout is not None
        relay = StreamRelay(proc.stdout, -1)
        try:
            assert proc.wait(timeout=5) == 0
        finally:
            proc.kill()
            relay.finish()


@pytest.mark.skipif(
    os.name != "posix", reason="uses POSIX process-group signals and pipe readiness"
)
def test_interrupt_does_not_wait_for_a_blocked_output_consumer(tmp_path):
    ready = tmp_path / "output-blocked"
    child = "import os; os.write(1, b'x' * 1000000)"
    runner = f"""
import pathlib, select, sys
from databricks_agentbricks.clients.databricks_cli import _run_until_ready

def probe():
    if not select.select([], [1], [], 0)[1]:
        pathlib.Path({str(ready)!r}).touch()
    return False

try:
    _run_until_ready([sys.executable, '-c', {child!r}],
                     cwd=None, env=None, probe=probe, on_ready=None)
except KeyboardInterrupt:
    sys.exit(130)
"""
    with subprocess.Popen(
        _python(runner),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    ) as proc:
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            assert ready.exists(), "relay did not fill its unread output pipe"
            os.killpg(proc.pid, signal.SIGINT)
            # Leave stdout unread so draining cannot hide a cancellation deadlock.
            assert proc.wait(timeout=3) == 130
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX pseudo-terminal")
@pytest.mark.parametrize("terminal_fd", [1, 2])
def test_mixed_terminal_and_redirected_stream_keep_original_behavior(tmp_path, terminal_fd):
    import pty

    child = (
        "import os, sys; "
        f"assert os.isatty({terminal_fd}); "
        f"print('STARTED', file=sys.{'stdout' if terminal_fd == 1 else 'stderr'}); "
        "sys.stdin.read(1); "
        f"print('redirected', file=sys.{'stderr' if terminal_fd == 1 else 'stdout'}); "
        "sys.exit(3)"
    )
    result_path = tmp_path / "result.json"
    runner = (
        "import json, pathlib, sys; "
        "from databricks_agentbricks.clients.databricks_cli import _run_until_ready; "
        f"result = _run_until_ready([sys.executable, '-c', {child!r}], "
        "cwd=None, env=None, probe=lambda: False, on_ready=None); "
        f"pathlib.Path({str(result_path)!r}).write_text("
        "json.dumps([result.returncode, result.stdout, result.stderr]))"
    )
    master, slave = pty.openpty()
    try:
        with subprocess.Popen(
            _python(runner),
            stdin=subprocess.PIPE,
            stdout=slave if terminal_fd == 1 else subprocess.PIPE,
            stderr=slave if terminal_fd == 2 else subprocess.PIPE,
        ) as proc:
            try:
                assert select.select([master], [], [], 5)[0], "startup output stayed buffered"
                assert b"STARTED" in os.read(master, 4096)
                assert proc.poll() is None
                stdout, stderr = proc.communicate(input=b"x", timeout=5)
                assert proc.returncode == 0
            finally:
                proc.kill()
    finally:
        os.close(master)
        os.close(slave)
    redirected = stderr if terminal_fd == 1 else stdout
    assert redirected == b"redirected\n"
    assert json.loads(result_path.read_text()) == (
        [3, "", "redirected\n"] if terminal_fd == 1 else [3, "redirected\n", ""]
    )
