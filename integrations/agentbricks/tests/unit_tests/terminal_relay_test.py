"""Unit tests for the pseudo-terminal relay behind `agentbricks dev`'s quiet-gap spinner."""

from __future__ import annotations

import sys
import time

import pytest

from databricks_agentbricks.clients import terminal_relay as tr

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="pseudo-terminals are POSIX only")


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(tr, "_QUIET_AFTER_S", 0.2)
    monkeypatch.setattr(tr, "_READY_POLL_S", 0.05)
    monkeypatch.setattr(tr, "_OUTPUT_GRACE_S", 0.5)


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _run(code: str, *, probe=lambda: False, on_ready=None):
    return tr.run(
        _python(code),
        cwd=None,
        env=None,
        probe=probe,
        on_ready=on_ready,
        quiet_status=lambda elapsed: "~ waiting",
    )


def test_output_passes_through_unchanged_and_the_child_sees_a_terminal(capfd):
    result = _run("import sys; print('hello', sys.stdout.isatty(), sys.stderr.isatty())")
    assert result.returncode == 0
    assert "hello True True" in capfd.readouterr().out


def test_status_appears_in_a_quiet_gap_and_is_erased_before_more_output(capfd):
    _run(
        "import time; print('installed', flush=True); time.sleep(0.8); print('started', flush=True)"
    )
    out = capfd.readouterr().out
    first, gap, rest = out.partition("~ waiting")
    assert "installed" in first and gap, out  # drawn after the quiet stretch began
    # erased (clear-line) before the next real output, so it never shares a line with a log
    assert rest.find("\r\x1b[2K") < rest.find("started"), repr(out)


def test_no_status_mid_line(capfd):
    # A prompt with no newline (a debugger, a y/n question) must not get a status line drawn over it.
    _run("import sys, time; sys.stdout.write('(Pdb) '); sys.stdout.flush(); time.sleep(0.8)")
    out = capfd.readouterr().out
    assert "(Pdb) " in out
    assert "~ waiting" not in out


def test_status_stops_once_ready(capfd):
    ready: list[bool] = []
    start = time.monotonic()

    def probe() -> bool:
        return time.monotonic() - start > 0.1

    _run(
        "import time; print('up', flush=True); time.sleep(1.0)",
        probe=probe,
        on_ready=lambda: ready.append(True),
    )
    out = capfd.readouterr().out
    assert ready == [True]
    assert "~ waiting" not in out  # ready before the output went quiet long enough


def test_exit_code_is_kept(capfd):
    assert _run("import sys; sys.exit(3)").returncode == 3


def test_a_child_holding_the_terminal_open_does_not_hang(capfd):
    code = (
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print('parent done', flush=True)\n"
    )
    start = time.monotonic()
    result = _run(code)
    assert time.monotonic() - start < 10
    assert result.returncode == 0
    assert "parent done" in capfd.readouterr().out


def test_supported_requires_a_terminal_on_both_streams(monkeypatch):
    monkeypatch.setattr(tr.sys, "platform", "darwin")
    monkeypatch.setattr(tr.os, "isatty", lambda fd: fd == 1)
    assert tr.supported() is False
    monkeypatch.setattr(tr.os, "isatty", lambda fd: True)
    assert tr.supported() is True
    monkeypatch.setattr(tr.sys, "platform", "win32")
    assert tr.supported() is False
