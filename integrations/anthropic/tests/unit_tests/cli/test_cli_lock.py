"""The single-instance lock: a second holder is refused, in-process and from another process."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from databricks_anthropic.cli import lock
from databricks_anthropic.cli.config import ConfigError

ENV = "env_test01"


def test_second_holder_is_refused_with_the_pid(tmp_path):
    path = lock.lock_path(tmp_path, ENV)
    first = lock.EnvironmentLock(path, ENV)
    first.acquire()
    try:
        second = lock.EnvironmentLock(path, ENV)
        with pytest.raises(lock.LockHeld) as err:
            second.acquire()
        assert err.value.pid == os.getpid()  # the holder wrote its pid
        assert str(err.value) == f"another dispatcher for env {ENV} is running (pid {os.getpid()})"
    finally:
        first.release()


def test_lock_is_reacquirable_after_release(tmp_path):
    path = lock.lock_path(tmp_path, ENV)
    with lock.EnvironmentLock(path, ENV):
        pass
    with lock.EnvironmentLock(path, ENV):
        assert lock.holder_pid(tmp_path, ENV) == os.getpid()  # held: the probe reports the holder


def test_probe_reports_free_when_nobody_holds_it(tmp_path):
    assert lock.holder_pid(tmp_path, ENV) is None
    assert lock.holder_pid(tmp_path, ENV) is None  # the probe itself leaves nothing held


def test_a_holder_in_another_process_blocks_us(tmp_path):
    """A real second process holds the lock; this process must be refused and told its pid."""
    script = (
        "import sys, time\n"
        "from databricks_anthropic.cli import lock\n"
        "from pathlib import Path\n"
        "l = lock.EnvironmentLock(lock.lock_path(Path(sys.argv[1]), sys.argv[2]), sys.argv[2])\n"
        "l.acquire()\n"
        "print('held', flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path), ENV],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "held"
        with pytest.raises(lock.LockHeld) as err:
            lock.EnvironmentLock(lock.lock_path(tmp_path, ENV), ENV).acquire()
        assert err.value.pid == child.pid
    finally:
        child.kill()
        child.wait()
    # The kernel released the lock when the child died: a stale lock never survives a crash.
    assert lock.holder_pid(tmp_path, ENV) is None


def test_lock_path_is_per_environment_and_path_safe(tmp_path):
    assert lock.lock_path(tmp_path, ENV) == tmp_path / f"{ENV}.lock"
    with pytest.raises(ConfigError):  # a path separator is not an environment id
        lock.lock_path(tmp_path, "../escape")


def test_platforms_without_fcntl_are_told_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "_fcntl", None)
    with pytest.raises(lock.LockUnsupported, match="needs fcntl"):
        lock.EnvironmentLock(lock.lock_path(tmp_path, ENV), ENV).acquire()
    with pytest.raises(lock.LockUnsupported):
        lock.holder_pid(tmp_path, ENV)
