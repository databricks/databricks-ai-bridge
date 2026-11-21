"""Single-instance lock: one local dispatcher per environment on this machine.

``fcntl.flock`` on ``<config dir>/<environment_id>.lock``, taken without waiting. The holder writes its pid into
the file so a second run can say whose it is. The kernel releases the lock when the process exits, however it
exits, so a crash never leaves a stale lock. The file itself is never deleted (deleting a lock file races).

POSIX only: on platforms without ``fcntl`` (Windows) :class:`LockUnsupported` is raised, and the CLI says so.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

from databricks_anthropic.cli.config import check_environment_id

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows
    _fcntl = None  # ty: ignore[invalid-assignment]

_BUSY_ERRNOS = (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK)


class LockError(RuntimeError):
    pass


class LockUnsupported(LockError):
    """This platform has no fcntl, so the single-instance lock is unavailable."""


class LockHeld(LockError):
    """Another process holds the environment's lock."""

    def __init__(self, environment_id: str, pid: int | None) -> None:
        self.environment_id = environment_id
        self.pid = pid
        who = f"(pid {pid})" if pid else "(pid unknown)"
        super().__init__(f"another dispatcher for env {environment_id} is running {who}")


def lock_path(config_dir: Path, environment_id: str) -> Path:
    return config_dir / f"{check_environment_id(environment_id)}.lock"


def _read_pid(fd: int) -> int | None:
    os.lseek(fd, 0, os.SEEK_SET)
    text = os.read(fd, 64).decode("ascii", "replace").strip()
    return int(text) if text.isdigit() else None


class EnvironmentLock:
    """Hold the lock for one environment; use as a context manager or call acquire/release."""

    def __init__(self, path: Path, environment_id: str) -> None:
        self.path = path
        self.environment_id = environment_id
        self._fd: int | None = None

    def acquire(self) -> None:
        if _fcntl is None:
            raise LockUnsupported(
                "the single-instance lock needs fcntl (Linux or macOS); this platform cannot run the dispatcher"
            )
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        except OSError as e:
            pid = _read_pid(fd)
            os.close(fd)
            if e.errno in _BUSY_ERRNOS:
                raise LockHeld(self.environment_id, pid) from None
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fd, self._fd = self._fd, None
        try:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> EnvironmentLock:
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def holder_pid(config_dir: Path, environment_id: str) -> int | None:
    """Probe the lock without keeping it: None when free, the holder's pid (or 0 when unknown) when held.

    Raises :class:`LockUnsupported` on platforms without fcntl.
    """
    lock = EnvironmentLock(lock_path(config_dir, environment_id), environment_id)
    try:
        lock.acquire()
    except LockHeld as held:
        return held.pid or 0
    lock.release()
    return None
