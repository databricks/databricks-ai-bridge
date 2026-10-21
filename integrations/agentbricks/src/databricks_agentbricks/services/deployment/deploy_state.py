"""The local record of a deploy in progress, and how `deployments wait` reads a deploy's outcome.

Every `agentbricks deploy` writes ``<source>/.agentbricks/deploy-<app>.json`` as it moves between
phases, and a background (``--no-wait``) deploy also logs to ``deploy-<app>.log`` beside it. The state
file lets a later command - `deployments wait`, or a second `deploy` that must not start a concurrent
rollout - tell whether the deploy is still running, which phase it is in, and how it ended:

.. code-block:: json

    {"pid": 1234, "started_at": "...", "phase": "Waiting for agent compute to start",
     "phase_started_at": "...", "status": "running", "error": null, "log": null}

On success the record also carries ``result``, the same object ``agentbricks -o json deploy`` prints.
Writes are atomic (a temp file renamed over the old one), so a reader never sees a partial file.

Without a local record, :func:`classify_app` reads the deploy's outcome from the App's own state.
Everything here is framework-neutral and takes its clock and process probe as arguments, so it can
be tested without real processes or waiting.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from databricks_agentbricks.reporting import Reporter, phase_name

STATE_DIR = ".agentbricks"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
# The phase a deploy reports before its first provisioning step.
STARTING_PHASE = "Starting the deploy"
EXITED_UNEXPECTEDLY = "deploy process exited unexpectedly"
# A `running` record whose phase hasn't moved in this long is stale even if its pid is alive (the pid
# may have been reused). The longest single phase is the 300 s compute wait plus busy-App retries.
STALE_AFTER_S = 20 * 60
# Lines of the deploy log `deployments wait` shows for a failed background deploy.
LOG_TAIL_LINES = 20


def state_path(source_dir: pathlib.Path, app: str) -> pathlib.Path:
    """Where the deploy of ``app`` from ``source_dir`` records its state."""
    return source_dir / STATE_DIR / f"deploy-{app}.json"


def log_path(source_dir: pathlib.Path, app: str) -> pathlib.Path:
    """Where a background deploy of ``app`` from ``source_dir`` writes its output."""
    return source_dir / STATE_DIR / f"deploy-{app}.log"


def timestamp(now: float) -> str:
    """An ISO-8601 UTC timestamp for a ``time.time()`` value."""
    return datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds")


def seconds_since(value: Any, now: float) -> float:
    """Seconds from an ISO-8601 timestamp to ``now`` (0 when the timestamp is missing or invalid)."""
    try:
        return max(0.0, now - datetime.fromisoformat(str(value)).timestamp())
    except ValueError:
        return 0.0


def read_state(path: pathlib.Path) -> Optional[dict[str, Any]]:
    """The recorded deploy state, or None when there is none (or it can't be parsed)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_state(path: pathlib.Path, state: dict[str, Any]) -> None:
    """Replace the recorded state atomically: write a temp file beside it, then rename it over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise


def pid_alive(pid: Any) -> bool:
    """Whether a process with this pid exists (one we may not signal still counts as alive)."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# Win32 constants: the process-query access right, the exit code GetExitCodeProcess reports for a
# live process, and the OpenProcess error for a process that exists but can't be opened.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


def _windows_pid_alive(pid: int) -> bool:
    """:func:`pid_alive` on Windows, where ``os.kill(pid, 0)`` sends CTRL_C_EVENT instead of probing."""
    if sys.platform != "win32":
        return False
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == _STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def resolve_state(
    state: dict[str, Any],
    *,
    now: float,
    is_alive: Callable[[Any], bool] = pid_alive,
) -> dict[str, Any]:
    """The state as it really is: a ``running`` record whose process is gone counts as failed.

    That covers a deploy that crashed or was killed (its pid is dead) and a record that hasn't moved
    in :data:`STALE_AFTER_S` (its pid may since have been reused by an unrelated process).
    """
    if state.get("status") != RUNNING:
        return state
    stale = seconds_since(state.get("phase_started_at"), now) > STALE_AFTER_S
    if is_alive(state.get("pid")) and not stale:
        return state
    return {**state, "status": FAILED, "error": EXITED_UNEXPECTEDLY}


def running_deploy(
    path: pathlib.Path,
    *,
    now: float,
    is_alive: Callable[[Any], bool] = pid_alive,
    own_pid: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """The live deploy recorded at ``path``, if one is still running in another process.

    ``own_pid`` excludes the caller itself: a background deploy finds the record its parent wrote
    for it, naming its own pid, and must not refuse itself.
    """
    state = read_state(path)
    if state is None or state.get("pid") == (os.getpid() if own_pid is None else own_pid):
        return None
    state = resolve_state(state, now=now, is_alive=is_alive)
    return state if state.get("status") == RUNNING else None


def log_tail(path: Optional[pathlib.Path], lines: int = LOG_TAIL_LINES) -> list[str]:
    """The last ``lines`` lines of a deploy log (empty when there is no log)."""
    if path is None:
        return []
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in deque(handle, maxlen=lines)]
    except OSError:
        return []


def _running_state(
    app: str, *, pid: int, started_at: str, now: str, log: Optional[pathlib.Path]
) -> dict[str, Any]:
    return {
        "deployment": app,
        "pid": pid,
        "started_at": started_at,
        "phase": STARTING_PHASE,
        "phase_started_at": now,
        "status": RUNNING,
        "error": None,
        "log": str(log) if log is not None else None,
    }


def record_background_start(
    source_dir: pathlib.Path, app: str, *, pid: int, now: float
) -> pathlib.Path:
    """Record a just-spawned background deploy as running, before its process records anything.

    Written by the parent so `deployments wait` finds a ``running`` record the moment
    `deploy --no-wait` returns. Skipped if the child already recorded itself (it started first).
    The log is recorded as an absolute path, so a `deployments wait` run from another directory
    still finds it. Returns the state file's path.
    """
    path = state_path(source_dir, app)
    previous = read_state(path) or {}
    if previous.get("pid") != pid:
        stamp = timestamp(now)
        write_state(
            path,
            _running_state(
                app, pid=pid, started_at=stamp, now=stamp, log=log_path(source_dir.absolute(), app)
            ),
        )
    return path


class DeployStateTracker:
    """Records one deploy's progress in its state file: started, each phase, then the outcome.

    ``result_payload`` turns the deploy's result into the object stored under ``result`` - the same
    one ``-o json deploy`` prints - so the CLI chooses that shape and this module stays free of it.
    A ``background`` deploy also records where its log is. A tracker that hasn't been started
    records nothing, so phases outside a deploy are ignored.
    """

    def __init__(
        self,
        *,
        result_payload: Callable[[Any], dict[str, Any]],
        background: bool = False,
        pid: Optional[int] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._result_payload = result_payload
        self._background = background
        self._pid = os.getpid() if pid is None else pid
        self._clock = clock
        self._path: Optional[pathlib.Path] = None
        self._state: dict[str, Any] = {}

    def start(self, source_dir: pathlib.Path, app: str) -> None:
        """Begin recording a deploy of ``app``.

        A background deploy keeps the start time and log its parent already recorded for it.
        """
        self._path = state_path(source_dir, app)
        now = timestamp(self._clock())
        previous = read_state(self._path) or {}
        inherited = previous.get("pid") == self._pid and previous.get("status") == RUNNING
        self._state = _running_state(
            app,
            pid=self._pid,
            started_at=previous.get("started_at", now) if inherited else now,
            now=now,
            log=log_path(source_dir, app) if self._background else None,
        )
        self._write()

    def phase(self, message: str) -> None:
        """Record that the deploy entered the phase named by a progress ``message``."""
        name = phase_name(message)
        if self._path is None or self._state.get("phase") == name:
            return
        self._state.update(phase=name, phase_started_at=timestamp(self._clock()))
        self._write()

    def succeeded(self, result: Any) -> None:
        """Record a finished deploy and its result."""
        if self._path is None:
            return
        self._state.update(status=SUCCEEDED, result=self._result_payload(result))
        self._write()
        self._path = None

    def failed(self, error: str) -> None:
        """Record a deploy that stopped with ``error``."""
        if self._path is None:
            return
        self._state.update(status=FAILED, error=error)
        self._write()
        self._path = None

    def _write(self) -> None:
        assert self._path is not None
        write_state(self._path, self._state)


class DeployStateReporter:
    """A :class:`Reporter` that records each progress/status step as the deploy's current phase.

    Wraps the real reporter, so the provisioners keep reporting progress as before and the state
    file follows the same phase names the user sees.
    """

    def __init__(self, inner: Reporter, tracker: DeployStateTracker) -> None:
        self._inner = inner
        self._tracker = tracker

    def status(self, message: str) -> AbstractContextManager[None]:
        self._tracker.phase(message)
        return self._inner.status(message)

    def progress(
        self, message: str, *, done_message: Optional[str] = None
    ) -> AbstractContextManager[None]:
        self._tracker.phase(message)
        return self._inner.progress(message, done_message=done_message)

    def timing(self, phase: str, seconds: float) -> None:
        self._inner.timing(phase, seconds)

    def note(self, message: str) -> None:
        self._inner.note(message)

    def echo(self, message: str, *, add_newline: bool = True) -> None:
        self._inner.echo(message, add_newline=add_newline)


# --- server-side state (no local record) ------------------------------------------------------


@dataclass(frozen=True)
class AppDeployStatus:
    """A deploy's outcome as read from the App: ``running``, ``succeeded``, or ``failed``."""

    status: str
    phase: str
    url: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


def _field(obj: Any, name: str) -> Any:
    """Read an Apps payload field across snake_case and camelCase CLI versions."""
    if not isinstance(obj, dict):
        return None
    if name in obj:
        return obj[name]
    parts = name.split("_")
    return obj.get(parts[0] + "".join(part.title() for part in parts[1:]))


def _deployment_status(deployment: Any) -> tuple[Optional[str], Optional[str]]:
    status = _field(deployment, "status")
    return _field(status, "state"), _field(status, "message")


def classify_app(app: str, payload: Optional[dict[str, Any]]) -> AppDeployStatus:
    """Classify an ``apps get`` payload (None when the App doesn't exist) as a deploy outcome.

    Succeeded: compute ACTIVE, nothing pending, and the active deployment SUCCEEDED. Failed: the
    latest deployment FAILED, the App is missing, compute is STOPPED/ERROR with nothing pending, or
    compute is ACTIVE with no deployment at all. Anything else is still deploying.
    """
    if payload is None:
        return AppDeployStatus(
            FAILED, "Checking the deployment", error=f"Deployment '{app}' does not exist."
        )
    compute = _field(_field(payload, "compute_status"), "state")
    compute_message = _field(_field(payload, "compute_status"), "message")
    pending = _field(payload, "pending_deployment")
    active = _field(payload, "active_deployment")
    pending_state, pending_message = _deployment_status(pending) if pending else (None, None)
    active_state, active_message = _deployment_status(active) if active else (None, None)
    url = _field(payload, "url") or None

    if pending and pending_state == "FAILED":
        return AppDeployStatus(
            FAILED,
            "Deploying the agent source",
            error=f"Could not deploy '{app}'.",
            message=pending_message,
        )
    if pending:
        return AppDeployStatus(
            RUNNING, "Deploying the agent source", message=pending_message or compute_message
        )
    if compute != "ACTIVE":
        if compute in ("STOPPED", "ERROR"):
            return AppDeployStatus(
                FAILED,
                "Waiting for agent compute to start",
                error=f"Deployment '{app}' compute is {compute}.",
                message=compute_message,
            )
        return AppDeployStatus(
            RUNNING, "Waiting for agent compute to start", message=compute_message
        )
    if active is None:
        return AppDeployStatus(
            FAILED,
            "Deploying the agent source",
            error=f"Deployment '{app}' has no deployed source.",
            message="Run `agentbricks deploy` to roll out the agent source.",
        )
    if active_state == "SUCCEEDED":
        return AppDeployStatus(SUCCEEDED, "Deployed", url=url, message=active_message)
    if active_state == "FAILED":
        return AppDeployStatus(
            FAILED,
            "Deploying the agent source",
            error=f"Could not deploy '{app}'.",
            message=active_message,
        )
    return AppDeployStatus(RUNNING, "Deploying the agent source", message=active_message)
