"""Thin helpers over the released ``databricks-sdk`` sandbox APIs.

Everything here uses sandbox CRUD plus ``execute_command_sync``. Facts these rely on
(verified against a live workspace):

- ``execute_command_sync`` is direct exec (no shell), so commands run through
  ``/bin/bash --noprofile --norc -c`` (never a login shell: the agent can write start-up files in its
  persistent home). Variable data travels in ``envs``, never in the script text.
- ``start_sandbox`` is rejected while STOPPING; wait for STOPPED first.
- Auto-stop is driven by exec API calls, not by processes running in the sandbox.
- A process daemonized with ``setsid nohup ... &`` outlives the RPC that started it.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.sandbox import (
    ComputeSpec,
    EnvironmentSpec,
    ExecuteCommandStatus,
    ExecuteCommandSyncResponse,
    Sandbox,
    SandboxMetadata,
    SandboxSpec,
    SandboxState,
)
from google.protobuf.duration_pb2 import (
    Duration,  # ty: ignore[unresolved-import] - generated module
)

_SHELL = "/bin/bash"
# Never a login shell: the agent can write ~/.profile and ~/.bashrc in the persistent home, and a login or
# interactive shell would run them inside our own commands (uploads carry secrets; the worker reads its
# config from the environment). The non-login exec environment already has PATH and the sandbox credential.
_SHELL_ARGS = ("--noprofile", "--norc", "-c")
# Variables the worker keeps from the exec environment (started under `env -i`); everything else is dropped.
WORKER_ENV_ALLOWLIST = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "USER",
    "LOGNAME",
    "SHELL",
    "DATABRICKS_CONFIG_FILE",  # the sandbox's built-in credential
    "DATABRICKS_TOKEN_CACHE_FILE",
)
# Base64 in one env var must stay under the kernel's per-variable limit (MAX_ARG_STRLEN,
# 128 KiB); a multiple of 4 keeps every chunk independently decodable.
_UPLOAD_B64_CHUNK = 120 * 1024


class SandboxCommandError(RuntimeError):
    """A command run in the sandbox did not complete with exit code 0."""


def sandbox_name(sandbox_id: str) -> str:
    return f"sandboxes/{sandbox_id}"


def state_of(w: WorkspaceClient, name: str) -> SandboxState | None:
    sandbox = w.sandbox.get_sandbox(name)
    return sandbox.status.state if sandbox.status else None


def wait_for_state(
    w: WorkspaceClient,
    name: str,
    target: SandboxState,
    *,
    timeout_s: float = 300,
    poll_s: float = 2,
) -> None:
    deadline = time.monotonic() + timeout_s
    last = None
    while time.monotonic() < deadline:
        last = state_of(w, name)
        if last == target:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"{name} did not reach {target} within {timeout_s}s (last={last})")


def create(
    w: WorkspaceClient,
    sandbox_id: str,
    *,
    inactivity_timeout_s: int | None,
    image_uri: str | None,
    display_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    compute = (
        ComputeSpec(inactivity_timeout=Duration(seconds=inactivity_timeout_s))
        if inactivity_timeout_s is not None
        else None
    )
    environment = EnvironmentSpec(image_uri=image_uri) if image_uri else None
    created = w.sandbox.create_sandbox(
        Sandbox(
            spec=SandboxSpec(compute=compute, environment=environment),
            display_name=display_name,
            metadata=SandboxMetadata(custom_tags=tags) if tags else None,
        ),
        sandbox_id=sandbox_id,
    )
    if not created.name:
        raise SandboxCommandError(f"create_sandbox returned no name for {sandbox_id}")
    return created.name


def ensure_running(w: WorkspaceClient, name: str, *, timeout_s: float = 300) -> None:
    """Start a stopped (or stopping) sandbox and wait for RUNNING. Raises NotFound."""
    state = state_of(w, name)
    if state == SandboxState.SANDBOX_STATE_STOPPING:
        wait_for_state(w, name, SandboxState.SANDBOX_STATE_STOPPED, timeout_s=timeout_s)
        state = SandboxState.SANDBOX_STATE_STOPPED
    if state == SandboxState.SANDBOX_STATE_STOPPED:
        w.sandbox.start_sandbox(name)
    wait_for_state(w, name, SandboxState.SANDBOX_STATE_RUNNING, timeout_s=timeout_s)


def delete_quietly(w: WorkspaceClient, name: str) -> None:
    try:
        w.sandbox.delete_sandbox(name)
    except NotFound:
        pass


def run(
    w: WorkspaceClient,
    name: str,
    script: str,
    *,
    envs: dict[str, str] | None = None,
    timeout_s: int = 120,
    check: bool = True,
) -> ExecuteCommandSyncResponse:
    response = w.sandbox.execute_command_sync(
        name,
        _SHELL,
        args=[*_SHELL_ARGS, script],
        envs=envs,
        execution_timeout=Duration(seconds=timeout_s),
    )
    completed = response.status == ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED
    if check and (not completed or response.exit_code != 0):
        raise SandboxCommandError(
            f"command failed in {name} (status={response.status}, exit={response.exit_code}): "
            f"{(response.stderr or response.stdout or '').strip()[-2000:]}"
        )
    return response


def upload_text(
    w: WorkspaceClient, name: str, path: str, content: str, *, mode: str = "644"
) -> None:
    """Write a file into the sandbox via chunked base64 over ``execute_command_sync``.

    Chunks go to ``<path>.part`` with a private umask and are renamed into place, so readers never see a
    partial file and restrictive modes (``600``) apply from the first byte.
    """
    if not re.fullmatch(r"[0-7]{3,4}", mode):
        raise ValueError(f"mode must be an octal file mode like 600, not {mode!r}")
    encoded = base64.b64encode(content.encode()).decode("ascii")
    pieces = [
        encoded[i : i + _UPLOAD_B64_CHUNK] for i in range(0, len(encoded), _UPLOAD_B64_CHUNK)
    ] or [""]
    for index, piece in enumerate(pieces):
        redirect = ">" if index == 0 else ">>"
        mkdir = 'mkdir -p "$(dirname "$DBX_PATH")"; ' if index == 0 else ""
        run(
            w,
            name,
            f'set -e -o pipefail; umask 077; {mkdir}printf %s "$DBX_B64" | base64 -d {redirect} "$DBX_PATH.part"',
            envs={"DBX_PATH": path, "DBX_B64": piece},
        )
    run(
        w,
        name,
        f'set -e; chmod {mode} "$DBX_PATH.part"; mv -f "$DBX_PATH.part" "$DBX_PATH"',
        envs={"DBX_PATH": path},
    )


def start_detached(
    w: WorkspaceClient,
    name: str,
    command: str,
    *,
    log_path: str,
    pid_path: str,
    envs: dict[str, str],
) -> int:
    """Daemonize ``command`` so the RPC returns while it keeps running; return its pid.

    The process starts under ``env -i`` with only ``WORKER_ENV_ALLOWLIST`` plus ``envs``, so nothing the
    agent may have planted (PYTHONPATH, LD_PRELOAD, a redirected base URL) reaches it. ``envs`` values reach
    the process environment only (never the command line or disk).
    """
    for key in envs:
        if not key.replace("_", "").isalnum():
            raise ValueError(f"invalid environment variable name {key!r}")
    keep = [*WORKER_ENV_ALLOWLIST, *envs]
    env_args = " ".join(f'{k}="${{{k}}}"' for k in keep)
    log_dir = shlex.quote(os.path.dirname(log_path))
    script = (
        f"mkdir -p {log_dir}; "
        f"setsid nohup env -i {env_args} "
        f"/bin/bash --noprofile --norc -c {shlex.quote(command)} "
        f"</dev/null >{shlex.quote(log_path)} 2>&1 & "
        f"pid=$!; echo $pid > {shlex.quote(pid_path)}; echo pid=$pid"
    )
    out = run(w, name, script, envs=envs).stdout or ""
    for token in out.split():
        if token.startswith("pid="):
            return int(token.split("=", 1)[1])
    raise SandboxCommandError(f"could not read pid from {out!r}")


def live_worker(
    w: WorkspaceClient, name: str, pid_path: str, work_id_path: str
) -> tuple[int, str] | None:
    """(pid, work id) of a still-running worker, or None if none is alive or it recorded no work id."""
    out = (
        run(
            w,
            name,
            f"p=$(cat {shlex.quote(pid_path)} 2>/dev/null); "
            f'if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then '
            f'echo "$p $(cat {shlex.quote(work_id_path)} 2>/dev/null)"; fi',
            check=False,
        ).stdout
        or ""
    ).split()
    if len(out) == 2 and out[0].isdigit():
        return int(out[0]), out[1]
    return None


def is_alive(w: WorkspaceClient, name: str, pid: int) -> bool:
    out = run(w, name, f"ps -p {int(pid)} >/dev/null && echo ALIVE || echo DEAD").stdout or ""
    return "ALIVE" in out


def stop_pidfile_process(
    w: WorkspaceClient, name: str, pid_path: str, *, marker_path: str, grace_s: int = 10
) -> None:
    """Terminate the process group recorded in ``pid_path`` (a previous worker) and wait for it to exit.

    ``marker_path`` exists while this runs: a worker that sees it on exit is being replaced, so it skips
    its delete/stop step (otherwise it would delete or stop the sandbox its replacement is about to use).
    TERM first (the worker flushes memory stores), KILL after ``grace_s``; the marker is removed once the
    old worker is gone, so the replacement's own exit step runs normally.
    """
    pid, marker = shlex.quote(pid_path), shlex.quote(marker_path)
    run(
        w,
        name,
        f'p=$(cat {pid} 2>/dev/null); if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then '
        f"touch {marker}; "
        'kill -TERM -"$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; '
        f'for _ in $(seq {int(grace_s) * 2}); do kill -0 "$p" 2>/dev/null || break; sleep 0.5; done; '
        'kill -KILL -"$p" 2>/dev/null; kill -KILL "$p" 2>/dev/null; '
        f"fi; rm -f {marker}; true",
        check=False,
        timeout_s=int(grace_s) + 30,
    )


def tail(w: WorkspaceClient, name: str, path: str, lines: int = 30) -> str:
    return run(w, name, f"tail -n {int(lines)} {shlex.quote(path)} 2>&1", check=False).stdout or ""
