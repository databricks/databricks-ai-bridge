"""Databricks-side helpers for the OpenAI Agents API sandbox example.

Provision a sandbox, launch ``codex exec-server`` in it, and run commands, using
only the released ``databricks-sdk`` unary APIs (sandbox CRUD plus
``execute_command_sync``). No streaming APIs are required. ``interactive.py``
drives the OpenAI Agents API session and builds on these helpers.
"""

from __future__ import annotations

import shlex
import time

from databricks.sdk.errors import NotFound
from databricks.sdk.service.sandbox import (
    ComputeSpec,
    ExecuteCommandStatus,
    Sandbox,
    SandboxSpec,
    SandboxState,
)
from google.protobuf.duration_pb2 import Duration

# /home/sandbox-agent is a dedicated, writable volume owned by the sandbox user that
# survives stop/restart. "/" and "/tmp" are an ephemeral overlay wiped on restart, and
# $HOME defaults to a root-owned, non-writable path, so put the workspace here.
HOME_DIR = "/home/sandbox-agent"
WORK_DIR = f"{HOME_DIR}/agent-workspace"

# codex ships in the sandbox image and is on the login-shell PATH, so launch it by
# name - no runtime install needed.
CODEX_BIN = "codex"
EXECUTOR_LOG = f"{HOME_DIR}/codex-exec-server.log"


class SandboxCommandError(RuntimeError):
    """A command run in the sandbox did not complete successfully."""


def run_command(workspace_client, name, script, *, envs=None, timeout_s=120, check=True):
    """Run a bash script in the sandbox via the unary ``execute_command_sync`` API.

    ``execute_command_sync`` is direct-exec (no shell, no login profile), so every
    command goes through ``/bin/bash -lc`` to get a login shell and a PATH.
    """
    response = workspace_client.sandbox.execute_command_sync(
        name,
        "/bin/bash",
        args=["-lc", script],
        envs=envs,
        execution_timeout=Duration(seconds=timeout_s),
    )
    completed = response.status == ExecuteCommandStatus.EXECUTE_COMMAND_STATUS_COMPLETED
    if check and (not completed or response.exit_code != 0):
        raise SandboxCommandError(
            f"command failed (status={response.status}, exit_code={response.exit_code}): "
            f"{response.stderr or response.stdout}"
        )
    return response


def _wait_for_state(workspace_client, name, target, *, timeout_s=180, poll_s=3):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if workspace_client.sandbox.get_sandbox(name).status.state == target:
            return
        time.sleep(poll_s)
    raise TimeoutError(f"sandbox {name} did not reach {target} within {timeout_s}s")


def provision_sandbox(workspace_client, sandbox_id, *, inactivity_timeout_s=900):
    """Create the sandbox (or reuse and start an existing one) and wait until RUNNING."""
    name = f"sandboxes/{sandbox_id}"
    try:
        sandbox = workspace_client.sandbox.get_sandbox(name)
    except NotFound:
        sandbox = workspace_client.sandbox.create_sandbox(
            Sandbox(
                spec=SandboxSpec(
                    compute=ComputeSpec(
                        inactivity_timeout=Duration(seconds=inactivity_timeout_s)
                    )
                )
            ),
            sandbox_id=sandbox_id,
        )
    if sandbox.status.state in (
        SandboxState.SANDBOX_STATE_STOPPED,
        SandboxState.SANDBOX_STATE_STOPPING,
    ):
        workspace_client.sandbox.start_sandbox(name)
    _wait_for_state(workspace_client, name, SandboxState.SANDBOX_STATE_RUNNING)
    return name


def prepare_workspace(workspace_client, name, *, work_dir=WORK_DIR):
    """Create the agent's workspace directory.

    The executor changes directory to this path for each tool call, so it must exist
    before the agent runs anything.
    """
    run_command(workspace_client, name, f"mkdir -p {shlex.quote(work_dir)}")


def _parse_pid(stdout):
    for token in stdout.split():
        if token.startswith("pid="):
            return int(token.split("=", 1)[1])
    raise SandboxCommandError(f"could not read exec-server pid from: {stdout!r}")


def start_executor(workspace_client, name, *, environment_id, remote_url, executor_api_key):
    """Start codex exec-server detached so the sync RPC returns while it stays connected.

    exec-server holds a long-lived outbound connection to OpenAI for the whole session,
    but ``execute_command_sync`` blocks until the command exits. ``setsid nohup ... &``
    daemonizes it (new session, detached from the RPC's shell) so the call returns
    immediately. The executor key is passed via env, never written to disk.
    """
    # CODEX_HOME under the persistent home, not /tmp: codex refuses to create its
    # PATH-alias helper binaries under a temp dir.
    script = (
        f"cx=$(mktemp -d -p {HOME_DIR}); rm -f {EXECUTOR_LOG}; "
        f"CODEX_HOME=$cx setsid nohup {CODEX_BIN} exec-server "
        f"--remote {shlex.quote(remote_url)} --environment-id {shlex.quote(environment_id)} "
        f"</dev/null >{EXECUTOR_LOG} 2>&1 & echo pid=$!"
    )
    response = run_command(
        workspace_client, name, script, envs={"CODEX_API_KEY": executor_api_key, "RUST_LOG": "info"}
    )
    return _parse_pid(response.stdout)


def confirm_executor(workspace_client, name, pid, *, settle_s=8):
    """Give the executor a moment to register, then confirm it is still running."""
    time.sleep(settle_s)
    out = run_command(
        workspace_client,
        name,
        f"if ps -p {pid} >/dev/null; then echo ALIVE; else echo DEAD; fi; "
        f"tail -n 20 {EXECUTOR_LOG}",
    ).stdout
    if "ALIVE" not in out:
        raise SandboxCommandError(f"codex exec-server exited early:\n{out}")
    return out


def stop_executor(workspace_client, name, pid):
    run_command(
        workspace_client,
        name,
        f"kill -TERM -{pid} 2>/dev/null; kill -TERM {pid} 2>/dev/null; true",
        check=False,
    )


def delete_sandbox(workspace_client, name):
    workspace_client.sandbox.delete_sandbox(name)
