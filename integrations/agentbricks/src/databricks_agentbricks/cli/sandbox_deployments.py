"""Invoke and debug the sandbox runtime bound to an App deployment."""

from __future__ import annotations

import json
import time
from uuid import uuid4

import click

from databricks_agentbricks.cli.environment_client import LauncherClient
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.deployment.names import _prefixed_name
from databricks_agentbricks.services.invoke.request import load_json


def request_run(launcher, method, path, invocation_id, **kwargs):
    try:
        return launcher.request(method, path, **kwargs)
    except AgentCliError as exc:
        # A lost response can follow a committed submission. Preserve the generated
        # identity in JSON errors too, so retrying does not create another run.
        exc.hint = (
            (exc.hint + "\n" if exc.hint else "")
            + f"Repeat with --invocation-id {invocation_id} and the same input to fetch the original run."
        )
        raise


@click.command("invoke")
@click.argument("app")
@click.option("--session-id", required=True, help="Session whose sandbox should be reused.")
@click.option("--invocation-id", help="Stable retry identifier (default: generated UUID).")
@click.option("--json", "json_value", required=True, help="Agent input as a JSON object.")
@click.option(
    "--retry",
    is_flag=True,
    help="Reattach to an existing invocation; failed runs require a new ID.",
)
@click.option("--stream", is_flag=True, help="Print incremental text while waiting.")
@click.option("--timeout", type=click.FloatRange(min=0.1), default=300.0, show_default=True)
@click.pass_obj
def invoke_deployment(obj, app, session_id, invocation_id, json_value, retry, stream, timeout):
    """Invoke a sandbox deployment; install its session sandbox on first use.

    APP accepts a base name or the full agent-bricks-prefixed deployment name.
    A timeout or disconnected stream does not cancel the invocation. Reuse the
    same session, invocation ID and input to retrieve its status or reconnect.
    """
    value = load_json(json_value)
    if not isinstance(value, dict):
        raise AgentCliError("--json must contain an object for the agent input.")
    if retry and not invocation_id:
        raise AgentCliError("--retry requires the original --invocation-id.")
    invocation_id = invocation_id or str(uuid4())
    launcher = LauncherClient(obj.profile, app=_prefixed_name(app))
    config = launcher.request("GET", "/api/config")
    agent_id = config.get("default_agent_id")
    if not agent_id:
        raise AgentCliError("This App has no sandbox deployment binding; deploy it with --sandbox.")
    run = request_run(
        launcher,
        "POST",
        "/api/runs",
        invocation_id,
        body={
            "agent_id": agent_id,
            "session_id": session_id,
            "invocation_id": invocation_id,
            "input": value,
            "retry": retry,
        },
        routing_key=session_id,
    )
    deadline = time.monotonic() + timeout
    stage = None
    printed = ""
    if stream:
        updates = launcher.watch(run["id"], routing_key=session_id, timeout=timeout)
        try:
            for update in updates:
                run = update
                text = run.get("display_text", (run.get("stream") or {}).get("text", ""))
                if text.startswith(printed):
                    click.echo(text[len(printed) :], nl=False, err=True)
                    printed = text
                if run["status"] not in {"queued", "running"}:
                    break
            else:
                raise AgentCliError(
                    f"Stream ended before run {run['id']} completed; it may still be running.",
                    hint=f"Repeat with --invocation-id {invocation_id} and the same input to reconnect.",
                )
        except AgentCliError as exc:
            exc.hint = (
                f"Repeat with --invocation-id {invocation_id} and the same input to reconnect."
            )
            raise
        finally:
            updates.close()
    while not stream and run["status"] in {"queued", "running"}:
        if obj.output == "text" and run["stage"] != stage:
            stage = run["stage"]
            click.echo(f"{run['id']}: {stage}", err=True)
        if time.monotonic() >= deadline:
            raise AgentCliError(
                f"Timed out waiting for run {run['id']}; it may still be running.",
                hint=f"Repeat with --invocation-id {invocation_id} and the same input to fetch its status/result.",
            )
        time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        run = request_run(
            launcher, "GET", f"/api/runs/{run['id']}", invocation_id, routing_key=session_id
        )
    if stream:
        click.echo(err=True)
    if run["status"] in {"failed", "interrupted"}:
        raise AgentCliError(
            f"Run {run['id']} {run['status']}: {run.get('error') or 'no diagnostic'}",
            hint="Inspect the session, then retry with a new --invocation-id and the same input.",
        )
    click.echo(json.dumps(run, indent=2, default=str))


@click.command("ssh")
@click.argument("app")
@click.option("--session-id", required=True, help="Session whose existing sandbox to debug.")
@click.option("--command", help="Run one shell command without a PTY and return its exit code.")
@click.pass_context
def deployments_ssh(ctx, app, session_id, command):
    """Open a debug shell in a sandbox deployment's existing session.

    Requires CAN_MANAGE on the App. Resumes stopped compute; never creates a sandbox.
    APP accepts a base name or the full agent-bricks-prefixed deployment name.
    """
    from databricks_agentbricks.services.environment_debug import connect

    if not session_id.strip() or len(session_id.encode("utf-8")) > 128:
        raise AgentCliError("--session-id must be non-blank and at most 128 UTF-8 bytes.")
    if command is not None and (not command or len(command) > 32768):
        raise AgentCliError("--command must contain 1–32768 characters.")
    launcher = LauncherClient(ctx.obj.profile, app=_prefixed_name(app))
    config = launcher.request("GET", "/api/config")
    # The current terminal protocol addresses the saved revision internally.
    # Resolve it from the deployment rather than exposing it as a CLI concept.
    binding = config.get("default_environment")
    if not binding:
        raise AgentCliError("This App has no sandbox deployment binding; deploy it with --sandbox.")
    ctx.exit(connect(launcher, binding, session_id, command=command))
