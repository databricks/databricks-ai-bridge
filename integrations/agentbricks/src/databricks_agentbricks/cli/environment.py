"""Inspect, invoke, and debug published sandbox environments."""

from __future__ import annotations

import json
import time
from uuid import uuid4

import click

from databricks_agentbricks.cli.environment_client import LauncherClient
from databricks_agentbricks.environment_source import environment_name
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke.request import load_json
from databricks_agentkit._api_client import _workspace_client


def launcher_options(command):
    command = click.option(
        "--auth/--no-auth",
        default=None,
        help="Inject Databricks OAuth (default: on for --app, off for --url).",
    )(command)
    command = click.option("--url", help="Base URL of a local or hosted sandbox launcher.")(command)
    return click.option("--app", help="Exact name of the deployed sandbox launcher App.")(command)


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


@click.group()
def environment():
    """Inspect, invoke, and debug published sandbox environments.

    Use deploy --sandbox to publish source and provision an orchestrator with its
    managed Runtime Store. Environment names identify published source revisions.
    """


@environment.command("get")
@click.argument("name")
@click.pass_obj
def get_environment(obj, name):
    """Inspect published metadata and current runtime state by name or manifest path."""
    from databricks_agentbricks.prepared_environment import read_manifest, services
    from databricks_agentkit.sandbox.environment import on_demand, sandbox_scope

    workspace = _workspace_client(obj.profile)
    manifest = read_manifest(workspace, name)
    if "runtime_store" in manifest:
        manifest["runtime"] = {"owner_app": manifest["runtime_store"]["owner_app"]}
    elif on_demand(manifest) and manifest.get("status") == "published":
        registry, _, _, _ = services(workspace, manifest)
        manifest["runtime"] = (
            {"sandbox_scope": "session", "sessions": registry.runtimes()}
            if sandbox_scope(manifest) == "session"
            else registry.runtime(manifest["runtime_key"])
        )
    click.echo(json.dumps(manifest, indent=2, default=str))


@environment.command("prepare")
@click.argument("name")
@click.pass_obj
def prepare_environment(obj, name):
    """Retry an artifact build from saved source, or legacy sandbox preparation."""
    from databricks_agentbricks.prepared_environment import prepare, read_manifest

    workspace = _workspace_client(obj.profile)
    result = prepare(
        workspace,
        read_manifest(workspace, name),
        progress=lambda stage: click.echo(stage, err=True),
    )
    click.echo(json.dumps(result, indent=2, default=str))


@environment.command("list")
@launcher_options
@click.pass_obj
def list_environments(obj, app, url, auth):
    """List environments registered with the sandbox launcher."""
    launcher = LauncherClient(obj.profile, app=app, url=url, auth=auth)
    click.echo(json.dumps(launcher.request("GET", "/api/environments"), indent=2))


@environment.command("ssh")
@click.argument("name")
@click.option("--session-id", required=True, help="Session whose existing sandbox to debug.")
@click.option("--command", help="Run one shell command without a PTY and return its exit code.")
@launcher_options
@click.pass_context
def ssh(ctx, name, session_id, command, app, url, auth):
    """Open a debug shell in a session's installed agent environment through its launcher.

    Requires CAN_MANAGE on the App. Resumes stopped compute; never creates a sandbox.
    Hosted Apps use HTTPS terminal transport; SSH is available when configured.
    """
    from databricks_agentbricks.services.environment_debug import connect

    name = environment_name(name)
    if not session_id.strip() or len(session_id.encode("utf-8")) > 128:
        raise AgentCliError("--session-id must be non-blank and at most 128 UTF-8 bytes.")
    if command is not None and (not command or len(command) > 32768):
        raise AgentCliError("--command must contain 1–32768 characters.")
    launcher = LauncherClient(ctx.obj.profile, app=app, url=url, auth=auth)
    ctx.exit(connect(launcher, name, session_id, command=command))


@environment.command("invoke")
@click.argument("name")
@click.option("--session-id", required=True, help="Session whose sandbox should be reused.")
@click.option("--invocation-id", help="Stable retry identifier (default: generated UUID).")
@click.option("--json", "json_value", required=True, help="Agent input as a JSON object.")
@click.option(
    "--retry",
    is_flag=True,
    help="Reattach to an existing invocation; failed runs require a new ID.",
)
@click.option("--stream", is_flag=True, help="Print incremental text while waiting for the run.")
@click.option(
    "--wait/--no-wait", default=True, show_default=True, help="Poll until the invocation finishes."
)
@click.option(
    "--timeout",
    type=click.FloatRange(min=0.1),
    default=300.0,
    show_default=True,
    help="Maximum polling time in seconds; timing out does not cancel the run.",
)
@launcher_options
@click.pass_obj
def invoke(
    obj, name, session_id, invocation_id, json_value, retry, stream, wait, timeout, app, url, auth
):
    """Invoke a session's pinned source revision in its sandbox."""
    name = environment_name(name)
    value = load_json(json_value)
    if not isinstance(value, dict):
        raise AgentCliError("--json must contain an object for the agent input.")
    if retry and not invocation_id:
        raise AgentCliError("--retry requires the original --invocation-id.")
    invocation_id = invocation_id or str(uuid4())
    if not app and not url:
        if not wait or stream:
            raise AgentCliError(
                "Standalone environment invocation waits for the result; --stream and --no-wait require an App launcher."
            )
        from databricks_agentbricks.prepared_environment import read_manifest, services
        from databricks_agentkit.sandbox.environment import bind_environment
        from databricks_agentkit.sandbox.jobs import RunManager

        workspace = _workspace_client(obj.profile)
        manifest = read_manifest(workspace, name)
        _, control, environments, orchestrator = services(workspace, manifest)
        bind_environment(orchestrator, manifest)
        from databricks_agentkit.runtime.types import InvocationFailedError

        manager = RunManager(control, orchestrator)
        try:
            manager.start()
            run = environments.invoke(name, session_id, invocation_id, value, retry=retry)
            manager.submit(run["id"])
            try:
                control.call(manager.runtime.wait(run["id"], timeout=timeout))
            except InvocationFailedError:
                pass
            except TimeoutError as exc:
                raise AgentCliError(
                    "Timed out waiting for the invocation; inspect the session before retrying."
                ) from exc
            run = control.run(run["id"])
        finally:
            manager.stop()
            control.close()
        if run["status"] in {"failed", "interrupted"}:
            raise AgentCliError(
                f"Run {run['id']} {run['status']}: {run.get('error')}",
                hint="Inspect the session, then retry with a new --invocation-id and the same input.",
            )
        click.echo(json.dumps(run, indent=2, default=str))
        return
    launcher = LauncherClient(obj.profile, app=app, url=url, auth=auth)
    run = request_run(
        launcher,
        "POST",
        f"/api/environments/{name}/runs",
        invocation_id,
        body={
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
    while wait and run["status"] in {"queued", "running"}:
        text = (run.get("stream") or {}).get("text", "")
        if stream and text.startswith(printed):
            click.echo(text[len(printed) :], nl=False, err=True)
            printed = text
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
    text = (run.get("stream") or {}).get("text", "")
    if stream and text.startswith(printed):
        click.echo(text[len(printed) :], err=True)
    if wait and run["status"] in {"failed", "interrupted"}:
        raise AgentCliError(
            f"Run {run['id']} {run['status']}: {run.get('error') or 'no diagnostic'}",
            hint="Inspect the session, then retry with a new --invocation-id and the same input.",
        )
    click.echo(json.dumps(run, indent=2, default=str))


@click.command("invoke")
@click.argument("app")
@click.option(
    "--environment", "environment_name", help="Environment name (default: deployment binding)."
)
@click.option("--session-id", required=True)
@click.option("--invocation-id")
@click.option("--json", "json_value", required=True)
@click.option(
    "--retry",
    is_flag=True,
    help="Reattach to an existing invocation; failed runs require a new ID.",
)
@click.option("--stream", is_flag=True, help="Print incremental text while waiting.")
@click.option("--timeout", type=click.FloatRange(min=0.1), default=300.0, show_default=True)
@click.pass_context
def invoke_deployment(
    ctx, app, environment_name, session_id, invocation_id, json_value, retry, stream, timeout
):
    """Invoke the environment served by an App; install its sandbox on first use."""
    from databricks_agentbricks.services.deployment.names import _prefixed_name

    app = _prefixed_name(app)
    if not environment_name:
        config = LauncherClient(ctx.obj.profile, app=app).request("GET", "/api/config")
        environment_name = config.get("default_environment")
    if not environment_name:
        raise AgentCliError("This App has no default environment; provide --environment.")
    return ctx.invoke(
        invoke,
        name=environment_name,
        session_id=session_id,
        invocation_id=invocation_id,
        json_value=json_value,
        retry=retry,
        stream=stream,
        wait=True,
        timeout=timeout,
        app=app,
        url=None,
        auth=None,
    )
