"""Click/rich presentation for the deploy service: the terminal ports and every verb's output.

``DeployService`` is CLI-framework-agnostic: it reaches the terminal through the
:class:`Reporter` / :class:`Prompter` ports and returns raw facts. This module is the only place
those facts become terminal output, so each verb's `--output json` payload and its human rendering
(the deploy success panel, the deployment table, the detail view, the action confirmations) are
chosen here, not inside the business logic.
"""

from __future__ import annotations

import pathlib
from contextlib import AbstractContextManager
from typing import Any, Optional

import click

from databricks_agentbricks import render
from databricks_agentbricks.cli.endpoint_examples import print_agent_invoke_command
from databricks_agentbricks.cli.tracing import TRACING_BIND_COMMAND, experiment_url
from databricks_agentbricks.render import field
from databricks_agentbricks.services.deploy_service import DeployResult
from databricks_agentkit import timefmt


class ClickReporter:
    """A :class:`Reporter` backed by ``render`` + ``click``: spinners, progress lines, and notices."""

    def status(self, message: str) -> AbstractContextManager[None]:
        return render.status(message)

    def progress(self, message: str) -> AbstractContextManager[None]:
        return render.progress(message)

    def note(self, message: str) -> None:
        click.echo(message, err=True)

    def echo(self, message: str, *, newline: bool = True) -> None:
        click.echo(message, nl=newline)


class ClickPrompter:
    """A :class:`Prompter` backed by ``click.confirm``: the interactive yes/no at the terminal."""

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        return click.confirm(prompt, default=default)


def _deployment_status(a: dict) -> Optional[str]:
    """The state to show for a deployment: the app's, else its compute's, else a bare top-level one.

    `apps list` and `apps get` report state in whichever of these sections the app version populates,
    so fall through them rather than pick one and show a blank pill.
    """
    for key in ("app_status", "compute_status"):
        section = a.get(key)
        if isinstance(section, dict) and field(section, "state"):
            return field(section, "state")
    return field(a, "state")


def present_deployments_list(items: list[dict], *, output: Optional[str]) -> None:
    """Show the agent deployments: the raw payloads for `--output json`, else a table."""
    if output == "json":
        render.emit_json(items)
        return
    rows = [
        [
            render.hyperlink(field(a, "name"), field(a, "url")),
            render.status_pill(_deployment_status(a)),
            timefmt.relative(field(a, "update_time")),
        ]
        for a in items
    ]
    render.resource_table(
        "Agent Deployments",
        [("Name", "left"), ("Status", "left"), ("Updated", "left")],
        rows,
    )


def present_deployment_detail(data: dict, name: str, *, output: Optional[str]) -> None:
    """Show one deployment: the raw payload for `--output json`, else a detail view."""
    if output == "json":
        render.emit_json(data)
        return
    url = field(data, "url")
    render.detail(
        "Agent Deployment",
        # The requested name is the fallback: an app that answered without a name still has the one
        # the user asked about.
        field(data, "name") or name,
        {
            "URL": render.hyperlink(url, url) if url else None,
            "Description": field(data, "description"),
            "Created": timefmt.absolute(field(data, "create_time")),
            "Updated": timefmt.absolute(field(data, "update_time")),
        },
        status=_deployment_status(data),
        snippets=[("open", "bash", f"open {url}")] if url else None,
    )


def present_started(name: str, *, output: Optional[str]) -> None:
    """Confirm a start: the machine payload for `--output json`, else a success line."""
    if output == "json":
        render.emit_json({"started": name})
        return
    render.success(f"Started deployment '{name}'")


def present_stopped(name: str, *, output: Optional[str]) -> None:
    """Confirm a stop: the machine payload for `--output json`, else a success line."""
    if output == "json":
        render.emit_json({"stopped": name})
        return
    render.success(f"Stopped deployment '{name}'")


def present_deleted(name: str, *, output: Optional[str]) -> None:
    """Confirm a delete: the machine payload for `--output json`, else a success line."""
    if output == "json":
        render.emit_json({"deleted": name})
        return
    render.success(f"Deleted deployment '{name}'")


def present_deploy_result(result: DeployResult, *, output: Optional[str]) -> None:
    """Report a finished deploy: the machine payload for `--output json`, else the success panel."""
    if output == "json":
        render.emit_json(
            {
                "deployment": result.deployment,
                "url": result.url,
                "workspace_path": result.workspace_path,
                "env": result.env,
                "trace_experiment_id": result.trace_experiment_id,
                "uc_trace_tables": result.uc_trace_tables,
                "trace_setup_error": result.trace_setup_error,
                "trace_grant": None
                if not result.trace_experiment_id
                else ("granted" if result.trace_grant_error is None else "failed"),
                "trace_grant_error": result.trace_grant_error,
                "memory_grant": "skipped"
                if not result.grants_memory
                else ("granted" if result.memory_grant_error is None else "failed"),
                "memory_grant_error": result.memory_grant_error,
                "session_grant": "skipped"
                if not result.grants_session
                else ("granted" if result.session_grant_error is None else "failed"),
                "session_grant_error": result.session_grant_error,
            }
        )
        return

    provisioned: dict[str, Any] = {}
    if result.memory_store:
        provisioned["Memory store"] = result.memory_store
    if result.session_store:
        provisioned["Session store"] = result.session_store
    if result.trace_experiment_id:
        provisioned["Traces"] = (
            experiment_url(result.client_host, result.trace_experiment_id)
            or result.trace_experiment_id
        )
    if result.pip_index_url:
        provisioned["Package index"] = result.pip_index_url
    if result.instances is not None:
        provisioned["Instances"] = str(result.instances)

    steps: list[str | tuple[str, str]] = [
        (f"agentbricks deployments get {result.deployment}", "Check its status and URL"),
        (f"agentbricks deployments logs {result.deployment}", "Tail its logs"),
    ]
    if result.url:
        steps.insert(0, f"Open the deployed agent: {result.url}")
    if result.scaffolded:
        steps.insert(
            0,
            f"Set a real `command:` in {pathlib.Path(result.source) / 'app.yaml'} "
            "(a placeholder was written)",
        )
    # Per-store failure next steps. Inserted session-then-memory so memory lands above session (each
    # insert(0) pushes to the front), matching the reconcile/grant order.
    if result.grants_session and result.session_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs read/write on its session store tables; that grant "
            "couldn't be applied automatically (it requires store ownership). "
            f"Cause: {result.session_grant_error}",
        )
    if result.grants_memory and result.memory_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs read/write on its memory store tables; that grant "
            "couldn't be applied automatically (it requires store ownership). "
            f"Cause: {result.memory_grant_error}",
        )
    if result.trace_experiment_id is None:
        # Deployed without tracing - either unbound, or a bound experiment that couldn't be set up.
        # Tell the developer (in case it wasn't intended) and point at `agentbricks tracing bind`; append the
        # cause when setup actually failed.
        step = (
            "Deployed without tracing. "
            f"Run `{TRACING_BIND_COMMAND}` and redeploy to trace this agent."
        )
        if result.trace_setup_error is not None:
            step += f" (Tracing setup failed: {result.trace_setup_error})"
        steps.insert(0, step)
    if result.trace_experiment_id and result.trace_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs write access to its trace experiment; that grant "
            f"couldn't be applied automatically. Cause: {result.trace_grant_error}",
        )
    if (
        (result.grants_memory or result.grants_session)
        and result.memory_grant_error is None
        and result.session_grant_error is None
    ):
        provisioned["Store access"] = "granted to app service principal"
    if result.trace_experiment_id and result.trace_grant_error is None:
        provisioned["Trace access"] = "granted to agent runtime service principal"
    fields = {"URL": result.url} if result.url else {}
    fields.update({"Workspace path": result.workspace_path, **provisioned})
    render.success(
        f"Deployed agent '{result.deployment}'",
        fields=fields,
        next_steps=steps,
    )
    print_agent_invoke_command(result.deployment, uses_runtime_api=result.uses_runtime_api)
