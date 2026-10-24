"""Click/rich presentation for deployment results and lifecycle commands.

``DeployService`` is CLI-framework-agnostic: it reaches the terminal through the
:class:`~databricks_agentbricks.reporting.Reporter` port and returns raw facts. This module
chooses each verb's `--output json` payload and human rendering; the separate
``presentation.reporter`` adapter handles progress. CLI commands handle confirmation.
"""

from __future__ import annotations

import pathlib
import shlex
from typing import Any, Optional

import click
from rich.text import Text

from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.endpoint import agent_invoke_step
from databricks_agentbricks.presentation.render import field
from databricks_agentbricks.presentation.tracing import TRACING_BIND_COMMAND, experiment_url
from databricks_agentbricks.reporting import format_elapsed
from databricks_agentbricks.services.deploy_service import DeleteResult, DeployResult, WaitResult
from databricks_agentbricks.services.deployment.app_provisioner import (
    APP_DELETE_TIMEOUT_S,
    source_delete_command,
)
from databricks_agentkit import timefmt
from databricks_agentkit.runtime import prompt_registry


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


def present_deleted(result: DeleteResult, *, output: Optional[str]) -> None:
    """Confirm a delete: the machine payload for `--output json`, else a success line."""
    path = result.source_path
    if output == "json":
        render.emit_json(
            {
                "deleted": result.deployment,
                "still_deleting": result.still_deleting,
                "wait_error": result.wait_error,
                "source_path": path,
                "source_path_status": result.source_folder_status,
                "source_path_error": result.source_folder_error,
            }
        )
        return
    check = f"`agentbricks deployments get {result.deployment}`"
    if result.still_deleting:
        last = f" The last check failed: {result.wait_error}" if result.wait_error else ""
        render.warning(
            f"Deployment '{result.deployment}' was still being deleted after "
            f"{APP_DELETE_TIMEOUT_S // 60} minutes.{last} Check {check}; it may finish on its own."
        )
    elif result.wait_error:
        render.warning(
            f"Could not confirm deployment '{result.deployment}' was deleted: {result.wait_error} "
            f"Check {check}; if the profile isn't signed in, run `agentbricks login` first."
        )
    else:
        render.success(f"Deleted deployment '{result.deployment}'")
    if path is None:
        return
    if result.source_folder_status == "kept":
        _print_source_delete_command(
            "Kept the source folder; to delete it"
            if result.confirmed
            else "Kept the source folder. Once the deployment is gone, delete it",
            path,
        )
    elif result.source_folder_status == "failed":
        render.warning(str(result.source_folder_error))
        _print_source_delete_command("To delete it yourself", path)
    elif result.source_folder_status == "kept_app_not_confirmed":
        _print_source_delete_command(
            "Kept the source folder: the deployment wasn't confirmed gone. Once it is, delete it",
            path,
        )


def _print_source_delete_command(intro: str, path: str) -> None:
    con = render.console()
    con.print(Text(intro, style=render.SECONDARY))
    # A separate, soft-wrapped line avoids copying panel borders or inserted line breaks.
    con.print(Text(source_delete_command(path), style=render.COMMAND), soft_wrap=True)


def deploy_result_payload(result: DeployResult) -> dict[str, Any]:
    """The `--output json` object for a finished deploy, also recorded in its local state file."""
    store_grant_attempted = result.memory_grant_attempted or result.session_grant_attempted
    store_grant_error = result.session_grant_error or result.memory_grant_error
    return {
        "deployment": result.deployment,
        "url": result.url,
        # True once the App answered; None when the profile couldn't check.
        "ready": result.ready,
        "workspace_path": result.workspace_path,
        "env": result.env,
        "memory_store": result.memory_store,
        "session_store": result.session_store,
        "trace_experiment_id": result.trace_experiment_id,
        "uc_trace_tables": result.uc_trace_tables,
        "trace_setup_error": result.trace_setup_error,
        "trace_grant": None
        if not result.trace_experiment_id
        else ("granted" if result.trace_grant_error is None else "failed"),
        "trace_grant_error": result.trace_grant_error,
        # TODO(vNext): add a versioned per-store schema before exposing separate memory and
        # session grant fields. Preserve the established machine contract until then.
        "store_grant": "skipped"
        if not store_grant_attempted
        else ("granted" if store_grant_error is None else "failed"),
        "store_grant_error": store_grant_error,
        "model_services": result.model_services,
        "model_service_grant": None
        if not result.model_services
        else ("granted" if result.model_service_grant_error is None else "failed"),
        "model_service_grant_error": result.model_service_grant_error,
        "prompts": result.prompts,
        "prompt_grant": None
        if not result.prompts
        else ("granted" if result.prompt_grant_error is None else "failed"),
        "prompt_grant_error": result.prompt_grant_error,
        "tool_access": None
        if result.tool_access is None
        else {
            "app_resources": result.tool_access.app_resources,
            "uc_grants": result.tool_access.uc_grants,
            "workspace_grants": result.tool_access.workspace_grants,
            "direct_resources_only": True,
            "uc_workspace_grants_additive": bool(
                result.tool_access.uc_grants or result.tool_access.workspace_grants
            ),
        },
    }


def present_deploy_result(
    result: DeployResult, *, output: Optional[str], elapsed_s: Optional[float] = None
) -> None:
    """Report a finished deploy: the machine payload for `--output json`, else the success panel.

    Text output ends with the deploy's wall time when ``elapsed_s`` is given; the JSON payload is
    unchanged so scripts keep their established contract.
    """
    if output == "json":
        render.emit_json(deploy_result_payload(result))
        return

    provisioned: dict[str, Any] = {}
    if result.memory_store:
        provisioned["Memory store"] = result.memory_store
    if result.session_store:
        provisioned["Session store"] = result.session_store
    if result.model_services:
        provisioned["Model services"] = ", ".join(
            f"{role}: {service}" for role, service in result.model_services.items()
        )
        if result.model_service_grant_error is None:
            provisioned["Model access"] = "granted to app service principal"
    prompt_schemas = sorted({prompt_registry.schema_of(p) for p in result.prompts.values()})
    if prompt_schemas and result.prompt_grant_error is None:
        provisioned["Prompt access"] = (
            f"granted to app service principal on {', '.join(prompt_schemas)}"
        )
    if result.trace_experiment_id:
        provisioned["Traces"] = (
            experiment_url(result.client_host, result.trace_experiment_id)
            or result.trace_experiment_id
        )
    if result.pip_index_url:
        provisioned["Package index"] = result.pip_index_url
    if result.instance_count is not None:
        provisioned["Instances"] = str(result.instance_count)
    if result.tool_access is not None:
        target_count = (
            result.tool_access.app_resources
            + result.tool_access.uc_grants
            + result.tool_access.workspace_grants
        )
        if target_count:
            additive_note = (
                "; UC/Workspace grants are additive"
                if result.tool_access.uc_grants or result.tool_access.workspace_grants
                else ""
            )
            provisioned["Tool access"] = (
                f"{target_count} explicit grant target"
                f"{'s' if target_count != 1 else ''} reconciled{additive_note}"
            )

    steps: list[str | tuple[str, str]] = [
        (f"agentbricks deployments get {result.deployment}", "Check its status and URL"),
        (f"agentbricks deployments logs {result.deployment}", "Tail its logs"),
    ]
    if result.memory_store:
        steps.append(
            (
                f"agentbricks memory stores get {shlex.quote(result.memory_store)}",
                "Look up the memory store",
            )
        )
    if result.session_store:
        steps.append(
            (
                f"agentbricks sessions stores get {shlex.quote(result.session_store)}",
                "Look up the session store",
            )
        )
    if result.url:
        steps.insert(0, f"Open the deployed agent: {result.url}")
    if result.created_app_yaml:
        steps.insert(
            0,
            f"Set a real `command:` in {pathlib.Path(result.source) / 'app.yaml'} "
            "(a placeholder was written)",
        )
    if result.model_services and result.model_service_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs EXECUTE on its model services "
            f"({', '.join(result.model_services.values())}) plus USE CATALOG / USE SCHEMA; "
            "that grant couldn't be applied automatically. "
            f"Cause: {result.model_service_grant_error}",
        )
    if prompt_schemas and result.prompt_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs USE SCHEMA, EXECUTE, CREATE FUNCTION, and MANAGE "
            f"on its prompt schemas ({', '.join(prompt_schemas)}) to load its prompts; "
            "that grant couldn't be applied automatically. "
            f"Cause: {result.prompt_grant_error}",
        )
    # Per-store failure next steps. Inserted session-then-memory so memory lands above session (each
    # insert(0) pushes to the front), matching the reconcile/grant order.
    if result.session_grant_attempted and result.session_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs read/write on its session store tables; that grant "
            "couldn't be applied automatically (it requires store ownership). "
            f"Cause: {result.session_grant_error}",
        )
    if result.memory_grant_attempted and result.memory_grant_error is not None:
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
    if result.trace_grant_error is not None:
        if result.trace_experiment_id:
            steps.insert(
                0,
                "The app's service principal needs write access to its trace experiment; that grant "
                f"couldn't be applied automatically. Cause: {result.trace_grant_error}",
            )
        else:
            steps.insert(
                0,
                "The previous tracing resources could not be removed automatically. "
                f"Cause: {result.trace_grant_error}",
            )
    if (
        (result.memory_grant_attempted or result.session_grant_attempted)
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
        next_steps=[
            *steps,
            agent_invoke_step(result.deployment, uses_runtime_api=result.uses_runtime_api),
        ],
    )
    if elapsed_s is not None:
        render.console().print(f"Deployed in {render.format_duration(elapsed_s)}")


def present_background_started(
    name: str, *, log: pathlib.Path, state_file: pathlib.Path, output: Optional[str]
) -> None:
    """Confirm a `deploy --no-wait` handoff and point at `deployments wait`."""
    if output == "json":
        render.emit_json(
            {
                "deployment": name,
                "status": "started",
                "log": str(log),
                "state_file": str(state_file),
            }
        )
        return
    click.echo(
        f"Deploy of {name} started in the background. Next: agentbricks deployments wait {name}"
    )


def present_wait_result(result: WaitResult, *, output: Optional[str]) -> None:
    """Report where a waited-on deploy stands; the command maps the status to its exit code."""
    if output == "json":
        render.emit_json(
            {
                "deployment": result.deployment,
                "status": result.status,
                "phase": result.phase,
                "elapsed_s": result.elapsed_s,
                "url": result.url,
                "error": result.error,
                "log_tail": result.log_tail,
            }
        )
        return
    if result.status == "succeeded":
        render.success(
            f"Deployed agent '{result.deployment}'",
            fields={"URL": result.url} if result.url else None,
            next_steps=[
                (f"agentbricks deployments logs {result.deployment}", "Tail its logs"),
                agent_invoke_step(result.deployment, uses_runtime_api=result.uses_runtime_api),
            ],
        )
        return
    if result.status == "failed":
        err = render.stderr_console()
        message, _, detail = (result.error or "The deploy failed.").partition("\n")
        render.diagnostic(
            "error",
            message,
            help=detail or "fix the cause, then re-run `agentbricks deploy`",
            con=err,
        )
        if result.log_tail:
            source = f"Last lines of {result.log}" if result.log else "App status"
            err.print(Text(f"{source}:", style="dim"), soft_wrap=True)
        for line in result.log_tail:
            err.print(Text(f"  {line}"), soft_wrap=True)
        return
    click.echo(
        f"still deploying: {result.phase}, {format_elapsed(result.elapsed_s)} elapsed — "
        "run the same command again"
    )
