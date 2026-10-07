"""Click/rich presentation for deployment results and lifecycle commands.

``DeployService`` is CLI-framework-agnostic: it reaches the terminal through the
:class:`~databricks_agentbricks.reporting.Reporter` port and returns raw facts. This module
chooses each verb's `--output json` payload and human rendering; the separate
``presentation.reporter`` adapter handles progress. CLI commands handle confirmation.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.endpoint import print_agent_invoke_command
from databricks_agentbricks.presentation.render import field
from databricks_agentbricks.presentation.tracing import TRACING_BIND_COMMAND, experiment_url
from databricks_agentbricks.services.deploy_service import DeployResult
from databricks_agentkit import timefmt


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
        store_grant_attempted = result.memory_grant_attempted or result.session_grant_attempted
        store_grant_error = result.session_grant_error or result.memory_grant_error
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
                # TODO(vNext): add a versioned per-store schema before exposing separate memory and
                # session grant fields. Preserve the established machine contract until then.
                "store_grant": "skipped"
                if not store_grant_attempted
                else ("granted" if store_grant_error is None else "failed"),
                "store_grant_error": store_grant_error,
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
    if result.url:
        steps.insert(0, f"Open the deployed agent: {result.url}")
    if result.created_app_yaml:
        steps.insert(
            0,
            f"Set a real `command:` in {pathlib.Path(result.source) / 'app.yaml'} "
            "(a placeholder was written)",
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
        next_steps=steps,
    )
    print_agent_invoke_command(result.deployment, uses_runtime_api=result.uses_runtime_api)
