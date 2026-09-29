"""Click/rich presentation for the deploy service: the `Reporter` adapter and the result output.

``DeployService`` is CLI-framework-agnostic — it reports progress through the :class:`Reporter` port
and returns raw facts. This module is the only place those facts become terminal output, so the
`--output json` payload and the human success panel are chosen here, not inside the business logic.
"""

from __future__ import annotations

import pathlib
from contextlib import AbstractContextManager
from typing import Any, Optional

import click

from databricks_agentbricks import render
from databricks_agentbricks.cli.endpoint_examples import print_agent_invoke_command
from databricks_agentbricks.cli.tracing import TRACING_BIND_COMMAND, experiment_url
from databricks_agentbricks.services.deploy_service import DeployResult


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
                "store_grant": "skipped"
                if not result.grants_stores
                else ("granted" if result.store_grant_error is None else "failed"),
                "store_grant_error": result.store_grant_error,
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
    if result.grants_stores and result.store_grant_error is not None:
        steps.insert(
            0,
            "The app's service principal needs read/write on its store tables; that grant couldn't "
            "be applied automatically (it requires store ownership). "
            f"Cause: {result.store_grant_error}",
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
    if result.grants_stores and result.store_grant_error is None:
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
