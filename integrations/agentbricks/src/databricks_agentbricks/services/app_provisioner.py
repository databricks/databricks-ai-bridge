"""Lifecycle orchestration for the deployed Databricks App itself.

``AppProvisioner`` owns the app the four
:class:`~databricks_agentbricks.services.provisioners.ResourceProvisioner`\\ s hang off - the
Databricks App that a deploy creates and rolls source out to. It deliberately does NOT implement the
three-phase ``ResourceProvisioner`` protocol: its moments are "ensure the app exists (and its compute
is active)" and "roll the source out", not ``reconcile``/``after_app_ready``/``grant``, so the naming
does not imply it is one of the resource provisioners.

Like the rest of the services layer it talks to the terminal only through the injected
:class:`Reporter`; it shells out through the injected ``AppsClient`` and opens a workspace client only
through ``api_client_factory``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from databricks_agentbricks.app_auth_client import AppUserScopeUpdatePlan
from databricks_agentbricks.apps_client import AppsClient
from databricks_agentbricks.deployment import _AGENT_COMPUTE_OUTPUT
from databricks_agentbricks.services.interaction import Reporter
from databricks_agentbricks.services.provisioners import ResourceContext


class AppProvisioner:
    """Create-or-scale the deployed App, wait for its compute, then sync and roll out the source."""

    def __init__(
        self,
        apps_client: AppsClient,
        api_client_factory: Callable[[], Any],
        reporter: Reporter,
    ) -> None:
        self._apps_client = apps_client
        self._api = api_client_factory
        self._reporter = reporter

    def ensure(
        self,
        ctx: ResourceContext,
        user_scope_plan: Optional[AppUserScopeUpdatePlan],
        instance_count: Optional[int],
        instance_args: list[str],
    ) -> None:
        name = ctx.project.name
        deployment_exists = ctx.deployment_exists
        #    `apps create` itself blocks for minutes (it provisions and waits for compute) and we capture
        #    its output to relabel "App compute" → "Agent compute", so nothing streams meanwhile. Wrap it
        #    in progress (persistent line + spinner) so the CLI isn't silent for the whole provision.
        if user_scope_plan is None and not deployment_exists:
            with self._reporter.progress(
                "Creating the agent and starting its compute (this can take a few minutes)…"
            ):
                out = self._apps_client.create(name, instance_args)
            old, new = _AGENT_COMPUTE_OUTPUT
            self._reporter.echo(out.replace(old, new), newline=False)
        elif user_scope_plan is None and instance_count is not None:
            out = self._apps_client.create_update_instances(name, instance_count)
            old, new = _AGENT_COMPUTE_OUTPUT
            self._reporter.echo(out.replace(old, new), newline=False)
        # `apps deploy` requires the app's compute to be ACTIVE — a just-created app may still be
        # starting, and an existing one may be STOPPED — so wait either way. Returns immediately when
        with self._reporter.progress(
            "Waiting for agent compute to start (this can take a few minutes)…"
        ):
            self._apps_client.wait_for_running(name)

    def rollout(self, ctx: ResourceContext, workspace_path: Optional[str]) -> str:
        ws_path = (
            workspace_path
            or f"/Workspace/Users/{self._api().current_user}/agentbricks_deployments/{ctx.project.name}"
        )
        self._apps_client.sync_source(ctx.project.name, ctx.project.source_dir, ws_path)
        self._apps_client.deploy(ctx.project.name, ws_path)
        return ws_path
