"""Preflight and conservative Apps scope reconciliation for request-user tools."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.apps import App, AppsAPI

from databricks_agentbricks.agent_project import AgentProject
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_types import AgentServer

_IDENTITY_DEFAULT_SCOPES = frozenset({"iam.access-control:read", "iam.current-user:read"})


def _validate_forwarding(app: App) -> None:
    if app.forward_user_access_token is False:
        raise AgentCliError(
            f"App '{app.name}' has forward_user_access_token=False; user auth requires forwarding.",
            hint="Enable user access token forwarding in Databricks Apps, then stop and start "
            "the App compute for the setting to take effect before retrying deployment.",
        )


def _validate_implicit_identity_scopes(app: App) -> None:
    if app.user_api_scopes is not None:
        return
    effective = set(app.effective_user_api_scopes or [])
    if effective - _IDENTITY_DEFAULT_SCOPES:
        raise AgentCliError(
            "Apps did not return configured scopes for an App with unexplained effective grants.",
            hint="Inspect the App's scope configuration with its owner before retrying the scope "
            "update.",
        )


def requires_user_auth(project: AgentProject | None) -> bool:
    """Infer request-user auth from tool bindings before any deployment mutation."""
    if project is None or not project.tools:
        return False
    managed = [
        tool
        for tool in project.tools
        if tool.source.kind in ("mcp", "sandbox", "genie_one", "genie_agent")
    ]
    user_auth = any(tool.auth == "user" for tool in managed)
    if user_auth and project.server != AgentServer.AGENTBRICKS:
        raise AgentCliError(
            "Managed tools with auth = 'user' require [agent].server = 'agentbricks'.",
            hint="Migrate to the request-auth-aware Agent Bricks DurableAgentServer template before enabling user "
            "auth. Failure recovery is unsupported for request-user attempts because the credential "
            "is transient.",
        )
    if user_auth:
        unspecified = [tool.id for tool in managed if tool.auth is None]
        if unspecified:
            raise AgentCliError(
                "Request-user tools require explicit auth on every managed "
                f"tool binding: {', '.join(unspecified)}.",
                hint="Choose auth = 'app' to preserve legacy identity, or explicitly choose 'user'.",
            )
    return user_auth


@dataclass(frozen=True)
class AppUserScopeUpdatePlan:
    apps: AppsAPI
    name: str
    existing_scopes: tuple[str, ...] | None
    scopes: tuple[str, ...]


def required_user_api_scopes(project: AgentProject | None) -> set[str]:
    """Return Databricks Apps user API scopes required by request-user tools."""
    scopes: set[str] = set()
    # TODO: Extend this least-privilege mapping for each supported request-user tool kind/service.
    for tool in project.tools if project else ():
        if tool.auth != "user":
            continue
        if tool.source.kind in ("genie_one", "genie_agent"):
            scopes.add("genie")
            continue
        if tool.source.kind not in ("mcp", "sandbox"):
            continue
        scopes.add("ai-gateway")
        if tool.source.service == "system.ai.dbsql":
            scopes.add("sql")
        if tool.source.service == "system.ai.genie_one_mcp":
            scopes.add("genie")
        if tool.source.kind == "sandbox":
            if any(scope.kind == "volume" for scope in tool.policy.downscope):
                scopes.add("files")
            if any(scope.kind == "workspace" for scope in tool.policy.downscope):
                print(
                    "[freshness-check workspace-scope-e2e] required_user_api_scopes",
                    file=sys.stderr,
                )
                scopes.add("workspace")
    return scopes


def plan_app_user_scope_update(
    name: str,
    profile: str | None,
    *,
    allow_existing_app_update: bool,
    required_scopes: set[str] | None = None,
) -> AppUserScopeUpdatePlan:
    """Plan user-scope creation or addition without changing the Databricks App.

    A new App can be created with the requested user API scopes automatically. An existing App is
    left unchanged unless all requested scopes are already present or the caller explicitly allows
    Agent Bricks to add the missing scopes.
    """
    try:
        apps = WorkspaceClient(profile=profile).apps
        try:
            existing = apps.get(name)
        except NotFound:
            existing = None
    except (DatabricksError, ValueError) as exc:
        raise AgentCliError(f"Could not read Apps user scopes for '{name}'.") from exc
    if existing is not None:
        _validate_forwarding(existing)
    if existing is not None:
        _validate_implicit_identity_scopes(existing)
    configured = tuple(sorted(set(existing.user_api_scopes or []))) if existing else None
    requested = {"ai-gateway"} if required_scopes is None else required_scopes
    scopes = tuple(sorted({*(configured or ()), *requested}))
    if existing is not None and scopes != configured and not allow_existing_app_update:
        raise AgentCliError(
            f"App '{name}' is missing required user API scopes; re-run with "
            "--allow-user-scope-update to add them.",
            hint="Review the existing App scopes and coordinate with other owners first. Agent Bricks "
            "preserves unrelated scopes; later deploys do not need the flag once all required "
            "scopes are present.",
        )
    return AppUserScopeUpdatePlan(apps=apps, name=name, existing_scopes=configured, scopes=scopes)


def apply_app_user_scope_update(
    plan: AppUserScopeUpdatePlan, *, instances: int | None = None, attempts: int = 12
) -> None:
    """Create a scoped App or add missing user API scopes to an existing App.

    New Apps enable user-token forwarding at creation. Existing Apps retain unrelated scopes and
    settings; the update mask includes only user API scopes and explicitly requested instance
    fields. The read-before-write check detects known drift, but Apps has no compare-and-swap
    contract, so owners must still coordinate concurrent updates. Scope removal is never automatic.
    """
    if not plan.scopes or not set(plan.existing_scopes or ()).issubset(plan.scopes):
        raise AgentCliError(
            "Automatic removal of Apps user scopes is unsupported.",
            hint="Remove scopes explicitly in Databricks Apps and verify the effective scopes. "
            "The SDK omits empty lists when serializing App; no removal was sent.",
        )
    desired = App(
        name=plan.name,
        user_api_scopes=list(plan.scopes),
        forward_user_access_token=True if plan.existing_scopes is None else None,
    )
    mask = ["user_api_scopes"]
    if instances is not None:
        desired.compute_min_instances = instances
        desired.compute_max_instances = instances
        mask.extend(["compute_min_instances", "compute_max_instances"])
    try:
        if plan.existing_scopes is None:
            plan.apps.create(desired)
        else:
            current = plan.apps.get(plan.name)
            _validate_forwarding(current)
            _validate_implicit_identity_scopes(current)
            if tuple(sorted(set(current.user_api_scopes or []))) != plan.existing_scopes:
                raise AgentCliError("Apps user scopes changed since preflight; review and retry.")
            if plan.scopes != plan.existing_scopes or instances is not None:
                plan.apps.create_update(plan.name, update_mask=",".join(mask), app=desired).result(
                    timeout=timedelta(minutes=5)
                )
        for attempt in range(attempts):
            current = plan.apps.get(plan.name)
            _validate_forwarding(current)
            desired_scopes = set(plan.scopes)
            effective = set(current.effective_user_api_scopes or [])
            if (
                set(current.user_api_scopes or []) == desired_scopes
                and desired_scopes.issubset(effective)
                and effective.issubset(desired_scopes | _IDENTITY_DEFAULT_SCOPES)
            ):
                return
            if attempt + 1 < attempts:
                time.sleep(5)
    except (DatabricksError, TimeoutError) as exc:
        raise AgentCliError(f"Could not reconcile Apps user scopes for '{plan.name}'.") from exc
    raise AgentCliError(
        f"App '{plan.name}' effective user scopes did not converge after {attempts} checks.",
        hint="Source deployment was stopped. Inspect requested/effective scopes in Databricks "
        "Apps; contact your platform administrator if propagation remains blocked. Then retry.",
    )
