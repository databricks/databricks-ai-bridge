"""Preflight and conservative Apps scope reconciliation for request-user tools."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.apps import App, AppsAPI

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.projects.types import AgentServer

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
    """Infer request-user auth from the manifest before any deployment mutation."""
    if project is None:
        return False
    managed = [
        tool
        for tool in project.tools
        if tool.source.kind in ("mcp", "sandbox", "genie_one", "genie_agent")
    ]
    managed_user_auth = any(tool.auth == "user" for tool in managed)
    user_auth = project.user_auth.required or managed_user_auth
    if user_auth and project.server != AgentServer.AGENTBRICKS:
        raise AgentCliError(
            "Request-user auth requires [agent].server = 'agentbricks'.",
            hint="Migrate to the request-auth-aware Agent Bricks DurableAgentServer template before enabling user "
            "auth. Failure recovery is unsupported for request-user attempts because the credential "
            "is transient.",
        )
    if managed_user_auth:
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


@dataclass(frozen=True)
class AppAuthPlan:
    """Validated request-user scope preflight, carried to App setup without mutating it."""

    scope_update: Optional[AppUserScopeUpdatePlan]

    @property
    def required(self) -> bool:
        return self.scope_update is not None

    @property
    def app_existed(self) -> Optional[bool]:
        if self.scope_update is None:
            return None
        return self.scope_update.existing_scopes is not None


def required_user_api_scopes(project: AgentProject | None) -> set[str]:
    """Union explicit additions with scopes inferred from request-user managed tools."""
    scopes = set(project.user_auth.additional_api_scopes) if project else set()
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
        if tool.source.kind == "sandbox" and any(
            scope.kind == "volume" for scope in tool.policy.downscope
        ):
            scopes.add("files")
        if tool.source.kind == "sandbox" and tool.policy.databricks_access_token_included:
            scopes.add("workspace.workspace")
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


def validate_app_user_scope_drift(plan: AppUserScopeUpdatePlan) -> None:
    """Re-read an existing App before update so known scope or forwarding drift fails closed."""
    try:
        current = plan.apps.get(plan.name)
    except (DatabricksError, ValueError) as exc:
        raise AgentCliError(f"Could not read Apps user scopes for '{plan.name}'.") from exc
    _validate_forwarding(current)
    _validate_implicit_identity_scopes(current)
    if tuple(sorted(set(current.user_api_scopes or []))) != plan.existing_scopes:
        raise AgentCliError("Apps user scopes changed since preflight; review and retry.")


def wait_for_app_user_scopes(plan: AppUserScopeUpdatePlan, *, attempts: int = 12) -> None:
    """Verify configured and effective scopes converge before source rollout."""
    try:
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


class AppsUserAuthClient:
    """Plan request-user App scopes and validate them without owning App lifecycle writes.

    Preflight uses the Python SDK to read an App and check scopes. The resulting plan is handed to
    ``AppProvisioner``, which owns creation, scaling, and readiness. This client is render-free.
    """

    def __init__(self, profile: Optional[str]) -> None:
        self._profile = profile

    def requires_user_auth(self, project) -> bool:
        """Whether the project binds a managed tool with ``auth = 'user'``, so the App needs OBO.

        False for a project with no tools or no agent.toml at all. Raises when the bindings are
        inconsistent (user auth on a non-Agent-Bricks server, or a managed tool left without an
        explicit ``auth``), so a bad combination fails pre-flight rather than mid-deploy.
        """
        return requires_user_auth(project)

    def required_user_api_scopes(self, project) -> set[str]:
        """The least-privilege Apps user API scopes the project's request-user tools need.

        Empty when nothing requests user auth; derived from the tool bindings only, so it reads no
        workspace state.
        """
        return required_user_api_scopes(project)

    def plan_user_auth(
        self,
        name: str,
        project: AgentProject | None,
        *,
        allow_existing_app_update: bool,
    ) -> AppAuthPlan:
        """Validate request-user auth and plan scopes without creating or scaling an App."""
        required = self.requires_user_auth(project)
        if allow_existing_app_update and not required:
            raise AgentCliError(
                "--allow-user-scope-update requires a managed tool with auth = 'user' in agent.toml."
            )
        if not required:
            return AppAuthPlan(scope_update=None)

        plan = plan_app_user_scope_update(
            name,
            self._profile,
            allow_existing_app_update=allow_existing_app_update,
            required_scopes=self.required_user_api_scopes(project),
        )
        return AppAuthPlan(scope_update=plan)
