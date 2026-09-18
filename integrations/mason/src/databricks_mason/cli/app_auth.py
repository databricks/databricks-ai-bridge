"""Preflight and conservative Apps scope reconciliation for request-user tools."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.apps import App, AppsAPI

from databricks_mason.agent_project import AgentProject
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import load_project_metadata
from databricks_mason.project_types import AgentServer

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
            hint="Inspect the App's scope configuration with its owner before retrying adoption.",
        )


def requires_user_auth(project: AgentProject | None) -> bool:
    """Validate the local contract before any deployment or store mutation."""
    if project is None or not project.tools:
        return False
    managed = [tool for tool in project.tools if tool.source.kind in ("mcp", "sandbox")]
    user_auth = any(tool.auth == "user" for tool in managed)
    metadata = None
    if (project.root / ".mason/project.toml").is_file():
        metadata = load_project_metadata(project.root)
    contract = metadata.request_auth_contract_version if metadata else None
    if user_auth and (contract != 1 or project.server != AgentServer.MASON):
        raise AgentCliError(
            "User auth requires request_auth_contract_version = 1 in .mason/project.toml.",
            hint="Migrate to the request-auth-aware Mason AgentApp template before setting the "
            "marker. Failure recovery is unsupported for request-user attempts because the "
            "credential is transient.",
        )
    if user_auth or contract == 1:
        unspecified = [tool.id for tool in managed if tool.auth is None]
        if unspecified:
            raise AgentCliError(
                "Request-auth contract migration requires explicit auth on every managed "
                f"MCP/sandbox binding: {', '.join(unspecified)}.",
                hint="Choose auth = 'app' to preserve legacy identity, or explicitly choose 'user'.",
            )
    return user_auth


@dataclass(frozen=True)
class AppAuthPlan:
    apps: AppsAPI
    name: str
    existing_scopes: tuple[str, ...] | None
    scopes: tuple[str, ...]


def required_user_scopes(project: AgentProject | None) -> set[str]:
    """Return baseline Apps scopes for request-user managed tools."""
    scopes: set[str] = set()
    for tool in project.tools if project else ():
        if tool.auth != "user" or tool.source.kind not in ("mcp", "sandbox"):
            continue
        scopes.add("ai-gateway")
        if tool.source.service == "system.ai.dbsql":
            scopes.add("sql")
    return scopes


def prepare_app_auth(
    name: str,
    profile: str | None,
    *,
    adopt: bool,
    required_scopes: set[str] | None = None,
) -> AppAuthPlan:
    """Read scope ownership before mutations; an existing App requires explicit adoption."""
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
    if existing is not None and not adopt:
        raise AgentCliError(
            f"App '{name}' already exists; user-auth scope management requires --adopt-user-auth.",
            hint="Review its existing scopes and coordinate with other owners first. Adoption "
            "preserves unrelated scopes; scope writes are not atomic with concurrent changes.",
        )
    if existing is not None:
        _validate_implicit_identity_scopes(existing)
    configured = tuple(sorted(set(existing.user_api_scopes or []))) if existing else None
    requested = {"ai-gateway"} if required_scopes is None else required_scopes
    scopes = tuple(sorted({*(configured or ()), *requested}))
    return AppAuthPlan(apps=apps, name=name, existing_scopes=configured, scopes=scopes)


def apply_app_auth(plan: AppAuthPlan, *, instances: int | None = None, attempts: int = 12) -> None:
    """Apply only nonempty scopes, preserving unrelated settings, then verify effective scopes.

    The read-before-write check detects known drift, not all races: Apps has no compare-and-swap
    contract here. Explicit adoption requires owners to coordinate writes. Scope removal is never
    automatic, including when the manifest becomes app-only.
    """
    if not plan.scopes or not set(plan.existing_scopes or ()).issubset(plan.scopes):
        raise AgentCliError(
            "Automatic removal of Apps user scopes is unsupported.",
            hint="Remove scopes explicitly in Databricks Apps and verify the effective scopes. "
            "The SDK omits empty lists when serializing App; no removal was sent.",
        )
    desired = App(name=plan.name, user_api_scopes=list(plan.scopes))
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
