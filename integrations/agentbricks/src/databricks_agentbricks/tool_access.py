"""Deploy-time access reconciliation for direct App-auth tool resources."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.catalog import PermissionsChange, Privilege, SecurableType
from databricks.sdk.service.workspace import (
    ObjectType,
    WorkspaceObjectAccessControlRequest,
    WorkspaceObjectPermissionLevel,
)

from databricks_agentbricks.agent_project import ToolSpec
from databricks_agentbricks.app_resources import apply_tool_resources
from databricks_agentbricks.errors import AgentCliError

_APP_PERMISSION_STRENGTH = {
    "EXECUTE": 1,
    "SELECT": 1,
    "MODIFY": 2,
    "READ_VOLUME": 1,
    "WRITE_VOLUME": 2,
}
_WORKSPACE_PERMISSION_STRENGTH = {
    WorkspaceObjectPermissionLevel.CAN_READ: 1,
    WorkspaceObjectPermissionLevel.CAN_EDIT: 2,
    WorkspaceObjectPermissionLevel.CAN_MANAGE: 3,
}
_WORKSPACE_OBJECT_TYPES = {
    ObjectType.DIRECTORY: "directories",
    ObjectType.FILE: "files",
    ObjectType.NOTEBOOK: "notebooks",
}
_MCP_SERVICE_SECURABLE_TYPE = "MCP_SERVICE"


@dataclass(frozen=True)
class UcGrant:
    """One effective Unity Catalog privilege required by a direct tool resource."""

    securable_type: SecurableType | str
    full_name: str
    privilege: Privilege


@dataclass(frozen=True)
class WorkspaceGrant:
    """One additive Workspace ACL required by an explicit Sandbox path."""

    path: str
    permission: WorkspaceObjectPermissionLevel


@dataclass(frozen=True)
class ToolAccessPlan:
    """Least-privilege access required by App-auth entries in ``agent.toml``."""

    app_resources: tuple[dict[str, Any], ...] = ()
    uc_grants: tuple[UcGrant, ...] = ()
    workspace_grants: tuple[WorkspaceGrant, ...] = ()


def _resource_name(kind: str, identifier: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{identifier}".encode()).hexdigest()[:16]
    return f"agentbricks-tool-{digest}"


def _uc_app_resource(securable_type: str, full_name: str, permission: str) -> dict[str, Any]:
    return {
        "name": _resource_name(securable_type, full_name),
        "uc_securable": {
            "securable_full_name": full_name,
            "securable_type": securable_type,
            "permission": permission,
        },
    }


def _mcp_grants(service: str) -> set[UcGrant]:
    catalog, schema, _ = service.split(".")
    return {
        UcGrant(SecurableType.CATALOG, catalog, Privilege.USE_CATALOG),
        UcGrant(SecurableType.SCHEMA, f"{catalog}.{schema}", Privilege.USE_SCHEMA),
        UcGrant(_MCP_SERVICE_SECURABLE_TYPE, service, Privilege.EXECUTE),
    }


def _securable_type_value(securable_type: SecurableType | str) -> str:
    return securable_type.value if isinstance(securable_type, SecurableType) else securable_type


def plan_tool_access(tools: Sequence[ToolSpec]) -> ToolAccessPlan:
    """Plan grants for direct App/default-auth resources without transitive discovery."""
    app_resources: dict[tuple[str, str], dict[str, Any]] = {}
    genie_names: dict[str, str] = {}
    uc_grants: set[UcGrant] = set()
    workspace_grants: dict[str, WorkspaceObjectPermissionLevel] = {}

    def add_uc_resource(securable_type: str, full_name: str, permission: str) -> None:
        key = (securable_type, full_name)
        current = app_resources.get(key)
        if current is not None:
            current_permission = current["uc_securable"]["permission"]
            if _APP_PERMISSION_STRENGTH[current_permission] >= _APP_PERMISSION_STRENGTH[permission]:
                return
        app_resources[key] = _uc_app_resource(securable_type, full_name, permission)

    for tool in tools:
        if tool.auth == "user":
            continue
        kind = tool.source.kind
        if kind == "uc_function":
            add_uc_resource("FUNCTION", tool.source.function or "", "EXECUTE")
        elif kind == "genie_agent":
            space_id = tool.source.space_id or ""
            genie_names[space_id] = min(tool.id, genie_names.get(space_id, tool.id))
        elif kind == "mcp":
            uc_grants.update(_mcp_grants(tool.source.service or ""))
        elif kind == "sandbox":
            uc_grants.update(_mcp_grants(tool.source.service or ""))
            for scope in tool.policy.downscope:
                if scope.kind == "table":
                    permission = "SELECT" if scope.permission == "read_only" else "MODIFY"
                    add_uc_resource("TABLE", scope.value, permission)
                elif scope.kind == "volume":
                    permission = (
                        "READ_VOLUME" if scope.permission == "read_only" else "WRITE_VOLUME"
                    )
                    add_uc_resource("VOLUME", scope.value, permission)
                elif scope.kind == "workspace":
                    permission = (
                        WorkspaceObjectPermissionLevel.CAN_READ
                        if scope.permission == "read_only"
                        else WorkspaceObjectPermissionLevel.CAN_EDIT
                    )
                    current = workspace_grants.get(scope.value)
                    if current is None or (
                        _WORKSPACE_PERMISSION_STRENGTH[permission]
                        > _WORKSPACE_PERMISSION_STRENGTH[current]
                    ):
                        workspace_grants[scope.value] = permission

    for space_id, name in genie_names.items():
        app_resources[("GENIE_SPACE", space_id)] = {
            "name": _resource_name("GENIE_SPACE", space_id),
            "genie_space": {"name": name, "space_id": space_id, "permission": "CAN_RUN"},
        }

    return ToolAccessPlan(
        app_resources=tuple(sorted(app_resources.values(), key=lambda resource: resource["name"])),
        uc_grants=tuple(
            sorted(
                uc_grants,
                key=lambda grant: (
                    _securable_type_value(grant.securable_type),
                    grant.full_name,
                    grant.privilege.value,
                ),
            )
        ),
        workspace_grants=tuple(
            WorkspaceGrant(path, workspace_grants[path]) for path in sorted(workspace_grants)
        ),
    )


def _effective_uc_privileges(client: Any, principal: str, grant: UcGrant) -> set[Privilege]:
    privileges: set[Privilege] = set()
    page_token: str | None = None
    try:
        while True:
            response = client.grants.get_effective(
                _securable_type_value(grant.securable_type),
                grant.full_name,
                max_results=0,
                principal=principal,
                **({"page_token": page_token} if page_token is not None else {}),
            )
            for assignment in response.privilege_assignments or ():
                if assignment.principal != principal:
                    continue
                privileges.update(
                    effective.privilege
                    for effective in assignment.privileges or ()
                    if effective.privilege is not None
                )
            page_token = response.next_page_token
            if not page_token:
                return privileges
    except DatabricksError as exc:
        raise AgentCliError(
            f"Could not read effective {grant.privilege.value} access on "
            f"{_securable_type_value(grant.securable_type)} {grant.full_name!r}."
        ) from exc


def _ensure_uc_grant(client: Any, principal: str, grant: UcGrant) -> None:
    """Add one missing UC privilege and require it to become effective."""
    if grant.privilege in _effective_uc_privileges(client, principal, grant):
        return
    try:
        client.grants.update(
            _securable_type_value(grant.securable_type),
            grant.full_name,
            changes=[PermissionsChange(principal=principal, add=[grant.privilege])],
        )
    except DatabricksError as exc:
        raise AgentCliError(
            f"Could not grant {grant.privilege.value} on "
            f"{_securable_type_value(grant.securable_type)} {grant.full_name!r} "
            "to the App service principal."
        ) from exc
    if grant.privilege not in _effective_uc_privileges(client, principal, grant):
        raise AgentCliError(
            f"{grant.privilege.value} on {_securable_type_value(grant.securable_type)} "
            f"{grant.full_name!r} did not become effective for the App service principal."
        )


def _has_workspace_permission(
    permissions: Any, principal: str, required: WorkspaceObjectPermissionLevel
) -> bool:
    required_strength = _WORKSPACE_PERMISSION_STRENGTH[required]
    for access_control in permissions.access_control_list or ():
        if access_control.service_principal_name != principal:
            continue
        for permission in access_control.all_permissions or ():
            if (
                _WORKSPACE_PERMISSION_STRENGTH.get(permission.permission_level, 0)
                >= required_strength
            ):
                return True
    return False


def _ensure_workspace_grant(client: Any, principal: str, grant: WorkspaceGrant) -> None:
    """Add one missing Workspace ACL entry and require it to become effective."""
    try:
        status = client.workspace.get_status(grant.path)
    except DatabricksError as exc:
        raise AgentCliError(f"Could not resolve Workspace object {grant.path!r}.") from exc
    object_type = _WORKSPACE_OBJECT_TYPES.get(status.object_type)
    if status.object_id is None or object_type is None:
        raise AgentCliError(
            f"Workspace object {grant.path!r} cannot be granted automatically.",
            hint="Sandbox Workspace scopes must resolve to a directory, file, or notebook.",
        )
    object_id = str(status.object_id)
    try:
        permissions = client.workspace.get_permissions(object_type, object_id)
    except DatabricksError as exc:
        raise AgentCliError(f"Could not read Workspace access for {grant.path!r}.") from exc
    if _has_workspace_permission(permissions, principal, grant.permission):
        return
    try:
        client.workspace.update_permissions(
            object_type,
            object_id,
            access_control_list=[
                WorkspaceObjectAccessControlRequest(
                    service_principal_name=principal,
                    permission_level=grant.permission,
                )
            ],
        )
        permissions = client.workspace.get_permissions(object_type, object_id)
    except DatabricksError as exc:
        raise AgentCliError(
            f"Could not grant {grant.permission.value} on Workspace object {grant.path!r} "
            "to the App service principal."
        ) from exc
    if not _has_workspace_permission(permissions, principal, grant.permission):
        raise AgentCliError(
            f"{grant.permission.value} on Workspace object {grant.path!r} did not become "
            "effective for the App service principal."
        )


def reconcile_tool_access(
    client: Any,
    app: str,
    principal: str | None,
    plan: ToolAccessPlan,
    profile: str | None,
) -> ToolAccessPlan:
    """Apply and verify direct tool access before a new source rollout."""
    if principal is None:
        raise AgentCliError(
            f"Could not resolve App {app!r}'s service principal for tool access grants."
        )
    for grant in plan.uc_grants:
        _ensure_uc_grant(client, principal, grant)
    for grant in plan.workspace_grants:
        _ensure_workspace_grant(client, principal, grant)
    resource_error = apply_tool_resources(app, plan.app_resources, profile)
    if resource_error is not None:
        raise AgentCliError(
            f"Could not attach the explicit tool resources required by App {app!r}.",
            hint=resource_error,
        )
    return plan
