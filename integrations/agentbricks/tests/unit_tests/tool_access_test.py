from __future__ import annotations

import re
from unittest.mock import Mock

import pytest
from databricks.sdk.errors import PermissionDenied
from databricks.sdk.service.catalog import (
    EffectivePermissionsList,
    EffectivePrivilege,
    EffectivePrivilegeAssignment,
    Privilege,
    PrivilegeAssignment,
    SecurableType,
    UpdatePermissionsResponse,
)
from databricks.sdk.service.workspace import (
    ObjectInfo,
    ObjectType,
    WorkspaceObjectAccessControlResponse,
    WorkspaceObjectPermission,
    WorkspaceObjectPermissionLevel,
    WorkspaceObjectPermissions,
)

from databricks_agentbricks import tool_access as ta
from databricks_agentbricks.agent_project import Scope, ToolSpec
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.tool_access import (
    UcGrant,
    WorkspaceGrant,
    _ensure_uc_grant,
    _ensure_workspace_grant,
    plan_tool_access,
    reconcile_tool_access,
)


def test_plan_maps_only_explicit_app_auth_resources_to_least_privilege():
    plan = plan_tool_access(
        [
            ToolSpec.uc_function("search", function="supervisor_agent.tools.search"),
            ToolSpec.genie_agent("genie", space_id="0" * 32, auth="app"),
            ToolSpec.sandbox(
                "sandbox",
                scopes=[
                    Scope.table("main.data.rows"),
                    Scope.volume("main.data.files"),
                    Scope.workspace("/Workspace/Shared/input", "read_write"),
                ],
            ),
            ToolSpec.mcp("search", service="supervisor_agent.tools.search"),
            ToolSpec.mcp("user-search", service="system.ai.docs", auth="user"),
        ]
    )

    uc_resources = {
        (
            resource["uc_securable"]["securable_full_name"],
            resource["uc_securable"]["securable_type"],
            resource["uc_securable"]["permission"],
        )
        for resource in plan.app_resources
        if "uc_securable" in resource
    }
    assert uc_resources == {
        ("supervisor_agent.tools.search", "FUNCTION", "EXECUTE"),
        ("main.data.rows", "TABLE", "SELECT"),
        ("main.data.files", "VOLUME", "READ_VOLUME"),
    }
    assert [
        resource["genie_space"] for resource in plan.app_resources if "genie_space" in resource
    ] == [{"name": "genie", "space_id": "0" * 32, "permission": "CAN_RUN"}]
    assert set(plan.uc_grants) == {
        UcGrant(SecurableType.CATALOG, "supervisor_agent", Privilege.USE_CATALOG),
        UcGrant(SecurableType.SCHEMA, "supervisor_agent.tools", Privilege.USE_SCHEMA),
        UcGrant("MCP_SERVICE", "supervisor_agent.tools.search", Privilege.EXECUTE),
    }
    assert plan.workspace_grants == (
        WorkspaceGrant(
            path="/Workspace/Shared/input",
            permission=WorkspaceObjectPermissionLevel.CAN_EDIT,
        ),
    )
    assert all(
        re.fullmatch(r"agentbricks-tool-[0-9a-f]{13}", r["name"]) for r in plan.app_resources
    )


def test_plan_deduplicates_targets_and_keeps_strongest_permission():
    tools = [
        ToolSpec.sandbox(
            "read",
            scopes=[
                Scope.table("main.data.rows"),
                Scope.volume("main.data.files"),
                Scope.workspace("/Workspace/Shared/input"),
            ],
        ),
        ToolSpec.sandbox(
            "write",
            scopes=[
                Scope.table("main.data.rows", "read_write"),
                Scope.volume("main.data.files", "read_write"),
                Scope.workspace("/Workspace/Shared/input", "read_write"),
            ],
        ),
    ]

    first = plan_tool_access(tools)
    second = plan_tool_access(list(reversed(tools)))

    assert first == second
    assert {
        resource["uc_securable"]["permission"]
        for resource in first.app_resources
        if "uc_securable" in resource
    } == {"MODIFY", "WRITE_VOLUME"}
    assert len(first.app_resources) == 2
    assert first.workspace_grants == (
        WorkspaceGrant(
            path="/Workspace/Shared/input",
            permission=WorkspaceObjectPermissionLevel.CAN_EDIT,
        ),
    )
    assert first.uc_grants == ()


def test_plan_ignores_user_auth_and_genie_one_without_an_explicit_resource():
    plan = plan_tool_access(
        [
            ToolSpec.mcp("search", service="system.ai.web_search", auth="user"),
            ToolSpec.sandbox(
                "sandbox",
                scopes=[Scope.table("main.data.rows")],
                auth="user",
            ),
            ToolSpec.genie_agent("genie", space_id="0" * 32, auth="user"),
            ToolSpec.genie_one("workspace_genie", auth="app"),
        ]
    )

    assert plan.app_resources == ()
    assert plan.uc_grants == ()
    assert plan.workspace_grants == ()


def test_plan_uses_platform_defaults_for_system_mcp_services():
    plan = plan_tool_access(
        [
            ToolSpec.mcp("search", service="system.ai.web_search"),
            ToolSpec.sandbox(
                "sandbox",
                scopes=[Scope.table("supervisor_agent.tools.rows")],
            ),
        ]
    )

    assert plan.uc_grants == ()


def test_plan_grants_external_mcp_service_in_supervisor_agent_catalog():
    plan = plan_tool_access([ToolSpec.mcp("search", service="supervisor_agent.tools.search")])

    assert set(plan.uc_grants) == {
        UcGrant(SecurableType.CATALOG, "supervisor_agent", Privilege.USE_CATALOG),
        UcGrant(SecurableType.SCHEMA, "supervisor_agent.tools", Privilege.USE_SCHEMA),
        UcGrant("MCP_SERVICE", "supervisor_agent.tools.search", Privilege.EXECUTE),
    }


def _effective(principal: str, privilege: Privilege) -> EffectivePermissionsList:
    return EffectivePermissionsList(
        privilege_assignments=[
            EffectivePrivilegeAssignment(
                principal=principal,
                privileges=[
                    EffectivePrivilege(
                        inherited_from_name="system.ai",
                        inherited_from_type=SecurableType.SCHEMA,
                        privilege=privilege,
                    )
                ],
            )
        ]
    )


def test_uc_grant_accepts_existing_effective_inherited_privilege():
    client = Mock()
    client.grants.get_effective.return_value = _effective("app-sp", Privilege.EXECUTE)
    grant = UcGrant("MCP_SERVICE", "system.ai.web_search", Privilege.EXECUTE)

    _ensure_uc_grant(client, "app-sp", grant)

    client.grants.update.assert_not_called()


def test_uc_grant_accepts_string_securable_type_for_minimum_sdk():
    client = Mock()
    client.grants.get_effective.return_value = _effective("app-sp", Privilege.EXECUTE)
    grant = UcGrant("MCP_SERVICE", "system.ai.web_search", Privilege.EXECUTE)

    _ensure_uc_grant(client, "app-sp", grant)

    client.grants.get_effective.assert_called_once_with(
        "MCP_SERVICE",
        "system.ai.web_search",
        max_results=0,
        principal="app-sp",
    )
    client.grants.update.assert_not_called()


def test_uc_grant_adds_missing_privilege_and_verifies_effective_access():
    client = Mock()
    client.grants.get_effective.side_effect = [
        EffectivePermissionsList(privilege_assignments=[]),
        _effective("app-sp", Privilege.USE_SCHEMA),
    ]
    grant = UcGrant(SecurableType.SCHEMA, "system.ai", Privilege.USE_SCHEMA)

    _ensure_uc_grant(client, "app-sp", grant)

    change = client.grants.update.call_args.kwargs["changes"][0]
    assert change.as_dict() == {"add": ["USE_SCHEMA"], "principal": "app-sp"}
    assert client.grants.update.call_args.args == ("SCHEMA", "system.ai")
    assert client.grants.get_effective.call_count == 2


def test_uc_grant_uses_update_response_when_effective_access_is_unreadable():
    client = Mock()
    client.grants.get_effective.side_effect = PermissionDenied(
        "User does not have READ METADATA on Catalog 'system'."
    )
    client.grants.update.return_value = UpdatePermissionsResponse(
        privilege_assignments=[
            PrivilegeAssignment(principal="app-sp", privileges=[Privilege.USE_CATALOG])
        ]
    )
    grant = UcGrant(SecurableType.CATALOG, "system", Privilege.USE_CATALOG)

    _ensure_uc_grant(client, "app-sp", grant)

    change = client.grants.update.call_args.kwargs["changes"][0]
    assert change.as_dict() == {"add": ["USE_CATALOG"], "principal": "app-sp"}


def test_uc_grant_fails_closed_when_update_response_omits_unreadable_grant():
    client = Mock()
    client.grants.get_effective.side_effect = PermissionDenied(
        "User does not have READ METADATA on Catalog 'system'."
    )
    client.grants.update.return_value = UpdatePermissionsResponse(privilege_assignments=[])
    grant = UcGrant(SecurableType.CATALOG, "system", Privilege.USE_CATALOG)

    with pytest.raises(AgentCliError, match="did not confirm"):
        _ensure_uc_grant(client, "app-sp", grant)


def test_uc_grant_follows_empty_effective_permission_pages():
    client = Mock()
    client.grants.get_effective.side_effect = [
        EffectivePermissionsList(privilege_assignments=[], next_page_token="page-2"),
        _effective("app-sp", Privilege.EXECUTE),
    ]
    grant = UcGrant("MCP_SERVICE", "system.ai.web_search", Privilege.EXECUTE)

    _ensure_uc_grant(client, "app-sp", grant)

    assert client.grants.get_effective.call_args_list[1].kwargs["page_token"] == "page-2"
    client.grants.update.assert_not_called()


def test_uc_grant_fails_when_update_does_not_become_effective():
    client = Mock()
    client.grants.get_effective.return_value = EffectivePermissionsList(privilege_assignments=[])
    grant = UcGrant("MCP_SERVICE", "system.ai.web_search", Privilege.EXECUTE)

    with pytest.raises(AgentCliError, match="did not become effective"):
        _ensure_uc_grant(client, "app-sp", grant)


def test_uc_grant_wraps_update_failure_with_target():
    client = Mock()
    client.grants.get_effective.return_value = EffectivePermissionsList(privilege_assignments=[])
    client.grants.update.side_effect = PermissionDenied("denied")
    grant = UcGrant("MCP_SERVICE", "system.ai.web_search", Privilege.EXECUTE)

    with pytest.raises(AgentCliError, match="system.ai.web_search"):
        _ensure_uc_grant(client, "app-sp", grant)


def _workspace_permissions(
    principal: str, permission: WorkspaceObjectPermissionLevel
) -> WorkspaceObjectPermissions:
    return WorkspaceObjectPermissions(
        access_control_list=[
            WorkspaceObjectAccessControlResponse(
                service_principal_name=principal,
                all_permissions=[
                    WorkspaceObjectPermission(inherited=True, permission_level=permission)
                ],
            )
        ],
        object_id="123",
        object_type="directories",
    )


@pytest.mark.parametrize(
    ("object_type", "api_type"),
    [
        (ObjectType.DIRECTORY, "directories"),
        (ObjectType.FILE, "files"),
        (ObjectType.NOTEBOOK, "notebooks"),
    ],
)
def test_workspace_grant_resolves_object_type_updates_and_verifies(object_type, api_type):
    client = Mock()
    client.workspace.get_status.return_value = ObjectInfo(object_id=123, object_type=object_type)
    client.workspace.get_permissions.side_effect = [
        WorkspaceObjectPermissions(access_control_list=[]),
        _workspace_permissions("app-sp", WorkspaceObjectPermissionLevel.CAN_EDIT),
    ]
    grant = WorkspaceGrant("/Workspace/Shared/input", WorkspaceObjectPermissionLevel.CAN_EDIT)

    _ensure_workspace_grant(client, "app-sp", grant)

    assert client.workspace.update_permissions.call_args.args == (api_type, "123")
    request = client.workspace.update_permissions.call_args.kwargs["access_control_list"][0]
    assert request.as_dict() == {
        "permission_level": "CAN_EDIT",
        "service_principal_name": "app-sp",
    }


def test_workspace_grant_accepts_stronger_effective_permission():
    client = Mock()
    client.workspace.get_status.return_value = ObjectInfo(
        object_id=123, object_type=ObjectType.DIRECTORY
    )
    client.workspace.get_permissions.return_value = _workspace_permissions(
        "app-sp", WorkspaceObjectPermissionLevel.CAN_MANAGE
    )
    grant = WorkspaceGrant("/Workspace/Shared/input", WorkspaceObjectPermissionLevel.CAN_READ)

    _ensure_workspace_grant(client, "app-sp", grant)

    client.workspace.update_permissions.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        ObjectInfo(object_type=ObjectType.DIRECTORY),
        ObjectInfo(object_id=123, object_type=ObjectType.REPO),
    ],
)
def test_workspace_grant_rejects_unmanageable_workspace_object(status):
    client = Mock()
    client.workspace.get_status.return_value = status
    grant = WorkspaceGrant("/Workspace/Shared/input", WorkspaceObjectPermissionLevel.CAN_READ)

    with pytest.raises(AgentCliError, match="Workspace object"):
        _ensure_workspace_grant(client, "app-sp", grant)

    client.workspace.update_permissions.assert_not_called()


def test_workspace_grant_fails_when_acl_does_not_become_effective():
    client = Mock()
    client.workspace.get_status.return_value = ObjectInfo(
        object_id=123, object_type=ObjectType.DIRECTORY
    )
    client.workspace.get_permissions.return_value = WorkspaceObjectPermissions(
        access_control_list=[]
    )
    grant = WorkspaceGrant("/Workspace/Shared/input", WorkspaceObjectPermissionLevel.CAN_EDIT)

    with pytest.raises(AgentCliError, match="did not become effective"):
        _ensure_workspace_grant(client, "app-sp", grant)


def test_reconcile_tool_access_requires_principal_before_direct_mutation(monkeypatch):
    applied = Mock(return_value=None)
    monkeypatch.setattr(ta, "apply_tool_resources", applied)
    plan = plan_tool_access(
        [ToolSpec.uc_function("search", function="supervisor_agent.tools.search")]
    )

    with pytest.raises(AgentCliError, match="service principal"):
        reconcile_tool_access(Mock(), "app", None, plan, "prof")

    applied.assert_not_called()


def test_reconcile_tool_access_applies_apps_uc_and_workspace_in_order(monkeypatch):
    events = []
    monkeypatch.setattr(
        ta,
        "apply_tool_resources",
        lambda app, resources, profile: events.append(("apps", app, resources, profile)),
    )
    monkeypatch.setattr(
        ta,
        "_ensure_uc_grant",
        lambda client, principal, grant: events.append(("uc", principal, grant)),
    )
    monkeypatch.setattr(
        ta,
        "_ensure_workspace_grant",
        lambda client, principal, grant: events.append(("workspace", principal, grant)),
    )
    plan = plan_tool_access(
        [
            ToolSpec.mcp("search", service="supervisor_agent.tools.search"),
            ToolSpec.sandbox(
                "sandbox",
                scopes=[Scope.workspace("/Workspace/Shared/input")],
            ),
        ]
    )

    assert reconcile_tool_access(Mock(), "app", "app-sp", plan, "prof") == plan

    assert [event[0] for event in events[:-1]] == [
        *("uc" for _ in plan.uc_grants),
        "workspace",
    ]
    assert events[-1] == ("apps", "app", plan.app_resources, "prof")


@pytest.mark.parametrize("failing_step", ["uc", "workspace"])
def test_reconcile_tool_access_additive_failure_does_not_replace_apps_resources(
    monkeypatch, failing_step
):
    applied = Mock(return_value=None)
    monkeypatch.setattr(ta, "apply_tool_resources", applied)

    def ensure_uc(*args):
        if failing_step == "uc":
            raise AgentCliError("UC grant failed")

    def ensure_workspace(*args):
        if failing_step == "workspace":
            raise AgentCliError("Workspace grant failed")

    monkeypatch.setattr(ta, "_ensure_uc_grant", ensure_uc)
    monkeypatch.setattr(ta, "_ensure_workspace_grant", ensure_workspace)
    plan = plan_tool_access(
        [
            ToolSpec.mcp("search", service="supervisor_agent.tools.search"),
            ToolSpec.sandbox(
                "sandbox",
                scopes=[Scope.workspace("/Workspace/Shared/input")],
            ),
        ]
    )

    with pytest.raises(AgentCliError, match="grant failed"):
        reconcile_tool_access(Mock(), "app", "app-sp", plan, "prof")

    applied.assert_not_called()


def test_reconcile_tool_access_surfaces_apps_resource_failure(monkeypatch):
    monkeypatch.setattr(ta, "apply_tool_resources", lambda *args: "denied: needs MANAGE")

    with pytest.raises(AgentCliError, match="explicit tool resources") as error:
        reconcile_tool_access(Mock(), "app", "app-sp", ta.ToolAccessPlan(), "prof")

    assert error.value.hint == "denied: needs MANAGE"
