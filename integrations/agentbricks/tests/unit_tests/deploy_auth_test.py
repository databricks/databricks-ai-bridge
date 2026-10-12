"""Request-user deployment contracts at the current Apps-auth client boundary.

User-scope planning moved out of the deploy command into ``AppsUserAuthClient`` and App creation/
updates belong to ``AppProvisioner``. These tests keep the old behavior assertions while exercising
those owners directly; CLI orchestration and rollout ordering are covered by deployment-service tests.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service.apps import App

from databricks_agentbricks.clients import apps_user_auth_client as auth_mod
from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.apps_user_auth_client import (
    AppsUserAuthClient,
    AppUserScopeUpdatePlan,
    plan_app_user_scope_update,
    required_user_api_scopes,
    requires_user_auth,
    validate_app_user_scope_drift,
    wait_for_app_user_scopes,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject, Scope, ToolSpec
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deploy_service import DeployRequest, DeployService
from databricks_agentbricks.services.deployment.provisioners import (
    AppProvisioner,
    GrantOutcome,
    ManifestPatch,
    MemoryStoreProvisioner,
    MemoryStoreState,
    RuntimeStoreProvisioner,
    RuntimeStoreState,
    SessionStoreProvisioner,
    SessionStoreState,
    TracingProvisioner,
    TracingState,
)
from databricks_agentbricks.services.deployment.tool_access import ToolAccessPlan
from databricks_agentbricks.services.deployment.tool_access_provisioner import ToolAccessProvisioner


def _project(root, *, auth="user", legacy=False):
    project = AgentProject.create(root, framework="langgraph", server="agentbricks")
    project.add_tool(ToolSpec.mcp("search", service="system.ai.web_search", auth=auth))
    if legacy:
        project.add_tool(ToolSpec.mcp("legacy", service="system.ai.python_exec"))
    project.write()
    return project


def _project_with_user_auth(root, *, server="agentbricks", scopes=("sql",)):
    project = AgentProject.create(root, framework="langgraph", server=server)
    project.write()
    rendered_scopes = ", ".join(f'"{scope}"' for scope in scopes)
    with project.path.open("a", encoding="utf-8") as manifest:
        manifest.write(
            f"\n[auth.user]\nrequired = true\nadditional_api_scopes = [{rendered_scopes}]\n"
        )
    return AgentProject.load(root)


def _sdk(monkeypatch, existing=None):
    apps = Mock()
    if existing is None:
        apps.get.side_effect = NotFound("absent")
    else:
        apps.get.return_value = existing
    workspace = Mock(return_value=SimpleNamespace(apps=apps))
    monkeypatch.setattr(auth_mod, "WorkspaceClient", workspace)
    return apps, workspace


def _real_auth_deploy_harness(tmp_path, sdk_apps):
    """Compose a real auth-aware DeployService while faking non-auth workspace operations."""
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.add_tool(ToolSpec.mcp("search", service="system.ai.web_search", auth="user"))
    project.write()

    resolver = Mock(spec=ProjectResolver)
    resolver.load.return_value = project
    resolver.resolve_deployment_name.return_value = "demo"
    resolver.resource_bindings.return_value = (None, None, None)

    cli_calls = []

    def run_databricks(args, profile, **kwargs):
        cli_calls.append((args, profile, kwargs))
        if args[:2] == ["apps", "get"]:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {"url": "https://demo.example", "compute_status": {"state": "ACTIVE"}}
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    apps_client = AppsClient("selected", runner=run_databricks)
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = SimpleNamespace(
        host="https://workspace.example",
        current_user="james@example.com",
        workspace_client=object(),
    )
    reporter = Mock(spec=Reporter)
    reporter.progress.return_value = nullcontext()
    reporter.status.return_value = nullcontext()
    app_provisioner = AppProvisioner(
        apps_client,
        provider,
        AppsUserAuthClient("selected"),
        reporter,
    )

    memory = Mock(spec=MemoryStoreProvisioner)
    memory.reconcile.return_value = MemoryStoreState(None, ManifestPatch({}))
    memory.grant.return_value = GrantOutcome.skipped()
    session = Mock(spec=SessionStoreProvisioner)
    session.reconcile.return_value = SessionStoreState(None, ManifestPatch({}))
    session.grant.return_value = GrantOutcome.skipped()
    tracing = Mock(spec=TracingProvisioner)
    tracing.reconcile.return_value = TracingState(None, (), None, ManifestPatch({}))
    tracing.grant.return_value = GrantOutcome.skipped()
    runtime = Mock(spec=RuntimeStoreProvisioner)
    runtime.reconcile.return_value = RuntimeStoreState(False, None, ManifestPatch({}))
    runtime.after_app_ready.return_value = ManifestPatch({})
    tool_access = Mock(spec=ToolAccessProvisioner)
    tool_access.plan.return_value = ToolAccessPlan()

    service = DeployService(
        project_resolver=resolver,
        apps_client=apps_client,
        api_client_provider=provider,
        app_provisioner=app_provisioner,
        memory_store_provisioner=memory,
        session_store_provisioner=session,
        tracing_provisioner=tracing,
        runtime_store_provisioner=runtime,
        tool_access_provisioner=tool_access,
        reporter=reporter,
    )
    return SimpleNamespace(
        project=project,
        sdk_apps=sdk_apps,
        cli_calls=cli_calls,
        provider=provider,
        app_provisioner=app_provisioner,
        tool_access=tool_access,
        service=service,
    )


def _auth_request(root, *, allow_user_scope_update=False):
    return DeployRequest(
        name="demo",
        source=str(root),
        pip_index_url=None,
        workspace_path="/Workspace/demo",
        instance_count=None,
        allow_user_scope_update=allow_user_scope_update,
    )


def test_declarative_user_auth_requires_no_managed_binding(tmp_path):
    project = _project_with_user_auth(tmp_path)

    assert requires_user_auth(project) is True
    assert required_user_api_scopes(project) == {"sql"}


def test_declarative_scopes_union_with_managed_inference(tmp_path):
    project = _project_with_user_auth(tmp_path, scopes=("sql", "ai-gateway", "sql"))
    project.add_tool(ToolSpec.mcp("search", service="system.ai.web_search", auth="user"))

    assert required_user_api_scopes(project) == {"sql", "ai-gateway"}


def test_declarative_user_auth_requires_agentbricks_server(tmp_path):
    project = _project_with_user_auth(tmp_path, server="custom")

    with pytest.raises(AgentCliError, match="server = 'agentbricks'"):
        requires_user_auth(project)


def test_user_auth_requires_explicit_auth_on_every_managed_binding(tmp_path):
    project = _project(tmp_path, legacy=True, auth="user")

    with pytest.raises(AgentCliError, match="explicit auth"):
        requires_user_auth(project)


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        (ToolSpec.mcp("search", service="system.ai.web_search", auth="user"), {"ai-gateway"}),
        (
            ToolSpec.mcp("dbsql", service="system.ai.dbsql", auth="user"),
            {"ai-gateway", "sql"},
        ),
        (ToolSpec.genie_one(auth="user"), {"genie"}),
        (ToolSpec.genie_agent("space", space_id="0" * 32, auth="user"), {"genie"}),
        (ToolSpec.mcp("dbsql", service="system.ai.dbsql", auth="app"), set()),
    ],
)
def test_required_user_api_scopes_follow_binding(tmp_path, binding, expected):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.add_tool(binding)

    assert required_user_api_scopes(project) == expected
    assert required_user_api_scopes(None) == set()


def test_scope_update_plan_uses_exact_required_scopes(monkeypatch):
    apps, _ = _sdk(monkeypatch)

    plan = plan_app_user_scope_update(
        "app", "selected", allow_existing_app_update=False, required_scopes={"genie"}
    )

    assert plan.apps is apps
    assert plan.existing_scopes is None
    assert plan.scopes == ("genie",)


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        (
            ToolSpec.sandbox("volume", scopes=[Scope.volume("cat.sch.vol")], auth="user"),
            {"ai-gateway", "files", "workspace.workspace"},
        ),
        (
            ToolSpec.sandbox(
                "volume-no-token",
                scopes=[Scope.volume("cat.sch.vol")],
                auth="user",
                databricks_access_token_included=False,
            ),
            {"ai-gateway", "files"},
        ),
        (ToolSpec.sandbox("volume-app", scopes=[Scope.volume("cat.sch.vol")], auth="app"), set()),
    ],
)
def test_sandbox_policy_requests_resource_and_token_scopes(tmp_path, binding, expected):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.add_tool(binding)

    assert required_user_api_scopes(project) == expected


def test_required_user_api_scopes_union_only_user_bindings(tmp_path):
    project = _project(tmp_path)
    project.add_tool(ToolSpec.mcp("genie", service="system.ai.genie_one_mcp", auth="user"))
    project.add_tool(ToolSpec.genie_one("first_class", auth="user"))

    assert required_user_api_scopes(project) == {"ai-gateway", "genie"}


@pytest.mark.parametrize(
    "binding",
    [
        ToolSpec.genie_one(auth="user"),
        ToolSpec.genie_agent("space", space_id="0" * 32, auth="user"),
    ],
)
def test_first_class_genie_requires_user_auth(tmp_path, binding):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.add_tool(binding)

    assert requires_user_auth(project) is True


def test_existing_app_requires_explicit_scope_update_permission(monkeypatch):
    _sdk(monkeypatch, App(name="app", user_api_scopes=["sql"]))

    with pytest.raises(AgentCliError, match="allow-user-scope-update"):
        plan_app_user_scope_update(
            "app", "selected", allow_existing_app_update=False, required_scopes={"ai-gateway"}
        )


def test_existing_scoped_app_does_not_require_repeated_scope_update_permission(monkeypatch):
    existing = App(
        name="app",
        user_api_scopes=["ai-gateway", "sql"],
        effective_user_api_scopes=["ai-gateway", "sql"],
    )
    _sdk(monkeypatch, existing)

    plan = plan_app_user_scope_update(
        "app", "selected", allow_existing_app_update=False, required_scopes={"ai-gateway", "sql"}
    )

    assert plan.existing_scopes == ("ai-gateway", "sql")
    assert plan.scopes == plan.existing_scopes


def test_sdk_read_permission_denied_is_not_treated_as_new_app(monkeypatch):
    apps, _ = _sdk(monkeypatch)
    apps.get.side_effect = PermissionDenied("denied")

    with pytest.raises(AgentCliError, match="read"):
        plan_app_user_scope_update("app", "selected", allow_existing_app_update=False)
    apps.create.assert_not_called()


def test_forwarding_disabled_fails_before_scope_planning(monkeypatch):
    existing = App(
        name="app",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=False,
    )
    _sdk(monkeypatch, existing)

    with pytest.raises(AgentCliError, match="forward_user_access_token"):
        plan_app_user_scope_update("app", "selected", allow_existing_app_update=False)


def test_unknown_configured_scopes_do_not_overwrite_effective_grants(monkeypatch):
    apps, _ = _sdk(monkeypatch, App(name="app", effective_user_api_scopes=["sql"]))

    with pytest.raises(AgentCliError, match="configured scopes"):
        plan_app_user_scope_update(
            "app", "selected", allow_existing_app_update=True, required_scopes={"ai-gateway"}
        )
    apps.create.assert_not_called()
    apps.create_update.assert_not_called()


def test_scope_update_does_not_request_implicit_identity_defaults(monkeypatch):
    defaults = ["iam.access-control:read", "iam.current-user:read"]
    existing = App(name="app", effective_user_api_scopes=defaults)
    _sdk(monkeypatch, existing)

    plan = plan_app_user_scope_update(
        "app", "selected", allow_existing_app_update=True, required_scopes={"ai-gateway"}
    )

    assert plan.existing_scopes == ()
    assert plan.scopes == ("ai-gateway",)


@pytest.mark.parametrize("extra", ["sql", "iam.access-control:write", "iam.unknown:read"])
def test_omitted_config_rejects_unexplained_effective_extras(monkeypatch, extra):
    _sdk(
        monkeypatch,
        App(name="app", effective_user_api_scopes=["iam.current-user:read", extra]),
    )

    with pytest.raises(AgentCliError, match="configured scopes"):
        plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)


def test_validate_scope_drift_fails_closed(monkeypatch):
    apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["sql"]))
    plan = plan_app_user_scope_update(
        "app", "selected", allow_existing_app_update=True, required_scopes={"sql"}
    )
    apps.get.return_value = App(name="app", user_api_scopes=["sql", "files"])

    with pytest.raises(AgentCliError, match="changed"):
        validate_app_user_scope_drift(plan)


def test_app_provisioner_scope_drift_stops_before_update():
    apps = Mock()
    apps.get.return_value = App(
        name="agent-bricks-app",
        user_api_scopes=["sql", "files"],
        effective_user_api_scopes=["sql", "files"],
        forward_user_access_token=True,
    )
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="agent-bricks-app",
        existing_scopes=("sql",),
        scopes=("ai-gateway", "sql"),
    )

    with pytest.raises(AgentCliError, match="changed"):
        AppProvisioner._reconcile_scoped_app(plan, instance_count=2)

    apps.create_update.assert_not_called()


@pytest.mark.parametrize(
    ("existing_scopes", "desired_scopes"),
    [
        (("ai-gateway",), ()),
        (("ai-gateway", "sql"), ("ai-gateway",)),
    ],
)
def test_app_provisioner_refuses_scope_removal(existing_scopes, desired_scopes):
    apps = Mock()
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="agent-bricks-app",
        existing_scopes=existing_scopes,
        scopes=desired_scopes,
    )

    with pytest.raises(AgentCliError, match="removal"):
        AppProvisioner._reconcile_scoped_app(plan, instance_count=None)

    apps.create.assert_not_called()
    apps.create_update.assert_not_called()


def test_app_provisioner_new_app_allows_credential_only_user_auth():
    apps = Mock()
    apps.get.return_value = App(
        name="agent-bricks-app",
        user_api_scopes=[],
        effective_user_api_scopes=["iam.access-control:read", "iam.current-user:read"],
        forward_user_access_token=True,
    )
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="agent-bricks-app",
        existing_scopes=None,
        scopes=(),
    )

    AppProvisioner._reconcile_scoped_app(plan, instance_count=None)

    assert apps.create.call_args.args[0].as_dict() == {
        "name": "agent-bricks-app",
        "forward_user_access_token": True,
    }


def test_wait_for_app_user_scopes_accepts_identity_defaults(monkeypatch):
    apps, _ = _sdk(monkeypatch)
    apps.get.side_effect = None
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="app",
        existing_scopes=("ai-gateway",),
        scopes=("ai-gateway",),
    )
    apps.get.return_value = App(
        name="app",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway", "iam.current-user:read"],
    )

    wait_for_app_user_scopes(plan, attempts=1)


def test_wait_for_app_user_scopes_fails_after_bounded_checks(monkeypatch):
    apps, _ = _sdk(monkeypatch)
    apps.get.side_effect = None
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="app",
        existing_scopes=("ai-gateway",),
        scopes=("ai-gateway", "genie"),
    )
    apps.get.return_value = App(
        name="app",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
    )
    monkeypatch.setattr(auth_mod.time, "sleep", lambda _: None)

    with pytest.raises(AgentCliError, match="did not converge"):
        wait_for_app_user_scopes(plan, attempts=3)
    assert apps.get.call_count == 3


@pytest.mark.parametrize(
    "effective_scopes",
    [
        ["ai-gateway"],
        ["ai-gateway", "genie", "sql"],
    ],
)
def test_wait_for_app_user_scopes_rejects_missing_or_unexplained_grants(effective_scopes):
    apps = Mock()
    apps.get.return_value = App(
        name="app",
        user_api_scopes=["ai-gateway", "genie"],
        effective_user_api_scopes=effective_scopes,
        forward_user_access_token=True,
    )
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="app",
        existing_scopes=("ai-gateway",),
        scopes=("ai-gateway", "genie"),
    )

    with pytest.raises(AgentCliError, match="did not converge"):
        wait_for_app_user_scopes(plan, attempts=1)


def test_wait_for_app_user_scopes_rejects_disabled_forwarding():
    apps = Mock()
    apps.get.return_value = App(
        name="app",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=False,
    )
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="app",
        existing_scopes=None,
        scopes=("ai-gateway",),
    )

    with pytest.raises(AgentCliError, match="forward_user_access_token"):
        wait_for_app_user_scopes(plan, attempts=1)


def test_app_provisioner_creates_new_scoped_app_and_verifies_effective_scopes():
    apps = Mock()
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="agent-bricks-app",
        existing_scopes=None,
        scopes=("ai-gateway", "sql"),
    )
    apps.get.return_value = App(
        name="agent-bricks-app",
        user_api_scopes=["ai-gateway", "sql"],
        effective_user_api_scopes=["ai-gateway", "sql"],
    )

    AppProvisioner._reconcile_scoped_app(plan, None)

    apps.create.assert_called_once()
    assert apps.create.call_args.args[0].as_dict() == {
        "name": "agent-bricks-app",
        "user_api_scopes": ["ai-gateway", "sql"],
        "forward_user_access_token": True,
    }


def test_app_provisioner_updates_existing_scopes_with_narrow_mask():
    apps = Mock()
    existing = App(name="agent-bricks-app", user_api_scopes=["sql"])
    ready = App(
        name="agent-bricks-app",
        user_api_scopes=["ai-gateway", "sql"],
        effective_user_api_scopes=["ai-gateway", "sql"],
    )
    apps.get.side_effect = [existing, ready]
    update_result = Mock()
    apps.create_update.return_value = update_result
    plan = AppUserScopeUpdatePlan(
        apps=apps,
        name="agent-bricks-app",
        existing_scopes=("sql",),
        scopes=("ai-gateway", "sql"),
    )

    AppProvisioner._reconcile_scoped_app(plan, 2)

    payload = apps.create_update.call_args.kwargs
    assert set(payload["update_mask"].split(",")) == {
        "user_api_scopes",
        "compute_min_instances",
        "compute_max_instances",
    }
    assert payload["app"].as_dict() == {
        "name": "agent-bricks-app",
        "user_api_scopes": ["ai-gateway", "sql"],
        "compute_min_instances": 2,
        "compute_max_instances": 2,
    }
    update_result.result.assert_called_once()


def test_app_auth_client_rejects_scope_update_flag_without_user_auth(tmp_path):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")

    with pytest.raises(AgentCliError, match="requires a managed tool"):
        AppsUserAuthClient("selected").plan_user_auth(
            "agent-bricks-app", project, allow_existing_app_update=True
        )


def test_declarative_user_auth_scope_update_checks_forwarding(monkeypatch, tmp_path):
    project = _project_with_user_auth(tmp_path, scopes=("sql",))
    _sdk(
        monkeypatch,
        App(
            name="agent-bricks-app",
            user_api_scopes=["sql"],
            effective_user_api_scopes=["sql"],
            forward_user_access_token=False,
        ),
    )

    with pytest.raises(AgentCliError, match="forward_user_access_token"):
        AppsUserAuthClient("selected").plan_user_auth(
            "agent-bricks-app", project, allow_existing_app_update=True
        )


def test_deploy_composes_real_auth_client_and_scoped_create_before_rollout(tmp_path, monkeypatch):
    sdk_apps, _ = _sdk(monkeypatch)
    ready = App(
        name="agent-bricks-demo",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=True,
    )
    sdk_apps.get.side_effect = [NotFound("absent"), ready]
    harness = _real_auth_deploy_harness(tmp_path, sdk_apps)

    result = harness.service.deploy(_auth_request(tmp_path))

    sdk_apps.create.assert_called_once()
    created = sdk_apps.create.call_args.args[0]
    assert created.as_dict() == {
        "name": "agent-bricks-demo",
        "user_api_scopes": ["ai-gateway"],
        "forward_user_access_token": True,
    }
    assert any(call[0][0] == "sync" for call in harness.cli_calls)
    assert any(call[0][:2] == ["apps", "deploy"] for call in harness.cli_calls)
    harness.tool_access.reconcile_before_rollout.assert_called_once()
    harness.tool_access.finalize_after_rollout.assert_called_once()
    assert result.url == "https://demo.example"


def test_deploy_existing_scope_gap_requires_flag_before_rollout(tmp_path, monkeypatch):
    existing = App(
        name="agent-bricks-demo",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=True,
    )
    sdk_apps, _ = _sdk(monkeypatch, existing)
    harness = _real_auth_deploy_harness(tmp_path, sdk_apps)
    harness.project.add_tool(ToolSpec.mcp("dbsql", service="system.ai.dbsql", auth="user"))
    harness.project.write()

    with pytest.raises(AgentCliError, match="allow-user-scope-update"):
        harness.service.deploy(_auth_request(tmp_path))

    harness.provider.get.assert_not_called()
    sdk_apps.create_update.assert_not_called()
    assert not any(call[0][0] == "sync" for call in harness.cli_calls)


def test_deploy_disabled_forwarding_fails_before_workspace_or_rollout(tmp_path, monkeypatch):
    existing = App(
        name="agent-bricks-demo",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=False,
    )
    sdk_apps, _ = _sdk(monkeypatch, existing)
    harness = _real_auth_deploy_harness(tmp_path, sdk_apps)

    with pytest.raises(AgentCliError, match="forward_user_access_token"):
        harness.service.deploy(_auth_request(tmp_path, allow_user_scope_update=True))

    harness.provider.get.assert_not_called()
    sdk_apps.create.assert_not_called()
    sdk_apps.create_update.assert_not_called()
    assert not any(call[0][0] == "sync" for call in harness.cli_calls)


def test_deploy_effective_scope_failure_stops_before_source_rollout(tmp_path, monkeypatch):
    existing = App(
        name="agent-bricks-demo",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=True,
    )
    not_converged = App(
        name="agent-bricks-demo",
        user_api_scopes=["ai-gateway", "sql"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=True,
    )
    sdk_apps, _ = _sdk(monkeypatch)
    responses = iter([existing, existing])

    def get_app(*args, **kwargs):
        return next(responses, not_converged)

    sdk_apps.get.side_effect = get_app
    update_result = Mock()
    sdk_apps.create_update.return_value = update_result
    harness = _real_auth_deploy_harness(tmp_path, sdk_apps)
    harness.project.add_tool(ToolSpec.mcp("dbsql", service="system.ai.dbsql", auth="user"))
    harness.project.write()
    monkeypatch.setattr(auth_mod.time, "sleep", lambda _: None)

    with pytest.raises(AgentCliError, match="did not converge"):
        harness.service.deploy(_auth_request(tmp_path, allow_user_scope_update=True))

    sdk_apps.create_update.assert_called_once()
    update_result.result.assert_called_once()
    harness.provider.get.assert_called_once()
    assert not any(call[0][0] == "sync" for call in harness.cli_calls)


def test_deploy_allow_user_scope_update_reaches_real_app_provisioner_and_rolls_out(
    tmp_path, monkeypatch
):
    existing = App(
        name="agent-bricks-demo",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=True,
    )
    converged = App(
        name="agent-bricks-demo",
        user_api_scopes=["ai-gateway", "sql"],
        effective_user_api_scopes=["ai-gateway", "sql"],
        forward_user_access_token=True,
    )
    sdk_apps, _ = _sdk(monkeypatch)
    sdk_apps.get.side_effect = [existing, existing, converged]
    update_result = Mock()
    sdk_apps.create_update.return_value = update_result
    harness = _real_auth_deploy_harness(tmp_path, sdk_apps)
    harness.project.add_tool(ToolSpec.mcp("dbsql", service="system.ai.dbsql", auth="user"))
    harness.project.write()

    result = harness.service.deploy(_auth_request(tmp_path, allow_user_scope_update=True))

    sdk_apps.create.assert_not_called()
    sdk_apps.create_update.assert_called_once()
    assert sdk_apps.create_update.call_args.args == ("agent-bricks-demo",)
    assert sdk_apps.create_update.call_args.kwargs["update_mask"] == ("user_api_scopes")
    updated = sdk_apps.create_update.call_args.kwargs["app"]
    assert updated.user_api_scopes == ["ai-gateway", "sql"]
    assert updated.forward_user_access_token is None
    update_result.result.assert_called_once()
    assert any(call[0][0] == "sync" for call in harness.cli_calls)
    assert any(call[0][:2] == ["apps", "deploy"] for call in harness.cli_calls)
    assert result.deployment == "agent-bricks-demo"
