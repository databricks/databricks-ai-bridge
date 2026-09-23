"""Request-user deploy contract and scoped Apps reconciliation tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service.apps import App

from databricks_mason.agent_project import AgentProject, Scope, ToolSpec
from databricks_mason.cli import deploy as deploy_mod
from databricks_mason.errors import AgentCliError


@pytest.fixture(autouse=True)
def _no_remote_provisioning(monkeypatch):
    monkeypatch.setattr(deploy_mod, "get_or_create_trace_experiment", lambda *args: None)
    monkeypatch.setattr(deploy_mod, "_USE_MANAGED_RUNTIME_STORE", True)
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda *args: "app-sp")
    runtime_store = Mock(
        return_value=SimpleNamespace(
            branch="projects/shared/branches/production",
            database_id="runtime-database",
            username="app-sp",
        )
    )
    monkeypatch.setattr(deploy_mod.managed_runtime_store, "get_or_create_backend", runtime_store)
    return runtime_store


def _project(root, *, auth="user", legacy=False):
    project = AgentProject.create(root, framework="langgraph", server="mason")
    project.add_tool(ToolSpec.mcp("search", service="system.ai.web_search", auth=auth))
    if legacy:
        project.add_tool(ToolSpec.mcp("legacy", service="system.ai.python_exec"))
    project.write()
    return project


def test_user_auth_requires_explicit_auth_on_every_managed_binding(tmp_path, monkeypatch):
    _project(tmp_path, legacy=True, auth="user")
    client = Mock()
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path)],
        obj=SimpleNamespace(profile="selected", output="text", client=client),
    )
    assert result.exit_code != 0
    assert "explicit auth" in result.output
    client.assert_not_called()


def test_invalid_auth_fails_before_cloud_or_stores(tmp_path, monkeypatch):
    project = _project(tmp_path)
    project.path.write_text(project.path.read_text().replace('auth = "user"', 'auth = "invalid"'))
    client = Mock()
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path)],
        obj=SimpleNamespace(profile="selected", output="text", client=client),
    )
    assert result.exit_code != 0
    assert "auth" in result.output
    client.assert_not_called()


def _sdk(monkeypatch, existing=None):
    from databricks_mason.cli import app_auth

    apps = Mock()
    apps.get.return_value = existing
    if existing is None:
        apps.get.side_effect = NotFound("absent")
    workspace = Mock(return_value=SimpleNamespace(apps=apps))
    monkeypatch.setattr(app_auth, "WorkspaceClient", workspace)
    return app_auth, apps, workspace


@pytest.mark.parametrize(
    "service,auth,expected",
    [
        ("system.ai.web_search", "user", {"ai-gateway"}),
        ("system.ai.dbsql", "user", {"ai-gateway", "sql"}),
        ("system.ai.dbsql", "app", set()),
        ("system.ai.dbsql", None, set()),
    ],
)
def test_required_user_api_scopes_follow_managed_auth(tmp_path, service, auth, expected):
    from databricks_mason.cli.app_auth import required_user_api_scopes

    project = AgentProject.create(tmp_path, framework="langgraph", server="mason")
    project.add_tool(ToolSpec.mcp("service", service=service, auth=auth))
    assert required_user_api_scopes(project) == expected
    assert required_user_api_scopes(None) == set()


def test_scope_update_plan_uses_exact_required_scopes(monkeypatch):
    app_auth, _, _ = _sdk(monkeypatch)

    plan = app_auth.plan_app_user_scope_update(
        "app", "selected", allow_existing_app_update=False, required_scopes={"genie"}
    )

    assert plan.scopes == ("genie",)


@pytest.mark.parametrize(
    "binding,expected",
    [
        (
            ToolSpec.sandbox("volume", scopes=[Scope.volume("cat.sch.vol")], auth="user"),
            {"ai-gateway", "files"},
        ),
        (
            ToolSpec.sandbox("table", scopes=[Scope.table("cat.sch.tbl")], auth="user"),
            {"ai-gateway"},
        ),
        (ToolSpec.sandbox("volume", scopes=[Scope.volume("cat.sch.vol")], auth="app"), set()),
    ],
)
def test_volume_downscope_requests_files_scope(tmp_path, binding, expected):
    from databricks_mason.cli.app_auth import required_user_api_scopes

    project = AgentProject.create(tmp_path, framework="langgraph", server="mason")
    project.add_tool(binding)
    assert required_user_api_scopes(project) == expected


def test_existing_app_requires_explicit_scope_update_permission(tmp_path, monkeypatch):
    _project(tmp_path)
    _, apps, workspace = _sdk(monkeypatch, App(name="agent-mason-test", user_api_scopes=["sql"]))
    client = Mock()
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path)],
        obj=SimpleNamespace(profile="selected", output="text", client=client),
    )
    assert result.exit_code != 0
    assert "--allow-user-scope-update" in result.output
    client.assert_not_called()
    apps.create.assert_not_called()
    apps.create_update.assert_not_called()
    workspace.assert_called_once_with(profile="selected")


def test_existing_scoped_app_does_not_require_repeated_scope_update_permission(monkeypatch):
    existing = App(
        name="app",
        user_api_scopes=["ai-gateway", "sql"],
        effective_user_api_scopes=["ai-gateway", "sql"],
    )
    app_auth, _, _ = _sdk(monkeypatch, existing)

    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=False)

    assert plan.existing_scopes == ("ai-gateway", "sql")
    assert plan.scopes == plan.existing_scopes


def test_sdk_read_permission_denied_is_not_treated_as_new_app(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch)
    apps.get.side_effect = PermissionDenied("denied")
    with pytest.raises(AgentCliError, match="read"):
        app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=False)
    apps.create.assert_not_called()


def test_existing_app_scope_update_preserves_unrelated_scopes_and_uses_narrow_mask(monkeypatch):
    existing = App(name="app", user_api_scopes=["sql"], effective_user_api_scopes=["sql"])
    app_auth, apps, _ = _sdk(monkeypatch, existing)
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    apps.get.side_effect = [
        existing,
        App(
            name="app",
            user_api_scopes=["ai-gateway", "sql"],
            effective_user_api_scopes=["ai-gateway", "sql"],
        ),
    ]
    app_auth.apply_app_user_scope_update(plan, instances=2)
    payload = apps.create_update.call_args.kwargs
    assert set(payload["update_mask"].split(",")) == {
        "user_api_scopes",
        "compute_min_instances",
        "compute_max_instances",
    }
    assert payload["app"].as_dict() == {
        "name": "app",
        "user_api_scopes": ["ai-gateway", "sql"],
        "compute_min_instances": 2,
        "compute_max_instances": 2,
    }
    apps.create.assert_not_called()


def test_changed_scopes_abort_instead_of_overwriting_another_owner(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["sql"]))
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    apps.get.return_value = App(name="app", user_api_scopes=["sql", "files"])
    with pytest.raises(AgentCliError, match="changed"):
        app_auth.apply_app_user_scope_update(plan)
    apps.create_update.assert_not_called()


def test_effective_scopes_poll_is_bounded_and_fails_closed(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["ai-gateway"]))
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    monkeypatch.setattr(app_auth.time, "sleep", lambda _: None)
    with pytest.raises(AgentCliError, match="effective"):
        app_auth.apply_app_user_scope_update(plan, attempts=3)
    assert apps.get.call_count <= 5


def test_empty_scope_removal_stops_before_sdk_drops_empty_list(monkeypatch):
    from dataclasses import replace

    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["ai-gateway"]))
    plan = replace(
        app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True),
        scopes=(),
    )
    with pytest.raises(AgentCliError, match="remov"):
        app_auth.apply_app_user_scope_update(plan)
    apps.create_update.assert_not_called()
    assert App(name="app", user_api_scopes=[]).as_dict() == {"name": "app"}


def test_nonempty_scope_removal_also_requires_manual_action(monkeypatch):
    from dataclasses import replace

    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["ai-gateway", "sql"]))
    plan = replace(
        app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True),
        scopes=("ai-gateway",),
    )
    with pytest.raises(AgentCliError, match="remov"):
        app_auth.apply_app_user_scope_update(plan)
    apps.create_update.assert_not_called()


def test_update_wait_failure_stops_before_effective_scope_success(monkeypatch):
    app_auth, apps, _ = _sdk(
        monkeypatch,
        App(name="app", user_api_scopes=["ai-gateway"], effective_user_api_scopes=["ai-gateway"]),
    )
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    apps.create_update.return_value.result.side_effect = PermissionDenied("update failed")
    with pytest.raises(AgentCliError, match="reconcile"):
        app_auth.apply_app_user_scope_update(plan, instances=2)


def test_unknown_configured_scopes_do_not_overwrite_effective_grants(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", effective_user_api_scopes=["sql"]))
    with pytest.raises(AgentCliError, match="configured scopes"):
        app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    apps.create_update.assert_not_called()


def test_scope_update_does_not_request_implicit_identity_defaults(monkeypatch):
    defaults = ["iam.access-control:read", "iam.current-user:read"]
    existing = App(name="app", effective_user_api_scopes=defaults)
    app_auth, apps, _ = _sdk(monkeypatch, existing)
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    assert plan.existing_scopes == ()
    assert plan.scopes == ("ai-gateway",)
    updated = App(
        name="app",
        user_api_scopes=list(plan.scopes),
        effective_user_api_scopes=[*plan.scopes, *defaults],
    )
    apps.get.side_effect = [existing, updated]
    app_auth.apply_app_user_scope_update(plan, attempts=1)
    assert apps.create_update.call_args.kwargs["app"].as_dict() == {
        "name": "app",
        "user_api_scopes": ["ai-gateway"],
    }


@pytest.mark.parametrize("extra", ["sql", "iam.access-control:write", "iam.unknown:read"])
def test_omitted_config_rejects_unexplained_effective_extras(monkeypatch, extra):
    app_auth, apps, _ = _sdk(
        monkeypatch, App(name="app", effective_user_api_scopes=["iam.current-user:read", extra])
    )
    with pytest.raises(AgentCliError, match="configured scopes"):
        app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    apps.create.assert_not_called()
    apps.create_update.assert_not_called()


@pytest.mark.parametrize("new_app", [True, False])
def test_effective_verification_allows_only_observed_identity_defaults(monkeypatch, new_app):
    configured = ["ai-gateway"]
    ready = App(
        name="app",
        user_api_scopes=configured,
        effective_user_api_scopes=[*configured, "iam.access-control:read", "iam.current-user:read"],
    )
    app_auth, apps, _ = _sdk(monkeypatch, None if new_app else ready)
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    apps.get.side_effect = None
    apps.get.return_value = ready
    app_auth.apply_app_user_scope_update(plan, attempts=1)
    if new_app:
        assert apps.create.call_args.args[0].as_dict() == {
            "name": "app",
            "user_api_scopes": configured,
            "forward_user_access_token": True,
        }
    else:
        apps.create_update.assert_not_called()


@pytest.mark.parametrize(
    "effective",
    [
        ["iam.access-control:read", "iam.current-user:read"],
        ["ai-gateway", "sql"],
        ["ai-gateway", "iam.current-user:read", "iam.current-user:write"],
    ],
)
def test_effective_verification_rejects_missing_required_or_unexplained_scopes(
    monkeypatch, effective
):
    app_auth, apps, _ = _sdk(
        monkeypatch,
        App(name="app", user_api_scopes=["ai-gateway"], effective_user_api_scopes=effective),
    )
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    with pytest.raises(AgentCliError, match="effective"):
        app_auth.apply_app_user_scope_update(plan, attempts=1)
    apps.create_update.assert_not_called()


def test_disabled_forwarding_fails_deploy_before_app_or_store_mutations(tmp_path, monkeypatch):
    project = _project(tmp_path)
    before = project.path.read_text()
    _, apps, _ = _sdk(
        monkeypatch,
        App(
            name="agent-mason-test",
            user_api_scopes=["ai-gateway"],
            effective_user_api_scopes=["ai-gateway"],
            forward_user_access_token=False,
        ),
    )
    client = Mock()
    cloud = Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cloud)
    monkeypatch.setattr(
        deploy_mod,
        "_wait_for_running",
        Mock(side_effect=AssertionError("unexpected compute lifecycle")),
    )
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path), "--allow-user-scope-update"],
        obj=SimpleNamespace(profile="selected", output="text", client=client),
    )
    assert result.exit_code != 0
    assert "forward_user_access_token" in result.output
    assert "stop" in result.output.lower() and "start" in result.output.lower()
    client.assert_not_called()
    cloud.assert_not_called()
    apps.create.assert_not_called()
    apps.create_update.assert_not_called()
    assert project.path.read_text() == before


def test_forwarding_disabled_after_preflight_stops_before_update(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["sql"]))
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    apps.get.return_value = App(
        name="app", user_api_scopes=["sql"], forward_user_access_token=False
    )
    with pytest.raises(AgentCliError, match="forward_user_access_token"):
        app_auth.apply_app_user_scope_update(plan, attempts=1)
    apps.create_update.assert_not_called()


def test_unexplained_effective_grants_after_preflight_stop_before_update(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app"))
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=True)
    apps.get.return_value = App(name="app", effective_user_api_scopes=["sql"])
    with pytest.raises(AgentCliError, match="configured scopes"):
        app_auth.apply_app_user_scope_update(plan, attempts=1)
    apps.create_update.assert_not_called()


def test_disabled_forwarding_during_verification_stops_deploy(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch)
    plan = app_auth.plan_app_user_scope_update("app", "selected", allow_existing_app_update=False)
    apps.get.side_effect = None
    apps.get.return_value = App(
        name="app",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=False,
    )
    with pytest.raises(AgentCliError, match="forward_user_access_token"):
        app_auth.apply_app_user_scope_update(plan, attempts=1)


@pytest.mark.parametrize("auth", [None, "app"])
def test_app_only_tools_keep_deployment_path(tmp_path, monkeypatch, auth, _no_remote_provisioning):
    _project(tmp_path, auth=auth)
    app_auth, apps, workspace = _sdk(monkeypatch)
    calls = []
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda *args: False)
    monkeypatch.setattr(deploy_mod, "_wait_for_running", lambda *args: None)

    def databricks(arguments, profile, **kwargs):
        calls.append(arguments)
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(deploy_mod, "_databricks", databricks)
    client = SimpleNamespace(host="https://workspace", current_user="user@example.com")
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path)],
        obj=SimpleNamespace(profile="selected", output="text", client=lambda: client),
    )
    assert result.exit_code == 0, result.output
    workspace.assert_not_called()
    _no_remote_provisioning.assert_called_once()
    assert ["apps", "create", "agent-mason-test"] in calls


def test_user_deploy_creates_scoped_app_and_runtime_store_before_source(
    tmp_path, monkeypatch, _no_remote_provisioning
):
    _project(tmp_path)
    app_auth, apps, workspace = _sdk(monkeypatch)
    calls = []

    def create(app):
        calls.append(("sdk-create", app.as_dict()))
        apps.get.side_effect = None
        apps.get.return_value = App(
            name=app.name,
            user_api_scopes=app.user_api_scopes,
            effective_user_api_scopes=app.user_api_scopes,
        )

    apps.create.side_effect = create

    def databricks(arguments, profile, **kwargs):
        calls.append((arguments[0], arguments))
        assert profile == "selected"
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(deploy_mod, "_databricks", databricks)
    monkeypatch.setattr(deploy_mod, "_wait_for_running", lambda *args: None)
    monkeypatch.setattr(deploy_mod, "get_or_create_trace_experiment", lambda *args: None)
    runtime_backend = _no_remote_provisioning.return_value
    _no_remote_provisioning.side_effect = lambda *args: (
        calls.append(("runtime-store", args)),
        runtime_backend,
    )[1]
    client = SimpleNamespace(host="https://workspace", current_user="user@example.com")
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path), "--instances", "2"],
        obj=SimpleNamespace(profile="selected", output="text", client=lambda: client),
    )
    assert result.exit_code == 0, result.output
    workspace.assert_called_once_with(profile="selected")
    assert calls[0] == (
        "sdk-create",
        {
            "name": "agent-mason-test",
            "user_api_scopes": ["ai-gateway"],
            "forward_user_access_token": True,
            "compute_min_instances": 2,
            "compute_max_instances": 2,
        },
    )
    assert any(kind == "sync" for kind, _ in calls)
    assert any(arguments[:2] == ["apps", "deploy"] for _, arguments in calls[1:])
    _no_remote_provisioning.assert_called_once()
    assert [kind for kind, _ in calls].index("runtime-store") < [kind for kind, _ in calls].index(
        "sync"
    )
    assert "re-consent" in result.output
