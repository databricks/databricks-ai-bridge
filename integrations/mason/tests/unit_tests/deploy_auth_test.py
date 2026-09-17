"""Request-user deploy contract and scoped Apps reconciliation tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import NotFound, PermissionDenied
from databricks.sdk.service.apps import App

from databricks_mason.agent_project import AgentProject, ToolSpec
from databricks_mason.cli import deploy as deploy_mod
from databricks_mason.errors import AgentCliError
from databricks_mason.project_config import write_project_metadata


@pytest.fixture(autouse=True)
def _no_remote_provisioning(monkeypatch):
    monkeypatch.setattr(deploy_mod, "resolve_trace_experiment_id", lambda *args: None)
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


def _project(root, *, marker=1, auth="user", legacy=False):
    project = AgentProject.create(root, framework="langgraph", server="mason")
    project.add_tool(ToolSpec.mcp("search", service="system.ai.web_search", auth=auth))
    if legacy:
        project.add_tool(ToolSpec.mcp("legacy", service="system.ai.python_exec"))
    project.write()
    if marker != "missing":
        write_project_metadata(root, framework="langgraph", template="agent-langgraph")
        if marker is not None:
            with (root / ".mason/project.toml").open("a") as metadata:
                metadata.write(f"request_auth_contract_version = {marker}\n")
    return project


@pytest.mark.parametrize("marker", ["missing", None, 0, 2, "true", '"1"'])
def test_invalid_contract_fails_before_any_deploy_side_effect(tmp_path, monkeypatch, marker):
    project = _project(tmp_path, marker=marker)
    before = project.path.read_text()
    client = Mock()
    cloud = Mock()
    monkeypatch.setattr(deploy_mod, "_databricks", cloud)
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path)],
        obj=SimpleNamespace(profile="selected", output="text", client=client),
    )
    assert result.exit_code != 0
    assert "request_auth_contract_version" in result.output
    client.assert_not_called()
    cloud.assert_not_called()
    assert project.path.read_text() == before
    assert not (tmp_path / "app.yaml").exists()


def test_missing_contract_hint_describes_request_auth_recovery_boundary(tmp_path):
    _project(tmp_path, marker="missing")
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path)],
        obj=SimpleNamespace(profile="selected", output="text", client=Mock()),
    )

    assert result.exit_code != 0
    assert "Failure recovery is unsupported for request-user attempts" in result.output
    assert "background execution is unsupported" not in result.output


@pytest.mark.parametrize("auth", ["user", "app"])
def test_contract_requires_explicit_auth_on_every_managed_binding(tmp_path, monkeypatch, auth):
    _project(tmp_path, legacy=True, auth=auth)
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
    ("binding", "expected"),
    [
        (
            ToolSpec.mcp("genie", service="system.ai.genie_one_mcp", auth="user"),
            {"ai-gateway", "genie"},
        ),
        (ToolSpec.genie_one(auth="user"), {"genie"}),
        (ToolSpec.genie_agent("space", space_id="0" * 32, auth="user"), {"genie"}),
        (ToolSpec.mcp("genie", service="system.ai.genie_one_mcp", auth="app"), set()),
        (ToolSpec.mcp("genie", service="system.ai.genie_one_mcp"), set()),
    ],
)
def test_required_user_scopes_follow_binding(tmp_path, binding, expected):
    from databricks_mason.cli.app_auth import required_user_scopes

    project = AgentProject.create(tmp_path, framework="langgraph", server="mason")
    project.add_tool(binding)
    assert required_user_scopes(project) == expected
    assert required_user_scopes(None) == set()


def test_prepare_app_auth_uses_exact_required_scopes(monkeypatch):
    app_auth, _, _ = _sdk(monkeypatch)

    plan = app_auth.prepare_app_auth("app", "selected", adopt=False, required_scopes={"genie"})

    assert plan.scopes == ("genie",)


def test_required_user_scopes_union_only_user_bindings(tmp_path):
    from databricks_mason.cli.app_auth import required_user_scopes

    project = _project(tmp_path)
    project.add_tool(ToolSpec.mcp("genie", service="system.ai.genie_one_mcp", auth="user"))
    project.add_tool(ToolSpec.genie_one("first_class", auth="user"))
    assert required_user_scopes(project) == {"ai-gateway", "genie"}


@pytest.mark.parametrize(
    "binding",
    [
        ToolSpec.genie_one(auth="user"),
        ToolSpec.genie_agent("space", space_id="0" * 32, auth="user"),
    ],
)
def test_first_class_genie_requires_request_auth_contract(tmp_path, binding):
    from databricks_mason.cli.app_auth import requires_user_auth

    project = AgentProject.create(tmp_path, framework="langgraph", server="mason")
    project.add_tool(binding)
    with pytest.raises(AgentCliError, match="request_auth_contract_version"):
        requires_user_auth(project)


def test_service_scopes_preserve_existing_scopes_and_verify_effective(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["model-serving"]))
    plan = app_auth.prepare_app_auth(
        "app", "selected", adopt=True, required_scopes={"ai-gateway", "genie"}
    )
    assert plan.scopes == ("ai-gateway", "genie", "model-serving")
    ready = App(
        name="app", user_api_scopes=list(plan.scopes), effective_user_api_scopes=list(plan.scopes)
    )
    apps.get.side_effect = [App(name="app", user_api_scopes=["model-serving"]), ready]
    app_auth.apply_app_auth(plan, attempts=1)
    assert apps.create_update.call_args.kwargs["app"].user_api_scopes == list(plan.scopes)


def test_genie_scope_not_effective_stops_rollout(monkeypatch):
    required = {"ai-gateway", "genie"}
    app_auth, apps, _ = _sdk(monkeypatch)
    plan = app_auth.prepare_app_auth("app", "selected", adopt=False, required_scopes=required)
    apps.get.side_effect = None
    apps.get.return_value = App(
        name="app",
        user_api_scopes=sorted(required),
        effective_user_api_scopes=["ai-gateway"],
    )
    with pytest.raises(AgentCliError, match="did not converge"):
        app_auth.apply_app_auth(plan, attempts=1)


def test_existing_app_requires_explicit_adoption(tmp_path, monkeypatch):
    _project(tmp_path)
    _, apps, workspace = _sdk(monkeypatch, App(name="agent-mason-test", user_api_scopes=["sql"]))
    client = Mock()
    result = CliRunner().invoke(
        deploy_mod.deploy,
        ["test", "--source", str(tmp_path)],
        obj=SimpleNamespace(profile="selected", output="text", client=client),
    )
    assert result.exit_code != 0
    assert "--adopt-user-auth" in result.output
    client.assert_not_called()
    apps.create.assert_not_called()
    apps.create_update.assert_not_called()
    workspace.assert_called_once_with(profile="selected")


def test_existing_scoped_app_does_not_require_repeated_adoption(monkeypatch):
    existing = App(
        name="app",
        user_api_scopes=["ai-gateway", "sql"],
        effective_user_api_scopes=["ai-gateway", "sql"],
    )
    app_auth, _, _ = _sdk(monkeypatch, existing)

    plan = app_auth.prepare_app_auth("app", "selected", adopt=False)

    assert plan.existing_scopes == ("ai-gateway", "sql")
    assert plan.scopes == plan.existing_scopes


def test_sdk_read_permission_denied_is_not_treated_as_new_app(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch)
    apps.get.side_effect = PermissionDenied("denied")
    with pytest.raises(AgentCliError, match="read"):
        app_auth.prepare_app_auth("app", "selected", adopt=False)
    apps.create.assert_not_called()


def test_adopted_app_preserves_unrelated_scopes_and_uses_narrow_mask(monkeypatch):
    existing = App(name="app", user_api_scopes=["sql"], effective_user_api_scopes=["sql"])
    app_auth, apps, _ = _sdk(monkeypatch, existing)
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    apps.get.side_effect = [
        existing,
        App(
            name="app",
            user_api_scopes=["ai-gateway", "sql"],
            effective_user_api_scopes=["ai-gateway", "sql"],
        ),
    ]
    app_auth.apply_app_auth(plan, instances=2)
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
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    apps.get.return_value = App(name="app", user_api_scopes=["sql", "files"])
    with pytest.raises(AgentCliError, match="changed"):
        app_auth.apply_app_auth(plan)
    apps.create_update.assert_not_called()


def test_effective_scopes_poll_is_bounded_and_fails_closed(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["ai-gateway"]))
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    monkeypatch.setattr(app_auth.time, "sleep", lambda _: None)
    with pytest.raises(AgentCliError, match="effective"):
        app_auth.apply_app_auth(plan, attempts=3)
    assert apps.get.call_count <= 5


def test_empty_scope_removal_stops_before_sdk_drops_empty_list(monkeypatch):
    from dataclasses import replace

    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["ai-gateway"]))
    plan = replace(app_auth.prepare_app_auth("app", "selected", adopt=True), scopes=())
    with pytest.raises(AgentCliError, match="remov"):
        app_auth.apply_app_auth(plan)
    apps.create_update.assert_not_called()
    assert App(name="app", user_api_scopes=[]).as_dict() == {"name": "app"}


def test_nonempty_scope_removal_also_requires_manual_action(monkeypatch):
    from dataclasses import replace

    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", user_api_scopes=["ai-gateway", "sql"]))
    plan = replace(app_auth.prepare_app_auth("app", "selected", adopt=True), scopes=("ai-gateway",))
    with pytest.raises(AgentCliError, match="remov"):
        app_auth.apply_app_auth(plan)
    apps.create_update.assert_not_called()


def test_update_wait_failure_stops_before_effective_scope_success(monkeypatch):
    app_auth, apps, _ = _sdk(
        monkeypatch,
        App(name="app", user_api_scopes=["ai-gateway"], effective_user_api_scopes=["ai-gateway"]),
    )
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    apps.create_update.return_value.result.side_effect = PermissionDenied("update failed")
    with pytest.raises(AgentCliError, match="reconcile"):
        app_auth.apply_app_auth(plan, instances=2)


def test_unknown_configured_scopes_do_not_overwrite_effective_grants(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app", effective_user_api_scopes=["sql"]))
    with pytest.raises(AgentCliError, match="configured scopes"):
        app_auth.prepare_app_auth("app", "selected", adopt=True)
    apps.create_update.assert_not_called()


def test_adoption_does_not_request_implicit_identity_defaults(monkeypatch):
    defaults = ["iam.access-control:read", "iam.current-user:read"]
    existing = App(name="app", effective_user_api_scopes=defaults)
    app_auth, apps, _ = _sdk(monkeypatch, existing)
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    assert plan.existing_scopes == ()
    assert plan.scopes == ("ai-gateway",)
    updated = App(
        name="app",
        user_api_scopes=list(plan.scopes),
        effective_user_api_scopes=[*plan.scopes, *defaults],
    )
    apps.get.side_effect = [existing, updated]
    app_auth.apply_app_auth(plan, attempts=1)
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
        app_auth.prepare_app_auth("app", "selected", adopt=True)
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
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    apps.get.side_effect = None
    apps.get.return_value = ready
    app_auth.apply_app_auth(plan, attempts=1)
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
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    with pytest.raises(AgentCliError, match="effective"):
        app_auth.apply_app_auth(plan, attempts=1)
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
        ["test", "--source", str(tmp_path), "--adopt-user-auth"],
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
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    apps.get.return_value = App(
        name="app", user_api_scopes=["sql"], forward_user_access_token=False
    )
    with pytest.raises(AgentCliError, match="forward_user_access_token"):
        app_auth.apply_app_auth(plan, attempts=1)
    apps.create_update.assert_not_called()


def test_unexplained_effective_grants_after_preflight_stop_before_update(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch, App(name="app"))
    plan = app_auth.prepare_app_auth("app", "selected", adopt=True)
    apps.get.return_value = App(name="app", effective_user_api_scopes=["sql"])
    with pytest.raises(AgentCliError, match="configured scopes"):
        app_auth.apply_app_auth(plan, attempts=1)
    apps.create_update.assert_not_called()


def test_disabled_forwarding_during_verification_stops_deploy(monkeypatch):
    app_auth, apps, _ = _sdk(monkeypatch)
    plan = app_auth.prepare_app_auth("app", "selected", adopt=False)
    apps.get.side_effect = None
    apps.get.return_value = App(
        name="app",
        user_api_scopes=["ai-gateway"],
        effective_user_api_scopes=["ai-gateway"],
        forward_user_access_token=False,
    )
    with pytest.raises(AgentCliError, match="forward_user_access_token"):
        app_auth.apply_app_auth(plan, attempts=1)


@pytest.mark.parametrize("auth", [None, "app"])
def test_app_only_contract_keeps_deployment_path(
    tmp_path, monkeypatch, auth, _no_remote_provisioning
):
    _project(tmp_path, marker=None, auth=auth)
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
    monkeypatch.setattr(deploy_mod, "resolve_trace_experiment_id", lambda *args: None)
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
