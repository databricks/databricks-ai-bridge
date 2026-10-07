"""Tests for deploy's model-service reconcile: create when missing, never repoint an existing one."""

from __future__ import annotations

import pathlib

import pytest
import yaml
from click.testing import CliRunner
from deploy_test import _FakeCtx

from databricks_agentbricks.agent_project import AgentProject
from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.errors import AgentCliError

SERVICE = "main.my_agent.llm"


class _FakeClient:
    def __init__(self, exists: bool):
        self.exists = exists
        self.created: list[tuple[str, str]] = []
        self.host = "https://ws"
        self.current_user = "tester@example.com"
        self.workspace_client = self

    def get_model_service(self, name):
        if not self.exists:
            raise AgentCliError("Resource not found", error_code="RESOURCE_DOES_NOT_EXIST")
        return {"name": f"model-services/{name}"}

    def create_model_service(self, name, model, *, comment=None):
        self.created.append((name, model))
        return {"name": f"model-services/{name}"}


def _project(tmp_path: pathlib.Path, default: str | None = "system.ai.claude-sonnet-4-5"):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.bind_model_service(SERVICE, default)
    project.write()
    return AgentProject.load(tmp_path)


def test_unbound_project_reconciles_nothing():
    assert deploy_mod._reconcile_model_services(None, _FakeClient(exists=False)) == {}


def test_missing_service_is_created_with_default(tmp_path):
    client = _FakeClient(exists=False)
    assert deploy_mod._reconcile_model_services(_project(tmp_path), client) == {"agent": SERVICE}
    assert client.created == [(SERVICE, "system.ai.claude-sonnet-4-5")]


def test_existing_service_is_left_alone(tmp_path):
    client = _FakeClient(exists=True)
    assert deploy_mod._reconcile_model_services(_project(tmp_path), client) == {"agent": SERVICE}
    assert client.created == []


def test_missing_service_without_default_explains_how_to_fix(tmp_path):
    with pytest.raises(AgentCliError, match="no default model") as exc_info:
        deploy_mod._reconcile_model_services(_project(tmp_path, default=None), _FakeClient(False))
    assert "--default" in (exc_info.value.hint or "")


def test_each_role_gets_its_own_service(tmp_path):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.bind_model_service(
        "main.my_agent.router_llm", "system.ai.claude-haiku-4-5", role="router"
    )
    project.bind_model_service(
        "main.my_agent.writer_llm", "system.ai.claude-sonnet-4-5", role="writer"
    )
    project.write()
    client = _FakeClient(exists=False)
    services = deploy_mod._reconcile_model_services(AgentProject.load(tmp_path), client)
    assert services == {"router": "main.my_agent.router_llm", "writer": "main.my_agent.writer_llm"}
    assert client.created == [
        ("main.my_agent.router_llm", "system.ai.claude-haiku-4-5"),
        ("main.my_agent.writer_llm", "system.ai.claude-sonnet-4-5"),
    ]


def test_one_service_cannot_back_two_roles(tmp_path):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.bind_model_service(SERVICE, role="router")
    with pytest.raises(AgentCliError, match="already bound to role 'router'"):
        project.bind_model_service(SERVICE, role="writer")


def test_loaded_bindings_are_plain_strings_deploy_can_write_to_app_yaml(tmp_path):
    # Found by a live deploy: tomlkit String values made yaml.safe_dump fail on app.yaml.
    import yaml

    project = AgentProject.load(_project(tmp_path).root)
    binding = project.model_services["agent"]
    assert type(binding.name) is str and type(binding.default) is str
    services = deploy_mod._reconcile_model_services(project, _FakeClient(exists=True))
    yaml.safe_dump({"env": [{"name": "AGENT_MODEL_SERVICE_AGENT", "value": services["agent"]}]})


def test_deploy_prunes_unbound_model_roles_and_preserves_bound_roles(tmp_path, monkeypatch):
    from types import SimpleNamespace

    project = AgentProject.create(tmp_path, framework="langgraph", server="custom")
    project.bind_model_service("main.my_agent.writer", "system.ai.claude-sonnet-4-5", role="writer")
    project.bind_model_service("main.my_agent.router", "system.ai.claude-sonnet-4-5", role="router")
    project.write()
    (tmp_path / "app.yaml").write_text(
        yaml.safe_dump(
            {
                "command": ["x"],
                "env": [
                    {"name": "AGENT_MODEL_SERVICE_ROUTER", "value": "main.my_agent.router"},
                    {"name": "AGENT_MODEL_SERVICE_WRITER", "value": "main.my_agent.writer"},
                    {"name": "KEEP", "value": "unchanged"},
                ],
            }
        )
    )
    project.unbind_model_service("router")
    project.write()
    client = _FakeClient(exists=True)
    monkeypatch.setattr(_FakeCtx, "client", lambda self: client)
    monkeypatch.setattr(deploy_mod, "_deployment_exists", lambda *a: True)
    monkeypatch.setattr(deploy_mod, "_app_compute_state", lambda *a: "ACTIVE")
    monkeypatch.setattr(deploy_mod, "_app_service_principal", lambda *a: None)
    monkeypatch.setattr(deploy_mod, "get_or_create_trace_experiment", lambda *a: None)
    monkeypatch.setattr(
        deploy_mod, "reconcile_tool_access", lambda client, app, principal, plan, profile: plan
    )
    monkeypatch.setattr(deploy_mod, "finalize_tool_access", lambda *a: None)
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="{}", stderr=""),
    )
    result = CliRunner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(tmp_path)], obj=_FakeCtx()
    )
    assert result.exit_code == 0, result.output
    env = {
        e["name"]: e["value"] for e in yaml.safe_load((tmp_path / "app.yaml").read_text())["env"]
    }
    assert "AGENT_MODEL_SERVICE_ROUTER" not in env
    assert env["AGENT_MODEL_SERVICE_WRITER"] == "main.my_agent.writer"
    assert env["KEEP"] == "unchanged"
    project.unbind_model_service("writer")
    project.write()
    result = CliRunner().invoke(
        deploy_mod.deploy, ["myapp", "--source", str(tmp_path)], obj=_FakeCtx()
    )
    assert result.exit_code == 0, result.output
    env = yaml.safe_load((tmp_path / "app.yaml").read_text())["env"]
    assert not any(e["name"].startswith("AGENT_MODEL_SERVICE_") for e in env)
