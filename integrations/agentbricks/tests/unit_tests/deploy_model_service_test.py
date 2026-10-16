"""Tests for deploy's model-service reconcile: create when missing, never repoint an existing one."""

from __future__ import annotations

import pathlib
from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import yaml
from deploy_service_test import _request, deployment_fixture  # noqa: F401

from databricks_agentbricks.clients.api_client_provider import ApiClientProvider
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.reporting import Reporter
from databricks_agentbricks.services.deployment.model_binding_provisioner import (
    ModelBindingProvisioner,
)
from databricks_agentbricks.services.deployment.names import DeploymentName
from databricks_agentbricks.services.deployment.provisioners import ProjectContext, ResourceContext

SERVICE = "main.my_agent.llm"


class _FakeClient:
    def __init__(self, exists: bool):
        self.exists = exists
        self.created: list[tuple[str, str]] = []
        self.host = "https://ws"
        self.current_user = "tester@example.com"
        self.workspace_client = self
        self.granted = []
        self.grant_error = None

    def get_model_service(self, name):
        if not self.exists:
            raise AgentCliError("Resource not found", error_code="RESOURCE_DOES_NOT_EXIST")
        return {"name": f"model-services/{name}"}

    def create_model_service(self, name, model, *, comment=None):
        self.created.append((name, model))
        return {"name": f"model-services/{name}"}

    def grant_model_service_execute(self, name, principal):
        if self.grant_error:
            raise AgentCliError(self.grant_error)
        self.granted.append(("model", name, principal))

    def grant_prompt_access(self, schema, principal):
        if self.grant_error:
            raise AgentCliError(self.grant_error)
        self.granted.append(("prompt", schema, principal))


def _reconcile(project, client):
    reporter = Mock(spec=Reporter)
    reporter.status.return_value = nullcontext()
    provider = Mock(spec=ApiClientProvider)
    provider.get.return_value = client
    provisioner = ModelBindingProvisioner(provider, reporter)
    ctx = ResourceContext(
        ProjectContext(pathlib.Path("."), DeploymentName("agent-bricks-test"), project),
        None,
        None,
        None,
    )
    return dict(provisioner.reconcile(ctx).model_services)


def _project(tmp_path: pathlib.Path, default: str | None = "system.ai.claude-sonnet-4-5"):
    project = AgentProject.create(tmp_path, framework="langgraph", server="agentbricks")
    project.bind_model_service(SERVICE, default)
    project.write()
    return AgentProject.load(tmp_path)


def test_unbound_project_reconciles_nothing():
    assert _reconcile(None, _FakeClient(exists=False)) == {}


def test_missing_service_is_created_with_default(tmp_path):
    client = _FakeClient(exists=False)
    assert _reconcile(_project(tmp_path), client) == {"agent": SERVICE}
    assert client.created == [(SERVICE, "system.ai.claude-sonnet-4-5")]


def test_existing_service_is_left_alone(tmp_path):
    client = _FakeClient(exists=True)
    assert _reconcile(_project(tmp_path), client) == {"agent": SERVICE}
    assert client.created == []


def test_missing_service_without_default_explains_how_to_fix(tmp_path):
    with pytest.raises(AgentCliError, match="no default model") as exc_info:
        _reconcile(_project(tmp_path, default=None), _FakeClient(False))
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
    services = _reconcile(AgentProject.load(tmp_path), client)
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
    services = _reconcile(project, _FakeClient(exists=True))
    yaml.safe_dump({"env": [{"name": "AGENT_MODEL_SERVICE_AGENT", "value": services["agent"]}]})


def test_deploy_prunes_unbound_model_roles_and_preserves_bound_roles(tmp_path, deployment_fixture):
    f = deployment_fixture
    project = f.project
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
    f.provider.get.return_value = client
    result = f.service.deploy(_request(tmp_path))
    assert result.model_services == {"writer": "main.my_agent.writer"}
    assert client.granted == [("model", "main.my_agent.writer", "sp-123")]
    env = {
        e["name"]: e["value"] for e in yaml.safe_load((tmp_path / "app.yaml").read_text())["env"]
    }
    assert "AGENT_MODEL_SERVICE_ROUTER" not in env
    assert env["AGENT_MODEL_SERVICE_WRITER"] == "main.my_agent.writer"
    assert env["KEEP"] == "unchanged"
    project.unbind_model_service("writer")
    project.write()
    result = f.service.deploy(_request(tmp_path))
    assert result.model_services == {}
    env = yaml.safe_load((tmp_path / "app.yaml").read_text())["env"]
    assert not any(e["name"].startswith("AGENT_MODEL_SERVICE_") for e in env)


def test_deploy_grants_each_prompt_schema_once(tmp_path, deployment_fixture):
    f = deployment_fixture
    f.project.bind_prompt("main.prompts.router", "router")
    f.project.bind_prompt("main.prompts.writer", "writer")
    f.project.bind_prompt("main.other.summary", "summary")
    client = _FakeClient(exists=True)
    f.provider.get.return_value = client
    result = f.service.deploy(_request(tmp_path))
    assert client.granted == [
        ("prompt", "main.other", "sp-123"),
        ("prompt", "main.prompts", "sp-123"),
    ]
    assert result.prompts == dict(f.project.prompts)
    assert result.prompt_grant_error is None


def test_deploy_reports_grant_failures_without_undoing_rollout(tmp_path, deployment_fixture):
    f = deployment_fixture
    f.project.bind_model_service(SERVICE, "system.ai.claude-sonnet-4-5")
    f.project.bind_prompt("main.prompts.writer", "writer")
    client = _FakeClient(exists=True)
    f.provider.get.return_value = client
    client.grant_error = "access denied"
    result = f.service.deploy(_request(tmp_path))
    f.app.deploy_source.assert_called_once()
    assert "access denied" in result.model_service_grant_error
    assert "access denied" in result.prompt_grant_error


def test_deploy_reports_missing_app_identity_for_bound_resources(tmp_path, deployment_fixture):
    f = deployment_fixture
    f.project.bind_model_service(SERVICE, "system.ai.claude-sonnet-4-5")
    f.project.bind_prompt("main.prompts.writer", "writer")
    client = _FakeClient(exists=True)
    f.provider.get.return_value = client
    f.apps.get_service_principal.return_value = None
    result = f.service.deploy(_request(tmp_path))
    assert "service principal" in result.model_service_grant_error
    assert "service principal" in result.prompt_grant_error
    assert client.granted == []
