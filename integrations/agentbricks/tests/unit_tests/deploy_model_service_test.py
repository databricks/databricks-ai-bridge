"""Tests for deploy's model-service reconcile: create when missing, never repoint an existing one."""

from __future__ import annotations

import pathlib

import pytest

from databricks_agentbricks.agent_project import AgentProject
from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.errors import AgentCliError

SERVICE = "main.my_agent.llm"


class _FakeClient:
    def __init__(self, exists: bool):
        self.exists = exists
        self.created: list[tuple[str, str]] = []

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
