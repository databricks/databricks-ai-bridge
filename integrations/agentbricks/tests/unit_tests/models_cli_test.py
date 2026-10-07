"""CLI tests for `agentbricks experimental models` bindings (bind / unbind / list / status / set).

Uses a real temp AgentProject and a fake client holding each model service's destination.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from click.testing import CliRunner

from databricks_agentbricks.agent_project import AgentProject, ModelServiceBinding
from databricks_agentbricks.cli.models import models
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.project_config import write_project_metadata

SERVICE = "main.my_agent.llm"
ROUTER = "main.my_agent.router_llm"
WRITER = "main.my_agent.writer_llm"


def _service(model: str, name: str = SERVICE) -> dict:
    return {
        "name": f"model-services/{name}",
        "config": {
            "routing": {
                "destinations": [
                    {
                        "name": "primary",
                        "destination_type": "DESTINATION_TYPE_PAY_PER_TOKEN_FOUNDATION_MODEL",
                        "pay_per_token_config": {"model": f"models/{model}"},
                        "traffic_percentage": 100,
                    }
                ]
            }
        },
    }


class _FakeClient:
    """Each model service starts on ``model``; ``models`` tracks per-service switches."""

    def __init__(self, model: str | None = "system.ai.claude-sonnet-4-5"):
        self.default_model = model
        self.models: dict[str, str] = {}
        self.calls: list[tuple] = []

    @property
    def model(self) -> str | None:
        return self.models.get(SERVICE, self.default_model)

    def get_model_service(self, name):
        self.calls.append(("get", name))
        model = self.models.get(name, self.default_model)
        if model is None:
            raise AgentCliError("not found", error_code="NOT_FOUND")
        return _service(model, name)

    def set_model_service_model(self, name, model):
        self.calls.append(("set", name, model))
        self.models[name] = model
        return _service(model, name)

    def list_chat_model_services(self):
        return ["system.ai.claude-haiku-4-5", "system.ai.claude-sonnet-4-5"]


class _Ctx:
    def __init__(self, client=None, output="text"):
        self._client = client
        self.output = output
        self.profile = None

    def client(self):
        if self._client is None:
            raise AssertionError("This command must not contact the workspace")
        return self._client


def _project(tmp_path: pathlib.Path, *, bind=True, tracing=True, compound=False) -> pathlib.Path:
    project = tmp_path / "agent-langgraph"
    (project / "agent").mkdir(parents=True)
    write_project_metadata(project, framework="langgraph", template="agent-langgraph")
    created = AgentProject.create(
        project,
        framework="langgraph",
        server="agentbricks",
        experiment_name="/Shared/agentbricks_traces/my-agent" if tracing else None,
    )
    if compound:
        created.bind_model_service(ROUTER, "system.ai.claude-sonnet-4-5", role="router")
        created.bind_model_service(WRITER, "system.ai.claude-sonnet-4-5", role="writer")
    elif bind:
        created.bind_model_service(SERVICE, "system.ai.claude-sonnet-4-5")
    created.write()
    return project


def _invoke(args, obj):
    return CliRunner().invoke(models, args, obj=obj)


# --- bind / unbind ----------------------------------------------------------------


@pytest.mark.parametrize("output", ["text", "json"])
def test_bind_writes_agent_toml_without_contacting_workspace(tmp_path, output):
    project = _project(tmp_path, bind=False)
    result = _invoke(
        ["bind", SERVICE, "--default", "claude-sonnet-4-5", "--source", str(project)],
        _Ctx(output=output),
    )
    assert result.exit_code == 0, result.output
    if output == "json":
        assert json.loads(result.output) == {
            "role": "agent",
            "model_service": SERVICE,
            "default": "system.ai.claude-sonnet-4-5",
            "manifest": str(project / "agent.toml"),
        }
    assert AgentProject.load(project).model_services == {
        "agent": ModelServiceBinding(SERVICE, "system.ai.claude-sonnet-4-5")
    }


def test_bind_keeps_recorded_default_when_omitted(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["bind", "main.my_agent.other", "--source", str(project)], _Ctx())
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).model_services == {
        "agent": ModelServiceBinding("main.my_agent.other", "system.ai.claude-sonnet-4-5")
    }


def test_bind_rejects_non_three_part_name(tmp_path):
    project = _project(tmp_path, bind=False)
    result = _invoke(["bind", "just-a-name", "--source", str(project)], _Ctx())
    assert result.exit_code != 0
    assert AgentProject.load(project).model_services == {}


def test_unbind_removes_the_table(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["unbind", "--source", str(project)], _Ctx())
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).model_services == {}
    assert "model_services" not in (project / "agent.toml").read_text()


# --- status / set ---------------------------------------------------------------


def test_status_reports_current_destination(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient(), output="json"))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_services"] == {
        "agent": {"model_service": SERVICE, "model": "system.ai.claude-sonnet-4-5"}
    }


def test_status_without_binding_points_at_bind(tmp_path):
    project = _project(tmp_path, bind=False)
    result = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient()))
    assert result.exit_code != 0
    assert "no model service bound" in result.output


def test_status_before_deploy_points_at_deploy(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient(model=None)))
    assert result.exit_code != 0
    assert "doesn't exist yet" in result.output


def test_set_json_without_yes_never_switches(tmp_path):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(["set", "claude-haiku-4-5", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["changed"] is False
    assert client.model == "system.ai.claude-sonnet-4-5"
    assert not any(call[0] == "set" for call in client.calls)


def test_set_to_current_model_is_a_no_op(tmp_path):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(
        ["set", "system.ai.claude-sonnet-4-5", "--yes", "--source", str(project)], _Ctx(client)
    )
    assert result.exit_code == 0, result.output
    assert not any(call[0] == "set" for call in client.calls)


# The SMU inputs every `models upgrade` call names (module:attr in the project).
_EVAL_FLAGS = [
    "--predict", "agent.eval:predict",
    "--train-data", "agent.eval:TRAIN",
    "--val-data", "agent.eval:VAL",
    "--scorer", "agent.eval:SCORERS",
]  # fmt: skip


def test_list_shows_chat_models(tmp_path):
    result = _invoke(["list"], _Ctx(_FakeClient(), "json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == [
        "system.ai.claude-haiku-4-5",
        "system.ai.claude-sonnet-4-5",
    ]


# --- compound agents: one model service per LLM call site ------------------------------------


def test_bind_two_roles_writes_one_table_each(tmp_path):
    project = _project(tmp_path, bind=False)
    for service, role in ((ROUTER, "router"), (WRITER, "writer")):
        result = _invoke(
            ["bind", service, "--role", role, "--default", "claude-sonnet-4-5",
             "--source", str(project)],
            _Ctx(),
        )  # fmt: skip
        assert result.exit_code == 0, result.output
    assert AgentProject.load(project).model_services == {
        "router": ModelServiceBinding(ROUTER, "system.ai.claude-sonnet-4-5"),
        "writer": ModelServiceBinding(WRITER, "system.ai.claude-sonnet-4-5"),
    }
    text = (project / "agent.toml").read_text()
    assert "[model_services.router]" in text and "[model_services.writer]" in text


def test_status_lists_every_role(tmp_path):
    project = _project(tmp_path, compound=True)
    result = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient(), "json"))
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.output)["model_services"]) == {"router", "writer"}


def test_set_needs_a_role_when_several_are_bound(tmp_path):
    project = _project(tmp_path, compound=True)
    client = _FakeClient()
    result = _invoke(["set", "claude-haiku-4-5", "--yes", "--source", str(project)], _Ctx(client))
    assert result.exit_code != 0
    assert "--role" in result.output
    result = _invoke(
        [
            "set",
            "claude-haiku-4-5",
            "--role",
            "router",
            "--yes",
            "--source",
            str(project),
        ],  # fmt: skip
        _Ctx(client),
    )
    assert result.exit_code == 0, result.output
    assert client.models == {ROUTER: "system.ai.claude-haiku-4-5"}


def test_set_switches_the_model_behind_the_service(tmp_path):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(
        ["set", "claude-haiku-4-5", "--yes", "--source", str(project)], _Ctx(client, "json")
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["changed"] is True
    assert client.model == "system.ai.claude-haiku-4-5"


# --- prompt bindings ---------------------------------------------------------------------


def test_bind_prompt_writes_agent_toml_and_status_lists_it(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["bind-prompt", "main.my_agent.writer", "--source", str(project)], _Ctx())
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).prompts == {"writer": "main.my_agent.writer"}
    assert '[prompts]\nwriter = "main.my_agent.writer"' in (project / "agent.toml").read_text()
    status = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient(), "json"))
    assert json.loads(status.output)["prompts"] == {"writer": "main.my_agent.writer"}


def test_bind_prompt_takes_an_explicit_key_and_rejects_bad_names(tmp_path):
    project = _project(tmp_path)
    result = _invoke(
        ["bind-prompt", "main.my_agent.v2_writer", "--key", "writer", "--source", str(project)],
        _Ctx(),
    )
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).prompts == {"writer": "main.my_agent.v2_writer"}
    bad = _invoke(["bind-prompt", "just-a-name", "--source", str(project)], _Ctx())
    assert bad.exit_code != 0


def test_unbind_prompt_removes_the_table_when_empty(tmp_path):
    project = _project(tmp_path)
    _invoke(["bind-prompt", "main.my_agent.writer", "--source", str(project)], _Ctx())
    result = _invoke(["unbind-prompt", "writer", "--source", str(project)], _Ctx())
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).prompts == {}
    assert "[prompts]" not in (project / "agent.toml").read_text()
