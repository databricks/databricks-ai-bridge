"""CLI tests for `agentbricks experimental models` (bind / unbind / status / set / rollback / upgrade).

Uses a real temp AgentProject and a fake client holding each model service's destination. The
optimizer and `promote_to_prod` are stubbed: `model_upgrade_test.py` covers the engine.
"""

from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from databricks_agentbricks import model_upgrade
from databricks_agentbricks.agent_project import AgentProject, ModelServiceBinding
from databricks_agentbricks.cli.models import models
from databricks_agentbricks.cli.prompts import prompts
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
        return [
            "system.ai.claude-haiku-4-5",
            "system.ai.claude-sonnet-4-5",
            "system.ai.gpt-5-4-mini",
            "system.ai.gpt-5-4-nano",
        ]


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
    # The module:attr references the upgrade tests pass (_EVAL_FLAGS); upgrade checks them statically.
    (project / "agent" / "eval.py").write_text(
        "def predict(inputs):\n    return ''\n\n\nTRAIN = []\nVAL = []\nSCORERS = []\n"
    )
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


def _invoke_prompts(args, obj):
    return CliRunner().invoke(prompts, args, obj=obj)


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


# --- status / set / rollback --------------------------------------------------------


def test_status_reports_current_destination(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient(), output="json"))
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model_services"] == {
        "agent": {"model_service": SERVICE, "model": "system.ai.claude-sonnet-4-5"}
    }
    assert payload["last_change"] is None
    assert payload["latest_run"] is None


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


def test_set_switches_and_records_history_then_rollback_restores(tmp_path):
    project = _project(tmp_path)
    client = _FakeClient()
    result = _invoke(
        ["set", "claude-haiku-4-5", "--yes", "--source", str(project)], _Ctx(client, "json")
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["changed"] is True
    assert client.model == "system.ai.claude-haiku-4-5"
    history = model_upgrade.read_history(project)["changes"]
    assert history[-1]["previous_model"] == "system.ai.claude-sonnet-4-5"
    assert history[-1]["model"] == "system.ai.claude-haiku-4-5"
    assert history[-1]["reason"] == "set"

    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert client.model == "system.ai.claude-sonnet-4-5"
    assert model_upgrade.read_history(project)["changes"][-1]["reason"] == "rollback"


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
    assert model_upgrade.read_history(project)["changes"] == []


def test_rollback_with_no_history_errors(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(_FakeClient()))
    assert result.exit_code != 0
    assert "No recorded switch" in result.output


# --- upgrade (job) / status / apply ------------------------------------------------------


# The SMU inputs every `models upgrade` call names (module:attr in the project).
_EVAL_FLAGS = [
    "--predict", "agent.eval:predict",
    "--train-data", "agent.eval:TRAIN",
    "--val-data", "agent.eval:VAL",
    "--scorer", "agent.eval:SCORERS",
]  # fmt: skip


def _rec(service: str, recommended: str) -> model_upgrade.ServiceRecommendation:
    return model_upgrade.ServiceRecommendation(
        service=service,
        current_model="system.ai.claude-sonnet-4-5",
        recommended_model=recommended,
        model_scores={"system.ai.claude-sonnet-4-5": 0.80, recommended: 0.82},
    )


def _report(recommended: str, **roles: str) -> model_upgrade.UpgradeReport:
    """A finished run's report: ``recommended`` for the single `agent` role, or one per role."""
    services = {"router": ROUTER, "writer": WRITER}
    recommendations = (
        {role: _rec(services[role], model) for role, model in roles.items()}
        if roles
        else {"agent": _rec(SERVICE, recommended)}
    )
    return model_upgrade.UpgradeReport(
        recommendations=recommendations,
        baseline_score=0.80,
        best_score=0.82,
        train_records=7,
        val_records=3,
        mlflow_run_id="r1",
    )


PROMPT_MOVE = {
    "name": "main.my_agent.system",
    "alias": "production",
    "prior_version": 3,
    "applied_version": 4,
}


@pytest.fixture
def stub_promote(monkeypatch):
    """Stand-in for promote_to_prod: switch each changed service on the fake client, and report a
    prompt-alias move for every prompt the report rewrote."""
    promoted: list = []

    def _promote(obj, report):
        promoted.append(report)
        for rec in report.recommendations.values():
            if rec.changed:
                obj.client().set_model_service_model(rec.service, rec.recommended_model)
        return [dict(PROMPT_MOVE, name=name) for name in report.prompt_changes]

    monkeypatch.setattr("databricks_agentbricks.cli.models._promote", _promote)
    return promoted


@pytest.fixture
def stub_restore(monkeypatch):
    """Records the prompt-alias moves `models rollback` reverses."""
    restored: list = []
    monkeypatch.setattr(
        "databricks_agentbricks.cli.prompts.prompt_alias_version",
        lambda obj, prompt: prompt["applied_version"],
    )
    monkeypatch.setattr(
        "databricks_agentbricks.cli.prompts.restore_prompt_aliases",
        lambda obj, prompts: restored.extend(prompts),
    )
    return restored


class _JobClient(_FakeClient):
    current_user = "me@example.com"

    def __init__(self, life_cycle="RUNNING", result=None, denied=()):
        super().__init__()
        self.life_cycle, self.result, self.denied = life_cycle, result, set(denied)

    def can_execute_model(self, model):
        return model not in self.denied

    def get_run(self, run_id):
        state = SimpleNamespace(life_cycle_state=self.life_cycle, result_state=self.result)
        return SimpleNamespace(state=state, run_page_url=f"https://ws/run/{run_id}")


@pytest.fixture
def stub_job(monkeypatch):
    calls: dict = {"synced": []}

    def _databricks(args, profile, **kwargs):
        calls["synced"].append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def _submit(client, *, workspace_path, config, timeout_hours):
        calls["submit"] = (workspace_path, config, timeout_hours)
        return 42, "https://ws/run/42"

    def _fetch(profile, trace_experiment, upgrade_id):
        calls["fetch"] = (trace_experiment, upgrade_id)
        return _report("system.ai.claude-haiku-4-5")

    monkeypatch.setattr("databricks_agentbricks.databricks_cli._databricks", _databricks)
    monkeypatch.setattr(model_upgrade, "submit_upgrade_run", _submit)
    monkeypatch.setattr(model_upgrade, "fetch_report", _fetch)
    return calls


def test_upgrade_uploads_project_and_submits_job_without_switching(tmp_path, stub_job):
    project = _project(tmp_path)
    client = _JobClient()
    result = _invoke(
        [
            "upgrade",
            *_EVAL_FLAGS,
            "-c",
            "claude-haiku-4-5,system.ai.gpt-5-4-mini",
            "--source",
            str(project),
        ],  # fmt: skip
        _Ctx(client, "json"),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["run_id"] == 42
    assert payload["run"] is None  # returned once submitted

    ((sync_args,),) = [stub_job["synced"]]
    ws_path = "/Workspace/Users/me@example.com/agentbricks_model_upgrades/agent-langgraph"
    assert sync_args[:3] == ["sync", str(project), ws_path]
    submitted_path, config, timeout_hours = stub_job["submit"]
    assert submitted_path == ws_path
    assert config.services == {"agent": SERVICE}
    assert config.predict_fn == "agent.eval:predict"
    assert (config.train_data, config.val_data) == ("agent.eval:TRAIN", "agent.eval:VAL")
    assert config.scorers == ["agent.eval:SCORERS"]
    assert config.prompt_uris == []
    assert config.candidates == {"agent": ["system.ai.claude-haiku-4-5", "system.ai.gpt-5-4-mini"]}
    assert config.trace_experiment == "/Shared/agentbricks_traces/my-agent"
    assert config.weights == (0.7, 0.2, 0.1)
    assert config.latency_gate == 60.0
    assert timeout_hours == 6.0
    # Submitting never switches the model.
    assert client.model == "system.ai.claude-sonnet-4-5"
    (run,) = model_upgrade.read_history(project)["runs"]
    assert run["run_id"] == 42 and run["upgrade_id"] == config.upgrade_id


def test_status_then_apply_switches_to_finished_runs_recommendation(
    tmp_path, stub_job, stub_promote
):
    project = _project(tmp_path)
    _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )

    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    status = _invoke(["status", "--source", str(project)], _Ctx(client, "json"))
    assert status.exit_code == 0, status.output
    latest = json.loads(status.output)["latest_run"]
    assert latest["state"] == "SUCCESS"
    recommendation = latest["report"]["recommendations"]["agent"]
    assert recommendation["recommended_model"] == "system.ai.claude-haiku-4-5"

    applied = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.output)["changed"] is True
    assert client.model == "system.ai.claude-haiku-4-5"
    change = model_upgrade.read_history(project)["changes"][-1]
    assert change["reason"] == "upgrade"
    assert change["best_score"] == pytest.approx(0.82)


def test_apply_before_the_run_finishes_refuses(tmp_path, stub_job):
    project = _project(tmp_path)
    _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )
    client = _JobClient(life_cycle="RUNNING")
    result = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(client))
    assert result.exit_code != 0
    assert "no recommendation" in result.output
    assert client.model == "system.ai.claude-sonnet-4-5"


def test_apply_with_no_runs_errors(tmp_path):
    project = _project(tmp_path)
    result = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(_JobClient()))
    assert result.exit_code != 0
    assert "No upgrade run" in result.output


def test_upgrade_refuses_candidates_the_owner_cannot_execute(tmp_path, stub_job):
    project = _project(tmp_path)
    client = _JobClient(denied={"system.ai.gpt-5-4-mini"})
    result = _invoke(
        [
            "upgrade",
            *_EVAL_FLAGS,
            "-c",
            "claude-haiku-4-5,gpt-5-4-mini",
            "--source",
            str(project),
        ],  # fmt: skip
        _Ctx(client),
    )
    assert result.exit_code != 0
    assert "gpt-5-4-mini" in result.output
    assert "submit" not in stub_job and stub_job["synced"] == []


def test_upgrade_requires_tracing(tmp_path, stub_job):
    project = _project(tmp_path, tracing=False)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--source", str(project)],  # fmt: skip
        _Ctx(_JobClient()),
    )
    assert result.exit_code != 0
    assert "tracing" in result.output
    assert "submit" not in stub_job


def test_upgrade_rejects_bad_weights(tmp_path, stub_job):
    project = _project(tmp_path)
    result = _invoke(
        [
            "upgrade",
            *_EVAL_FLAGS,
            "-c",
            "claude-haiku-4-5",
            "--weights",
            "0.5,0.5,0.5",
            "--source",
            str(project),
        ],  # fmt: skip
        _Ctx(_JobClient()),
    )
    assert result.exit_code != 0
    assert "submit" not in stub_job


def test_list_shows_chat_models(tmp_path):
    result = _invoke(["list"], _Ctx(_FakeClient(), "json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == _FakeClient().list_chat_model_services()


def test_upgrade_passes_prompt_uris_to_the_job(tmp_path, stub_job):
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5",
         "--prompt", "prompts:/main.my_agent.system@production", "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    _, config, _ = stub_job["submit"]
    assert config.prompt_uris == ["prompts:/main.my_agent.system@production"]


def test_upgrade_rejects_a_prompt_that_is_not_a_uri(tmp_path, stub_job):
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--prompt", "main.my_agent.system",
         "--source", str(project)],
        _Ctx(_JobClient()),
    )  # fmt: skip
    assert result.exit_code != 0
    assert "submit" not in stub_job


def test_apply_with_rewritten_prompts_promotes_through_promote_to_prod(
    tmp_path, stub_job, stub_promote, monkeypatch
):
    report = _report("system.ai.claude-haiku-4-5")
    report.prompt_changes = ["main.my_agent.system"]
    monkeypatch.setattr(model_upgrade, "fetch_report", lambda *args: report)
    project = _project(tmp_path)
    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--source", str(project)],
        _Ctx(client, "json"),
    )
    result = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert stub_promote == [report]
    assert json.loads(result.output)["prompts"] == ["main.my_agent.system"]
    (change,) = model_upgrade.read_history(project)["changes"]
    assert change["model"] == "system.ai.claude-haiku-4-5"


# --- compound agents: one model service per LLM call site ------------------------------------


def test_promote_records_registered_version_and_alias_reads_bypass_cache(mocker):
    from databricks_agentbricks.cli import models as models_mod
    from databricks_agentbricks.cli import prompts as prompts_mod
    from databricks_agentkit.model_upgrades import optimization as opt

    target = opt._PromptTarget(
        uri="prompts:/main.my_agent.system@production",
        name="main.my_agent.system",
        alias="production",
        version=None,
        short_name="system",
        template="old",
        required_vars=[],
        prior_version=2,
    )
    result = opt.Result(
        best_candidate={"prompt:system": "rewritten"},
        best_score=0.9,
        baseline_score=0.5,
        prompt_uris=[target.uri],
        gateway_endpoints={},
        prompt_targets=[target],
        endpoint_targets=[],
        gepa_result=None,
    )
    mocker.patch.object(model_upgrade, "load_promotion", return_value=result)
    mocker.patch.object(prompts_mod, "registry_mlflow", return_value=opt.mlflow)
    load = mocker.patch.object(
        opt.mlflow.genai, "load_prompt", return_value=SimpleNamespace(version=2)
    )
    mocker.patch.object(
        opt.mlflow.genai, "register_prompt", return_value=SimpleNamespace(version=5)
    )
    mocker.patch.object(opt.mlflow.genai, "set_prompt_alias")
    moves = models_mod._promote(_Ctx(), SimpleNamespace(mlflow_run_id="run"))
    assert moves == [
        {"name": target.name, "alias": target.alias, "prior_version": 2, "applied_version": 5}
    ]
    load.return_value.version = 5
    assert prompts_mod.prompt_alias_version(_Ctx(), moves[0]) == 5
    assert load.call_args_list == [mocker.call(target.uri, cache_ttl_seconds=0)] * 2


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


def test_upgrade_takes_candidates_per_role(tmp_path, stub_job):
    project = _project(tmp_path, compound=True)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "router=claude-haiku-4-5,gpt-5-4-nano",
         "-c", "writer=claude-haiku-4-5", "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    _, config, _ = stub_job["submit"]
    assert config.services == {"router": ROUTER, "writer": WRITER}
    assert config.candidates == {
        "router": ["system.ai.claude-haiku-4-5", "system.ai.gpt-5-4-nano"],
        "writer": ["system.ai.claude-haiku-4-5"],
    }


def test_upgrade_needs_role_prefixes_when_several_roles_are_bound(tmp_path, stub_job):
    project = _project(tmp_path, compound=True)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--source", str(project)],  # fmt: skip
        _Ctx(_JobClient()),
    )
    assert result.exit_code != 0
    assert "router=claude-haiku-4-5" in result.output
    assert "submit" not in stub_job


def test_upgrade_rejects_an_unbound_role(tmp_path, stub_job):
    project = _project(tmp_path, compound=True)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "critic=claude-haiku-4-5", "--source", str(project)],
        _Ctx(_JobClient()),
    )
    assert result.exit_code != 0
    assert "isn't a bound role" in result.output


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


def test_apply_switches_every_changed_role_and_records_each(
    tmp_path, stub_job, stub_promote, monkeypatch
):
    report = _report("", router="system.ai.claude-haiku-4-5", writer="system.ai.claude-sonnet-4-5")
    monkeypatch.setattr(model_upgrade, "fetch_report", lambda *args: report)
    project = _project(tmp_path, compound=True)
    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "router=claude-haiku-4-5", "--source", str(project)],
        _Ctx(client, "json"),
    )
    result = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.output)["models"]) == {"router"}  # writer kept its model
    assert client.models == {ROUTER: "system.ai.claude-haiku-4-5"}
    (change,) = model_upgrade.read_history(project)["changes"]
    assert change["model_service"] == ROUTER


def test_apply_refuses_when_a_service_moved_since_the_run(
    tmp_path, stub_job, stub_promote, monkeypatch
):
    report = _report("system.ai.claude-haiku-4-5")
    monkeypatch.setattr(model_upgrade, "fetch_report", lambda *args: report)
    project = _project(tmp_path)
    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--source", str(project)],
        _Ctx(client, "json"),
    )
    client.models[SERVICE] = "system.ai.gpt-5-4-mini"  # someone ran `models set` meanwhile
    result = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(client))
    assert result.exit_code != 0
    assert "switched models after" in result.output
    assert stub_promote == []


# --- rollback undoes the whole last action --------------------------------------------------


def _applied(tmp_path, monkeypatch, report, *, compound=False):
    """A project whose latest upgrade run (``report``) has been applied."""
    monkeypatch.setattr(model_upgrade, "fetch_report", lambda *args: report)
    project = _project(tmp_path, compound=compound)
    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    candidates = "router=claude-haiku-4-5" if compound else "claude-haiku-4-5"
    _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", candidates, "--source", str(project)],
        _Ctx(client, "json"),
    )
    result = _invoke(["apply", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    return project, client


def test_rollback_undoes_the_models_and_prompts_an_apply_changed(
    tmp_path, stub_job, stub_promote, stub_restore, monkeypatch
):
    report = _report("system.ai.claude-haiku-4-5")
    report.prompt_changes = ["main.my_agent.system"]
    project, client = _applied(tmp_path, monkeypatch, report)
    assert client.model == "system.ai.claude-haiku-4-5"

    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["undid"] == "apply"
    assert client.model == "system.ai.claude-sonnet-4-5"
    assert stub_restore == [PROMPT_MOVE]  # @production back on the version before the apply
    (action,) = model_upgrade.read_history(project)["actions"]
    assert action["rolled_back"] is True

    again = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client))
    assert again.exit_code != 0
    assert "No recorded switch or upgrade" in again.output


def test_rollback_after_a_compound_apply_restores_every_switched_role(
    tmp_path, stub_job, stub_promote, stub_restore, monkeypatch
):
    report = _report("", router="system.ai.claude-haiku-4-5", writer="system.ai.gpt-5-4-mini")
    project, client = _applied(tmp_path, monkeypatch, report, compound=True)
    assert client.models == {ROUTER: "system.ai.claude-haiku-4-5", WRITER: "system.ai.gpt-5-4-mini"}

    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert client.models == {
        ROUTER: "system.ai.claude-sonnet-4-5",
        WRITER: "system.ai.claude-sonnet-4-5",
    }
    assert stub_restore == []  # nothing to move: the apply rewrote no prompts


def test_rollback_with_a_role_switches_only_that_model(
    tmp_path, stub_job, stub_promote, stub_restore, monkeypatch
):
    report = _report("", router="system.ai.claude-haiku-4-5", writer="system.ai.gpt-5-4-mini")
    report.prompt_changes = ["main.my_agent.system"]
    project, client = _applied(tmp_path, monkeypatch, report, compound=True)

    result = _invoke(
        ["rollback", "--role", "router", "--yes", "--source", str(project)], _Ctx(client, "json")
    )
    assert result.exit_code == 0, result.output
    assert client.models == {
        ROUTER: "system.ai.claude-sonnet-4-5",
        WRITER: "system.ai.gpt-5-4-mini",
    }
    assert stub_restore == []

    remaining = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert remaining.exit_code == 0, remaining.output
    assert client.models == {
        ROUTER: "system.ai.claude-sonnet-4-5",
        WRITER: "system.ai.claude-sonnet-4-5",
    }
    assert stub_restore == [PROMPT_MOVE]
    assert model_upgrade.read_history(project)["actions"][-1]["rolled_back"] is True


def test_failed_rollback_resumes_without_switching_restored_roles_again(
    tmp_path, stub_job, stub_promote, stub_restore, monkeypatch
):
    report = _report("", router="system.ai.claude-haiku-4-5", writer="system.ai.gpt-5-4-mini")
    report.prompt_changes = ["main.my_agent.system"]
    project, client = _applied(tmp_path, monkeypatch, report, compound=True)
    original = client.set_model_service_model
    fail = True

    def set_model(name, model):
        if name == WRITER and fail:
            raise AgentCliError("temporary failure")
        return original(name, model)

    monkeypatch.setattr(client, "set_model_service_model", set_model)
    first = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert first.exit_code != 0
    action = model_upgrade.read_history(project)["actions"][-1]
    assert action["models"][0]["rolled_back"] is True
    assert action["rolled_back"] is False
    assert stub_restore == []
    calls = list(client.calls)
    fail = False
    retry = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert retry.exit_code == 0, retry.output
    assert not any(call[:2] == ("set", ROUTER) for call in client.calls[len(calls) :])
    assert client.models == {
        ROUTER: "system.ai.claude-sonnet-4-5",
        WRITER: "system.ai.claude-sonnet-4-5",
    }
    assert stub_restore == [PROMPT_MOVE]
    assert model_upgrade.read_history(project)["actions"][-1]["rolled_back"] is True


def test_rollback_resumes_after_prompt_failure(
    tmp_path, stub_job, stub_promote, stub_restore, monkeypatch
):
    report = _report("system.ai.claude-haiku-4-5")
    report.prompt_changes = ["main.my_agent.system", "main.my_agent.writer"]
    project, client = _applied(tmp_path, monkeypatch, report)
    restored = []
    fail = True

    def restore(obj, prompts):
        prompt = prompts[0]
        if prompt["name"] == "main.my_agent.writer" and fail:
            raise AgentCliError("registry unavailable")
        restored.append(prompt["name"])

    monkeypatch.setattr("databricks_agentbricks.cli.prompts.restore_prompt_aliases", restore)
    first = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert first.exit_code != 0
    assert restored == ["main.my_agent.system"]
    action = model_upgrade.read_history(project)["actions"][-1]
    assert action["models"][0]["rolled_back"] is True
    assert action["prompts"][0]["rolled_back"] is True
    fail = False
    retry = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert retry.exit_code == 0, retry.output
    assert restored == ["main.my_agent.system", "main.my_agent.writer"]
    assert model_upgrade.read_history(project)["actions"][-1]["rolled_back"] is True


def test_rollback_accepts_restored_models_when_progress_was_not_saved(
    tmp_path, stub_job, stub_promote, stub_restore, monkeypatch
):
    project, client = _applied(tmp_path, monkeypatch, _report("system.ai.claude-haiku-4-5"))
    client.models[SERVICE] = "system.ai.claude-sonnet-4-5"
    calls = list(client.calls)
    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code == 0, result.output
    assert not any(call[0] == "set" for call in client.calls[len(calls) :])
    assert model_upgrade.last_undoable_action(project) is None


def test_rollback_refuses_changed_prompt_before_restoring_models(
    tmp_path, stub_job, stub_promote, stub_restore, monkeypatch
):
    report = _report("system.ai.claude-haiku-4-5")
    report.prompt_changes = ["main.my_agent.system"]
    project, client = _applied(tmp_path, monkeypatch, report)
    monkeypatch.setattr(
        "databricks_agentbricks.cli.prompts.prompt_alias_version", lambda obj, prompt: 7
    )
    calls = list(client.calls)
    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client, "json"))
    assert result.exit_code != 0
    assert "moved after the last apply" in result.output
    assert not any(call[0] == "set" for call in client.calls[len(calls) :])
    assert stub_restore == []


def test_role_rollback_consumes_history_instead_of_toggling(tmp_path):
    project = _project(tmp_path)
    client = _FakeClient()
    for model in ("claude-haiku-4-5", "gpt-5-4-mini"):
        result = _invoke(["set", model, "--yes", "--source", str(project)], _Ctx(client, "json"))
        assert result.exit_code == 0, result.output
    for expected in ("system.ai.claude-haiku-4-5", "system.ai.claude-sonnet-4-5"):
        result = _invoke(
            ["rollback", "--role", "agent", "--yes", "--source", str(project)], _Ctx(client, "json")
        )
        assert result.exit_code == 0, result.output
        assert client.model == expected
    again = _invoke(
        ["rollback", "--role", "agent", "--yes", "--source", str(project)], _Ctx(client, "json")
    )
    assert again.exit_code != 0
    assert client.model == "system.ai.claude-sonnet-4-5"


def test_rollback_refuses_when_a_model_moved_since_the_action(
    tmp_path, stub_job, stub_promote, stub_restore, monkeypatch
):
    project, client = _applied(tmp_path, monkeypatch, _report("system.ai.claude-haiku-4-5"))
    client.models[SERVICE] = "system.ai.gpt-5-4-mini"
    result = _invoke(["rollback", "--yes", "--source", str(project)], _Ctx(client))
    assert result.exit_code != 0
    assert "switched models after the last apply" in result.output
    assert stub_restore == []


# --- where the search runs, applying from upgrade, and the summary ---------------------------


def test_upgrade_can_run_locally_without_a_job(tmp_path, stub_job, monkeypatch):
    ran = []

    def _run_and_record(root, config):
        ran.append((root, config))
        return _report("system.ai.claude-haiku-4-5")

    monkeypatch.setattr(model_upgrade, "run_and_record", _run_and_record)
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--run-on", "local",
         "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert len(ran) == 1 and ran[0][0] == project
    assert stub_job["synced"] == [] and "submit" not in stub_job  # nothing uploaded or submitted
    (run,) = model_upgrade.read_history(project)["runs"]
    assert run["run_on"] == "local" and run["report"]["recommendations"]["agent"]["changed"]


def test_upgrade_with_apply_auto_waits_and_applies(tmp_path, stub_job, stub_promote, monkeypatch):
    # The root `agentbricks -o json` sets this; it keeps the wait spinner out of the JSON.
    monkeypatch.setattr("databricks_agentbricks.errors._OUTPUT_MODE", "json")
    project = _project(tmp_path)
    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--apply", "auto",
         "--source", str(project)],
        _Ctx(client, "json"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["applied"]["changed"] is True
    assert client.model == "system.ai.claude-haiku-4-5"
    assert len(model_upgrade.read_history(project)["actions"]) == 1  # rollback can undo it


def test_upgrade_with_apply_ask_leaves_things_alone_when_declined(tmp_path, stub_job, stub_promote):
    project = _project(tmp_path)
    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    result = CliRunner().invoke(
        models,
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--apply", "ask",
         "--source", str(project)],
        obj=_Ctx(client),
        input="n\n",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "Not applied" in result.output
    assert client.model == "system.ai.claude-sonnet-4-5"
    assert stub_promote == []


def test_upgrade_summary_shows_each_roles_change_and_the_score(tmp_path, stub_job):
    project = _project(tmp_path)
    client = _JobClient(life_cycle="TERMINATED", result="SUCCESS")
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--wait", "--source", str(project)],
        _Ctx(client),
    )
    assert result.exit_code == 0, result.output
    out = " ".join(result.output.split())
    assert "7 train / 3 eval records" in out
    assert "claude-haiku-4-5" in out and "switch" in out
    assert "0.800 → 0.820 (+0.020)" in out
    assert "agentbricks experimental models apply" in out


# --- prompt bindings ---------------------------------------------------------------------


def test_bind_prompt_writes_agent_toml_and_status_lists_it(tmp_path):
    project = _project(tmp_path)
    result = _invoke_prompts(["bind", "main.my_agent.writer", "--source", str(project)], _Ctx())
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).prompts == {"writer": "main.my_agent.writer"}
    assert '[prompts]\nwriter = "main.my_agent.writer"' in (project / "agent.toml").read_text()
    status = _invoke(["status", "--source", str(project)], _Ctx(_FakeClient(), "json"))
    assert json.loads(status.output)["prompts"] == {"writer": "main.my_agent.writer"}


def test_bind_prompt_takes_an_explicit_key_and_rejects_bad_names(tmp_path):
    project = _project(tmp_path)
    result = _invoke_prompts(
        ["bind", "main.my_agent.v2_writer", "--key", "writer", "--source", str(project)],
        _Ctx(),
    )
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).prompts == {"writer": "main.my_agent.v2_writer"}
    bad = _invoke_prompts(["bind", "just-a-name", "--source", str(project)], _Ctx())
    assert bad.exit_code != 0


def test_unbind_prompt_removes_the_table_when_empty(tmp_path):
    project = _project(tmp_path)
    _invoke_prompts(["bind", "main.my_agent.writer", "--source", str(project)], _Ctx())
    result = _invoke_prompts(["unbind", "writer", "--source", str(project)], _Ctx())
    assert result.exit_code == 0, result.output
    assert AgentProject.load(project).prompts == {}
    assert "[prompts]" not in (project / "agent.toml").read_text()


def test_upgrade_optimizes_the_bound_prompts_by_default(tmp_path, stub_job):
    project = _project(tmp_path)
    _invoke_prompts(["bind", "main.my_agent.writer", "--source", str(project)], _Ctx())
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )
    assert result.exit_code == 0, result.output
    _, config, _ = stub_job["submit"]
    assert config.prompt_uris == ["prompts:/main.my_agent.writer@production"]


def test_upgrade_models_only_skips_the_bound_prompts(tmp_path, stub_job):
    project = _project(tmp_path)
    _invoke_prompts(["bind", "main.my_agent.writer", "--source", str(project)], _Ctx())
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--models-only",
         "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    _, config, _ = stub_job["submit"]
    assert config.prompt_uris == []


def test_upgrade_without_candidates_optimizes_only_the_bound_prompts(tmp_path, stub_job):
    project = _project(tmp_path)
    _invoke_prompts(["bind", "main.my_agent.writer", "--source", str(project)], _Ctx())
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )
    assert result.exit_code == 0, result.output
    _, config, _ = stub_job["submit"]
    assert config.candidates == {}
    assert config.prompt_uris == ["prompts:/main.my_agent.writer@production"]


def test_upgrade_without_candidates_or_prompts_errors(tmp_path, stub_job):
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )
    assert result.exit_code != 0
    assert "submit" not in stub_job


def test_upgrade_models_only_without_candidates_errors(tmp_path, stub_job):
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "--models-only", "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )
    assert result.exit_code != 0
    assert "submit" not in stub_job


def test_prompts_list_shows_each_bound_prompts_production_version(tmp_path, monkeypatch):
    project = _project(tmp_path)
    _invoke_prompts(["bind", "main.my_agent.writer", "--source", str(project)], _Ctx())
    _invoke_prompts(["bind", "main.my_agent.router", "--source", str(project)], _Ctx())

    def _load_prompt(uri):
        if "router" in uri:
            raise RuntimeError("alias not found")
        return SimpleNamespace(version=3)

    fake_mlflow = SimpleNamespace(genai=SimpleNamespace(load_prompt=_load_prompt))
    monkeypatch.setattr(
        "databricks_agentbricks.cli.prompts.registry_mlflow", lambda obj: fake_mlflow
    )
    result = _invoke_prompts(["list", "--source", str(project)], _Ctx(output="json"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "writer": {"prompt": "main.my_agent.writer", "production_version": "3"},
        "router": {"prompt": "main.my_agent.router", "production_version": None},
    }


def test_upgrade_rejects_a_model_the_workspace_doesnt_have(tmp_path, stub_job):
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "not-a-real-model", "--source", str(project)],
        _Ctx(_JobClient()),
    )
    assert result.exit_code != 0
    assert "No such model: system.ai.not-a-real-model" in result.output
    assert "submit" not in stub_job


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        ("agent.nope:predict", "No module 'agent.nope'"),
        ("agent.eval:predikt", "doesn't define 'predikt'"),
        ("agent.eval", "isn't a module:attr reference"),
    ],
)
def test_upgrade_rejects_a_reference_that_cant_resolve(tmp_path, stub_job, ref, message):
    project = _project(tmp_path)
    flags = list(_EVAL_FLAGS)
    flags[1] = ref
    result = _invoke(
        ["upgrade", *flags, "-c", "claude-haiku-4-5", "--source", str(project)],
        _Ctx(_JobClient()),
    )
    assert result.exit_code != 0
    assert message in result.output
    assert "submit" not in stub_job


def test_check_ref_follows_imports_and_star_imports(tmp_path):
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "__init__.py").write_text("")
    (tmp_path / "agent" / "small.py").write_text("from agent.eval import TRAIN as T, VAL\n")
    (tmp_path / "agent" / "star.py").write_text("from agent.eval import *\n")
    model_upgrade.check_ref(tmp_path, "agent.small:T")
    model_upgrade.check_ref(tmp_path, "agent.small:VAL")
    model_upgrade.check_ref(tmp_path, "agent.star:ANYTHING")  # a star import can define any name
    with pytest.raises(AgentCliError, match="doesn't define 'TRAIN'"):
        model_upgrade.check_ref(tmp_path, "agent.small:TRAIN")


def test_upgrade_passes_latency_gate_to_the_job(tmp_path, stub_job):
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--latency-gate", "300",
         "--source", str(project)],
        _Ctx(_JobClient(), "json"),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    _, config, _ = stub_job["submit"]
    assert config.latency_gate == 300.0
    assert model_upgrade.JobConfig.from_param(config.to_param()).latency_gate == 300.0


def test_report_where_every_record_scored_zero_says_so(capsys):
    from databricks_agentbricks.cli.models import _render_report

    report = _report("system.ai.claude-sonnet-4-5")
    report.baseline_score = report.best_score = 0.0
    _render_report(report)
    out = capsys.readouterr().out
    assert "Every record scored 0" in out
    assert "--latency-gate" in out
    assert "Already on the best" not in out


def test_local_run_with_prompts_needs_litellm(tmp_path, stub_job, monkeypatch):
    """Without litellm, GEPA can't write a single prompt rewrite, so fail before the search."""
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "litellm" else real_find_spec(name, *a, **k),
    )
    project = _project(tmp_path)
    result = _invoke(
        ["upgrade", *_EVAL_FLAGS, "-c", "claude-haiku-4-5", "--run-on", "local",
         "--prompt", "prompts:/main.my_agent.writer@production", "--source", str(project)],
        _Ctx(_JobClient()),
    )  # fmt: skip
    assert result.exit_code != 0
    assert "--run-on local needs litellm installed here" in result.output


def test_binding_a_new_role_keeps_a_blank_line_before_the_next_table(tmp_path):
    project = _project(tmp_path)
    _invoke_prompts(["bind", "main.my_agent.writer", "--source", str(project)], _Ctx())
    result = _invoke(
        ["bind", ROUTER, "--role", "router", "--default", "claude-haiku-4-5",
         "--source", str(project)],
        _Ctx(),
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    text = (project / "agent.toml").read_text()
    assert "\n\n[prompts]\n" in text
    assert '"system.ai.claude-haiku-4-5"\n[' not in text
