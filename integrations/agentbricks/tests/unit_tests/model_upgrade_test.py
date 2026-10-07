"""Unit tests for the `agentbricks experimental models upgrade` engine: names, loading, job config, search."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from databricks_agentbricks import model_upgrade
from databricks_agentbricks.errors import AgentCliError


def test_model_names_round_trip():
    assert model_upgrade.system_ai_name("claude-haiku-4-5") == "system.ai.claude-haiku-4-5"
    assert (
        model_upgrade.system_ai_name("system.ai.claude-haiku-4-5") == "system.ai.claude-haiku-4-5"
    )
    assert model_upgrade.bare_name("system.ai.claude-haiku-4-5") == "claude-haiku-4-5"


def test_load_object_imports_from_the_project(tmp_path):
    (tmp_path / "evalmod_for_test.py").write_text("RECORDS = [{'inputs': {'q': 1}}]\n")
    assert model_upgrade.load_object(tmp_path, "evalmod_for_test:RECORDS") == [{"inputs": {"q": 1}}]


@pytest.mark.parametrize("ref", ["no_colon", ":attr", "module:"])
def test_load_object_rejects_bad_refs(tmp_path, ref):
    with pytest.raises(AgentCliError, match="module:attr"):
        model_upgrade.load_object(tmp_path, ref)


def test_load_object_missing_attr_errors(tmp_path):
    (tmp_path / "evalmod_missing.py").write_text("X = 1\n")
    with pytest.raises(AgentCliError, match="Could not load"):
        model_upgrade.load_object(tmp_path, "evalmod_missing:Y")


def test_load_scorers_flattens_lists(tmp_path):
    (tmp_path / "scorermod_for_test.py").write_text("A = 'a'\nBOTH = ['b', 'c']\n")
    refs = ["scorermod_for_test:A", "scorermod_for_test:BOTH"]
    assert model_upgrade._load_scorers(tmp_path, refs) == ["a", "b", "c"]


def test_actions_are_undone_newest_first(tmp_path):
    assert model_upgrade.last_undoable_action(tmp_path) is None
    model_upgrade.record_action(tmp_path, "set", models=[{"model_service": "a.b.c"}])
    model_upgrade.record_action(
        tmp_path,
        "apply",
        models=[],
        prompts=[{"name": "a.b.p", "alias": "production", "prior_version": 2}],
    )
    index, action = model_upgrade.last_undoable_action(tmp_path)
    assert (index, action["kind"]) == (1, "apply")
    model_upgrade.mark_rolled_back(tmp_path, index)
    index, action = model_upgrade.last_undoable_action(tmp_path)
    assert (index, action["kind"]) == (0, "set")


def test_history_round_trip(tmp_path):
    assert model_upgrade.read_history(tmp_path) == {"runs": [], "changes": [], "actions": []}
    model_upgrade.record_run(tmp_path, {"upgrade_id": "u1", "run_id": 7, "state": "PENDING"})
    model_upgrade.update_run(tmp_path, "u1", state="SUCCESS")
    model_upgrade.record_change(
        tmp_path, service="a.b.c", previous="system.ai.x", model="system.ai.y", reason="set"
    )
    history = model_upgrade.read_history(tmp_path)
    assert history["runs"][0]["state"] == "SUCCESS"
    (entry,) = history["changes"]
    assert entry["previous_model"] == "system.ai.x"
    assert entry["model"] == "system.ai.y"
    assert (tmp_path / ".agentbricks" / "model_upgrades.json").is_file()


def test_job_config_round_trips_through_the_job_parameter():
    config = model_upgrade.JobConfig(
        upgrade_id="u1",
        services={"router": "a.b.router", "writer": "a.b.writer"},
        candidates={"router": ["system.ai.x"], "writer": []},
        prompt_uris=["prompts:/a.b.system@production"],
        predict_fn="agent.eval:predict",
        train_data="agent.eval:TRAIN",
        val_data="agent.eval:VAL",
        scorers=["agent.eval:SCORERS"],
        trace_experiment="/Shared/t",
        budget=None,
        weights=(0.7, 0.2, 0.1),
    )
    assert model_upgrade.JobConfig.from_param(config.to_param()) == config


def _rec(current: str, recommended: str, service: str = "a.b.c"):
    return model_upgrade.ServiceRecommendation(service, current, recommended, {recommended: 0.6})


def test_report_round_trips_through_json():
    report = model_upgrade.UpgradeReport(
        recommendations={
            "router": _rec("system.ai.x", "system.ai.y", "a.b.router"),
            "writer": _rec("system.ai.z", "system.ai.z", "a.b.writer"),
        },
        baseline_score=0.5,
        best_score=0.6,
        train_records=7,
        val_records=3,
        prompt_changes=["a.b.system"],
        mlflow_run_id="r1",
    )
    assert model_upgrade.UpgradeReport.from_json(report.to_json()) == report
    assert report.changed
    assert report.recommendations["router"].changed and not report.recommendations["writer"].changed


def test_report_with_only_a_prompt_rewrite_is_a_change():
    report = model_upgrade.UpgradeReport({"agent": _rec("system.ai.x", "system.ai.x")}, 0.5, 0.6)
    assert not report.changed
    report.prompt_changes = ["a.b.system"]
    assert report.changed


def test_run_upgrade_searches_every_role_jointly(monkeypatch):
    calls = {}

    class _Target:
        name, short_name, template = "a.b.system", "system", "old"

    def _optimize(predict_fn, train, val, **kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            best_candidate={"model:a.b.router": "claude-haiku-4-5", "prompt:system": "new"},
            baseline_score=0.7,
            best_score=0.8,
            prompt_targets=[_Target()],
            gepa_result=None,
        )

    monkeypatch.setattr(
        model_upgrade, "_optimizer", lambda: SimpleNamespace(optimize_prompts_and_models=_optimize)
    )
    report, _ = model_upgrade.run_upgrade(
        services={"router": "a.b.router", "writer": "a.b.writer"},
        current_models={
            "router": "system.ai.claude-sonnet-4-5",
            "writer": "system.ai.claude-sonnet-4-5",
        },
        candidates={"router": ["system.ai.claude-haiku-4-5"], "writer": []},
        prompt_uris=["prompts:/a.b.system@production"],
        predict_fn=lambda inputs: "",
        train_data=[{}] * 3,
        val_data=[{}] * 2,
        scorers=[],
        budget=None,
        weights=(0.7, 0.2, 0.1),
    )
    # Only roles with candidates are searched; both services are reported.
    assert calls["gateway_endpoints"] == {"a.b.router": ["claude-haiku-4-5"]}
    assert calls["prompt_uris"] == ["prompts:/a.b.system@production"]
    assert calls["max_metric_calls"] == 20
    assert report.recommendations["router"].recommended_model == "system.ai.claude-haiku-4-5"
    assert not report.recommendations["writer"].changed
    assert report.prompt_changes == ["a.b.system"]


def test_promotion_round_trips_into_a_result_promote_to_prod_accepts(monkeypatch):
    optimization = pytest.importorskip("databricks_agentkit.model_upgrades.optimization")
    result = optimization.Result(
        best_candidate={"prompt:system": "new {{ q }}", "model:a.b.c": "claude-haiku-4-5"},
        best_score=0.9,
        baseline_score=0.8,
        prompt_uris=["prompts:/a.b.system@production"],
        gateway_endpoints={"a.b.c": ["claude-haiku-4-5"]},
        prompt_targets=[
            optimization._PromptTarget(
                "prompts:/a.b.system@production",
                "a.b.system",
                "production",
                None,
                "system",
                "old {{ q }}",
                ["q"],
                3,
            )
        ],
        endpoint_targets=[
            optimization._EndpointTarget("a.b.c", ["claude-haiku-4-5"], "claude-sonnet-4-5")
        ],
        gepa_result=object(),  # not serializable; must be dropped
    )
    payload = json.loads(json.dumps(model_upgrade.promotion_payload(result)))
    monkeypatch.setattr("mlflow.artifacts.load_dict", lambda uri: payload)
    rebuilt = model_upgrade.load_promotion("r1")
    assert rebuilt.prompt_targets == result.prompt_targets
    assert rebuilt.endpoint_targets == result.endpoint_targets
    assert rebuilt.best_candidate == result.best_candidate
    assert rebuilt.gepa_result is None


def test_upgrade_requirement_override(monkeypatch):
    monkeypatch.setenv(
        model_upgrade.UPGRADE_REQUIREMENT_ENV, "databricks-agentbricks[upgrade] @ git+x"
    )
    assert model_upgrade.upgrade_requirement() == "databricks-agentbricks[upgrade] @ git+x"


def test_submit_upgrade_run_uploads_runner_and_installs_project(monkeypatch):
    monkeypatch.setenv(
        model_upgrade.UPGRADE_REQUIREMENT_ENV, "databricks-agentbricks[upgrade]==9.9"
    )

    class _Client:
        def __init__(self):
            self.uploaded, self.submitted = {}, {}

        def upload_workspace_file(self, path, content):
            self.uploaded[path] = content

        def submit_serverless_python_run(self, **kwargs):
            self.submitted = kwargs
            return 9

        def get_run(self, run_id):
            return SimpleNamespace(run_page_url="https://ws/run/9")

    client = _Client()
    config = model_upgrade.JobConfig(
        "u1",
        {"agent": "a.b.c"},
        {"agent": ["system.ai.x"]},
        [],
        "m:p",
        "m:t",
        "m:v",
        ["m:s"],
        "/Shared/t",
        40,
        (1.0, 0.0, 0.0),
    )
    run_id, url = model_upgrade.submit_upgrade_run(
        client, workspace_path="/Workspace/Users/me/p", config=config
    )
    assert (run_id, url) == (9, "https://ws/run/9")
    runner = "/Workspace/Users/me/p/.agentbricks/model_upgrade_job.py"
    assert "job_main" in client.uploaded[runner]
    # Set before any import: psycopg's bundled OpenSSL aborts on FIPS compute.
    assert client.uploaded[runner].index("PSYCOPG_IMPL") < client.uploaded[runner].index(
        "import job_main"
    )
    assert client.submitted["python_file"] == runner
    assert client.submitted["parameters"] == ["/Workspace/Users/me/p", config.to_param()]
    assert client.submitted["dependencies"] == [
        "/Workspace/Users/me/p",
        "databricks-agentbricks[upgrade]==9.9",
    ]
    assert client.submitted["environment_version"] == model_upgrade.SERVERLESS_ENVIRONMENT_VERSION


def test_job_submission_only_passes_arguments_the_sdk_accepts(monkeypatch):
    # A live run failed with "JobsAPI.submit() got an unexpected keyword argument 'tags'": check the
    # call against the installed SDK's real signature, not a fake.
    import inspect

    from databricks.sdk.service.jobs import JobsAPI

    from databricks_agentkit._api_client import _AgentBricksApiClient

    accepted = set(inspect.signature(JobsAPI.submit).parameters) - {"self"}
    seen = {}

    class _Jobs:
        def submit(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(run_id=7)

    client = object.__new__(_AgentBricksApiClient)
    client._w = SimpleNamespace(jobs=_Jobs())
    run_id = client.submit_serverless_python_run(
        run_name="r",
        python_file="/Workspace/p.py",
        parameters=[],
        dependencies=[],
        environment_version="5",
        timeout_seconds=60,
    )
    assert run_id == 7
    assert set(seen) <= accepted, set(seen) - accepted
