"""How the optimizer drives model services: per-model `_exp` clones and the redirect to them."""

from __future__ import annotations

import pytest

from databricks_agentkit.model_upgrades import optimization as opt

_MS = "databricks_agentkit.model_upgrades.optimization.model_services"


def _target(name="main.agent.llm", initial="claude-sonnet-4-5", experiment_id="test"):
    return opt._EndpointTarget(
        name=name,
        candidate_models=["claude-haiku-4-5"],
        initial_model=initial,
        experiment_id=experiment_id,
    )


class _State:
    def __init__(self, targets):
        self.endpoint_targets = targets
        self.active_clones = {}
        self.created_exp_endpoints = []


SONNET_CLONE = "main.agent.llm_exp_test_claude_sonnet_4_5"
HAIKU_CLONE = "main.agent.llm_exp_test_claude_haiku_4_5"


def test_each_model_gets_its_own_clone_name():
    assert _target().exp_names == {
        "claude-sonnet-4-5": SONNET_CLONE,
        "claude-haiku-4-5": HAIKU_CLONE,
    }
    assert _target().exp_name("system.ai.claude-haiku-4-5") == HAIKU_CLONE


def test_read_endpoint_destination_strips_schema(mocker):
    mocker.patch(f"{_MS}.get_model", return_value="system.ai.claude-sonnet-4-5")
    assert opt._read_endpoint_destination("main.agent.llm") == "claude-sonnet-4-5"


def test_ensure_exp_endpoints_creates_one_clone_per_model_and_waits_for_each(mocker):
    create = mocker.patch(f"{_MS}.create")
    wait = mocker.patch(f"{opt.__name__}._wait_until_serving")
    state = _State([_target()])
    opt._ensure_exp_endpoints(state)
    created = sorted(call.args[1:] for call in create.call_args_list)
    assert created == [
        (HAIKU_CLONE, "system.ai.claude-haiku-4-5"),
        (SONNET_CLONE, "system.ai.claude-sonnet-4-5"),
    ]
    assert all("role=experimental" in c.kwargs["comment"] for c in create.call_args_list)
    assert sorted(state.created_exp_endpoints) == [HAIKU_CLONE, SONNET_CLONE]
    assert sorted(call.args[0] for call in wait.call_args_list) == [HAIKU_CLONE, SONNET_CLONE]


def test_existing_clone_is_neither_reused_nor_deleted(mocker):
    mocker.patch(f"{_MS}.create", side_effect=RuntimeError("already exists"))
    delete = mocker.patch(f"{_MS}.delete")
    state = _State([_target()])
    with pytest.raises(RuntimeError, match="already exists"):
        opt._ensure_exp_endpoints(state)
    opt._cleanup_exp_endpoints(state)
    delete.assert_not_called()


def test_partial_clone_creation_failure_cleans_up_only_created_services(mocker):
    mocker.patch(f"{_MS}.get_model", return_value="system.ai.m1")
    created = []

    def _create(client, name, model, **kwargs):
        if name.endswith("_m2") and "writer" in name:
            raise RuntimeError("creation failed")
        created.append(name)

    create = mocker.patch(f"{_MS}.create", side_effect=_create)
    delete = mocker.patch(f"{_MS}.delete")
    with pytest.raises(RuntimeError, match="creation failed"):
        opt.optimize_prompts_and_models(
            lambda inputs: "answer",
            [],
            [],
            gateway_endpoints={"main.agent.router": ["m2"], "main.agent.writer": ["m2"]},
            scorers=[lambda inputs, expectations, answer: 1.0],
            max_metric_calls=1,
        )
    assert create.call_count == 4  # router and writer, each with m1 and m2
    assert sorted(call.args[1] for call in delete.call_args_list) == sorted(created)
    assert len(created) == 3


def test_switching_candidates_redirects_without_updating_any_service(mocker):
    set_model = mocker.patch(f"{_MS}.set_model")
    state = _State([_target()])
    opt._sync_destinations({"model:main.agent.llm": "claude-haiku-4-5"}, state)
    assert state.active_clones == {"main.agent.llm": HAIKU_CLONE}
    opt._sync_destinations({"model:main.agent.llm": "claude-sonnet-4-5"}, state)
    assert state.active_clones == {"main.agent.llm": SONNET_CLONE}
    set_model.assert_not_called()


def test_cleanup_deletes_clone_and_hints_on_failure(mocker, capsys):
    mocker.patch(f"{_MS}.create")
    mocker.patch(f"{_MS}.delete", side_effect=RuntimeError("boom"))
    state = _State([_target()])
    opt._ensure_exp_endpoints(state)
    opt._cleanup_exp_endpoints(state)
    out = capsys.readouterr().out
    for clone in (SONNET_CLONE, HAIKU_CLONE):
        assert f"databricks api delete /api/2.1/unity-catalog/model-services/{clone}" in out


def test_system_ai_names_pass_through_resolution():
    # Agent Bricks agents name models as system.ai.*; resolution must not double the prefix.
    assert opt._resolve_system_ai_name("system.ai.claude-haiku-4-5") == "system.ai.claude-haiku-4-5"


def test_cost_prices_calls_to_the_service_as_the_routed_candidate(mocker):
    # Autolog records the requested name -- the model service's `_exp` clone -- not a model.
    priced = []

    def _cost(model, in_t, out_t):
        priced.append(model)
        return 0.001

    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._mlflow_model_cost", side_effect=_cost
    )
    et = _target()
    calls = [{"model": "main.agent.llm", "input": 100, "output": 50}]
    cost = opt._estimate_cost_usd(
        {"model:main.agent.llm": "claude-haiku-4-5"}, [et], {}, {}, llm_calls=calls
    )
    assert cost == 0.001
    assert priced == ["claude-haiku-4-5"]


def test_cost_prices_a_clone_as_the_one_model_it_routes_to(mocker):
    priced = []
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._mlflow_model_cost",
        side_effect=lambda model, in_t, out_t: priced.append(model) or 0.001,
    )
    et = _target()
    calls = [{"model": SONNET_CLONE, "requested": SONNET_CLONE, "input": 100, "output": 50}]
    opt._estimate_cost_usd(
        {"model:main.agent.llm": "claude-haiku-4-5"}, [et], {}, {}, llm_calls=calls
    )
    assert priced == ["claude-sonnet-4-5"]


def test_cost_prices_an_uncataloged_backend_id_as_the_routed_candidate():
    # Workspaces echo different backend IDs; one the catalog lacks must not drop to the fallback
    # rate, which is below Haiku's price and would make Sonnet look cheaper than Haiku.
    et = _target()
    calls = [
        {
            "model": "workspace-specific-sonnet-id",
            "requested": SONNET_CLONE,
            "input": 1_000_000,
            "output": 0,
        }
    ]
    cost = opt._estimate_cost_usd(
        {"model:main.agent.llm": "claude-sonnet-4-5"}, [et], {}, {}, llm_calls=calls
    )
    assert cost == opt._mlflow_model_cost("claude-sonnet-4-5", 1_000_000, 0)
    assert cost != opt._rate_cost_usd(1_000_000, 0, opt._FALLBACK_TOKEN_RATE)


def test_cost_prefers_a_priceable_backend_id_over_the_routed_candidate(mocker):
    priced = []

    def _cost(model, in_t, out_t):
        priced.append(model)
        return 0.002

    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._mlflow_model_cost", side_effect=_cost
    )
    et = _target()
    calls = [
        {
            "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "requested": SONNET_CLONE,
            "input": 100,
            "output": 50,
        }
    ]
    cost = opt._estimate_cost_usd(
        {"model:main.agent.llm": "claude-sonnet-4-5"}, [et], {}, {}, llm_calls=calls
    )
    assert cost == 0.002
    assert priced == ["global.anthropic.claude-sonnet-4-5-20250929-v1:0"]


def test_trace_summary_records_the_requested_service_for_each_llm_call(mocker):
    span = mocker.Mock(
        outputs={"model": "workspace-specific-sonnet-id"},
        start_time_ns=0,
        end_time_ns=1,
    )
    span.name = "ChatCompletion"
    span.get_attribute.side_effect = lambda key: {
        opt._TOKEN_USAGE_KEY: {"input_tokens": 10, "output_tokens": 5},
        opt._LLM_MODEL_ATTR_KEY: "main.agent.llm_exp_test",
    }.get(key)
    mocker.patch(
        f"{opt.__name__}._get_active_trace",
        return_value=mocker.Mock(data=mocker.Mock(spans=[span])),
    )
    calls = opt._extract_trace_summary()["llm_calls"]
    assert calls == [
        {
            "model": "workspace-specific-sonnet-id",
            "requested": "main.agent.llm_exp_test",
            "input": 10,
            "output": 5,
        }
    ]


def test_overlapping_runs_keep_their_models_and_cleanup_independent(mocker):
    services = {}
    mocker.patch(
        f"{_MS}.create",
        side_effect=lambda client, name, model, **kw: services.update({name: model}),
    )
    mocker.patch(f"{_MS}.delete", side_effect=lambda client, name: services.pop(name))
    target_a = opt._EndpointTarget("main.agent.llm", ["m1", "m2"], "m1")
    target_b = opt._EndpointTarget("main.agent.llm", ["m1", "m2"], "m1")
    assert set(target_a.exp_names.values()).isdisjoint(target_b.exp_names.values())
    state_a, state_b = _State([target_a]), _State([target_b])
    opt._ensure_exp_endpoints(state_a)
    opt._ensure_exp_endpoints(state_b)
    opt._cleanup_exp_endpoints(state_a)
    assert services == {
        target_b.exp_name("m1"): "system.ai.m1",
        target_b.exp_name("m2"): "system.ai.m2",
    }
