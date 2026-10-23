"""How the optimizer drives model services: the `_exp` clone lifecycle and destination syncs."""

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
        self.last_gateway_synced = {}
        self.backend_controls = {}
        self.expected_backends = {}
        self.created_exp_endpoints = []


def test_exp_clone_name_uses_underscore_leaf():
    assert _target().exp_name == "main.agent.llm_exp_test"


def test_read_endpoint_destination_strips_schema(mocker):
    mocker.patch(f"{_MS}.get_model", return_value="system.ai.claude-sonnet-4-5")
    assert opt._read_endpoint_destination("main.agent.llm") == "claude-sonnet-4-5"


def test_ensure_exp_endpoints_creates_clone_at_seed_model(mocker):
    mocker.patch(f"{_MS}.get_model", side_effect=LookupError("missing"))
    create = mocker.patch(f"{_MS}.create")
    opt._ensure_exp_endpoints(_State([_target()]))
    _, name, model = create.call_args.args
    assert (name, model) == ("main.agent.llm_exp_test", "system.ai.claude-sonnet-4-5")
    assert "role=experimental" in create.call_args.kwargs["comment"]


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
    create = mocker.patch(f"{_MS}.create", side_effect=[None, RuntimeError("creation failed")])
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
    assert create.call_count == 2
    delete.assert_called_once_with(
        create.call_args_list[0].args[0], create.call_args_list[0].args[1]
    )


def test_sync_destinations_repoints_clone_once(mocker):
    set_model = mocker.patch(f"{_MS}.set_model")
    state = _State([_target()])
    candidate = {"model:main.agent.llm": "claude-haiku-4-5"}
    opt._sync_destinations(candidate, state, use_exp=True)
    opt._sync_destinations(candidate, state, use_exp=True)
    set_model.assert_called_once()
    assert set_model.call_args.args[1:] == ("main.agent.llm_exp_test", "system.ai.claude-haiku-4-5")


def test_cleanup_deletes_clone_and_hints_on_failure(mocker, capsys):
    mocker.patch(f"{_MS}.create")
    mocker.patch(f"{_MS}.delete", side_effect=RuntimeError("boom"))
    state = _State([_target()])
    opt._ensure_exp_endpoints(state)
    opt._cleanup_exp_endpoints(state)
    assert (
        "databricks api delete /api/2.1/unity-catalog/model-services/main.agent.llm_exp_test"
        in (capsys.readouterr().out)
    )


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
    calls = [{"model": et.exp_name, "input": 100, "output": 50}]
    cost = opt._estimate_cost_usd(
        {"model:main.agent.llm": "claude-haiku-4-5"}, [et], {}, {}, llm_calls=calls
    )
    assert cost == 0.001
    assert priced == ["claude-haiku-4-5"]


def test_cost_prices_an_uncataloged_backend_id_as_the_routed_candidate():
    # Workspaces echo different backend IDs; one the catalog lacks must not drop to the fallback
    # rate, which is below Haiku's price and would make Sonnet look cheaper than Haiku.
    et = _target()
    calls = [
        {
            "model": "workspace-specific-sonnet-id",
            "requested": et.exp_name,
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
            "requested": et.exp_name,
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
    mocker.patch(
        f"{_MS}.set_model", side_effect=lambda client, name, model: services.update({name: model})
    )
    mocker.patch(f"{_MS}.delete", side_effect=lambda client, name: services.pop(name))
    target_a = opt._EndpointTarget("main.agent.llm", ["m1", "m2"], "m1")
    target_b = opt._EndpointTarget("main.agent.llm", ["m1", "m2"], "m1")
    assert target_a.exp_name != target_b.exp_name
    state_a, state_b = _State([target_a]), _State([target_b])
    opt._ensure_exp_endpoints(state_a)
    opt._ensure_exp_endpoints(state_b)
    opt._sync_destinations({"model:main.agent.llm": "m1"}, state_a, use_exp=True)
    opt._sync_destinations({"model:main.agent.llm": "m2"}, state_b, use_exp=True)
    opt._sync_destinations({"model:main.agent.llm": "m1"}, state_a, use_exp=True)
    assert services[target_a.exp_name] == "system.ai.m1"
    assert services[target_b.exp_name] == "system.ai.m2"
    opt._cleanup_exp_endpoints(state_a)
    assert services == {target_b.exp_name: "system.ai.m2"}
