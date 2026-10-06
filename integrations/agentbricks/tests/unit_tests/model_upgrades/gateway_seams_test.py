"""How the optimizer drives model services: the `_exp` clone lifecycle and destination syncs."""

from __future__ import annotations

from databricks_agentkit.model_upgrades import optimization as opt

_MS = "databricks_agentkit.model_upgrades.optimization.model_services"


def _target(name="main.agent.llm", initial="claude-sonnet-4-5"):
    return opt._EndpointTarget(name=name, candidate_models=["claude-haiku-4-5"], initial_model=initial)


class _State:
    def __init__(self, targets):
        self.endpoint_targets = targets
        self.last_gateway_synced = {}


def test_exp_clone_name_uses_underscore_leaf():
    assert _target().exp_name == "main.agent.llm_exp"


def test_read_endpoint_destination_strips_schema(mocker):
    mocker.patch(f"{_MS}.get_model", return_value="system.ai.claude-sonnet-4-5")
    assert opt._read_endpoint_destination("main.agent.llm") == "claude-sonnet-4-5"


def test_ensure_exp_endpoints_creates_clone_at_seed_model(mocker):
    mocker.patch(f"{_MS}.get_model", side_effect=LookupError("missing"))
    create = mocker.patch(f"{_MS}.create")
    opt._ensure_exp_endpoints(_State([_target()]))
    _, name, model = create.call_args.args
    assert (name, model) == ("main.agent.llm_exp", "system.ai.claude-sonnet-4-5")
    assert "role=experimental" in create.call_args.kwargs["comment"]


def test_ensure_exp_endpoints_reuses_existing_clone(mocker):
    mocker.patch(f"{_MS}.get_model", return_value="system.ai.claude-sonnet-4-5")
    create = mocker.patch(f"{_MS}.create")
    opt._ensure_exp_endpoints(_State([_target()]))
    create.assert_not_called()


def test_sync_destinations_repoints_clone_once(mocker):
    set_model = mocker.patch(f"{_MS}.set_model")
    state = _State([_target()])
    candidate = {"model:main.agent.llm": "claude-haiku-4-5"}
    opt._sync_destinations(candidate, state, use_exp=True)
    opt._sync_destinations(candidate, state, use_exp=True)
    set_model.assert_called_once()
    assert set_model.call_args.args[1:] == ("main.agent.llm_exp", "system.ai.claude-haiku-4-5")


def test_cleanup_deletes_clone_and_hints_on_failure(mocker, capsys):
    mocker.patch(f"{_MS}.delete", side_effect=RuntimeError("boom"))
    opt._cleanup_exp_endpoints(_State([_target()]))
    assert "databricks api delete /api/2.1/unity-catalog/model-services/main.agent.llm_exp" in (
        capsys.readouterr().out
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

    mocker.patch("databricks_agentkit.model_upgrades.optimization._mlflow_model_cost", side_effect=_cost)
    et = _target()
    calls = [{"model": "main.agent.llm_exp", "input": 100, "output": 50}]
    cost = opt._estimate_cost_usd(
        {"model:main.agent.llm": "claude-haiku-4-5"}, [et], {}, {}, llm_calls=calls
    )
    assert cost == 0.001
    assert priced == ["claude-haiku-4-5"]
