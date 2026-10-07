"""Tests for promote_to_prod."""

from databricks_agentkit.model_upgrades import promote_to_prod
from databricks_agentkit.model_upgrades.optimization import (
    Result,
    _EndpointTarget,
    _PromptTarget,
)


def _result(candidate, *, prompt_targets=(), endpoint_targets=()):
    pts, ets = list(prompt_targets), list(endpoint_targets)
    return Result(
        best_candidate=candidate,
        best_score=0.9,
        baseline_score=0.5,
        prompt_uris=[pt.uri for pt in pts],
        gateway_endpoints={et.name: et.candidate_models for et in ets},
        prompt_targets=pts,
        endpoint_targets=ets,
        gepa_result=None,
    )


def _pt(prior_version=2):
    return _PromptTarget(
        uri="prompts:/cat.schema.foo@production",
        name="cat.schema.foo",
        alias="production",
        version=None,
        short_name="foo",
        template="hello {{name}}",
        required_vars=["name"],
        prior_version=prior_version,
    )


def test_unchanged_prompt_skipped(mocker):
    pt = _pt()
    register = mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.register_prompt"
    )
    set_alias = mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.set_prompt_alias"
    )

    out = promote_to_prod(_result({"prompt:foo": "hello {{name}}"}, prompt_targets=[pt]))
    assert out["cat.schema.foo"] is None
    register.assert_not_called()
    set_alias.assert_not_called()


def test_changed_prompt_registers_new_version_with_rollback(mocker):
    pt = _pt()
    register = mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.register_prompt",
        return_value=mocker.Mock(version=3),
    )
    set_alias = mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.set_prompt_alias"
    )

    promote_to_prod(_result({"prompt:foo": "hi {{name}}!"}, prompt_targets=[pt]))

    register.assert_called_once()
    assert register.call_args.kwargs["template"] == "hi {{name}}!"
    calls = set_alias.call_args_list
    assert calls[0].kwargs == {
        "name": "cat.schema.foo",
        "alias": "production_previous",
        "version": 2,
    }
    assert calls[1].kwargs == {
        "name": "cat.schema.foo",
        "alias": "production",
        "version": 3,
    }


def test_endpoint_unchanged_skipped(mocker):
    et = _EndpointTarget(
        name="ep1", candidate_models=["m1"], initial_model="databricks-claude-sonnet-4"
    )
    update = mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.set_model"
    )
    promote_to_prod(_result({"model:ep1": "databricks-claude-sonnet-4"}, endpoint_targets=[et]))
    update.assert_not_called()


def test_endpoint_changed_updated(mocker):
    et = _EndpointTarget(name="ep1", candidate_models=["m1", "m2"], initial_model="m1")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.m2", "display_name": "M2", "description": ""},
    )
    update = mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.set_model"
    )
    out = promote_to_prod(_result({"model:ep1": "m2"}, endpoint_targets=[et]))
    assert out["ep1"] == "updated:m1->m2"
    update.assert_called_once()
    assert update.call_args.args[1] == "ep1"
    assert update.call_args.args[1:] == ("ep1", "system.ai.m2")


def test_promote_dry_run_does_not_apply(mocker):
    et = _EndpointTarget(name="ep1", candidate_models=["m1", "m2"], initial_model="m1")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.m2", "display_name": "M2", "description": ""},
    )
    update = mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.set_model"
    )
    register = mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.register_prompt"
    )

    out = promote_to_prod(_result({"model:ep1": "m2"}, endpoint_targets=[et]), dry_run=True)
    update.assert_not_called()
    register.assert_not_called()
    assert out["ep1"] == "would-update:m1->m2"


def test_promote_rolls_back_endpoints_on_prompt_failure(mocker):
    """If endpoints succeed but a prompt registration fails, endpoints should be reverted."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1", "m2"], initial_model="m1")
    pt = _pt()
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.m2", "display_name": "M2", "description": ""},
    )

    update_calls = []

    def fake_update(client, name, model):
        update_calls.append((name, model))

    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.set_model",
        side_effect=fake_update,
    )

    mocker.patch("databricks_agentkit.model_upgrades.optimization.mlflow.genai.set_prompt_alias")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.register_prompt",
        side_effect=RuntimeError("registry down"),
    )

    import pytest

    with pytest.raises(RuntimeError, match="registry down"):
        promote_to_prod(
            _result(
                {"model:ep1": "m2", "prompt:foo": "hi {{name}}!"},
                prompt_targets=[pt],
                endpoint_targets=[et],
            )
        )

    assert ("ep1", "system.ai.m2") in update_calls
    assert len(update_calls) == 2


def test_promote_rolls_back_on_endpoint_failure(mocker):
    et1 = _EndpointTarget(name="ep1", candidate_models=["m1", "m2"], initial_model="m1")
    et2 = _EndpointTarget(name="ep2", candidate_models=["m1", "m2"], initial_model="m1")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.m2", "display_name": "M2", "description": ""},
    )

    calls = []

    def fake_update(client, name, model):
        calls.append((name, model))
        if name == "ep2" and model == "system.ai.m2":
            raise RuntimeError("gateway exploded")

    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.set_model",
        side_effect=fake_update,
    )

    import pytest

    with pytest.raises(RuntimeError, match="gateway exploded"):
        promote_to_prod(
            _result(
                {"model:ep1": "m2", "model:ep2": "m2"},
                endpoint_targets=[et1, et2],
            )
        )

    assert ("ep1", "system.ai.m2") in calls
    assert ("ep2", "system.ai.m2") in calls
    assert ("ep1", "system.ai.m2") == calls[0]
    rollback_calls = [c for c in calls if c[0] == "ep1"]
    assert len(rollback_calls) == 2
