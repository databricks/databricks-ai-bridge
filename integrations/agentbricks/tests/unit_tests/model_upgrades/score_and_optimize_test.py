"""Tests for score and optimize_prompts_and_models wrappers."""

import pytest

from databricks_agentkit.model_upgrades import optimize_prompts_and_models, score


@pytest.fixture
def fake_predict():
    return lambda inputs: f"answer for {inputs}"


@pytest.fixture
def fake_scorers():
    return [lambda inputs, expected, answer: 1.0]


def test_score_runs_predict_over_val_data(fake_predict, fake_scorers):
    val = [
        {"inputs": {"x": 1}, "expectations": {"expected_response": "y"}},
        {"inputs": {"x": 2}, "expectations": {"expected_response": "z"}},
    ]
    assert score(fake_predict, val, scorers=fake_scorers) == pytest.approx(1.0)


def test_score_returns_zero_on_empty(fake_predict, fake_scorers):
    assert score(fake_predict, [], scorers=fake_scorers) == 0.0


def test_score_requires_scorers(fake_predict):
    with pytest.raises(TypeError):
        score(fake_predict, [])  # ty: ignore[missing-argument]


def test_optimize_prompts_and_models_requires_at_least_one_target(fake_predict, fake_scorers):
    with pytest.raises(ValueError, match="prompt_uris or gateway_endpoints"):
        optimize_prompts_and_models(
            fake_predict,
            [],
            [],
            scorers=fake_scorers,
            max_metric_calls=10,
        )


def test_optimize_prompts_and_models_requires_scorers(fake_predict):
    with pytest.raises(TypeError):
        optimize_prompts_and_models(fake_predict, [], [], max_metric_calls=10)  # ty: ignore[missing-argument]


def test_optimize_prompts_and_models_rejects_unbalanced_weights(fake_predict, fake_scorers):
    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        optimize_prompts_and_models(
            fake_predict,
            [],
            [],
            prompt_uris=["prompts:/cat.schema.foo@production"],
            scorers=fake_scorers,
            max_metric_calls=10,
            weight_quality=1.0,
            weight_latency=0.5,
            weight_cost=0.5,
        )


def test_optimize_prompts_and_models_rejects_negative_weights(fake_predict, fake_scorers):
    with pytest.raises(ValueError, match="weight_latency must be >= 0"):
        optimize_prompts_and_models(
            fake_predict,
            [],
            [],
            prompt_uris=["prompts:/cat.schema.foo@production"],
            scorers=fake_scorers,
            max_metric_calls=10,
            weight_quality=1.2,
            weight_latency=-0.1,
            weight_cost=-0.1,
        )


def test_optimize_prompts_and_models_no_preflight_cost_raise(mocker, fake_predict, fake_scorers):
    """Unpriceable candidate models no longer raise at build time -- cost is priced
    at runtime by the resolved model, so priceability can't be known up front."""
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.get_model",
        return_value="system.ai.unknown-model-x",
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.create")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.delete")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.set_model")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.unknown-model-x", "display_name": "X", "description": ""},
    )
    fake_gepa = mocker.patch("databricks_agentkit.model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={"model:ep1": "unknown-model-x"},
        val_aggregate_scores=[0.5, 0.6],
        best_idx=1,
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization._AgentAdapter")

    # Should NOT raise, even with weight_cost > 0 and no token_costs for the model.
    optimize_prompts_and_models(
        fake_predict,
        [],
        [{"inputs": {}, "expectations": {"expected_response": "x"}}],
        gateway_endpoints={"ep1": ["unknown-model-x", "another-unknown"]},
        scorers=fake_scorers,
        max_metric_calls=10,
    )


def test_optimize_prompts_and_models_accepts_token_costs_for_unknown_model(
    mocker, fake_predict, fake_scorers
):
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.get_model",
        return_value="system.ai.custom-model",
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.create")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.delete")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.set_model")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.custom-model", "display_name": "X", "description": ""},
    )
    fake_gepa = mocker.patch("databricks_agentkit.model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={"model:ep1": "custom-model"},
        val_aggregate_scores=[0.5, 0.6],
        best_idx=1,
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization._AgentAdapter")

    # Should not raise.
    optimize_prompts_and_models(
        fake_predict,
        [],
        [{"inputs": {}, "expectations": {"expected_response": "x"}}],
        gateway_endpoints={"ep1": ["custom-model"]},
        scorers=fake_scorers,
        max_metric_calls=10,
        token_costs={"custom-model": {"input": 1.0, "output": 5.0}},
    )


def test_optimize_prompts_and_models_threads_inputs_to_gepa_optimize(
    mocker, fake_predict, fake_scorers
):
    """End-to-end mock: prompt loading, endpoint reads, exp lifecycle, gepa.optimize."""
    pv = mocker.Mock(template="Answer the {{question}} succinctly.", version=3)
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv
    )

    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.get_model",
        return_value="system.ai.databricks-claude-sonnet-4",
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.create")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.delete")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.set_model")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.x", "display_name": "X", "description": ""},
    )

    fake_gepa = mocker.patch("databricks_agentkit.model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={
            "prompt:foo": "Answer the {{question}}.",
            "model:ep1": "databricks-gpt-5-4-mini",
        },
        val_aggregate_scores=[0.50, 0.85],
        best_idx=1,
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization._AgentAdapter")

    result = optimize_prompts_and_models(
        fake_predict,
        [],
        [{"inputs": {"question": "q"}, "expectations": {"expected_response": "a"}}],
        prompt_uris=["prompts:/cat.schema.foo@production"],
        gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini", "databricks-claude-sonnet-4"]},
        scorers=fake_scorers,
        max_metric_calls=10,
    )

    assert fake_gepa.optimize.call_count == 1
    kwargs = fake_gepa.optimize.call_args.kwargs
    assert kwargs["seed_candidate"] == {
        "prompt:foo": "Answer the {{question}} succinctly.",
        "model:ep1": "databricks-claude-sonnet-4",
    }
    assert kwargs["max_metric_calls"] == 10
    assert kwargs["reflection_lm"] == "databricks/databricks-claude-opus-5-5"
    assert "prompt:foo" in kwargs["reflection_prompt_template"]
    assert "model:ep1" in kwargs["reflection_prompt_template"]
    # Default model_selection="bandit" wires a custom proposer for model choices.
    assert callable(kwargs["custom_candidate_proposer"])
    assert result.prompt_uris == ["prompts:/cat.schema.foo@production"]
    assert result.gateway_endpoints == {
        "ep1": ["databricks-gpt-5-4-mini", "databricks-claude-sonnet-4"]
    }
    assert result.baseline_score == 0.50
    assert result.best_score == 0.85


def _mock_optimize_env(mocker):
    """Patch the endpoint/prompt/gepa plumbing so optimize runs offline."""
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.get_model",
        return_value="system.ai.databricks-claude-sonnet-4",
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.create")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.delete")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.set_model")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.x", "display_name": "X", "description": ""},
    )
    fake_gepa = mocker.patch("databricks_agentkit.model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={"model:ep1": "databricks-gpt-5-4-mini"},
        val_aggregate_scores=[0.5, 0.6],
        best_idx=1,
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization._AgentAdapter")
    return fake_gepa


def test_reflection_mode_omits_custom_proposer(mocker, fake_predict, fake_scorers):
    """model_selection='reflection' keeps the original LLM-driven model path."""
    fake_gepa = _mock_optimize_env(mocker)
    optimize_prompts_and_models(
        fake_predict,
        [],
        [{"inputs": {}, "expectations": {"expected_response": "x"}}],
        gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini", "databricks-claude-sonnet-4"]},
        scorers=fake_scorers,
        max_metric_calls=10,
        model_selection="reflection",
    )
    kwargs = fake_gepa.optimize.call_args.kwargs
    assert "custom_candidate_proposer" not in kwargs
    assert "model:ep1" in kwargs["reflection_prompt_template"]


def test_invalid_model_selection_rejected(mocker, fake_predict, fake_scorers):
    with pytest.raises(ValueError, match="model_selection"):
        optimize_prompts_and_models(
            fake_predict,
            [],
            [{"inputs": {}, "expectations": {"expected_response": "x"}}],
            gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini"]},
            scorers=fake_scorers,
            max_metric_calls=10,
            model_selection="ucb2",
        )


def test_patched_endpoints_rewrites_known_models_and_restores():
    """The context manager should rewrite `model=<ep>` to `<ep>_exp` and restore on exit."""
    Completions = pytest.importorskip("openai.resources.chat.completions").Completions

    from databricks_agentkit.model_upgrades.optimization import _EndpointTarget, _patched_endpoints

    seen = []
    original_create = Completions.create

    def fake_create(self, *args, **kwargs):
        seen.append(kwargs.get("model"))
        return "ok"

    Completions.create = fake_create
    try:
        targets = [
            _EndpointTarget(name="wb-supervisor", candidate_models=["m"], initial_model="m"),
            _EndpointTarget(name="wb-rewriter", candidate_models=["m"], initial_model="m"),
        ]
        with _patched_endpoints(targets):
            Completions.create(None, model="wb-supervisor", messages=[])
            Completions.create(None, model="wb-rewriter", messages=[])
            Completions.create(None, model="some-other-endpoint", messages=[])
        assert seen == [targets[0].exp_name, targets[1].exp_name, "some-other-endpoint"]
        # After exit, the patch is removed and our fake is back at the top.
        assert Completions.create is fake_create
    finally:
        Completions.create = original_create


def test_patched_endpoints_covers_responses_api():
    """Responses.create should be patched alongside Completions.create."""
    Responses = pytest.importorskip("openai.resources.responses").Responses

    from databricks_agentkit.model_upgrades.optimization import _EndpointTarget, _patched_endpoints

    seen = []
    original = Responses.create

    def fake_create(self, *args, **kwargs):
        seen.append(kwargs.get("model"))
        return "ok"

    Responses.create = fake_create
    try:
        targets = [_EndpointTarget(name="wb-supervisor", candidate_models=["m"], initial_model="m")]
        with _patched_endpoints(targets):
            Responses.create(None, model="wb-supervisor", input=[])
            Responses.create(None, model="some-other", input=[])
        assert seen == [targets[0].exp_name, "some-other"]
        assert Responses.create is fake_create
    finally:
        Responses.create = original


def test_preflight_runs_predict_once_before_gepa(mocker, fake_scorers):
    """Pre-flight should call predict_fn on the first record before launching gepa."""
    pv = mocker.Mock(template="answer the {{question}}", version=3)
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.get_model",
        return_value="system.ai.databricks-gpt-5-4-mini",
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.create")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.delete")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.set_model")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.x", "display_name": "X", "description": ""},
    )

    fake_gepa = mocker.patch("databricks_agentkit.model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={
            "prompt:foo": "answer the {{question}}",
            "model:ep1": "databricks-gpt-5-4-mini",
        },
        val_aggregate_scores=[0.5, 0.5],
        best_idx=1,
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization._AgentAdapter")

    calls = []

    def predict(inputs):
        calls.append(inputs)
        return "ok"

    optimize_prompts_and_models(
        predict,
        [{"inputs": {"question": "first"}, "expectations": {"expected_response": "a"}}],
        [],
        prompt_uris=["prompts:/cat.schema.foo@production"],
        gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini"]},
        scorers=fake_scorers,
        max_metric_calls=10,
    )
    assert calls == [{"question": "first"}]


def _patch_for_preflight(mocker):
    pv = mocker.Mock(template="answer the {{question}}", version=3)
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.get_model",
        return_value="system.ai.databricks-gpt-5-4-mini",
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.create")
    delete = mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.delete")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.set_model")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.x", "display_name": "X", "description": ""},
    )
    fake_gepa = mocker.patch("databricks_agentkit.model_upgrades.optimization.gepa")
    fake_gepa.optimize.return_value = mocker.Mock(
        best_candidate={
            "prompt:foo": "answer the {{question}}",
            "model:ep1": "databricks-gpt-5-4-mini",
        },
        val_aggregate_scores=[0.0, 0.0],
        best_idx=1,
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization._AgentAdapter")
    return fake_gepa, delete


_THREE = [
    {"inputs": {"question": f"q{i}"}, "expectations": {"expected_response": "a"}} for i in range(3)
]


def test_preflight_tolerates_one_failing_record(mocker, fake_scorers, capsys):
    """A hard first record (or a transient error) warns, and the run goes ahead once one passes."""
    fake_gepa, _ = _patch_for_preflight(mocker)

    def flaky(inputs):
        if inputs["question"] == "q0":
            raise ValueError("hard record")
        return "ok"

    optimize_prompts_and_models(
        flaky,
        _THREE,
        [],
        prompt_uris=["prompts:/cat.schema.foo@production"],
        gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini"]},
        scorers=fake_scorers,
        max_metric_calls=10,
    )
    fake_gepa.optimize.assert_called_once()
    out = capsys.readouterr().out
    assert "WARN" in out and "hard record" in out and "Pre-flight passed" in out


def test_preflight_aborts_when_every_record_fails(mocker, fake_scorers):
    """A systemic failure (a missing prompt, broken auth) stops the run before GEPA spends hours
    scoring every candidate 0 on quality, and still cleans up the _exp copies."""
    fake_gepa, delete = _patch_for_preflight(mocker)

    def broken(inputs):
        raise ValueError("Prompt with name wb_scratch_enrichment does not exist")

    with pytest.raises(RuntimeError, match="failed on all 3 pre-flight records"):
        optimize_prompts_and_models(
            broken,
            _THREE,
            [],
            prompt_uris=["prompts:/cat.schema.foo@production"],
            gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini"]},
            scorers=fake_scorers,
            max_metric_calls=10,
        )
    fake_gepa.optimize.assert_not_called()
    delete.assert_called()


def test_patched_endpoints_no_targets_is_noop():
    """Passing an empty target list should not touch the OpenAI client."""
    Completions = pytest.importorskip("openai.resources.chat.completions").Completions

    from databricks_agentkit.model_upgrades.optimization import _patched_endpoints

    before = Completions.create
    with _patched_endpoints([]):
        assert Completions.create is before
    assert Completions.create is before


def test_optimize_prompts_and_models_cleans_up_exp_endpoints_on_failure(
    mocker, fake_predict, fake_scorers
):
    pv = mocker.Mock(template="x {{var}}", version=1)
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.get_model",
        return_value="system.ai.databricks-gpt-5-4-mini",
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.create")
    delete = mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.delete")

    fake_gepa = mocker.patch("databricks_agentkit.model_upgrades.optimization.gepa")
    fake_gepa.optimize.side_effect = RuntimeError("boom")
    mocker.patch("databricks_agentkit.model_upgrades.optimization._AgentAdapter")

    with pytest.raises(RuntimeError, match="boom"):
        optimize_prompts_and_models(
            fake_predict,
            [],
            [],
            prompt_uris=["prompts:/cat.schema.foo@production"],
            gateway_endpoints={"ep1": ["databricks-gpt-5-4-mini"]},
            scorers=fake_scorers,
            max_metric_calls=5,
        )
    assert delete.call_args.args[1].startswith("ep1_exp_")
