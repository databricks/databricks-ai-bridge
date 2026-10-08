"""End-to-end tests for the scoring path inside `_AgentAdapter` and its helpers.

These tests exercise the actual scoring formula, scorer dispatch, latency
gate, cost estimation, warn-once paths, and the scorer-success ratchet --
behaviors that the optimize_prompts_and_models tests mock out.
"""

import pytest

from databricks_agentkit.model_upgrades import optimization as opt
from databricks_agentkit.model_upgrades.optimization import (
    _DBU_TO_USD,
    _SCORER_WARNINGS_SEEN,
    _AgentAdapter,
    _convert_to_numeric,
    _EndpointTarget,
    _estimate_cost_usd,
    _make_bandit_proposer,
    _mlflow_model_cost,
    _read_span_model,
    _run_scorers,
    _State,
    _ucb1_select,
    _warn_once,
    optimize_prompts_and_models,
)


def _state(
    *,
    predict_fn=None,
    scorers=(),
    endpoint_targets=(),
    prompt_targets=(),
    weight_quality=0.7,
    weight_latency=0.2,
    weight_cost=0.1,
    latency_hard_gate=60.0,
    cost_soft_gate=0.02,
    token_costs=None,
):
    return _State(
        predict_fn=predict_fn or (lambda x: "ok"),
        prompt_targets=list(prompt_targets),
        endpoint_targets=list(endpoint_targets),
        scorers=list(scorers),
        weight_quality=weight_quality,
        weight_latency=weight_latency,
        weight_cost=weight_cost,
        latency_hard_gate=latency_hard_gate,
        cost_soft_gate=cost_soft_gate,
        reflection_model="reflection",
        token_costs=token_costs or {},
    )


@pytest.fixture(autouse=True)
def _clear_warning_dedup():
    """Each test sees fresh warning dedup; warn_once is module-global by design."""
    _SCORER_WARNINGS_SEEN.clear()


# ---------------------------------------------------------------------------
# _run_scorers matrix
# ---------------------------------------------------------------------------


def test_run_scorers_numeric_callable_averages():
    s1 = lambda inputs, expected, answer: 0.6
    s2 = lambda inputs, expected, answer: 0.8
    state = _state(scorers=[s1, s2])
    assert _run_scorers([s1, s2], {}, "x", "y", state=state) == pytest.approx(0.7)
    assert state.scorer_attempted == 1
    assert state.scorer_succeeded == 1


def test_run_scorers_raising_callable_warns_and_skips(recwarn):
    def bad(inputs, expected, answer):
        raise TypeError("boom")

    s_ok = lambda inputs, expected, answer: 1.0
    state = _state(scorers=[bad, s_ok])

    assert _run_scorers([bad, s_ok], {}, "x", "y", state=state) == pytest.approx(1.0)
    assert state.scorer_attempted == 1
    assert state.scorer_succeeded == 1  # at least one numeric came back
    msgs = [str(w.message) for w in recwarn.list]
    assert any("TypeError: boom" in m for m in msgs)


def test_run_scorers_all_fail_returns_zero_and_does_not_increment_succeeded(recwarn):
    def bad(inputs, expected, answer):
        raise RuntimeError("nope")

    state = _state(scorers=[bad])
    assert _run_scorers([bad], {}, "x", "y", state=state) == 0.0
    assert state.scorer_attempted == 1
    assert state.scorer_succeeded == 0


def test_run_scorers_categorical_rating_yes_no_coerced():
    from mlflow.genai.judges import CategoricalRating

    yes = lambda inputs, expected, answer: CategoricalRating.YES
    no = lambda inputs, expected, answer: CategoricalRating.NO
    state = _state(scorers=[yes, no])
    assert _run_scorers([yes, no], {}, "x", "y", state=state) == pytest.approx(0.5)


def test_run_scorers_feedback_value_unwrapped():
    from mlflow.entities import Feedback

    fb = lambda inputs, expected, answer: Feedback(name="s", value=0.42)
    state = _state(scorers=[fb])
    assert _run_scorers([fb], {}, "x", "y", state=state) == pytest.approx(0.42)


def test_run_scorers_non_numeric_warns_and_skips(recwarn):
    weird = lambda inputs, expected, answer: "not a number"
    state = _state(scorers=[weird])
    assert _run_scorers([weird], {}, "x", "y", state=state) == 0.0
    msgs = [str(w.message) for w in recwarn.list]
    assert any("returned non-numeric" in m for m in msgs)


def test_run_scorers_passes_trace_to_mlflow_scorer(mocker):
    """MLflow Scorer.run gets the trace kwarg so trace-aware judges (e.g.
    make_judge with a Trace template field) can read per-component spans."""
    from mlflow.genai.scorers.base import Scorer as MlflowScorer

    captured = {}

    class FakeScorer(MlflowScorer):
        name: str = "fake"

        def __call__(self, *args, **kwargs):  # pragma: no cover -- abstract guard
            return 1.0

        def run(self, *, inputs=None, outputs=None, expectations=None, trace=None):  # ty: ignore[invalid-method-override]
            captured["inputs"] = inputs
            captured["outputs"] = outputs
            captured["expectations"] = expectations
            captured["trace"] = trace
            return 0.9

    sentinel_trace = object()
    score = _run_scorers(
        [FakeScorer()],
        inputs={"q": "?"},
        expectations={"expected_response": "a"},
        answer="answer",
        trace=sentinel_trace,
    )
    assert score == pytest.approx(0.9)
    assert captured["inputs"] == {"q": "?"}
    assert captured["outputs"] == "answer"
    assert captured["expectations"] == {"expected_response": "a"}
    assert captured["trace"] is sentinel_trace


def test_run_scorers_trace_defaults_to_none_for_mlflow_scorer():
    """If callers don't pass trace, MLflow Scorer.run still gets trace=None
    (not a missing kwarg) -- preserves the documented Scorer.run signature."""
    from mlflow.genai.scorers.base import Scorer as MlflowScorer

    captured = {}

    class FakeScorer(MlflowScorer):
        name: str = "fake"

        def __call__(self, *args, **kwargs):  # pragma: no cover
            return 1.0

        def run(self, *, inputs=None, outputs=None, expectations=None, trace=None):  # ty: ignore[invalid-method-override]
            captured["trace"] = trace
            return 1.0

    _run_scorers(
        [FakeScorer()],
        inputs={},
        expectations={},
        answer="a",
    )
    assert "trace" in captured
    assert captured["trace"] is None


def test_warn_once_dedupes_within_process(recwarn):
    _warn_once("scorer_x", "raised", "first")
    _warn_once("scorer_x", "raised", "second")
    msgs = [str(w.message) for w in recwarn.list]
    relevant = [m for m in msgs if "scorer_x" in m]
    assert len(relevant) == 1


def test_convert_to_numeric_handles_all_types():
    from mlflow.entities import Feedback
    from mlflow.genai.judges import CategoricalRating

    assert _convert_to_numeric(0.5) == 0.5
    assert _convert_to_numeric(1) == 1.0
    assert _convert_to_numeric(True) == 1.0
    assert _convert_to_numeric(CategoricalRating.YES) == 1.0
    assert _convert_to_numeric(CategoricalRating.NO) == 0.0
    assert _convert_to_numeric(Feedback(name="s", value=0.3)) == pytest.approx(0.3)
    assert _convert_to_numeric("hi") is None
    assert _convert_to_numeric(None) is None


# ---------------------------------------------------------------------------
# _estimate_cost_usd
# ---------------------------------------------------------------------------


def test_estimate_cost_no_endpoints_is_zero():
    assert _estimate_cost_usd({}, [], {"input": 100, "output": 50}, {}) == 0.0


def test_estimate_cost_real_tokens_uses_per_model_rates():
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    token_costs = {"m1": {"input": 1000.0, "output": 2000.0}}
    cost = _estimate_cost_usd(
        {"model:ep1": "m1"},
        [et],
        {"input": 1_000_000, "output": 1_000_000},
        token_costs,
    )
    # 1M input * 1000/1M + 1M output * 2000/1M = 3000 DBU * 0.07 = 210 USD
    assert cost == pytest.approx(210.0)


def test_estimate_cost_no_tokens_uses_fallback():
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    token_costs = {"m1": {"input": 1.0, "output": 1.0}}
    cost = _estimate_cost_usd({"model:ep1": "m1"}, [et], {}, token_costs)
    # 500 in * 1/1M + 200 out * 1/1M = 0.0007 DBU * 0.07 = ~4.9e-8
    assert cost > 0
    assert cost == pytest.approx(700 / 1_000_000 * 0.07)


def test_estimate_cost_zero_input_only_still_uses_real_tokens():
    """We previously fell back to 500/200 when input was 0. Now: only fallback
    if BOTH are zero. Verifies the simplify-pass C fix."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    token_costs = {"m1": {"input": 1.0, "output": 1.0}}
    # input=0, output=200
    cost_real = _estimate_cost_usd(
        {"model:ep1": "m1"},
        [et],
        {"input": 0, "output": 200},
        token_costs,
    )
    cost_fallback = _estimate_cost_usd({"model:ep1": "m1"}, [et], {}, token_costs)
    assert cost_real != cost_fallback
    # cost_real used 0 input, 200 output: 200/1M * 0.07 = 1.4e-8
    assert cost_real == pytest.approx(200 / 1_000_000 * 0.07)


def test_estimate_cost_uses_mlflow_pricing_when_no_override():
    """No token_costs override -> price via MLflow (stubbed 1e-6 in / 3e-6 out)."""
    et = _EndpointTarget(
        name="ep1", candidate_models=["databricks-x"], initial_model="databricks-x"
    )
    cost = _estimate_cost_usd(
        {"model:ep1": "databricks-x"},
        [et],
        {"input": 1000, "output": 500},
        {},
    )
    # stub: 1000 * 1e-6 + 500 * 3e-6 = 0.001 + 0.0015 = 0.0025
    assert cost == pytest.approx(0.0025)


def test_estimate_cost_override_beats_mlflow():
    """An explicit token_costs entry takes priority over MLflow pricing."""
    et = _EndpointTarget(
        name="ep1", candidate_models=["databricks-x"], initial_model="databricks-x"
    )
    token_costs = {"databricks-x": {"input": 0.0, "output": 0.0}}  # forced free
    cost = _estimate_cost_usd(
        {"model:ep1": "databricks-x"},
        [et],
        {"input": 1000, "output": 500},
        token_costs,
    )
    assert cost == 0.0


def test_estimate_cost_falls_back_when_mlflow_cannot_price():
    """Model MLflow can't price (stub returns None for non-databricks) and no
    override -> fallback rate, not a crash."""
    et = _EndpointTarget(name="ep1", candidate_models=["mystery"], initial_model="mystery")
    cost = _estimate_cost_usd(
        {"model:ep1": "mystery"},
        [et],
        {"input": 1_000_000, "output": 0},
        {},
    )
    # fallback input rate 10 DBU/1M: 1M * 10/1M * 0.07 = 0.7
    assert cost == pytest.approx(1_000_000 * 10.0 / 1_000_000 * _DBU_TO_USD)


@pytest.mark.real_pricing
def test_mlflow_model_cost_prices_known_databricks_model():
    """The real wrapper prices a model in MLflow's offline catalog (Claude Sonnet
    is bundled, so this holds without network). Guards the databricks-provider wiring."""
    cost = _mlflow_model_cost("databricks-claude-sonnet-4", 1_000_000, 1_000_000)
    assert cost is not None
    assert cost > 0


@pytest.mark.real_pricing
def test_mlflow_model_cost_returns_none_for_endpoint_name():
    """Gateway endpoint names are not catalog models -> None (so the caller
    falls back rather than mispricing)."""
    assert _mlflow_model_cost("wanderbricks-supervisor-exp", 1000, 500) is None


@pytest.mark.real_pricing
def test_mlflow_model_cost_strips_date_snapshot_suffix():
    """FMAPI echoes a dated snapshot name; the catalog keys on the undated name.
    The dated name must still price (via the strip-and-retry), matching the base."""
    dated = _mlflow_model_cost("gpt-5.4-mini-2026-03-17", 1_000_000, 1_000_000)
    base = _mlflow_model_cost("gpt-5.4-mini", 1_000_000, 1_000_000)
    assert dated is not None and base is not None
    assert dated == base


# ---------------------------------------------------------------------------
# _estimate_cost_usd -- per-call (resolved-model) path
# ---------------------------------------------------------------------------


def test_estimate_cost_prices_per_llm_call_by_resolved_model():
    """With per-call trace data, price each call by its resolved model via MLflow
    (stub: 1e-6 in / 3e-6 out for priceable names), independent of endpoint count."""
    et = _EndpointTarget(
        name="ep1",
        candidate_models=["databricks-gpt-5-4-mini"],
        initial_model="databricks-gpt-5-4-mini",
    )
    llm_calls = [
        {"model": "gpt-5.4-mini-2026-03-17", "input": 1000, "output": 500},
        {"model": "gpt-5.4-nano-2026-03-17", "input": 200, "output": 100},
    ]
    cost = _estimate_cost_usd(
        {"model:ep1": "databricks-gpt-5-4-mini"},
        [et],
        {"input": 1300, "output": 600},
        {},
        llm_calls=llm_calls,
    )
    # call1: 1000*1e-6 + 500*3e-6 = 0.0025 ; call2: 200*1e-6 + 100*3e-6 = 0.0005
    assert cost == pytest.approx(0.0030)


def test_estimate_cost_per_call_override_by_resolved_name():
    """A token_costs override keyed by the resolved model name wins over MLflow."""
    et = _EndpointTarget(
        name="ep1",
        candidate_models=["databricks-gpt-5-4-mini"],
        initial_model="databricks-gpt-5-4-mini",
    )
    llm_calls = [{"model": "gpt-5.4-mini-2026-03-17", "input": 1000, "output": 500}]
    token_costs = {"gpt-5.4-mini-2026-03-17": {"input": 0.0, "output": 0.0}}
    cost = _estimate_cost_usd(
        {"model:ep1": "databricks-gpt-5-4-mini"},
        [et],
        {"input": 1000, "output": 500},
        token_costs,
        llm_calls=llm_calls,
    )
    assert cost == 0.0


def test_estimate_cost_per_call_falls_back_and_warns(recwarn):
    """A call whose model MLflow can't price (stub -> None) uses the fallback rate
    and warns once, rather than raising."""
    et = _EndpointTarget(name="ep1", candidate_models=["mystery"], initial_model="mystery")
    llm_calls = [{"model": "mystery-model", "input": 1_000_000, "output": 0}]
    cost = _estimate_cost_usd(
        {"model:ep1": "mystery"}, [et], {"input": 1_000_000, "output": 0}, {}, llm_calls=llm_calls
    )
    assert cost == pytest.approx(1_000_000 * 10.0 / 1_000_000 * _DBU_TO_USD)
    assert any("unpriced_model" in str(w.message) for w in recwarn.list)


def test_estimate_cost_per_call_ignores_endpoint_count():
    """Per-call pricing must NOT divide by number of endpoints (that's the fallback
    path's behavior). Two endpoints, one real call -> priced once, in full."""
    e1 = _EndpointTarget(
        name="ep1", candidate_models=["databricks-x"], initial_model="databricks-x"
    )
    e2 = _EndpointTarget(
        name="ep2", candidate_models=["databricks-y"], initial_model="databricks-y"
    )
    llm_calls = [{"model": "gpt-5.4-mini", "input": 1000, "output": 0}]
    cost = _estimate_cost_usd({}, [e1, e2], {"input": 1000, "output": 0}, {}, llm_calls=llm_calls)
    # single call priced in full: 1000 * 1e-6 = 0.001 (no /2 split)
    assert cost == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# _read_span_model
# ---------------------------------------------------------------------------


class _FakeSpan:
    def __init__(self, outputs=None, attrs=None):
        self.outputs = outputs
        self._attrs = attrs or {}

    def get_attribute(self, key):
        return self._attrs.get(key)


def test_read_span_model_prefers_resolved_outputs_model():
    span = _FakeSpan(
        outputs={"model": "gpt-5.4-mini-2026-03-17", "choices": []},
        attrs={"mlflow.llm.model": "toy-endpoint-exp"},
    )
    assert _read_span_model(span) == "gpt-5.4-mini-2026-03-17"


def test_read_span_model_falls_back_to_attr():
    span = _FakeSpan(outputs={"choices": []}, attrs={"mlflow.llm.model": "toy-endpoint-exp"})
    assert _read_span_model(span) == "toy-endpoint-exp"


def test_read_span_model_parses_json_outputs():
    span = _FakeSpan(outputs='{"model": "claude-sonnet-4-6"}')
    assert _read_span_model(span) == "claude-sonnet-4-6"


def test_read_span_model_none_when_absent():
    span = _FakeSpan(outputs={"choices": []}, attrs={})
    assert _read_span_model(span) is None


# ---------------------------------------------------------------------------
# _AgentAdapter._run_one composite-score formula
# ---------------------------------------------------------------------------


def test_run_one_composite_score(mocker):
    """Verify quality * 0.7 + latency * 0.2 + cost * 0.1 with mocked trace + scorers."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=lambda x: "answer",
        scorers=[lambda inputs, expected, answer: 1.0],  # quality = 1.0
        endpoint_targets=[et],
        token_costs={"m1": {"input": 0.0, "output": 0.0}},  # cost = 0 USD
        latency_hard_gate=10.0,
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {"input": 100, "output": 50}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    score, obj, answer, _, _ = adapter._run_one(
        {"model:ep1": "m1"},
        inputs={},
        expectations={"expected_response": "answer"},
    )
    # quality=1.0 (perfect scorer)
    # latency very small -> lat_score ~= 1.0
    # cost = 0 -> cost_score = 1.0
    # composite ~= 0.7*1 + 0.2*1 + 0.1*1 = 1.0
    assert obj["quality"] == 1.0
    assert obj["cost"] == pytest.approx(1.0)
    assert obj["latency"] > 0.99
    assert score == pytest.approx(1.0, abs=0.01)
    assert answer == "answer"


def test_run_one_latency_hard_gate(mocker):
    """Predict that takes longer than latency_hard_gate scores 0 across the board."""
    import time as _time

    def slow_predict(x):
        _time.sleep(0.05)
        return "ok"

    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=slow_predict,
        scorers=[lambda inputs, expected, answer: 1.0],
        endpoint_targets=[et],
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
        latency_hard_gate=0.01,  # 10ms gate -- the sleep blows past it
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    score, obj, _, feedback, _ = adapter._run_one(
        {"model:ep1": "m1"},
        inputs={},
        expectations={"expected_response": "ok"},
    )
    assert score == 0.0
    assert obj == {"quality": 0.0, "latency": 0.0, "cost": 0.0}
    assert feedback.startswith("REJECTED: latency")


def test_run_one_warns_when_a_record_goes_over_the_latency_gate(mocker, monkeypatch):
    """A rejection used to be silent, so a slow agent read as "already on the best models"."""
    import time as _time

    monkeypatch.setattr(opt, "_SCORER_WARNINGS_SEEN", set())
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=lambda x: _time.sleep(0.05) or "ok",
        scorers=[lambda inputs, expected, answer: 1.0],
        endpoint_targets=[et],
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
        latency_hard_gate=0.01,
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )
    with pytest.warns(UserWarning, match="over latency_hard_gate"):
        _AgentAdapter(state)._run_one(
            {"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "ok"}
        )


def test_run_one_predict_failure_yields_error_string(mocker):
    """Predict raising should produce 'ERROR: ...' answer for the scorer to see."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")

    def boom(x):
        raise ValueError("agent broke")

    seen = []

    def scorer(inputs, expected, answer):
        seen.append(answer)
        return 0.0

    state = _state(
        predict_fn=boom,
        scorers=[scorer],
        endpoint_targets=[et],
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    _, _, answer, _, _ = adapter._run_one(
        {"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "x"}
    )
    assert answer.startswith("ERROR:")
    assert "agent broke" in answer
    assert seen == [answer]


def test_run_one_cost_fallback_warns_when_no_tokens(mocker, recwarn):
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=lambda x: "ok",
        scorers=[lambda inputs, expected, answer: 1.0],
        endpoint_targets=[et],
        weight_cost=0.1,
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    adapter._run_one({"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "ok"})

    msgs = [str(w.message) for w in recwarn.list]
    assert any("no_token_usage" in m for m in msgs)


def test_run_one_no_warn_when_weight_cost_zero(mocker, recwarn):
    """If the customer has explicitly disabled the cost component, no warn."""
    et = _EndpointTarget(name="ep1", candidate_models=["m1"], initial_model="m1")
    state = _state(
        predict_fn=lambda x: "ok",
        scorers=[lambda inputs, expected, answer: 1.0],
        endpoint_targets=[et],
        weight_quality=0.8,
        weight_latency=0.2,
        weight_cost=0.0,
        token_costs={"m1": {"input": 0.0, "output": 0.0}},
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._extract_trace_summary",
        return_value={"total_tokens": {}, "spans": []},
    )

    adapter = _AgentAdapter(state)
    adapter._run_one({"model:ep1": "m1"}, inputs={}, expectations={"expected_response": "ok"})

    msgs = [str(w.message) for w in recwarn.list]
    assert not any("no_token_usage" in m for m in msgs)


# ---------------------------------------------------------------------------
# Scorer-success ratchet (raises if <10% of evals produced a numeric score)
# ---------------------------------------------------------------------------


def test_optimize_raises_when_scorer_success_under_10pct(mocker, recwarn):
    """If almost every scorer call fails, the optimize call should surface
    the issue at the end rather than silently reporting delta=0."""
    pv = mocker.Mock(template="answer the {{question}}", version=3)
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.mlflow.genai.load_prompt", return_value=pv
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.model_services.get_model",
        return_value="system.ai.m1",
    )
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.create")
    mocker.patch("databricks_agentkit.model_upgrades.optimization.model_services.delete")
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._resolve_model_info",
        return_value={"name": "system.ai.m1", "display_name": "M1", "description": ""},
    )

    # Drive _AgentAdapter.evaluate via a fake gepa.optimize that pumps records
    # through the adapter so _run_scorers fires and increments the counters.
    def fake_optimize(*, adapter, valset, **kwargs):
        for _ in range(20):
            adapter.evaluate(valset, kwargs["seed_candidate"])
        return mocker.Mock(
            best_candidate=kwargs["seed_candidate"],
            val_aggregate_scores=[0.0, 0.0],
            best_idx=1,
        )

    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization.gepa.optimize", side_effect=fake_optimize
    )

    def always_fails(inputs, expected, answer):
        raise RuntimeError("scorer broken")

    val = [{"inputs": {"question": "q"}, "expectations": {"expected_response": "a"}}]
    with pytest.raises(RuntimeError, match="produced a numeric scorer score"):
        optimize_prompts_and_models(
            predict_fn=lambda inputs: "x",
            train_data=[],
            val_data=val,
            prompt_uris=["prompts:/cat.schema.foo@production"],
            scorers=[always_fails],
            max_metric_calls=10,
        )


# ---------------------------------------------------------------------------
# UCB1 bandit model selection
# ---------------------------------------------------------------------------


def test_ucb1_cold_start_pulls_untried_arm_first():
    """Any candidate with no observations is pulled before exploiting."""
    chosen = _ucb1_select({"a": [0.9, 0.9]}, ["a", "b", "c"])
    assert chosen == "b"


def test_ucb1_exploits_clear_winner_at_equal_counts():
    """With equal pull counts, the higher-mean arm wins (bonus is equal)."""
    chosen = _ucb1_select({"a": [0.9, 0.9], "b": [0.1, 0.1]}, ["a", "b"])
    assert chosen == "a"


def test_ucb1_exploration_bonus_favors_undersampled_arm():
    """A slightly-worse but barely-sampled arm beats a well-sampled leader."""
    chosen = _ucb1_select({"a": [0.6] * 40, "b": [0.55]}, ["a", "b"])
    assert chosen == "b"


def test_ucb1_zero_exploration_is_greedy():
    """c=0 collapses UCB1 to pure exploitation of the sample mean."""
    chosen = _ucb1_select({"a": [0.6] * 40, "b": [0.55]}, ["a", "b"], c=0.0)
    assert chosen == "a"


def test_ucb1_empty_candidates_raises():
    with pytest.raises(ValueError, match="non-empty"):
        _ucb1_select({}, [])


def _bandit_state(endpoint_targets):
    return _State(
        predict_fn=lambda x: "ok",
        prompt_targets=[],
        endpoint_targets=list(endpoint_targets),
        scorers=[],
        weight_quality=0.7,
        weight_latency=0.2,
        weight_cost=0.1,
        latency_hard_gate=60.0,
        cost_soft_gate=0.02,
        reflection_model="reflection",
        token_costs={},
        model_selection="bandit",
    )


def test_bandit_proposer_picks_model_without_llm():
    """The model branch returns an allowed name and never touches the LM."""
    et = _EndpointTarget(name="ep1", candidate_models=["m_a", "m_b"], initial_model="m_a")
    adapter = _AgentAdapter(_bandit_state([et]))
    # m_a has strong history, m_b untried -> cold start should pick m_b.
    adapter._model_history["ep1"]["m_a"].extend([0.9, 0.9])
    proposer = _make_bandit_proposer(adapter, templates={})

    out = proposer({"model:ep1": "m_a"}, {}, ["model:ep1"])
    assert out == {"model:ep1": "m_b"}


def test_bandit_proposer_delegates_prompt_to_reflection_lm(mocker):
    """The prompt branch calls InstructionProposalSignature.run (the LM path)."""
    fake_run = mocker.patch(
        "gepa.strategies.instruction_proposal.InstructionProposalSignature.run",
        return_value={"new_instruction": "IMPROVED"},
    )
    # No endpoints -> no litellm import needed; LM is built lazily only if used,
    # and run() is mocked so the lazy builder is the only thing to guard.
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._make_litellm_reflection_lm",
        return_value=lambda p: "unused",
    )
    adapter = _AgentAdapter(_bandit_state([]))
    proposer = _make_bandit_proposer(
        adapter, templates={"prompt:foo": "T <curr_param> <side_info>"}
    )

    reflective = {"prompt:foo": [{"Score": "0.5"}]}
    out = proposer({"prompt:foo": "old prompt"}, reflective, ["prompt:foo"])

    assert out == {"prompt:foo": "IMPROVED"}
    assert fake_run.call_count == 1


def test_bandit_proposer_skips_prompt_with_no_reflective_data(mocker):
    """A prompt component absent from the reflective dataset is skipped, no LM call."""
    fake_run = mocker.patch(
        "gepa.strategies.instruction_proposal.InstructionProposalSignature.run",
    )
    adapter = _AgentAdapter(_bandit_state([]))
    proposer = _make_bandit_proposer(adapter, templates={"prompt:foo": "T"})

    out = proposer({"prompt:foo": "old"}, {}, ["prompt:foo"])
    assert out == {}
    fake_run.assert_not_called()


def test_bandit_proposer_coupled_dispatch_model_and_prompt_together(mocker):
    """Under the 'all' component selector GEPA passes BOTH kinds in one call; the
    proposer must return every component -- UCB1 for model:, reflection for prompt:."""
    fake_run = mocker.patch(
        "gepa.strategies.instruction_proposal.InstructionProposalSignature.run",
        return_value={"new_instruction": "IMPROVED"},
    )
    mocker.patch(
        "databricks_agentkit.model_upgrades.optimization._make_litellm_reflection_lm",
        return_value=lambda p: "unused",
    )
    et = _EndpointTarget(name="ep1", candidate_models=["m_a", "m_b"], initial_model="m_a")
    adapter = _AgentAdapter(_bandit_state([et]))
    adapter._model_history["ep1"]["m_a"].extend([0.9, 0.9])  # m_b untried -> picked
    proposer = _make_bandit_proposer(
        adapter, templates={"prompt:foo": "T <curr_param> <side_info>"}
    )

    out = proposer(
        {"prompt:foo": "old", "model:ep1": "m_a"},
        {"prompt:foo": [{"Score": "0.5"}]},
        ["prompt:foo", "model:ep1"],
    )
    assert out == {"prompt:foo": "IMPROVED", "model:ep1": "m_b"}
    assert fake_run.call_count == 1  # exactly one LM call, for the prompt only


def test_bandit_proposer_independent_per_endpoint(mocker):
    """Multiple model: components are each proposed from their own arm history."""
    et1 = _EndpointTarget(name="ep1", candidate_models=["a1", "b1"], initial_model="a1")
    et2 = _EndpointTarget(name="ep2", candidate_models=["a2", "b2"], initial_model="a2")
    adapter = _AgentAdapter(_bandit_state([et1, et2]))
    # ep1: both arms tried, a1 clearly better -> exploit a1.
    adapter._model_history["ep1"]["a1"].extend([0.9, 0.9])
    adapter._model_history["ep1"]["b1"].extend([0.1, 0.1])
    # ep2: b2 untried -> cold-start picks b2.
    adapter._model_history["ep2"]["a2"].extend([0.5])
    proposer = _make_bandit_proposer(adapter, templates={})

    out = proposer(
        {"model:ep1": "a1", "model:ep2": "a2"},
        {},
        ["model:ep1", "model:ep2"],
    )
    assert out == {"model:ep1": "a1", "model:ep2": "b2"}


def test_get_active_trace_flushes_async_logging_before_reading(monkeypatch):
    """A trace still in MLflow's async export queue reads back as corrupted, so flush first."""
    calls = []
    monkeypatch.setattr(opt.mlflow, "get_last_active_trace_id", lambda: "tr-1")
    monkeypatch.setattr(opt.mlflow, "flush_trace_async_logging", lambda: calls.append("flush"))
    monkeypatch.setattr(opt.mlflow, "get_trace", lambda trace_id: calls.append(trace_id) or "trace")
    assert opt._get_active_trace() == "trace"
    assert calls == ["flush", "tr-1"]
