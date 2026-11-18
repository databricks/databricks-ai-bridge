"""Candidate scores must come from the candidate's own clones, which each serve one fixed model."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from databricks_agentkit.model_upgrades import optimization as opt

pytestmark = pytest.mark.route_readiness

HAIKU_CLONE = "main.agent.llm_exp_test_haiku"
SONNET_CLONE = "main.agent.llm_exp_test_sonnet"


def _fake_time(monkeypatch, **overrides):
    # Isolate the optimizer's clock from sleeps in background MLflow threads.
    clock = SimpleNamespace(
        monotonic=time.monotonic, perf_counter=time.perf_counter, sleep=time.sleep
    )
    for name, fn in overrides.items():
        setattr(clock, name, fn)
    monkeypatch.setattr(opt, "time", clock)


def _state():
    state = opt._State(
        predict_fn=Mock(return_value="answer"),
        prompt_targets=[],
        endpoint_targets=[opt._EndpointTarget("main.agent.llm", ["haiku"], "sonnet", "test")],
        scorers=[Mock(return_value=1.0)],
        weight_quality=1.0,
        weight_latency=0.0,
        weight_cost=0.0,
        latency_hard_gate=60.0,
        cost_soft_gate=0.02,
        reflection_model="unused",
        token_costs={},
    )
    opt._sync_destinations({"model:main.agent.llm": "haiku"}, state)
    return state


@pytest.fixture
def gateway(monkeypatch):
    elapsed = [0.0]
    _fake_time(
        monkeypatch,
        monotonic=lambda: elapsed[0],
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
    )
    api = Mock()
    monkeypatch.setattr(opt, "_client", lambda: SimpleNamespace(api_client=api))
    return api


def _run(monkeypatch, state, *calls):
    """Score one record whose prediction sends LLM calls to ``calls`` through the redirect."""
    state.predict_fn = Mock(side_effect=lambda inputs: state.redirected_calls.extend(calls) or "a")
    monkeypatch.setattr(opt, "_get_active_trace", lambda previous_trace_id: None)
    monkeypatch.setattr(opt, "_extract_trace_summary", lambda trace: {})
    return opt._AgentAdapter(state)._run_one({"model:main.agent.llm": "haiku"}, {}, {})


def test_new_clone_not_found_is_retried_until_it_answers(gateway):
    from databricks.sdk.errors import NotFound, TemporarilyUnavailable

    gateway.do.side_effect = [NotFound("propagating"), TemporarilyUnavailable("warming"), {}]
    opt._wait_until_serving(HAIKU_CLONE)
    assert gateway.do.call_count == 3
    assert {call.kwargs["body"]["model"] for call in gateway.do.call_args_list} == {HAIKU_CLONE}


def test_permission_errors_surface_instead_of_waiting(gateway):
    from databricks.sdk.errors import PermissionDenied

    gateway.do.side_effect = PermissionDenied("not allowed")
    with pytest.raises(PermissionDenied, match="not allowed"):
        opt._wait_until_serving(HAIKU_CLONE)


def test_a_clone_that_never_answers_times_out(gateway):
    from databricks.sdk.errors import NotFound

    gateway.do.side_effect = NotFound("propagating")
    with pytest.raises(TimeoutError, match="did not start serving"):
        opt._wait_until_serving(HAIKU_CLONE, timeout=6.0)


def test_readiness_failure_still_cleans_up_every_clone(monkeypatch):
    state = _state()
    state.active_clones.clear()
    monkeypatch.setattr(opt, "_build_state", lambda *args: state)
    monkeypatch.setattr(opt, "_client", Mock())
    monkeypatch.setattr(opt.model_services, "create", Mock())
    delete = Mock()
    monkeypatch.setattr(opt.model_services, "delete", delete)
    monkeypatch.setattr(opt, "_wait_until_serving", Mock(side_effect=TimeoutError("not serving")))

    with pytest.raises(TimeoutError, match="not serving"):
        opt.optimize_prompts_and_models(
            state.predict_fn,
            [{"inputs": {}, "expectations": {}}],
            [],
            scorers=state.scorers,
            max_metric_calls=1,
        )
    assert sorted(call.args[1] for call in delete.call_args_list) == [HAIKU_CLONE, SONNET_CLONE]


def test_a_call_to_the_active_clone_is_scored(monkeypatch):
    state = _state()
    score, _, _, _, _ = _run(monkeypatch, state, HAIKU_CLONE)
    assert score == 1.0
    assert state.predict_fn.call_count == 1


@pytest.mark.parametrize(
    "calls, error",
    [
        ((), "no LLM call was redirected"),  # the agent bypassed the OpenAI client
        ((SONNET_CLONE,), "reached " + SONNET_CLONE),  # another candidate's clone
        ((HAIKU_CLONE, SONNET_CLONE), "reached " + SONNET_CLONE),
    ],
)
def test_misrouted_or_unverifiable_predictions_never_reach_scorers(monkeypatch, calls, error):
    state = _state()
    with pytest.raises(RuntimeError, match=error):
        _run(monkeypatch, state, *calls)
    state.scorers[0].assert_not_called()


def test_calls_from_the_previous_prediction_do_not_count(monkeypatch):
    state = _state()
    state.redirected_calls.append(HAIKU_CLONE)
    with pytest.raises(RuntimeError, match="no LLM call was redirected"):
        _run(monkeypatch, state)


def test_roles_the_agent_does_not_call_are_not_required(monkeypatch):
    state = _state()
    unused = opt._EndpointTarget("main.agent.unused", ["haiku"], "sonnet", "test")
    state.endpoint_targets.append(unused)
    opt._sync_destinations(
        {"model:main.agent.llm": "haiku", "model:main.agent.unused": "haiku"}, state
    )
    score, _, _, _, _ = _run(monkeypatch, state, HAIKU_CLONE)
    assert score == 1.0


def test_a_failed_prediction_is_scored_zero_without_a_routing_check(monkeypatch):
    state = _state()
    state.predict_fn = Mock(side_effect=RuntimeError("agent crashed"))
    monkeypatch.setattr(opt, "_get_active_trace", lambda previous_trace_id: None)
    monkeypatch.setattr(opt, "_extract_trace_summary", lambda trace: {})
    score, _, _, feedback, _ = opt._AgentAdapter(state)._run_one(
        {"model:main.agent.llm": "haiku"}, {}, {}
    )
    assert score == 0.0
    assert "prediction failed" in feedback


def test_untraced_prediction_does_not_reuse_the_previous_predictions_trace(monkeypatch):
    state = _state()
    state.token_costs = {"haiku": {"input": 1.0, "output": 0.0}}
    active_id = [None]
    span = SimpleNamespace(
        name="chat",
        span_type="LLM",
        inputs={},
        outputs={"model": "haiku"},
        attributes={opt._TOKEN_USAGE_KEY: {"input_tokens": 1_000_000, "output_tokens": 0}},
    )
    trace = SimpleNamespace(data=SimpleNamespace(spans=[span]))
    monkeypatch.setattr(opt.mlflow, "get_last_active_trace_id", lambda: active_id[0])
    monkeypatch.setattr(opt.mlflow, "flush_trace_async_logging", Mock())
    get_trace = Mock(return_value=trace)
    monkeypatch.setattr(opt.mlflow, "get_trace", get_trace)
    scorer = Mock(return_value=1.0)
    monkeypatch.setattr(opt, "_run_scorers", scorer)

    def predict(inputs):
        state.redirected_calls.append(HAIKU_CLONE)
        if inputs["traced"]:
            active_id[0] = "tr-fresh"
        return "answer"

    state.predict_fn = predict
    adapter = opt._AgentAdapter(state)
    candidate = {"model:main.agent.llm": "haiku"}
    first = adapter._run_one(candidate, {"traced": True}, {})
    assert first[4]["total_tokens"]["input"] == 1_000_000
    assert scorer.call_args.kwargs["trace"] is trace
    assert adapter._last_metrics["cost_usd"] == pytest.approx(0.07)

    second = adapter._run_one(candidate, {"traced": False}, {})
    assert second[0] == 1.0  # Redirect verification works without tracing.
    assert second[4] == {}
    assert scorer.call_args.kwargs["trace"] is None
    assert adapter._last_metrics["cost_usd"] == pytest.approx(500 / 1_000_000 * 0.07)
    get_trace.assert_called_once_with("tr-fresh")
