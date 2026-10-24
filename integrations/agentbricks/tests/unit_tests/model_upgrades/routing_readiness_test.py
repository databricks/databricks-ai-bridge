"""Candidate scores must describe the backend that actually handled inference."""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from databricks_agentkit.model_upgrades import optimization as opt

pytestmark = pytest.mark.route_readiness


def _fake_time(monkeypatch, **overrides):
    # Swap only this module's clock: patching the shared time module would let other threads'
    # sleeps advance it (or drain it) mid-test.
    clock = SimpleNamespace(
        monotonic=time.monotonic, perf_counter=time.perf_counter, sleep=time.sleep
    )
    for name, fn in overrides.items():
        setattr(clock, name, fn)
    monkeypatch.setattr(opt, "time", clock)


def _state():
    return opt._State(
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
    monkeypatch.setattr(opt, "_set_endpoint_model", Mock())
    return api


def _trace(model, *, service="main.agent.llm_exp_test"):
    span = SimpleNamespace(inputs={"model": service}, outputs={"model": model})
    return SimpleNamespace(data=SimpleNamespace(spans=[span]))


def test_sync_waits_for_consecutive_inference_matches_not_destination_readback(gateway):
    state = _state()
    # One direct control, then mixed old/new replicas before three consecutive matches.
    identities = [
        "haiku-id",
        "sonnet-id",
        "haiku-id",
        "sonnet-id",
        "haiku-id",
        "haiku-id",
        "haiku-id",
    ]
    gateway.do.side_effect = [{"model": model} for model in identities]
    candidate = {"model:main.agent.llm": "haiku"}

    opt._sync_destinations(candidate, state, use_exp=True)

    assert gateway.do.call_count == len(identities)
    assert gateway.do.call_args_list[0].kwargs["body"]["model"] == "system.ai.haiku"
    assert {call.kwargs["body"]["model"] for call in gateway.do.call_args_list[1:]} == {
        "main.agent.llm_exp_test"
    }
    assert state.last_gateway_synced == {"main.agent.llm_exp_test": "haiku"}
    assert state.expected_backends == {"main.agent.llm_exp_test": "haiku-id"}
    opt._sync_destinations(candidate, state, use_exp=True)
    assert gateway.do.call_count == len(identities)


def test_timeout_invalidates_previous_ready_route_and_does_not_predict(gateway, monkeypatch):
    monkeypatch.setattr(opt, "_ROUTE_READY_TIMEOUT_S", 4.0, raising=False)
    state = _state()
    state.last_gateway_synced["main.agent.llm_exp_test"] = "sonnet"
    gateway.do.side_effect = lambda *args, **kwargs: {
        "model": "haiku-id" if kwargs["body"]["model"] == "system.ai.haiku" else "sonnet-id"
    }

    with pytest.raises(TimeoutError, match="main.agent.llm_exp_test.*haiku-id.*sonnet-id"):
        opt._AgentAdapter(state).evaluate(
            [{"inputs": {}, "expectations": {}}], {"model:main.agent.llm": "haiku"}
        )

    assert state.last_gateway_synced == {}
    state.predict_fn.assert_not_called()
    state.scorers[0].assert_not_called()


@pytest.mark.parametrize("response", [{}, {"model": ""}, {"model": None}])
def test_missing_control_identity_fails_closed(gateway, response):
    gateway.do.return_value = response
    state = _state()
    with pytest.raises(RuntimeError, match="backend model identity"):
        opt._sync_destinations({"model:main.agent.llm": "haiku"}, state, use_exp=True)
    assert state.last_gateway_synced == {}


def test_wrong_backend_after_readiness_is_retried_before_scoring(monkeypatch):
    state = _state()
    state.expected_backends = {"main.agent.llm_exp_test": "haiku-id"}
    monkeypatch.setattr(
        opt, "_get_active_trace", Mock(side_effect=[_trace("sonnet-id"), _trace("haiku-id")])
    )
    monkeypatch.setattr(opt, "_extract_trace_summary", lambda: {})
    wait = Mock()
    monkeypatch.setattr(opt, "_wait_for_endpoint_model", wait, raising=False)
    adapter = opt._AgentAdapter(state)

    score, _, _, _, _ = adapter._run_one({"model:main.agent.llm": "haiku"}, {}, {})

    assert score == 1.0
    assert state.predict_fn.call_count == 2
    assert state.scorers[0].call_count == 1
    assert adapter._model_history["main.agent.llm"]["haiku"] == [1.0]
    assert wait.call_args.args == ("main.agent.llm_exp_test", "haiku-id")


@pytest.mark.parametrize(
    "trace", [None, _trace(None), _trace("haiku-id", service="system.ai.sonnet")]
)
def test_unverifiable_evaluation_never_reaches_scorers(monkeypatch, trace):
    state = _state()
    state.expected_backends = {"main.agent.llm_exp_test": "haiku-id"}
    monkeypatch.setattr(opt, "_get_active_trace", lambda: trace)
    monkeypatch.setattr(opt, "_extract_trace_summary", lambda: {})

    with pytest.raises(RuntimeError, match="verify.*backend"):
        opt._AgentAdapter(state)._run_one({"model:main.agent.llm": "haiku"}, {}, {})
    state.scorers[0].assert_not_called()


def test_ready_probe_failure_still_cleans_up_experimental_service(monkeypatch):
    state = _state()
    monkeypatch.setattr(opt, "_build_state", lambda *args: state)
    monkeypatch.setattr(opt, "_client", Mock())
    monkeypatch.setattr(opt.model_services, "create", Mock())
    delete = Mock()
    monkeypatch.setattr(opt.model_services, "delete", delete)
    monkeypatch.setattr(
        opt, "_sync_destinations", Mock(side_effect=TimeoutError("route not ready"))
    )

    with pytest.raises(TimeoutError, match="route not ready"):
        opt.optimize_prompts_and_models(
            state.predict_fn,
            [{"inputs": {}, "expectations": {}}],
            [],
            scorers=state.scorers,
            max_metric_calls=1,
        )
    assert delete.call_args.args[1] == "main.agent.llm_exp_test"


def test_mixed_backends_after_probing_time_out_without_scoring(gateway, monkeypatch):
    monkeypatch.setattr(opt, "_ROUTE_READY_TIMEOUT_S", 8.0)
    state = _state()
    state.expected_backends = {"main.agent.llm_exp_test": "haiku-id"}
    gateway.do.return_value = {"model": "haiku-id"}
    monkeypatch.setattr(opt, "_get_active_trace", lambda: _trace("sonnet-id"))
    monkeypatch.setattr(opt, "_extract_trace_summary", lambda: {})
    adapter = opt._AgentAdapter(state)

    with pytest.raises(TimeoutError, match="Candidate evaluation stopped"):
        adapter._run_one({"model:main.agent.llm": "haiku"}, {}, {})
    assert state.predict_fn.call_count > 1
    state.scorers[0].assert_not_called()
    assert not adapter._model_history["main.agent.llm"]


def test_wrong_call_cannot_hide_behind_another_matching_call():
    first = _trace("sonnet-id").data.spans[0]
    second = _trace("haiku-id").data.spans[0]
    trace = SimpleNamespace(data=SimpleNamespace(spans=[first, second]))
    assert opt._candidate_route_mismatches(trace, {"main.agent.llm_exp_test": "haiku-id"}) == {
        "main.agent.llm_exp_test": "sonnet-id"
    }


def test_trace_check_allows_conditional_roles_that_are_not_called():
    assert (
        opt._candidate_route_mismatches(
            _trace("haiku-id"),
            {"main.agent.llm_exp_test": "haiku-id", "main.agent.unused_exp_test": "other-id"},
        )
        == {}
    )


def test_new_service_not_found_is_retried_but_permission_errors_surface(gateway):
    from databricks.sdk.errors import NotFound, PermissionDenied

    gateway.do.side_effect = [NotFound("route propagating"), *[{"model": "haiku-id"}] * 3]
    opt._wait_for_endpoint_model("main.agent.llm_exp_test", "haiku-id")
    assert gateway.do.call_count == 4
    gateway.do.side_effect = PermissionDenied("not allowed")
    with pytest.raises(PermissionDenied, match="not allowed"):
        opt._wait_for_endpoint_model("main.agent.llm_exp_test", "haiku-id")


def test_stale_attempt_latency_is_excluded(monkeypatch):
    state = _state()
    state.expected_backends = {"main.agent.llm_exp_test": "haiku-id"}
    clock = iter([0.0, 100.0, 110.0, 111.0])
    _fake_time(monkeypatch, perf_counter=lambda: next(clock))
    monkeypatch.setattr(
        opt, "_get_active_trace", Mock(side_effect=[_trace("sonnet-id"), _trace("haiku-id")])
    )
    monkeypatch.setattr(opt, "_extract_trace_summary", lambda: {})
    monkeypatch.setattr(opt, "_wait_for_endpoint_model", Mock())
    score, objectives, _, _, _ = opt._AgentAdapter(state)._run_one(
        {"model:main.agent.llm": "haiku"}, {}, {}
    )
    assert score == 1.0
    assert objectives["latency"] == pytest.approx(1 - 1 / 60)


def test_previous_predictions_trace_cannot_verify_an_untraced_record(monkeypatch):
    state = _state()
    state.expected_backends = {"main.agent.llm_exp_test": "haiku-id"}
    previous = _trace("haiku-id")
    previous.info = SimpleNamespace(trace_id="previous-record")
    monkeypatch.setattr(opt.mlflow, "get_last_active_trace_id", lambda: "previous-record")
    monkeypatch.setattr(opt, "_get_active_trace", lambda: previous)
    monkeypatch.setattr(opt, "_extract_trace_summary", lambda: {})
    with pytest.raises(RuntimeError, match="fresh.*trace"):
        opt._AgentAdapter(state)._run_one({"model:main.agent.llm": "haiku"}, {}, {})
    state.scorers[0].assert_not_called()


def test_request_model_attribute_and_serialized_span_payloads_are_supported():
    trace = _trace("haiku-id")
    span = trace.data.spans[0]
    span.inputs = "{}"
    span.outputs = '{"model": "haiku-id"}'
    span.attributes = {"mlflow.llm.model": "main.agent.llm_exp_test"}
    assert opt._candidate_route_mismatches(trace, {"main.agent.llm_exp_test": "haiku-id"}) == {}


def test_request_model_get_attribute_is_supported():
    span = SimpleNamespace(
        inputs={},
        outputs={"model": "haiku-id"},
        get_attribute=lambda key: "main.agent.llm_exp_test",
    )
    trace = SimpleNamespace(data=SimpleNamespace(spans=[span]))
    assert opt._candidate_route_mismatches(trace, {"main.agent.llm_exp_test": "haiku-id"}) == {}


def test_unrelated_text_and_non_string_models_do_not_hide_candidate_calls():
    spans = [
        SimpleNamespace(inputs="plain text", outputs="plain text"),
        SimpleNamespace(inputs={"model": []}, outputs={}),
        *_trace("haiku-id").data.spans,
    ]
    trace = SimpleNamespace(data=SimpleNamespace(spans=spans))
    assert opt._candidate_route_mismatches(trace, {"main.agent.llm_exp_test": "haiku-id"}) == {}
