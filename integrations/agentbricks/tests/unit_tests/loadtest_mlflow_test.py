"""MLflow admission, bounded observation handling, coverage, and safe result summaries."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from databricks_agentbricks.clients.evaluation_client import (
    EvaluationClient,
    EvaluationFacts,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.evaluation_service import EvaluationService


def _client():
    client = Mock(spec=EvaluationClient)
    client.experiment_by_id.return_value = SimpleNamespace(tags={})
    client.get_scorer.side_effect = lambda name, experiment_id: name
    client.evaluation_run.return_value = nullcontext("evaluation-run")
    client.browser_url.return_value = "https://workspace/ml/experiments/123/runs/evaluation-run"
    client.evaluate.side_effect = lambda batch, scorer: EvaluationFacts(
        {f"{scorer}/mean": 1.0}, len(batch), 0
    )
    return client


def _observation(case_index=0, trace_id=None):
    return {
        "case_index": case_index,
        "inputs": {"query": "PRIVATE-QUESTION"},
        "outputs": "PRIVATE-ANSWER",
        "trace_id": trace_id,
    }


def test_preflight_validates_selected_experiment_scorers_and_uc_warehouse_once():
    client = _client()
    client.resolve_warehouse_id.return_value = "warehouse"
    service = EvaluationService(
        "staging",
        "123",
        scorer_names=("quality", "quality"),
        traces=True,
        warehouse_id="warehouse",
        client=client,
    )
    service.prepare()
    service.prepare()
    client.select_workspace.assert_called_once_with()
    client.get_scorer.assert_called_once_with("quality", "123")
    client.require_warehouse_for_uc_read.assert_called_once_with(
        client.experiment_by_id.return_value, "warehouse"
    )
    client.evaluate.assert_not_called()


def test_preflight_fails_without_experiment_or_scorer_and_hides_exception_content():
    client = _client()
    client.experiment_by_id.return_value = None
    with pytest.raises(AgentCliError, match="does not exist"):
        EvaluationService("staging", "123", scorer_names=("quality",), client=client).prepare()
    client.get_scorer.assert_not_called()
    client.experiment_by_id.side_effect = RuntimeError("PRIVATE-QUESTION PRIVATE-ANSWER")
    with pytest.raises(AgentCliError) as error:
        EvaluationService("staging", "123", client=client).prepare()
    assert "RuntimeError" in str(error.value)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.parametrize("expectations", [{"expected_response": "answer"}, ["answer"]])
def test_expectations_require_array_of_case_objects(expectations):
    client = _client()
    with pytest.raises(AgentCliError, match="array of objects"):
        EvaluationService("staging", "123", expectations=expectations, client=client).prepare()
    client.select_workspace.assert_not_called()


def test_finish_uses_bounded_answers_and_case_expectations_without_public_content():
    client = _client()
    service = EvaluationService(
        "staging",
        "123",
        scorer_names=("quality",),
        max_samples=2,
        expectations=[{"expected_response": "FIRST"}, {"expected_response": "SECOND"}],
        client=client,
    )
    result = service.finish([_observation(1), _observation(0), _observation(0)])
    records, scorer = client.evaluate.call_args.args
    assert scorer == "quality"
    assert len(records) == 2
    assert records[0]["outputs"] == "PRIVATE-ANSWER"
    assert records[0]["expectations"] == {"expected_response": "SECOND"}
    assert records[1]["expectations"] == {"expected_response": "FIRST"}
    assert result["sampled_successes"] == 2
    assert result["evaluation"]["scorers"][0]["coverage"] == 1
    assert result["traces"]["status"] == "disabled"
    assert "PRIVATE" not in json.dumps(result)
    assert "SECOND" not in json.dumps(result)


def test_trace_coverage_uses_explicit_ids_only_and_profiles_span_type_without_names():
    client = _client()
    client.get_trace.side_effect = [
        SimpleNamespace(
            data=SimpleNamespace(
                spans=[
                    SimpleNamespace(
                        name="PRIVATE-SPAN-NAME",
                        span_type="LLM",
                        start_time_ns=0,
                        end_time_ns=20_000_000,
                    ),
                    SimpleNamespace(
                        name="tool",
                        span_type="PRIVATE-CUSTOM-TYPE",
                        start_time_ns=0,
                        end_time_ns=10_000_000,
                    ),
                ]
            )
        ),
        None,
        PermissionError("PRIVATE-ERROR"),
    ]
    service = EvaluationService(
        "staging", "123", scorer_names=("quality",), traces=True, client=client
    )
    result = service.finish(
        [
            _observation(trace_id="a"),
            _observation(trace_id="b"),
            _observation(),
            _observation(trace_id="c"),
        ]
    )
    assert [call.args[0] for call in client.get_trace.call_args_list] == ["a", "b", "c"]
    trace = result["traces"]
    assert (
        trace["selected"],
        trace["correlated"],
        trace["fetched"],
        trace["missing"],
        trace["errors"],
        trace["coverage"],
    ) == (4, 3, 1, 3, 1, 0.25)
    assert trace["spans"] == [
        {"span_type": "LLM", "count": 1, "total_duration_ms": 20, "mean_duration_ms": 20},
        {"span_type": "UNKNOWN", "count": 1, "total_duration_ms": 10, "mean_duration_ms": 10},
    ]
    assert len(client.evaluate.call_args_list) == 2
    assert "trace" in client.evaluate.call_args_list[0].args[0][0]
    assert "trace" not in client.evaluate.call_args_list[1].args[0][0]
    assert result["evaluation"]["scorers"][0]["metrics"] == {
        "traced/quality/mean": 1,
        "untraced/quality/mean": 1,
    }
    assert "PRIVATE" not in json.dumps(result)


def test_one_scorer_failure_keeps_other_scores_and_performance_result_available():
    client = _client()
    client.evaluate.side_effect = [
        RuntimeError("PRIVATE-ANSWER"),
        EvaluationFacts({"good/mean": 0.5}, 1, 0),
    ]
    result = EvaluationService(
        "staging", "123", scorer_names=("bad", "good"), client=client
    ).finish([_observation(), _observation()])
    assert result["evaluation"]["status"] == "partial"
    bad, good = result["evaluation"]["scorers"]
    assert (bad["errors"], bad["coverage"], bad["status"]) == (2, 0, "failed")
    assert (good["scored"], good["coverage"], good["status"]) == (1, 0.5, "partial")
    assert "PRIVATE" not in json.dumps(result)


def test_no_samples_does_not_create_evaluation_run_or_report_zero_coverage():
    client = _client()
    result = EvaluationService(
        "staging", "123", scorer_names=("quality",), traces=True, client=client
    ).finish([])
    assert result["evaluation"]["status"] == result["traces"]["status"] == "no_samples"
    assert result["traces"]["coverage"] is None
    client.evaluation_run.assert_not_called()


def test_native_evaluate_counts_rows_not_feedbacks_and_omits_rationales(monkeypatch, capsys):
    import mlflow.genai

    frame = pd.DataFrame(
        {
            "first/value": [True, False, None, True],
            "second/value": [1, 0, None, None],
            "first/error_message": [None, None, "PRIVATE-ERROR", None],
            "first/rationale": ["PRIVATE-RATIONALE"] * 4,
        }
    )
    records = [{"inputs": {"query": "PRIVATE"}, "outputs": "PRIVATE"}]
    observed = []

    def evaluate(**kwargs):
        observed.append(kwargs)
        sys.stdout.write("PRIVATE-ANSWER")  # native display must not leak into public logs
        logging.getLogger("mlflow.test").error("PRIVATE-ERROR")
        return SimpleNamespace(
            metrics={"quality/mean": 0.5, "bad": float("nan"), "rationale": "PRIVATE"},
            result_df=frame,
        )

    monkeypatch.setattr(mlflow.genai, "evaluate", evaluate)
    previous_logging = logging.root.manager.disable
    facts = EvaluationClient("staging").evaluate(records, "quality")
    assert observed == [{"data": records, "scorers": ["quality"]}]
    assert facts == EvaluationFacts({"quality/mean": 0.5}, 3, 1)
    assert logging.root.manager.disable == previous_logging
    assert "PRIVATE" not in capsys.readouterr().out


def test_unknown_native_coverage_remains_unknown(monkeypatch):
    import mlflow.genai

    monkeypatch.setattr(
        mlflow.genai, "evaluate", lambda **kwargs: SimpleNamespace(metrics={}, result_df=None)
    )
    assert EvaluationClient("staging").evaluate([], "quality") == EvaluationFacts({}, None, None)
    client = _client()
    client.evaluate.return_value = EvaluationFacts({}, None, None)
    client.evaluate.side_effect = None
    result = EvaluationService("staging", "123", scorer_names=("quality",), client=client).finish(
        [_observation()]
    )
    assert result["evaluation"]["scorers"][0]["coverage"] is None
    assert result["evaluation"]["scorers"][0]["scored"] is None
