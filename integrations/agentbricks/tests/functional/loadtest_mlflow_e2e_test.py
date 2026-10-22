"""Native SQLite MLflow registration, trace readback, and judging of captured outputs."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from pathlib import Path

import mlflow
import pytest
from mlflow.genai.scorers import scorer

from databricks_agentbricks.clients.loadtest_mlflow_client import MLflowLoadTestClient
from databricks_agentbricks.services.invoke.loadtest import run_loadtest
from databricks_agentbricks.services.invoke.loadtest_mlflow import LoadTestMLflowService
from databricks_agentbricks.services.invoke.transport import EndpointRequest


def test_registered_scorer_evaluates_existing_traced_and_untraced_answers(
    tmp_path: Path, monkeypatch
):
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("loadtest-mlflow-e2e")
    mlflow.set_experiment(experiment_id=experiment_id)

    @scorer
    def matches_reference(outputs, expectations, trace=None):
        return outputs == expectations["expected_response"] and (
            trace is None or len(trace.data.spans) > 0
        )

    matches_reference.register(name="matches_reference", experiment_id=experiment_id)

    class LocalClient(MLflowLoadTestClient):
        def select_workspace(self):
            self.module.set_tracking_uri(tracking_uri)

        def browser_url(self, *args, **kwargs):
            return None

    service = LoadTestMLflowService(
        None,
        experiment_id,
        traces=True,
        max_requests=4,
        scorer_names=("matches_reference",),
        expectations=[{"expected_response": "PRIVATE-ANSWER"}] * 2,
        client=LocalClient(None),
    )
    seen = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            output = "PRIVATE-ANSWER" if body["correct"] else "WRONG-ANSWER"
            trace_id = None
            if body["correct"]:
                with mlflow.start_span(name="PRIVATE-root-name", span_type="CHAIN") as span:
                    span.set_inputs(body)
                    span.set_outputs(output)
                    trace_id = span.trace_id
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if trace_id:
                self.send_header("X-MLflow-Trace-ID", trace_id)
            self.end_headers()
            self.wfile.write(json.dumps({"output": output}).encode())

    cases = [
        {"query": "PRIVATE-QUESTION", "correct": True},
        {"query": "PRIVATE-QUESTION", "correct": False},
    ]
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            load = run_loadtest(
                EndpointRequest(
                    f"http://127.0.0.1:{server.server_port}/run",
                    "POST",
                    {"Content-Type": "application/json"},
                    cases[0],
                    3,
                    body_set=True,
                ),
                concurrency=2,
                duration=15,
                max_requests=6,
                cases=cases,
                output="text",
                per_user_routing=False,
                mlflow_service=service,
                report_dir=tmp_path / "report",
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)
    assert load.returncode == 0
    assert len(seen) == 7  # One excluded readiness call + six measured calls; judging adds none.
    assert load.summary["counts"]["succeeded"] == 6
    result = load.summary["mlflow"]
    assert result["sampled_successes"] == 4
    assert result["eligible_successes"] == 6

    assert result["evaluation"]["status"] == "completed", result
    outcome = result["evaluation"]["scorers"][0]
    assert outcome["scored"] == 4
    assert outcome["coverage"] == 1
    assert outcome["metrics"]["traced/matches_reference/mean"] == 1
    assert outcome["metrics"]["untraced/matches_reference/mean"] == 0
    assert result["traces"]["fetched"] == 2
    assert result["traces"]["coverage"] == 0.5
    assert result["traces"]["spans"][0]["span_type"] == "CHAIN"
    run = mlflow.get_run(result["evaluation"]["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["traced/matches_reference/mean"] == 1
    assert run.data.metrics["untraced/matches_reference/mean"] == 0
    assert json.loads((tmp_path / "report" / "summary.json").read_text())["mlflow"] == result
    assert "PRIVATE" not in json.dumps(result)
    assert "WRONG-ANSWER" not in json.dumps(result)
