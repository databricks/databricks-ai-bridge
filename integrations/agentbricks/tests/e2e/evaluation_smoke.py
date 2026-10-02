"""Real HTTP runtime + local MLflow smoke test, without model or cloud calls.

Run with full MLflow installed (either framework extra):
    uv run python tests/e2e/evaluation_smoke.py --output /tmp/agentbricks-eval-smoke

This verifies the evaluator's transport, case failure handling, extension and MLflow persistence.
Use the scaffold's evals/run.py against a real project agent separately to verify model/tool calls.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import threading
import time

import mlflow
import uvicorn

from databricks_agentbricks.evaluation import load_cases, run_evaluation
from databricks_agentkit import DurableAgentServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    calls = []
    app = DurableAgentServer()

    @app.invoke
    async def invoke(value, context):
        query = value["messages"][-1]["content"]
        calls.append({"query": query, "session_id": context.session_id})
        return {"status": "completed", "output": [{"role": "assistant", "content": query}]}

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    worker = threading.Thread(target=server.run, daemon=True)
    worker.start()
    deadline = time.monotonic() + 15
    while not server.started and worker.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("Runtime did not start")
    tracking_uri = f"sqlite:///{(args.output / 'evaluations.db').resolve()}"
    results = []
    try:
        for label, expected_pass, cases in (
            (
                "failing-expectation",
                False,
                [{"inputs": {"query": "hello"}, "expectations": {"contains": ["missing"]}}],
            ),
            (
                "passing-and-extended",
                True,
                [
                    {"inputs": {"query": query}, "expectations": {"contains": [query]}}
                    for query in ("hello", "evaluation-ready", "additional-case")
                ],
            ),
        ):
            path = args.output / f"{label}.jsonl"
            path.write_text("\n".join(json.dumps(case) for case in cases) + "\n")
            run_id, passed = run_evaluation(
                load_cases(path),
                url=f"http://127.0.0.1:{port}",
                authorization=None,
                timeout=10,
                tracking_uri=tracking_uri,
                experiment="evaluator-smoke",
            )
            assert passed == expected_pass, (label, passed)
            run = mlflow.get_run(run_id)
            mlflow.flush_trace_async_logging()
            traces = mlflow.search_traces(
                locations=[run.info.experiment_id], run_id=run_id, return_type="list"
            )
            assert len(traces) == len(cases)
            assert all(
                trace.data.spans
                and {"completed_answer", "expected_content"}
                <= {assessment.name for assessment in trace.info.assessments}
                for trace in traces
            )
            results.append(
                {"scenario": label, "run_id": run_id, "passed": passed, "metrics": run.data.metrics}
            )
        assert len({call["session_id"] for call in calls}) == len(calls)
        assert len(calls) == 4, "Each case should invoke the agent exactly once."
    finally:
        server.should_exit = True
        worker.join(timeout=10)
    (args.output / "results.json").write_text(
        json.dumps({"tracking_uri": tracking_uri, "runs": results, "calls": calls}, indent=2)
    )
    print(json.dumps(results, indent=2))  # noqa: T201 - standalone e2e report


if __name__ == "__main__":
    main()
