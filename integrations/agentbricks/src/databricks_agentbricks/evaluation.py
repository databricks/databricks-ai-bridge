"""Small offline-dataset evaluation runner for the managed runtime's actual HTTP agent."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from typing import Any
from uuid import uuid4

import click

from databricks_agentbricks.cli.endpoint import _authorization_header, _resolve_endpoint
from databricks_agentbricks.cli.endpoint_transport import EndpointRequest, HttpSession
from databricks_agentbricks.errors import AgentCliError


def load_cases(path: pathlib.Path) -> list[dict[str, Any]]:
    cases = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            if (
                not isinstance(case.get("inputs", {}).get("query"), str)
                or not case["inputs"]["query"].strip()
            ):
                raise ValueError("inputs.query must be a nonempty string")
            expected = case.get("expectations", {}).get("contains", [])
            if not isinstance(expected, list) or not all(
                isinstance(s, str) and s for s in expected
            ):
                raise ValueError("expectations.contains must be a list of nonempty strings")
            cases.append(case)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid evaluation case at {path}:{number}: {exc}") from exc
    if not cases:
        raise ValueError("The evaluation dataset is empty.")
    return cases


def response_text(response: Any) -> str:
    """Read only final assistant text from the managed adapter response, never tool output."""
    if not isinstance(response, dict) or response.get("status") != "completed":
        raise ValueError("Agent invocation did not complete successfully.")
    payload = response.get("output")
    if not isinstance(payload, dict) or payload.get("status") != "completed":
        raise ValueError("Agent did not produce a completed answer (it may require approval).")
    texts = []
    for message in payload.get("output", []):
        if not isinstance(message, dict) or message.get("role", message.get("type")) not in {
            "assistant",
            "ai",
        }:
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") in {"text", "output_text"}
                and isinstance(block.get("text"), str)
            )
    return "\n".join(text for text in texts if text).strip()


def invoke_case(
    url: str, query: str, *, authorization: str | None, timeout: float
) -> dict[str, Any]:
    session_id = str(uuid4())
    headers = {"Content-Type": "application/json", "X-Routing-Key": session_id}
    if authorization:
        headers["Authorization"] = authorization
    try:
        response = HttpSession().send(
            EndpointRequest(
                url=f"{url}/api/invocations",
                method="POST",
                headers=headers,
                timeout=timeout,
                body={
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "input": {"messages": [{"role": "user", "content": query}]},
                    "stream": False,
                    "background": False,
                },
                body_set=True,
            )
        )
        if response.status_code != 200:
            # Do not log an HTML auth page, response headers, or arbitrary server error bodies.
            raise ValueError(f"Agent returned HTTP {response.status_code}; inspect the agent logs.")
        return {
            "answer": response_text(response.body),
            "error": None,
            "latency_seconds": response.elapsed_seconds,
        }
    except (AgentCliError, ValueError) as exc:
        return {"answer": "", "error": str(exc), "latency_seconds": None}


def run_evaluation(
    cases: list[dict[str, Any]],
    *,
    url: str,
    authorization: str | None,
    timeout: float,
    tracking_uri: str,
    experiment: str,
) -> tuple[str, bool]:
    import mlflow
    from mlflow.genai.scorers import scorer

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    outcomes: list[dict[str, Any]] = []

    @mlflow.trace(name="evaluate_project_agent", span_type="AGENT")
    def predict_fn(query: str) -> dict[str, Any]:
        output = invoke_case(url, query, authorization=authorization, timeout=timeout)
        outcomes.append(output)
        return output

    @scorer
    def completed_answer(outputs: dict[str, Any]) -> bool:
        return not outputs.get("error") and bool(outputs.get("answer", "").strip())

    @scorer
    def expected_content(outputs: dict[str, Any], expectations: dict[str, Any]) -> bool:
        return (
            not outputs.get("error")
            and bool(outputs.get("answer", "").strip())
            and all(
                text.casefold() in outputs["answer"].casefold()
                for text in expectations.get("contains", [])
            )
        )

    # genai.evaluate logs each input/output/error and scorer result, plus aggregate metrics. The
    # HTTP route invokes the project's real agent, including its tools and instructions.
    # MLflow's default validation invokes the first case an extra time. Skip that preflight: the
    # agent may call real tools, so an unscored duplicate would create unnecessary side effects.
    key = "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION"
    previous = os.environ.get(key)
    os.environ[key] = "true"
    try:
        result = mlflow.genai.evaluate(
            data=cases,
            predict_fn=predict_fn,
            scorers=[completed_answer, expected_content],
        )
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    passed = len(outcomes) == len(cases) and all(
        not row["error"] and row["answer"] for row in outcomes
    )
    passed = passed and all(
        result.metrics.get(f"{name}/mean", 0) == 1
        for name in ("completed_answer", "expected_content")
    )
    return result.run_id, passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the project agent against a JSONL dataset."
    )
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("evals/cases.jsonl"))
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--url", help="Local agent URL (default: http://127.0.0.1:8000).")
    target.add_argument("--app", help="Databricks deployment name; requires --profile with OAuth.")
    parser.add_argument("--profile", help="Explicit Databricks OAuth profile for --app.")
    parser.add_argument("--tracking-uri", default="sqlite:///.agentbricks/evaluations.db")
    parser.add_argument("--experiment", default="agentbricks-starter-evaluation")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if args.app and not args.profile:
        parser.error("--app requires --profile <name>")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    try:
        cases = load_cases(args.data)
        url, authenticate = _resolve_endpoint(
            args.app, None if args.app else args.url or "http://127.0.0.1:8000", args.profile
        )
        authorization = _authorization_header(args.profile) if authenticate else None
        if args.tracking_uri == "sqlite:///.agentbricks/evaluations.db":
            pathlib.Path(".agentbricks").mkdir(exist_ok=True)
        run_id, passed = run_evaluation(
            cases,
            url=url,
            authorization=authorization,
            timeout=args.timeout,
            tracking_uri=args.tracking_uri,
            experiment=args.experiment,
        )
        click.echo(
            f"MLflow run: {run_id}\nTracking URI: {args.tracking_uri}\nResult: {'PASS' if passed else 'FAIL'}"
        )
        if args.tracking_uri.startswith("sqlite:"):
            click.echo(
                f"View per-case results: uv run mlflow ui --backend-store-uri {args.tracking_uri}"
            )
        return 0 if passed else 1
    except (OSError, ValueError, AgentCliError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
