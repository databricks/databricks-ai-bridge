"""Run Locust separately so its gevent patches do not affect the CLI."""

from __future__ import annotations

import json
import math
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from uuid import uuid4

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke.loadtest_mlflow import LoadTestMLflowService
from databricks_agentbricks.services.invoke.loadtest_streaming import StreamMetrics
from databricks_agentbricks.services.invoke.transport import EndpointRequest, HttpSession


@dataclass(frozen=True)
class LoadtestResult:
    returncode: int
    summary: dict[str, Any]


def run_loadtest(
    request: EndpointRequest,
    *,
    concurrency: int,
    duration: int,
    output: str,
    per_user_routing: bool,
    max_requests: int | None = None,
    cases: list[Any] | None = None,
    report_dir: Path | None = None,
    follow_events: bool = False,
    mlflow_service: LoadTestMLflowService | None = None,
) -> LoadtestResult:
    """Delegate scheduling, statistics, and failure reporting to Locust."""
    if find_spec("locust") is None:
        raise AgentCliError(
            "Locust is required for --loadtest.",
            hint="Install the loadtest extra: pip install 'databricks-agentbricks[loadtest]'",
        )
    if follow_events and (
        request.method.upper() != "POST"
        or any(
            not isinstance(body, dict)
            or body.get("background") is not True
            or body.get("stream") is not True
            or not isinstance(body.get("id"), str)
            or not body["id"]
            for body in (cases if cases is not None else [request.body])
        )
    ):
        raise AgentCliError(
            "--follow-events requires POST bodies with an id, background=true and stream=true."
        )
    if mlflow_service is not None:
        mlflow_service.prepare()
    if report_dir is not None:
        report_dir = report_dir.resolve()
        if report_dir.exists() and (not report_dir.is_dir() or any(report_dir.iterdir())):
            raise AgentCliError("--report-dir must be a new or empty directory.")
        report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # One excluded readiness request catches bad URLs/auth/payloads before generating load.
    body = cases[0] if cases is not None else request.body
    probe = replace(
        request,
        body=json.loads(json.dumps(body).replace("{{uuid}}", str(uuid4()))),
        body_set=cases is not None or request.body_set,
    )
    if follow_events:
        import requests

        from databricks_agentbricks.services.invoke.loadtest_request import perform_request

        try:
            with requests.Session() as session:
                session.headers.update(probe.headers)
                perform_request(
                    session,
                    asdict(probe),
                    json.dumps(probe.body),
                    time.perf_counter(),
                    follow_events=True,
                )
        except Exception as exc:
            raise AgentCliError(f"Load test preflight failed: {type(exc).__name__}.") from exc
    else:
        response = HttpSession(follow_redirects=False).send(probe)
        if not 200 <= response.status_code < 300:
            raise AgentCliError(f"Load test preflight failed: HTTP {response.status_code}.")
        if request.sse and response.events:
            stream = StreamMetrics()
            try:
                for event in response.events:
                    stream.observe(event["data"], event.get("event", ""))
                stream.finish()
            except ValueError as exc:
                raise AgentCliError(f"Load test preflight failed: {exc}.") from exc
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(Path(__file__).with_name("_locustfile.py")),
        "--headless",
        "--users",
        str(concurrency),
        "--spawn-rate",
        str(concurrency),
        "--run-time",
        f"{duration}s",
        "--stop-timeout",
        str(math.ceil(request.timeout)),
    ]
    if output == "json":
        command.append("--json")
    if report_dir is not None:
        command.extend(
            [
                "--html",
                str(report_dir / "locust.html"),
                "--csv",
                str(report_dir / "locust"),
                "--csv-full-history",
            ]
        )
    with TemporaryDirectory(prefix="agentbricks-loadtest-") as temporary:
        summary_file = Path(temporary) / "summary.json"
        observations_file = Path(temporary) / "observations.json"
        config = {
            "request": asdict(request),
            "per_user_routing": per_user_routing,
            "concurrency": concurrency,
            "duration": duration,
            "max_requests": max_requests,
            "cases": cases,
            "summary_file": str(summary_file),
            "follow_events": follow_events,
            "mlflow_max_requests": mlflow_service.max_requests if mlflow_service else 0,
            "observations_file": str(observations_file),
        }
        # A separate process group lets Ctrl-C stop dispatch and drain before reporting.
        # Credentials and payloads travel only through stdin, never arguments or files.
        with subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as process:
            cancelled = False
            previous_terminate = signal.signal(
                signal.SIGTERM, lambda signum, frame: process.send_signal(signum)
            )
            try:
                process.communicate(input=json.dumps(config))
            except KeyboardInterrupt:
                cancelled = True
                process.send_signal(signal.SIGINT)
                process.communicate()
            finally:
                signal.signal(signal.SIGTERM, previous_terminate)
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        returncode = 130 if cancelled else process.returncode
        if cancelled and summary:
            summary["outcome"] = "cancelled"
        if mlflow_service is not None and summary:
            if cancelled or process.returncode in (130, 143):
                summary["mlflow"] = {"status": "not_run", "reason": "cancelled"}
            else:
                observations = (
                    json.loads(observations_file.read_text()) if observations_file.exists() else []
                )
                analysis_exit = 130

                def interrupt_analysis(signum, frame):
                    nonlocal analysis_exit
                    analysis_exit = 128 + signum
                    raise KeyboardInterrupt

                previous_terminate = signal.signal(signal.SIGTERM, interrupt_analysis)
                try:
                    summary["mlflow"] = mlflow_service.finish(observations)
                    summary["mlflow"]["eligible_successes"] = summary.get("counts", {}).get(
                        "succeeded", 0
                    )
                except KeyboardInterrupt:
                    returncode = analysis_exit
                    summary["mlflow"] = {"status": "not_run", "reason": "analysis_cancelled"}
                finally:
                    signal.signal(signal.SIGTERM, previous_terminate)
                if not returncode and (
                    summary["mlflow"].get("evaluation", {}).get("status") in {"failed", "partial"}
                    or summary["mlflow"].get("traces", {}).get("errors", 0)
                ):
                    returncode = 1
        if report_dir is not None and summary:
            (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return LoadtestResult(returncode, summary)
