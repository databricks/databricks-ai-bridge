"""Run Locust separately so its gevent patches do not affect the CLI."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.services.invoke.transport import EndpointRequest
from databricks_agentbricks.services.loadtest.plan import LoadTestPlan


@dataclass(frozen=True)
class LoadtestResult:
    returncode: int
    summary: dict[str, Any]


def _save_responses(
    source: Path, destination: Path, case_count: int, *, capture_failed: bool = False
) -> None:
    """Stream private JSONL into the public export schema without retaining a run in RAM."""
    created = False
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f'{{"schema_version": 1, "case_count": {case_count}, "observations": [')
            separator = ""
            with source.open(encoding="utf-8") as observations:
                for line in observations:
                    if line.strip():
                        try:
                            observation = json.loads(line)
                        except ValueError:
                            # A failed disk write can leave one partial record after complete
                            # successes. Preserve that valid prefix and still fail the run below.
                            if capture_failed:
                                break
                            raise
                        stream.write(separator)
                        json.dump(observation, stream)
                        separator = ","
            stream.write("]}\n")
    except (OSError, ValueError) as exc:
        if created:
            destination.unlink(missing_ok=True)
        raise AgentCliError(
            "Could not save responses; any completed performance report is preserved."
        ) from exc


def run_loadtest(
    request: EndpointRequest,
    *,
    plan: LoadTestPlan,
    per_user_routing: bool,
) -> LoadtestResult:
    """Execute an already validated, resolved plan with Locust's native scheduler and reports."""
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(Path(__file__).with_name("_locustfile.py")),
        "--headless",
        "--users",
        str(plan.concurrency),
        "--spawn-rate",
        str(plan.concurrency),
        "--run-time",
        f"{plan.duration}s",
        "--stop-timeout",
        str(math.ceil(request.timeout)),
    ]
    if plan.report_dir is not None:
        command.extend(
            [
                "--html",
                str(plan.report_dir / "locust.html"),
                "--csv",
                str(plan.report_dir / "locust"),
                "--csv-full-history",
            ]
        )
    with TemporaryDirectory(prefix="agentbricks-loadtest-") as temporary:
        summary_file = Path(temporary) / "summary.json"
        observations_file = Path(temporary) / "observations.jsonl"
        if plan.responses_file is not None:
            observations_file.touch(mode=0o600)
        wire_request = asdict(request)
        # Bodies have one normalized source in the plan, including the explicit JSON null case.
        del wire_request["body"]
        config = {
            "request": wire_request,
            "bodies": plan.bodies,
            "per_user_routing": per_user_routing,
            "concurrency": plan.concurrency,
            "duration": plan.duration,
            "max_requests": plan.max_requests,
            "summary_file": str(summary_file),
            "warmup_requested": plan.warmup,
            "observations_file": str(observations_file)
            if plan.responses_file is not None
            else None,
        }
        # Credentials travel only through stdin, never arguments or files. Locust diagnostics
        # stay on stderr; the CLI owns stdout, including its machine-readable summary.
        with subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=sys.stderr,
            text=True,
            start_new_session=True,
        ) as process:
            cancellation_code = None

            def terminate(signum, frame):
                nonlocal cancellation_code
                cancellation_code = 143
                process.send_signal(signum)

            previous_terminate = signal.signal(signal.SIGTERM, terminate)
            try:
                process.communicate(input=json.dumps(config))
            except KeyboardInterrupt:
                cancellation_code = 130
                process.send_signal(signal.SIGINT)
                process.communicate()
            finally:
                signal.signal(signal.SIGTERM, previous_terminate)
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        if summary.get("warmup_error"):
            raise AgentCliError(
                f"Load test warm-up failed: {summary['warmup_error']}. No measured requests were sent."
            )
        returncode = cancellation_code or process.returncode
        if cancellation_code and summary:
            summary["outcome"] = "cancelled"
        if plan.report_dir is not None and summary:
            (plan.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if plan.responses_file is not None:
            _save_responses(
                observations_file,
                plan.responses_file,
                len(plan.bodies or (None,)),
                capture_failed=bool(summary.get("capture_error")),
            )
        if summary.get("capture_error"):
            raise AgentCliError(
                "Response capture failed; any completed performance report and captured responses are preserved."
            )
        if not summary and not cancellation_code:
            raise AgentCliError(
                "Load test runner exited without a summary; see diagnostics on stderr."
            )
        return LoadtestResult(returncode, summary)
