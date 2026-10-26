"""Locust task loaded by ``python -m locust``, after gevent is initialized."""

import atexit
import csv
import json
import logging
import os
import signal
import sys
import time
from collections import Counter
from io import StringIO
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import gevent
import requests
from locust import HttpUser, events, task
from locust.exception import StopUser
from locust.html import get_html_report
from locust.stats import PERCENTILES_TO_REPORT, RequestStats, StatsCSV, StatsEntry

from databricks_agentbricks.services.loadtest.request import materialize_body, perform_request
from databricks_agentbricks.services.loadtest.streaming import StreamError

_CONFIG = json.load(sys.stdin)
_REQUEST = _CONFIG["request"]
_BODIES = _CONFIG["bodies"] if _CONFIG["bodies"] is not None else [None]
_WARMUP_PERFORMED = False
if _CONFIG.get("warmup_requested", False):
    try:
        with requests.Session() as warmup_session:
            warmup_session.headers.update(_REQUEST["headers"])
            warmup_user = str(uuid4())
            if _CONFIG["per_user_routing"]:
                warmup_session.headers["X-Routing-Key"] = warmup_user
            with gevent.Timeout(_REQUEST["timeout"], TimeoutError("Request deadline exceeded")):
                perform_request(
                    warmup_session,
                    _REQUEST,
                    materialize_body(_BODIES[0], warmup_user),
                    time.perf_counter(),
                )
        _WARMUP_PERFORMED = True
    except Exception as error:
        detail = str(error) if isinstance(error, StreamError) else type(error).__name__
        Path(_CONFIG["summary_file"]).write_text(json.dumps({"warmup_error": detail}))
        logging.error("Load test warm-up failed: %s", detail)
        sys.exit(1)
_STARTED = 0
_COMPLETED = 0
_PEAK = 0
_ACTIVE = 0
_START_TIME: float | None = None
_END_TIME: float | None = None
_CANCELLED = False
_ERRORS: Counter[str] = Counter()
_SUCCESSFUL = StatsEntry(RequestStats(), "successful latency", "METRIC")
_TTFT = StatsEntry(RequestStats(), "answer-text TTFT", "METRIC")
_USAGE_COUNT = 0
_OUTPUT_TOKENS = 0
_CAPTURE_ERROR = None


@events.test_start.add_listener
def start(environment, **kwargs):
    global _START_TIME
    _START_TIME = time.perf_counter()

    def cancel(exit_code):
        global _CANCELLED
        if not _CANCELLED:
            _CANCELLED = True
            environment.process_exit_code = exit_code
            gevent.spawn(environment.runner.quit)

    # Stop and drain before Locust saves its native HTML report.
    gevent.signal_handler(signal.SIGINT, cancel, 130)
    gevent.signal_handler(signal.SIGTERM, cancel, 143)


@events.request.add_listener
def record(response_time, exception, context, **kwargs):
    global _COMPLETED, _USAGE_COUNT, _OUTPUT_TOKENS, _CAPTURE_ERROR
    observation = context.get("loadtest")
    if observation is None:
        return
    _COMPLETED += 1
    if exception is not None:
        _ERRORS[observation.get("error") or type(exception).__name__] += 1
        return
    if observation.get("captured") is not None:
        try:
            # Append each completed success immediately; never retain the run's outputs in RAM.
            descriptor = os.open(_CONFIG["observations_file"], os.O_WRONLY | os.O_APPEND)
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(observation["captured"]) + "\n")
        except OSError as error:
            _CAPTURE_ERROR = f"Response capture failed ({type(error).__name__})"
            environment = context["environment"]
            environment.process_exit_code = 1
            gevent.spawn(environment.runner.quit)
    _SUCCESSFUL.log(response_time, 0)
    if observation.get("ttft_ms") is not None:
        _TTFT.log(observation["ttft_ms"], 0)
    if observation.get("output_tokens") is not None:
        _USAGE_COUNT += 1
        _OUTPUT_TOKENS += observation["output_tokens"]


@events.test_stop.add_listener
def stopped(environment, **kwargs):
    global _END_TIME
    _END_TIME = time.perf_counter()


def _timing(entry):
    """Use Locust's bounded histograms without registering extra HTTP requests."""
    count = entry.num_requests
    return {
        "count": count,
        "min": entry.min_response_time if count else None,
        "mean": entry.avg_response_time if count else None,
        "max": entry.max_response_time if count else None,
        **{
            f"p{p}": entry.get_response_time_percentile(p / 100) if count else None
            for p in (50, 90, 95, 99)
        },
    }


def _save_final_csv(environment):
    writer = StatsCSV(environment, PERCENTILES_TO_REPORT)
    for suffix, write in (
        ("_stats.csv", writer.requests_csv),
        ("_failures.csv", writer.failures_csv),
        ("_exceptions.csv", writer.exceptions_csv),
    ):
        content = StringIO(newline="")
        # Locust's writerow return annotation differs from stdlib csv.writer.
        write(csv.writer(content))  # ty: ignore[invalid-argument-type]
        # Save drained counts after native handles close so their buffered snapshots
        # cannot overwrite the final CSV statistics.
        atexit.register(
            Path(environment.parsed_options.csv_prefix + suffix).write_text,
            content.getvalue(),
            encoding="utf-8",
            newline="",
        )


@events.quitting.add_listener
def finish(environment, **kwargs):
    # Also drain if Locust shuts down before stopping its users.
    if environment.runner.state != "stopped":
        environment.runner.stop()
    html_file = environment.parsed_options.html_file
    # Locust's SIGTERM shutdown path skips its usual HTML export.
    if html_file and not Path(html_file).exists():
        Path(html_file).write_text(
            get_html_report(environment, show_download_link=False), encoding="utf-8"
        )
    elapsed = (
        max((_END_TIME or time.perf_counter()) - _START_TIME, 0.001)
        if _START_TIME is not None
        else 0.001
    )
    incomplete = _STARTED - _COMPLETED
    outcome = (
        "incomplete" if incomplete or not _STARTED else "cancelled" if _CANCELLED else "completed"
    )
    summary = {
        "schema_version": 1,
        "workload": {
            "concurrency": _CONFIG["concurrency"],
            "duration_seconds": _CONFIG["duration"],
            "max_requests": _CONFIG.get("max_requests"),
            "cases": len(_BODIES),
            "warmup_requests": int(_WARMUP_PERFORMED),
        },
        "outcome": outcome,
        "counts": {
            "started": _STARTED,
            "completed": _COMPLETED,
            "succeeded": _SUCCESSFUL.num_requests,
            "failed": _COMPLETED - _SUCCESSFUL.num_requests,
            "incomplete": incomplete,
        },
        "observed_peak_in_flight": _PEAK,
        "elapsed_seconds_including_drain": elapsed,
        "completed_rps": _COMPLETED / elapsed,
        "successful_rps": _SUCCESSFUL.num_requests / elapsed,
        "error_rate": (_COMPLETED - _SUCCESSFUL.num_requests) / _COMPLETED if _COMPLETED else None,
        "latency_ms": {
            "all_completed": _timing(environment.stats.total),
            "successful": _timing(_SUCCESSFUL),
        },
        "ttft_ms": _timing(_TTFT),
        "ttft_coverage": {
            "measured": _TTFT.num_requests,
            "unknown": _SUCCESSFUL.num_requests - _TTFT.num_requests,
            "eligible_successes": _SUCCESSFUL.num_requests,
        },
        "usage": {
            "observed_successes": _USAGE_COUNT,
            "eligible_successes": _SUCCESSFUL.num_requests,
            "output_tokens": _OUTPUT_TOKENS if _USAGE_COUNT else None,
        },
        "errors": dict(_ERRORS),
        "generator_cpu_warning": environment.runner.cpu_warning_emitted,
    }
    if _CAPTURE_ERROR:
        summary["capture_error"] = _CAPTURE_ERROR
    Path(_CONFIG["summary_file"]).write_text(json.dumps(summary, indent=2) + "\n")
    if environment.parsed_options.csv_prefix:
        _save_final_csv(environment)
    if outcome != "completed" and environment.process_exit_code is None:
        environment.process_exit_code = 1


class EndpointUser(HttpUser):
    _origin = urlsplit(_REQUEST["url"])
    _hostname = _origin.hostname or "localhost"
    if ":" in _hostname:
        _hostname = f"[{_hostname}]"
    host = f"{_origin.scheme}://{_hostname}"
    if _origin.port is not None:
        host += f":{_origin.port}"

    def on_start(self):
        self._session = requests.Session()
        self._session.headers.update(_REQUEST["headers"])
        self._user_id = str(uuid4())
        if _CONFIG["per_user_routing"]:
            self._session.headers["X-Routing-Key"] = self._user_id

    def on_stop(self):
        self._session.close()

    @task
    def invoke(self):
        global _STARTED, _ACTIVE, _PEAK
        if _CAPTURE_ERROR:
            raise StopUser()
        if _CONFIG.get("max_requests") is not None and _STARTED >= _CONFIG["max_requests"]:
            raise StopUser()
        # Assignment and admission happen before yielding to another user.
        case = _STARTED % len(_BODIES)
        body = materialize_body(_BODIES[case], self._user_id)
        _STARTED += 1
        _ACTIVE += 1
        _PEAK = max(_PEAK, _ACTIVE)
        capture_output = _CONFIG.get("observations_file") is not None
        observation = {}
        failure = None
        completed = False
        started = time.perf_counter()
        name = urlsplit(_REQUEST["url"]).path or "/"
        if len(_BODIES) > 1:
            name += f" [case {case + 1}]"
        try:
            with gevent.Timeout(_REQUEST["timeout"], TimeoutError("Request deadline exceeded")):
                measured = perform_request(
                    self._session,
                    _REQUEST,
                    body,
                    started,
                    capture_output=capture_output,
                )
                observation.update(
                    ttft_ms=measured.get("ttft_ms"), output_tokens=measured.get("output_tokens")
                )
                if capture_output:
                    observation["captured"] = {
                        "case_index": case,
                        "inputs": json.loads(body) if body is not None else None,
                        "outputs": measured.get("outputs"),
                        "trace_id": measured.get("trace_id"),
                    }
        except Exception as error:
            observation["error"] = (
                str(error) if isinstance(error, StreamError) else type(error).__name__
            )
            failure = RuntimeError(observation["error"])
            completed = True
        else:
            completed = True
        finally:
            if completed:
                self.environment.events.request.fire(
                    request_type=_REQUEST["method"],
                    name=name,
                    response_time=(time.perf_counter() - started) * 1000,
                    response_length=0,
                    exception=failure,
                    context={"loadtest": observation, "environment": self.environment},
                )
            _ACTIVE -= 1
            if (
                _CONFIG.get("max_requests") is not None
                and _STARTED >= _CONFIG["max_requests"]
                and not _ACTIVE
            ):
                gevent.spawn(self.environment.runner.quit)
