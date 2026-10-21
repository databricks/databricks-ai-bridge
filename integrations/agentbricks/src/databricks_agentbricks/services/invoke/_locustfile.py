"""Locust task loaded by ``python -m locust``, after gevent is initialized."""

import atexit
import csv
import json
import signal
import sys
import time
from collections import Counter
from io import StringIO
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import gevent
from locust import HttpUser, events, task
from locust.exception import StopUser
from locust.html import get_html_report
from locust.stats import PERCENTILES_TO_REPORT, RequestStats, StatsCSV, StatsEntry

from databricks_agentbricks.services.invoke.loadtest_streaming import (
    StreamError,
    StreamMetrics,
    output_tokens,
)

_CONFIG = json.load(sys.stdin)
_REQUEST = _CONFIG["request"]
_CASES = _CONFIG.get("cases")
_BODIES = (
    [json.dumps(case) for case in _CASES]
    if _CASES is not None
    else [json.dumps(_REQUEST["body"]) if _REQUEST["body_set"] else None]
)
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
    global _COMPLETED, _USAGE_COUNT, _OUTPUT_TOKENS
    observation = context.get("loadtest")
    if observation is None:
        return
    observation["recorded"] = True
    _COMPLETED += 1
    if exception is not None:
        _ERRORS[observation.get("error") or type(exception).__name__] += 1
        return
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
        # Capture before native JSON output resets per-case stats; write after
        # native handles close so their buffered snapshots cannot overwrite it.
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
        self.client.headers.update(_REQUEST["headers"])
        if _CONFIG["per_user_routing"]:
            self.client.headers["X-Routing-Key"] = str(uuid4())

    @task
    def invoke(self):
        global _STARTED, _ACTIVE, _PEAK
        if _CONFIG.get("max_requests") is not None and _STARTED >= _CONFIG["max_requests"]:
            raise StopUser()
        # Assignment and admission happen before yielding to another user.
        case = _STARTED % len(_BODIES)
        body = _BODIES[case]
        body = body.replace("{{uuid}}", str(uuid4())) if body is not None else None
        _STARTED += 1
        _ACTIVE += 1
        _PEAK = max(_PEAK, _ACTIVE)
        observation = {}
        started = time.perf_counter()
        name = urlsplit(_REQUEST["url"]).path or "/"
        if _CASES is not None:
            name += f" [case {case + 1}]"
        try:
            with gevent.Timeout(_REQUEST["timeout"], TimeoutError("Request deadline exceeded")):
                with self.client.request(
                    _REQUEST["method"],
                    _REQUEST["url"],
                    name=name,
                    data=body,
                    timeout=_REQUEST["timeout"],
                    allow_redirects=False,
                    stream=True,
                    catch_response=True,
                    context={"loadtest": observation},
                ) as response:
                    try:
                        if not response.status_code:
                            response.raise_for_status()
                        if not 200 <= response.status_code < 300:
                            # Include the entire failed response in its client latency.
                            _ = response.content
                            observation["error"] = f"HTTP {response.status_code}"
                            response.failure(observation["error"])
                        elif (
                            response.headers.get("Content-Type", "").split(";", 1)[0]
                            == "text/event-stream"
                        ):
                            metrics = StreamMetrics()
                            metrics.read(response, started)
                            observation.update(
                                ttft_ms=metrics.ttft_ms, output_tokens=metrics.output_tokens
                            )
                        else:
                            content = response.content
                            try:
                                observation["output_tokens"] = output_tokens(json.loads(content))
                            except (ValueError, TypeError):
                                pass
                    except Exception as error:
                        observation["error"] = (
                            f"HTTP {response.status_code}"
                            if response.status_code and not 200 <= response.status_code < 300
                            else str(error)
                            if isinstance(error, StreamError)
                            else type(error).__name__
                        )
                        response.failure(observation["error"])
                    finally:
                        response.request_meta["response_time"] = (
                            time.perf_counter() - started
                        ) * 1000
                        if response.raw is not None:
                            response.request_meta["response_length"] = response.raw.tell()
                        response.close()
        except Exception as error:
            # Failures before a response exists must still appear in native stats.
            if not observation.get("recorded"):
                observation["error"] = type(error).__name__
                self.environment.events.request.fire(
                    request_type=_REQUEST["method"],
                    name=name,
                    response_time=(time.perf_counter() - started) * 1000,
                    response_length=0,
                    exception=RuntimeError(observation["error"]),
                    context={"loadtest": observation},
                )
        finally:
            _ACTIVE -= 1
            if (
                _CONFIG.get("max_requests") is not None
                and _STARTED >= _CONFIG["max_requests"]
                and not _ACTIVE
            ):
                gevent.spawn(self.environment.runner.quit)
