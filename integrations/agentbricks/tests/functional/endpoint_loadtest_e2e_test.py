"""Real CLI + Locust against a local endpoint; opt-in with the loadtest extra installed."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.util import find_spec
from pathlib import Path

import pytest


def _assert_final_csv_counts(report_dir, summary, case_counts=None):
    with (report_dir / "locust_stats.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    aggregate = next(row for row in rows if row["Name"] == "Aggregated")
    assert int(aggregate["Request Count"]) == summary["counts"]["completed"]
    assert int(aggregate["Failure Count"]) == summary["counts"]["failed"]
    requests = [row for row in rows if row["Type"] == "POST"]
    assert sum(int(row["Request Count"]) for row in requests) == summary["counts"]["completed"]
    if case_counts is not None:
        assert {row["Name"]: int(row["Request Count"]) for row in requests} == case_counts
    with (report_dir / "locust_failures.csv").open(newline="") as stream:
        failures = list(csv.DictReader(stream))
    assert sum(int(row["Occurrences"]) for row in failures) == summary["counts"]["failed"]


@pytest.mark.parametrize("streaming", [False, True])
def test_loadtest_caps_cases_and_saves_a_complete_report(tmp_path, streaming):
    """Exercise the installed entrypoint, excluded warm-up, drain, and saved statistics."""
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    agentbricks = Path(sys.executable).with_name("agentbricks")
    if not agentbricks.is_file():
        pytest.skip("requires the installed agentbricks CLI")
    seen = []
    active = peak = 0
    lock = threading.Lock()

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            nonlocal active, peak
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                seen.append(body)
                active += 1
                peak = max(peak, active)
            try:
                self.send_response(200 if streaming or body["case"] == "ok" else 503)
                self.send_header(
                    "Content-Type", "text/event-stream" if streaming else "application/json"
                )
                self.end_headers()
                if streaming:
                    # A metadata event must not count as content. The request remains active
                    # after the first actual text delta, so full latency must include the tail.
                    self.wfile.write(b'data: {"type":"response.created"}\n\n')
                    self.wfile.flush()
                    time.sleep(0.06)
                    self.wfile.write(
                        b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
                    )
                    self.wfile.flush()
                    time.sleep(0.16)
                    self.wfile.write(
                        b'data: {"type":"response.completed","response":'
                        b'{"usage":{"output_tokens":4}}}\n\ndata: [DONE]\n\n'
                    )
                else:
                    time.sleep(0.06)
                    self.wfile.write(b'{"usage":{"completion_tokens":4},"text":"hello"}')
            finally:
                with lock:
                    active -= 1

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps(
            [
                {"case": "ok", "id": "{{request_id}}", "session_id": "{{user_id}}"},
                {
                    "case": "alternate" if streaming else "fail",
                    "id": "{{request_id}}",
                    "session_id": "{{user_id}}",
                },
            ]
        )
    )
    report_dir = tmp_path / "report"
    responses_file = tmp_path / "responses.json"
    config = tmp_path / "empty.databrickscfg"
    config.write_text("")
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            command = [
                str(agentbricks),
                "--output",
                "json",
                "endpoint",
                "loadtest",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--path",
                "/run",
                "--concurrency",
                "3",
                "--duration",
                "20",
                "--max-requests",
                "48",
                "--timeout",
                "3",
                "--requests",
                str(cases),
                "--report-dir",
                str(report_dir),
                "--save-responses",
                str(responses_file),
            ]
            if streaming:
                command.append("--sse")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=40,
                cwd=tmp_path,
                env={
                    "PATH": f"{agentbricks.parent}:/usr/bin:/bin",
                    "HOME": str(tmp_path),
                    "DATABRICKS_CONFIG_FILE": str(config),
                    "AGENTBRICKS_DISABLE_TELEMETRY": "1",
                    # Preserve wheel/editable mode without ambient workspace credentials.
                    **(
                        {"PYTHONPATH": os.environ["PYTHONPATH"]}
                        if "PYTHONPATH" in os.environ
                        else {}
                    ),
                },
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert result.returncode == (0 if streaming else 1), result.stdout + result.stderr
    # JSON stdout is the same complete, versioned summary saved in the report.
    summary = json.loads((report_dir / "summary.json").read_text())
    assert json.loads(result.stdout) == summary
    assert summary["counts"] == {
        "started": 48,
        "completed": 48,
        "succeeded": 48 if streaming else 24,
        "failed": 0 if streaming else 24,
        "incomplete": 0,
    }
    assert len(seen) == 49  # One excluded warm-up followed by exactly forty-eight load requests.
    assert Counter(body["case"] for body in seen[1:]) == {
        "ok": 24,
        "alternate" if streaming else "fail": 24,
    }
    _assert_final_csv_counts(report_dir, summary, {"/run [case 1]": 24, "/run [case 2]": 24})
    assert summary["observed_peak_in_flight"] == 3
    assert peak == 3
    assert len({body["id"] for body in seen}) == 49
    for body in seen:
        assert uuid.UUID(body["id"])
        assert uuid.UUID(body["session_id"])
    assert 1 <= len({body["session_id"] for body in seen[1:]}) <= 3
    assert summary["latency_ms"]["successful"]["count"] == (48 if streaming else 24)
    assert summary["latency_ms"]["all_completed"]["min"] >= 40
    assert summary["usage"]["output_tokens"] == (192 if streaming else 96)
    assert summary["completed_rps"] > 0
    assert summary["elapsed_seconds_including_drain"] < 20
    if streaming:
        assert summary["ttft_ms"]["count"] == 48
        assert summary["ttft_coverage"]["measured"] == 48
        assert summary["ttft_ms"]["min"] >= 40
        assert summary["latency_ms"]["successful"]["min"] >= 190
        assert summary["latency_ms"]["successful"]["mean"] > summary["ttft_ms"]["mean"] + 100
    else:
        assert sum(summary["errors"].values()) == 24
    for artifact in ("index.html", "locust.html", "locust_stats.csv", "locust_stats_history.csv"):
        assert (report_dir / artifact).is_file(), artifact

    responses = json.loads(responses_file.read_text())
    assert responses["schema_version"] == 1
    assert responses["case_count"] == 2
    observations = responses["observations"]
    assert len(observations) == (48 if streaming else 24)
    assert responses_file.stat().st_mode & 0o777 == 0o600
    assert {item["inputs"]["id"] for item in observations}.issubset(
        {body["id"] for body in seen[1:]}
    )
    for item in observations:
        assert item["outputs"] == (
            "hello" if streaming else {"usage": {"completion_tokens": 4}, "text": "hello"}
        )
    assert all("hello" not in file.read_text() for file in report_dir.iterdir())


@pytest.mark.parametrize(
    ("stop_signal", "exit_code"), [(signal.SIGINT, 130), (signal.SIGTERM, 143)]
)
def test_interrupt_stops_dispatch_and_reports_drained_requests(tmp_path, stop_signal, exit_code):
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    agentbricks = Path(sys.executable).with_name("agentbricks")
    if not agentbricks.is_file():
        pytest.skip("requires the installed agentbricks CLI")
    loading = threading.Event()
    requests = 0
    responses_file = tmp_path / "responses.json"
    lock = threading.Lock()

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            nonlocal requests
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                requests += 1
                if requests == 4:  # Warm-up plus three in-flight load requests.
                    loading.set()
            time.sleep(0.5)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"text":"finished"}')

    config = tmp_path / "empty.databrickscfg"
    config.write_text("")
    report_dir = tmp_path / "report"
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        process = subprocess.Popen(
            [
                str(agentbricks),
                "endpoint",
                "loadtest",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--path",
                "/run",
                "--concurrency",
                "3",
                "--duration",
                "60",
                "--timeout",
                "3",
                "--report-dir",
                str(report_dir),
                "--save-responses",
                str(responses_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
            env={
                "PATH": f"{agentbricks.parent}:/usr/bin:/bin",
                "HOME": str(tmp_path),
                "DATABRICKS_CONFIG_FILE": str(config),
                "AGENTBRICKS_DISABLE_TELEMETRY": "1",
            },
        )
        try:
            assert loading.wait(timeout=15), "CLI did not start concurrent requests"
            process.send_signal(stop_signal)
            stdout, stderr = process.communicate(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            server.shutdown()
            worker.join(timeout=5)

    assert process.returncode == exit_code, stdout + stderr
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["outcome"] == "cancelled"
    counts = summary["counts"]
    assert counts["started"] >= 3
    assert counts["completed"] == counts["started"]
    assert counts["succeeded"] == counts["started"]
    assert counts["incomplete"] == counts["failed"] == 0
    assert requests == counts["started"] + 1
    _assert_final_csv_counts(report_dir, summary)
    assert len(json.loads(responses_file.read_text())["observations"]) == counts["succeeded"]
    assert responses_file.stat().st_mode & 0o777 == 0o600
    assert "cancelled" in stderr
    assert (report_dir / "index.html").is_file()
    assert (report_dir / "locust.html").is_file()


def test_redirect_warmup_never_generates_load(tmp_path):
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    agentbricks = Path(sys.executable).with_name("agentbricks")
    if not agentbricks.is_file():
        pytest.skip("requires the installed agentbricks CLI")
    paths = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            paths.append(self.path)
            self.send_response(302 if self.path == "/run" else 200)
            self.send_header("Location", "/target")
            self.end_headers()

    config = tmp_path / "empty.databrickscfg"
    config.write_text("")
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = subprocess.run(
                [
                    str(agentbricks),
                    "endpoint",
                    "loadtest",
                    "--url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--path",
                    "/run",
                    "--duration",
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                cwd=tmp_path,
                env={
                    "PATH": f"{agentbricks.parent}:/usr/bin:/bin",
                    "HOME": str(tmp_path),
                    "DATABRICKS_CONFIG_FILE": str(config),
                    "AGENTBRICKS_DISABLE_TELEMETRY": "1",
                },
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert result.returncode != 0
    assert "warm-up failed: HTTP 302" in result.stderr
    assert paths == ["/run"]


def test_natural_duration_drains_and_final_csv_matches_all_cases(tmp_path):
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    agentbricks = Path(sys.executable).with_name("agentbricks")
    if not agentbricks.is_file():
        pytest.skip("requires the installed agentbricks CLI")
    seen = []
    lock = threading.Lock()

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                seen.append(body)
            # Requests remain in flight when the native timer stops dispatch.
            time.sleep(0.31)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"text":"finished"}')

    cases = tmp_path / "cases.json"
    cases.write_text('[{"case": 1}, {"case": 2}]')
    config = tmp_path / "empty.databrickscfg"
    config.write_text("")
    report_dir = tmp_path / "report"
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = subprocess.run(
                [
                    str(agentbricks),
                    "--output",
                    "json",
                    "endpoint",
                    "loadtest",
                    "--url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--path",
                    "/run",
                    "--concurrency",
                    "2",
                    "--duration",
                    "2",
                    "--max-requests",
                    "1000",
                    "--timeout",
                    "3",
                    "--requests",
                    str(cases),
                    "--report-dir",
                    str(report_dir),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                cwd=tmp_path,
                env={
                    "PATH": f"{agentbricks.parent}:/usr/bin:/bin",
                    "HOME": str(tmp_path),
                    "DATABRICKS_CONFIG_FILE": str(config),
                    "AGENTBRICKS_DISABLE_TELEMETRY": "1",
                },
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)

    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["outcome"] == "completed"
    counts = summary["counts"]
    assert 4 <= counts["started"] < 1000  # The duration ended the run before its cap.
    assert counts["started"] == counts["completed"] == counts["succeeded"]
    assert counts["failed"] == counts["incomplete"] == 0
    assert len(seen) == counts["completed"] + 1
    case_counts = Counter(body["case"] for body in seen[1:])
    _assert_final_csv_counts(
        report_dir,
        summary,
        {f"/run [case {case}]": count for case, count in case_counts.items()},
    )
    assert summary["latency_ms"]["all_completed"]["count"] == counts["completed"]
    assert json.loads(result.stdout) == summary
    assert summary["workload"]["warmup_requests"] == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_captures_all_successes_in_private_observation_file(tmp_path, streaming):
    """Capture exceeds twenty successes, writes incrementally, and keeps reports content-free."""
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    from databricks_agentbricks.services.loadtest import streaming as loadtest_streaming

    seen = []
    observed_incrementally = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            if observations_file.exists():
                observed_incrementally.append(len(observations_file.read_text().splitlines()))
            assert self.headers.get("Authorization") == "Bearer private-token"
            assert self.headers.get("X-Routing-Key") == body["user"]
            self.send_response(200 if body["case"] == "ok" else 503)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            self.send_header("X-MLflow-Trace-ID", "tr-explicit-header-id")
            self.end_headers()
            if not streaming:
                self.wfile.write(b'{"output":"private-answer","trace_id":"tr-explicit-body-id"}')
                return
            self.wfile.write(
                b'data: {"type":"response.output_text.delta","delta":"private-answer"}\n\n'
            )
            self.wfile.write(
                b'data: {"type":"response.completed","trace_id":"tr-explicit-stream-id"}\n\n'
            )

    summary_file = tmp_path / "summary.json"
    observations_file = tmp_path / "observations.jsonl"
    observations_file.touch(mode=0o600)
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "locust",
                    "-f",
                    str(Path(loadtest_streaming.__file__).with_name("_locustfile.py")),
                    "--headless",
                    "--users",
                    "2",
                    "--spawn-rate",
                    "2",
                    "--run-time",
                    "10s",
                    "--stop-timeout",
                    "3",
                ],
                input=json.dumps(
                    {
                        "request": {
                            "url": f"http://127.0.0.1:{server.server_port}/run",
                            "method": "POST",
                            "headers": {"Authorization": "Bearer private-token"},
                            "body_set": True,
                            "timeout": 3,
                            "sse": True,
                        },
                        "bodies": [
                            json.dumps(
                                {
                                    "case": "ok",
                                    "id": "{{request_id}}",
                                    "user": "{{user_id}}",
                                    "input": "private-question",
                                }
                            ),
                            json.dumps(
                                {
                                    "case": "fail",
                                    "id": "{{request_id}}",
                                    "user": "{{user_id}}",
                                    "input": "private-question",
                                }
                            ),
                        ],
                        "per_user_routing": True,
                        "concurrency": 2,
                        "duration": 10,
                        "max_requests": 48,
                        "summary_file": str(summary_file),
                        "observations_file": str(observations_file),
                    }
                ),
                capture_output=True,
                text=True,
                timeout=20,
                env={**os.environ, "AGENTBRICKS_DISABLE_TELEMETRY": "1"},
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)
    assert result.returncode == 1, result.stdout + result.stderr
    summary_text = summary_file.read_text()
    summary = json.loads(summary_text)
    assert summary["counts"] == {
        "started": 48,
        "completed": 48,
        "succeeded": 24,
        "failed": 24,
        "incomplete": 0,
    }
    observations = [json.loads(line) for line in observations_file.read_text().splitlines()]
    assert len(observations) == 24
    assert max(observed_incrementally) >= 20
    assert len({body["id"] for body in seen}) == 48
    assert 1 <= len({body["user"] for body in seen}) <= 2
    assert observations_file.stat().st_mode & 0o777 == 0o600
    for observation in observations:
        assert observation["case_index"] == 0
        assert observation["inputs"]["id"] in {body["id"] for body in seen}
        assert observation["inputs"]["input"] == "private-question"
        assert observation["outputs"] == "private-answer"
        assert observation["trace_id"] == (
            "tr-explicit-stream-id" if streaming else "tr-explicit-body-id"
        )
    for private in ("private-question", "private-answer", "private-token", "tr-explicit"):
        assert private not in summary_text
        assert private not in result.stdout + result.stderr


@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("json_null", [False, True])
def test_one_http_202_call_preserves_bodyless_or_json_null_and_excludes_only_warmup(
    tmp_path, warmup, json_null
):
    """A 202 is an ordinary success; no background URL is followed and cold calls can count."""
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    agentbricks = Path(sys.executable).with_name("agentbricks")
    if not agentbricks.is_file():
        pytest.skip("requires the installed agentbricks CLI")
    seen = []
    get_calls = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            seen.append(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            sequence = len(seen)
            if sequence == 1:
                time.sleep(0.1)
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"sequence": sequence, "events_url": "/never"}).encode())

        def do_GET(self):
            get_calls.append(self.path)
            self.send_response(500)
            self.end_headers()

    config = tmp_path / "empty.databrickscfg"
    config.write_text("")
    report = tmp_path / "report"
    responses = tmp_path / "responses.json"
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            command = [
                str(agentbricks),
                "--output",
                "json",
                "endpoint",
                "loadtest",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--path",
                "/run",
                "--concurrency",
                "1",
                "--duration",
                "10",
                "--max-requests",
                "3",
                "--report-dir",
                str(report),
                "--save-responses",
                str(responses),
            ]
            if not warmup:
                command.append("--no-warmup")
            if json_null:
                requests_file = tmp_path / "requests.json"
                requests_file.write_text("[null]")
                command.extend(["--requests", str(requests_file)])
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=20,
                cwd=tmp_path,
                env={
                    "PATH": f"{agentbricks.parent}:/usr/bin:/bin",
                    "HOME": str(tmp_path),
                    "DATABRICKS_CONFIG_FILE": str(config),
                    "AGENTBRICKS_DISABLE_TELEMETRY": "1",
                    "AGENTBRICKS_DISABLE_AGENT_HINT": "1",
                },
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(result.stdout)
    assert summary == json.loads((report / "summary.json").read_text())
    assert summary["workload"]["warmup_requests"] == int(warmup)
    assert summary["counts"]["succeeded"] == summary["counts"]["completed"] == 3
    assert len(seen) == 3 + int(warmup)
    assert seen == ([b"null"] if json_null else [b""]) * len(seen)
    assert not get_calls
    observations = json.loads(responses.read_text())["observations"]
    assert [row["outputs"]["sequence"] for row in observations] == (
        [2, 3, 4] if warmup else [1, 2, 3]
    )
    assert all(row["inputs"] is None for row in observations)
    if not warmup:
        assert summary["latency_ms"]["successful"]["max"] >= 90


def test_warmup_deadline_bounds_a_stream_that_never_completes_a_line(tmp_path):
    """Socket activity cannot extend the warm-up beyond its total request deadline."""
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    agentbricks = Path(sys.executable).with_name("agentbricks")
    if not agentbricks.is_file():
        pytest.skip("requires the installed agentbricks CLI")
    calls = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            calls.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b"data: private-incomplete-event")
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

    config = tmp_path / "empty.databrickscfg"
    config.write_text("")
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        started = time.monotonic()
        try:
            result = subprocess.run(
                [
                    str(agentbricks),
                    "endpoint",
                    "loadtest",
                    "--url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--path",
                    "/run",
                    "--duration",
                    "10",
                    "--max-requests",
                    "3",
                    "--timeout",
                    "0.2",
                    "--sse",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmp_path,
                env={
                    "PATH": f"{agentbricks.parent}:/usr/bin:/bin",
                    "HOME": str(tmp_path),
                    "DATABRICKS_CONFIG_FILE": str(config),
                    "AGENTBRICKS_DISABLE_TELEMETRY": "1",
                    "AGENTBRICKS_DISABLE_AGENT_HINT": "1",
                },
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)
    assert result.returncode == 1, result.stdout + result.stderr
    assert time.monotonic() - started < 5
    assert calls == ["/run"]
    assert any(
        f"warm-up failed: {name}" in result.stderr for name in ("TimeoutError", "ConnectionError")
    )
    assert "No measured requests were sent" in " ".join(result.stderr.split())
    assert "private-incomplete-event" not in result.stderr + result.stdout


def test_response_capture_io_failure_stops_load_and_reports_a_content_free_error(tmp_path):
    if find_spec("locust") is None:
        pytest.skip("requires databricks-agentbricks[loadtest]")
    from databricks_agentbricks.services.loadtest import streaming

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"output":"private-answer"}')

    summary_file = tmp_path / "summary.json"
    invalid_capture = tmp_path / "private-capture-directory"
    invalid_capture.mkdir(mode=0o700)
    with ThreadingHTTPServer(("127.0.0.1", 0), Endpoint) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "locust",
                    "-f",
                    str(Path(streaming.__file__).with_name("_locustfile.py")),
                    "--headless",
                    "--users",
                    "1",
                    "--spawn-rate",
                    "1",
                    "--run-time",
                    "10s",
                    "--stop-timeout",
                    "3",
                ],
                input=json.dumps(
                    {
                        "request": {
                            "url": f"http://127.0.0.1:{server.server_port}/run",
                            "method": "POST",
                            "headers": {},
                            "body_set": False,
                            "timeout": 3,
                            "sse": False,
                        },
                        "bodies": None,
                        "per_user_routing": False,
                        "concurrency": 1,
                        "duration": 10,
                        "max_requests": 1000,
                        "summary_file": str(summary_file),
                        "observations_file": str(invalid_capture),
                    }
                ),
                capture_output=True,
                text=True,
                timeout=15,
                env={**os.environ, "AGENTBRICKS_DISABLE_TELEMETRY": "1"},
            )
        finally:
            server.shutdown()
            worker.join(timeout=5)
    assert result.returncode == 1, result.stdout + result.stderr
    summary = json.loads(summary_file.read_text())
    assert summary["capture_error"] == "Response capture failed (IsADirectoryError)"
    assert summary["counts"]["started"] < 1000
    assert summary["counts"]["started"] == summary["counts"]["succeeded"]
    for private in ("private-answer", str(invalid_capture)):
        assert private not in summary_file.read_text() + result.stdout + result.stderr
