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
    """Exercise the installed entrypoint, excluded preflight, drain, and saved statistics."""
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
                {"case": "ok", "id": "{{uuid}}", "session_id": "{{uuid}}"},
                {
                    "case": "alternate" if streaming else "fail",
                    "id": "{{uuid}}",
                    "session_id": "{{uuid}}",
                },
            ]
        )
    )
    report_dir = tmp_path / "report"
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
                "invoke",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--path",
                "/run",
                "--loadtest",
                "--concurrency",
                "3",
                "--duration",
                "20",
                "--max-requests",
                "8",
                "--timeout",
                "3",
                "--cases",
                str(cases),
                "--report-dir",
                str(report_dir),
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
    assert "Load test report" in result.stderr
    # Native JSON stdout remains parseable even though the friendly report is printed.
    assert isinstance(json.loads(result.stdout), list)
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["counts"] == {
        "started": 8,
        "completed": 8,
        "succeeded": 8 if streaming else 4,
        "failed": 0 if streaming else 4,
        "incomplete": 0,
    }
    assert len(seen) == 9  # One excluded preflight followed by exactly eight load requests.
    assert Counter(body["case"] for body in seen[1:]) == {
        "ok": 4,
        "alternate" if streaming else "fail": 4,
    }
    _assert_final_csv_counts(report_dir, summary, {"/run [case 1]": 4, "/run [case 2]": 4})
    assert summary["observed_peak_in_flight"] == 3
    assert peak == 3
    assert len({body["id"] for body in seen}) == 9
    for body in seen:
        assert uuid.UUID(body["id"])
        assert body["id"] == body["session_id"]
    assert summary["latency_ms"]["successful"]["count"] == (8 if streaming else 4)
    assert summary["latency_ms"]["all_completed"]["min"] >= 40
    assert summary["usage"]["output_tokens"] == (32 if streaming else 16)
    assert summary["completed_rps"] > 0
    assert summary["elapsed_seconds_including_drain"] < 20
    if streaming:
        assert summary["ttft_ms"]["count"] == 8
        assert summary["ttft_coverage"]["measured"] == 8
        assert summary["ttft_ms"]["min"] >= 40
        assert summary["latency_ms"]["successful"]["min"] >= 190
        assert summary["latency_ms"]["successful"]["mean"] > summary["ttft_ms"]["mean"] + 100
    else:
        assert sum(summary["errors"].values()) == 4
    for artifact in ("index.html", "locust.html", "locust_stats.csv", "locust_stats_history.csv"):
        assert (report_dir / artifact).is_file(), artifact


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
    lock = threading.Lock()

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            nonlocal requests
            self.rfile.read(int(self.headers["Content-Length"]))
            with lock:
                requests += 1
                if requests == 4:  # Preflight plus three in-flight load requests.
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
                "invoke",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--path",
                "/run",
                "--json",
                "{}",
                "--loadtest",
                "--concurrency",
                "3",
                "--duration",
                "60",
                "--timeout",
                "3",
                "--report-dir",
                str(report_dir),
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
    assert "cancelled" in stderr
    assert (report_dir / "index.html").is_file()
    assert (report_dir / "locust.html").is_file()


def test_redirect_preflight_never_generates_load(tmp_path):
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
                    "invoke",
                    "--url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--path",
                    "/run",
                    "--loadtest",
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
    assert "preflight failed: HTTP 302" in result.stderr
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
                    "invoke",
                    "--url",
                    f"http://127.0.0.1:{server.server_port}",
                    "--path",
                    "/run",
                    "--loadtest",
                    "--concurrency",
                    "2",
                    "--duration",
                    "2",
                    "--max-requests",
                    "1000",
                    "--timeout",
                    "3",
                    "--cases",
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
    native = json.loads(result.stdout)
    assert sum(row["num_requests"] for row in native) == counts["completed"]
