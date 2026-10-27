"""Unit coverage for the deployed-App readiness probe, against a local HTTP stub."""

from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Iterator

import pytest

from databricks_agentbricks.clients.app_probe import AppProbe, ProbeOutcome, ProbeResult
from databricks_agentbricks.errors import AgentCliError


class _Stub(http.server.BaseHTTPRequestHandler):
    status = 200
    seen: list[tuple[str, str | None]] = []

    def do_GET(self) -> None:  # noqa: N802 - http.server's handler name
        type(self).seen.append((self.path, self.headers.get("Authorization")))
        self.send_response(type(self).status)
        if 300 <= type(self).status < 400:
            self.send_header("Location", "/login")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - base signature
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Stub.seen = []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _probe() -> AppProbe:
    return AppProbe(lambda: "Bearer tok", timeout_s=2)


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (200, "answered"),
        (404, "answered"),
        (302, "answered"),
        (502, "booting"),
        (503, "booting"),
        (504, "booting"),
        (401, "unverifiable"),
        (403, "unverifiable"),
    ],
)
def test_probe_classifies_the_status(server: str, status: int, outcome: ProbeOutcome):
    _Stub.status = status

    assert _probe().probe(server) == ProbeResult(outcome, f"HTTP {status}")


def test_probe_gets_the_root_with_the_bearer_and_does_not_follow_redirects(server: str):
    _Stub.status = 302

    _probe().probe(server + "/")

    # One request: the redirect to /login was reported, not followed.
    assert _Stub.seen == [("/", "Bearer tok")]


def test_probe_counts_an_unreachable_app_as_booting():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    result = _probe().probe(f"http://127.0.0.1:{port}")

    assert result.outcome == "booting"


def test_probe_without_an_oauth_token_is_unverifiable_and_sends_nothing(server: str):
    def pat_profile() -> str:
        raise AgentCliError(
            "Databricks Apps API routes require OAuth; the selected profile uses a PAT."
        )

    result = AppProbe(pat_profile).probe(server)

    assert result.outcome == "unverifiable"
    assert result.detail is not None and "PAT" in result.detail
    assert _Stub.seen == []


def test_probe_resolves_the_token_once(server: str):
    calls: list[int] = []

    def header() -> str:
        calls.append(1)
        return "Bearer tok"

    probe = AppProbe(header)
    probe.probe(server)
    probe.probe(server)

    assert len(calls) == 1
