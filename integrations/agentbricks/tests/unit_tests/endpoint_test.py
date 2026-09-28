from __future__ import annotations

import io
import json
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli import endpoint as endpoint_mod
from databricks_agentbricks.cli import endpoint_output as endpoint_output_mod
from databricks_agentbricks.cli import endpoint_request as endpoint_request_mod
from databricks_agentbricks.cli import endpoint_transport as endpoint_transport_mod
from databricks_agentbricks.cli.endpoint import endpoint
from databricks_agentbricks.cli.endpoint_transport import EndpointRequest, EndpointResponse


class _Ctx:
    profile = "profile"
    output = "json"


def _response(
    body,
    *,
    status_code: int = 200,
    url: str = "https://app/api/invocations",
    events=(),
) -> EndpointResponse:
    return EndpointResponse(
        url=url,
        status_code=status_code,
        headers={"Content-Type": "application/json"},
        body=body,
        elapsed_seconds=0.01,
        events=events,
    )


def test_request_url_merges_query_parameters():
    assert (
        endpoint_request_mod.request_url(
            "https://app.example/base",
            "/runs?existing=yes",
            {"after": "10"},
        )
        == "https://app.example/runs?existing=yes&after=10"
    )


def test_request_url_rejects_absolute_path():
    with pytest.raises(endpoint_mod.AgentCliError, match="must be relative"):
        endpoint_request_mod.request_url(
            "https://app.example",
            "https://other.example/run",
            {},
        )


def test_build_request_is_generic_json_http():
    request = endpoint_request_mod.build_request(
        base_url="https://app.example",
        method="patch",
        path="/custom/run",
        query=("mode=fast",),
        json_value='{"question":"hello"}',
        timeout=12,
        sse=False,
    )

    assert request == EndpointRequest(
        url="https://app.example/custom/run?mode=fast",
        method="PATCH",
        headers={"Content-Type": "application/json"},
        body={"question": "hello"},
        timeout=12,
        sse=False,
        body_set=True,
    )


def test_build_request_without_body_does_not_set_content_type():
    request = endpoint_request_mod.build_request(
        base_url="https://app.example",
        method="GET",
        path="/health",
        query=(),
        json_value=None,
        timeout=12,
        sse=False,
    )

    assert request.body is None
    assert request.body_set is False
    assert request.headers == {}


def test_json_null_is_sent_as_a_request_body():
    request = endpoint_request_mod.build_request(
        base_url="https://app.example",
        method="POST",
        path="/run",
        query=(),
        json_value="null",
        timeout=12,
        sse=False,
    )

    assert request.body is None
    assert request.body_set is True


def test_invalid_json_is_rejected_locally():
    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--url", "http://localhost:8000", "--path", "/run", "--json", "{"],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "Invalid JSON request body" in result.output


def test_invoke_deployed_app_resolves_oauth_and_explicit_session_header(monkeypatch):
    captured = {}

    class FakeSession:
        def send(self, request, *, on_event=None):
            captured["request"] = request
            return _response({"ok": True}, url=request.url)

    monkeypatch.setattr(endpoint_mod, "_app_url", lambda name, profile: "https://app.example")
    monkeypatch.setattr(endpoint_mod, "_authorization_header", lambda profile: "Bearer token")
    monkeypatch.setattr(endpoint_mod, "HttpSession", FakeSession)

    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "my-agent",
            "--path",
            "/api/invocations",
            "--session-id",
            "conversation-123",
            "--json",
            '{"input":[]}',
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.url == "https://app.example/api/invocations"
    assert request.headers["Authorization"] == "Bearer token"
    assert request.headers["X-Databricks-Session-Id"] == "conversation-123"
    assert "Cookie" not in request.headers
    assert request.body == {"input": []}


def test_invoke_url_uses_explicit_session_header_without_auth(monkeypatch):
    captured = {}

    class FakeSession:
        def send(self, request, *, on_event=None):
            captured["request"] = request
            return _response({"ok": True}, url=request.url)

    monkeypatch.setattr(endpoint_mod, "HttpSession", FakeSession)

    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "http://localhost:8000",
            "--path",
            "/api/invocations",
            "--session-id",
            "local-session",
            "--json",
            '{"input":[]}',
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert "Authorization" not in request.headers
    assert request.headers["X-Databricks-Session-Id"] == "local-session"
    assert "Cookie" not in request.headers


@pytest.mark.parametrize("path", ["/api/invocations", "/api/invocations/", "/api/invocations?x=1"])
@pytest.mark.parametrize(
    "body",
    [
        {"id": "invocation-1", "input": []},
        {"id": "invocation-1", "input": {"session_id": "body-session"}},
        {"id": "invocation-1", "session_id": "body-session", "input": []},
    ],
)
def test_managed_invocation_requires_explicit_session_option(path, body):
    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--url", "http://localhost:1", "--path", path, "--json", json.dumps(body)],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "--session-id is required for POST /api/invocations" in result.output
    assert "reuse it for every turn" in result.output


@pytest.mark.parametrize(
    "session_id", ["", " ", " leading", "trailing ", "a\nb", "a\rb", "a\tb", "a\x7fb"]
)
def test_invalid_session_option_is_rejected_locally(session_id):
    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "http://localhost:1",
            "--path",
            "/api/invocations",
            "--session-id",
            session_id,
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "--session-id must be nonblank" in result.output


@pytest.fixture
def local_http_endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length)) if length else None
            requests.append({"headers": dict(self.headers), "body": body})
            response = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def do_GET(self):
            self.do_POST()

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("mode", [{}, {"background": True}, {"stream": True}])
def test_invoke_sends_same_session_header_for_multiple_turns(local_http_endpoint, mode):
    url, requests = local_http_endpoint
    for invocation_id in ("turn-1", "turn-2"):
        body = {"id": invocation_id, "input": {"messages": []}, **mode}
        args = [
            "invoke",
            "--url",
            url,
            "--path",
            "/api/invocations",
            "--session-id",
            "chat-123",
            "--json",
            json.dumps(body),
        ]
        if mode.get("stream"):
            args.append("--sse")
        result = CliRunner().invoke(endpoint, args, obj=_Ctx())
        assert result.exit_code == 0, result.output

    assert [request["body"]["id"] for request in requests] == ["turn-1", "turn-2"]
    for request in requests:
        headers = {name.lower(): value for name, value in request["headers"].items()}
        assert headers["x-databricks-session-id"] == "chat-123"
        assert "cookie" not in headers
        assert request["body"]["input"] == {"messages": []}
        assert "session_id" not in request["body"]


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/api/invocations/turn-1"),
        ("GET", "/api/invocations/turn-1/events"),
        ("POST", "/invocations"),
    ],
)
def test_reads_and_custom_endpoints_do_not_require_session(local_http_endpoint, method, path):
    url, requests = local_http_endpoint
    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--url", url, "--method", method, "--path", path],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    headers = {name.lower(): value for name, value in requests[0]["headers"].items()}
    assert "x-databricks-session-id" not in headers
    assert "cookie" not in headers


def test_deployed_app_read_does_not_generate_session(local_http_endpoint, monkeypatch):
    url, requests = local_http_endpoint
    # Only workspace discovery/authentication are replaced; the CLI and HTTP transport run normally.
    monkeypatch.setattr(endpoint_mod, "_app_url", lambda name, profile: url)
    monkeypatch.setattr(endpoint_mod, "_authorization_header", lambda profile: "Bearer test-token")
    result = CliRunner().invoke(
        endpoint,
        ["invoke", "my-agent", "--method", "GET", "--path", "/api/invocations/turn-1"],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    headers = {name.lower(): value for name, value in requests[0]["headers"].items()}
    assert headers["authorization"] == "Bearer test-token"
    assert "x-databricks-session-id" not in headers
    assert "cookie" not in headers


def test_url_can_explicitly_request_oauth(monkeypatch):
    captured = {}

    class FakeSession:
        def send(self, request, *, on_event=None):
            captured["request"] = request
            return _response({"ok": True}, url=request.url)

    monkeypatch.setattr(endpoint_mod, "_authorization_header", lambda profile: "Bearer token")
    monkeypatch.setattr(endpoint_mod, "HttpSession", FakeSession)

    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "https://app.example",
            "--path",
            "/run",
            "--auth",
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert captured["request"].headers["Authorization"] == "Bearer token"


def test_app_and_url_are_mutually_exclusive():
    result = CliRunner().invoke(
        endpoint,
        ["invoke", "my-agent", "--url", "http://localhost:8000", "--path", "/run"],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "APP and --url are mutually exclusive" in result.output


def test_app_or_url_is_required():
    result = CliRunner().invoke(endpoint, ["invoke", "--path", "/run"], obj=_Ctx())

    assert result.exit_code != 0
    assert "Provide a Databricks App name or --url" in result.output
    assert "localhost:8000" in result.output


def test_non_success_status_is_an_error(monkeypatch):
    class FakeSession:
        def send(self, request, *, on_event=None):
            return _response({"error": "bad request"}, status_code=400, url=request.url)

    monkeypatch.setattr(endpoint_mod, "HttpSession", FakeSession)

    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--url", "http://localhost:8000", "--path", "/run"],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "Endpoint returned HTTP 400" in result.output
    assert "bad request" in result.output


def test_sse_response_is_returned_as_generic_events(monkeypatch):
    events = (
        {"event": "delta", "data": {"content": "hello"}},
        {"data": "[DONE]"},
    )

    class FakeSession:
        def send(self, request, *, on_event=None):
            assert request.sse is True
            return _response(None, url=request.url, events=events)

    monkeypatch.setattr(endpoint_mod, "HttpSession", FakeSession)

    result = CliRunner().invoke(
        endpoint,
        ["invoke", "--url", "http://localhost:8000", "--path", "/events", "--sse"],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["events"] == list(events)


def test_sse_parser_decodes_json_and_done_markers():
    response = io.BytesIO(
        b'id: 1\nevent: delta\ndata: {"type":"delta","content":"hi"}\n\ndata: [DONE]\n\n'
    )

    events = list(endpoint_transport_mod.iter_sse(response, on_event=None))

    assert events == [
        {
            "id": "1",
            "event": "delta",
            "data": {"type": "delta", "content": "hi"},
        },
        {"data": "[DONE]"},
    ]


def test_sse_printer_does_not_assume_an_agent_event_schema():
    printer = endpoint_output_mod.SsePrinter(enabled=True)

    with CliRunner().isolation() as streams:
        printer({"data": {"arbitrary": "value"}})
        printer({"data": "[DONE]"})

    assert streams[0].getvalue().decode() == '{"arbitrary": "value"}\n[DONE]\n'


def test_http_session_wraps_connection_errors():
    class FailingOpener:
        def open(self, request, timeout):
            raise urllib.error.URLError("connection refused")

    session = endpoint_transport_mod.HttpSession()
    session._opener = FailingOpener()

    with pytest.raises(endpoint_mod.AgentCliError, match="Could not reach endpoint"):
        session.send(
            EndpointRequest(
                url="http://localhost:1/run",
                method="POST",
                headers={"Content-Type": "application/json"},
                body={},
                timeout=1,
            )
        )


def test_help_exposes_only_low_level_options():
    result = CliRunner().invoke(endpoint, ["invoke", "--help"], obj=_Ctx())

    assert result.exit_code == 0, result.output
    for option in ("--url", "--method", "--path", "--query", "--json", "--sse", "--session-id"):
        assert option in result.output
    for removed in (
        "--preset",
        "--message",
        "--background",
        "--wait",
        "--id",
        "--poll-interval",
        "--expect-status",
        "--routing-key",
        "--header",
        "--json-file",
    ):
        assert removed not in result.output


def test_endpoint_has_no_loadtest_command():
    assert set(endpoint.commands) == {"invoke"}
