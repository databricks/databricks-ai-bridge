from __future__ import annotations

import io
import json
import urllib.error

import pytest
from click.testing import CliRunner

from databricks_mason import endpoint as endpoint_mod
from databricks_mason import endpoint_output as endpoint_output_mod
from databricks_mason import endpoint_request as endpoint_request_mod
from databricks_mason import endpoint_transport as endpoint_transport_mod
from databricks_mason.endpoint import endpoint
from databricks_mason.endpoint_transport import EndpointRequest, EndpointResponse


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


def test_invoke_deployed_app_resolves_oauth_and_generated_session(monkeypatch):
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
            "--json",
            '{"input":[]}',
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.url == "https://app.example/api/invocations"
    assert request.headers["Authorization"] == "Bearer token"
    assert request.headers["Cookie"].startswith("__Host-databricks-app-router=")
    assert request.body == {"input": []}


def test_invoke_url_uses_explicit_routing_session_without_auth(monkeypatch):
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
    assert request.headers["Cookie"] == "__Host-databricks-app-router=local-session"


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
