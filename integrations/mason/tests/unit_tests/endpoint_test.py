from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Iterator

import pytest
from click.testing import CliRunner

from databricks_mason import endpoint as endpoint_mod
from databricks_mason import endpoint_loadtest as endpoint_loadtest_mod
from databricks_mason import endpoint_output as endpoint_output_mod
from databricks_mason import endpoint_request as endpoint_request_mod
from databricks_mason import endpoint_transport as endpoint_transport_mod
from databricks_mason.endpoint import EndpointResponse, endpoint
from databricks_mason.endpoint_presets import build_preset_body, get_preset, polling_path


class _Ctx:
    profile = "profile"
    output = "json"


def _response(
    body,
    *,
    status_code: int = 200,
    url: str = "https://app/api/invocations",
) -> EndpointResponse:
    return EndpointResponse(
        url=url,
        status_code=status_code,
        headers={"Content-Type": "application/json"},
        body=body,
        elapsed_seconds=0.01,
    )


def test_durable_preset_builds_message_request_with_id_and_flags():
    preset = get_preset("mason-durable")
    assert preset is not None

    body = build_preset_body(
        preset,
        None,
        message="hello",
        stream=True,
        background=True,
        request_id="request-id",
    )

    assert body == {
        "id": "request-id",
        "input": [{"role": "user", "content": "hello"}],
        "stream": True,
        "background": True,
    }


def test_mason_preset_preserves_complete_json_body():
    preset = get_preset("mason")
    assert preset is not None

    body = build_preset_body(
        preset,
        {"input": {"question": "hello"}, "model": "test-model"},
        message=None,
        stream=False,
        background=False,
    )

    assert body == {"input": {"question": "hello"}, "model": "test-model"}


def test_polling_path_supports_both_mason_presets():
    durable = get_preset("mason-durable")
    mason = get_preset("mason")
    assert durable is not None and mason is not None
    assert polling_path(durable, {"status_url": "/api/invocations/one"}) == ("/api/invocations/one")
    assert polling_path(mason, {"id": "two"}) == "/api/invocations/two"


def test_request_url_accepts_absolute_status_url_and_merges_query():
    assert (
        endpoint_request_mod.request_url(
            "https://app.example/base",
            "https://status.example/runs/one?existing=yes",
            {"after": "10"},
        )
        == "https://status.example/runs/one?existing=yes&after=10"
    )


def test_invoke_durable_preset_resolves_app_auth_and_body(monkeypatch):
    captured = {}

    class FakeSession:
        def send(self, request, *, on_event=None):
            captured["request"] = request
            return _response({"id": request.body["id"], "status": "completed", "output": "ok"})

    monkeypatch.setattr(
        endpoint_mod,
        "resolve_target",
        lambda **kwargs: ("https://app", True, "my-app"),
    )
    monkeypatch.setattr(
        endpoint_request_mod,
        "auth_headers",
        lambda profile: {"Authorization": "Bearer token"},
    )
    monkeypatch.setattr(endpoint_mod, "HttpSession", FakeSession)

    result = CliRunner().invoke(
        endpoint,
        ["invoke", "my-app", "--preset", "mason-durable", "--message", "hello"],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.url == "https://app/api/invocations"
    assert request.headers["Authorization"] == "Bearer token"
    assert request.headers["Cookie"].startswith("__Host-databricks-app-router=")
    assert request.body["input"] == [{"role": "user", "content": "hello"}]
    assert isinstance(request.body["id"], str)
    assert json.loads(result.output)["body"]["output"] == "ok"


def test_invoke_generic_url_sends_body_without_auth(monkeypatch):
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
            "/custom/run",
            "--json",
            '{"question":"hello"}',
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    request = captured["request"]
    assert request.url == "http://localhost:8000/custom/run"
    assert "Authorization" not in request.headers
    assert request.body == {"question": "hello"}


def test_message_requires_preset():
    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "http://localhost:8000",
            "--path",
            "/custom/run",
            "--message",
            "hello",
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "--message requires --preset" in result.output


def test_background_wait_polls_to_terminal(monkeypatch):
    responses: Iterator[EndpointResponse] = iter(
        [
            _response(
                {"id": "run-1", "status": "queued", "status_url": "/api/invocations/run-1"},
                status_code=202,
            ),
            _response(
                {"id": "run-1", "status": "completed", "output": "done"},
                url="https://app/api/invocations/run-1",
            ),
        ]
    )
    sent = []

    class FakeSession:
        def send(self, request, *, on_event=None):
            sent.append(request)
            return next(responses)

    monkeypatch.setattr(
        endpoint_mod,
        "resolve_target",
        lambda **kwargs: ("https://app", False, None),
    )
    monkeypatch.setattr(endpoint_mod, "HttpSession", FakeSession)
    monkeypatch.setattr(endpoint_mod.time, "sleep", lambda _: None)

    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "https://app",
            "--preset",
            "mason-durable",
            "--message",
            "hello",
            "--background",
            "--wait",
            "--no-auth",
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert [request.method for request in sent] == ["POST", "GET"]
    assert sent[1].url == "https://app/api/invocations/run-1"
    assert json.loads(result.output)["body"]["output"] == "done"


def test_background_wait_uses_absolute_status_url(monkeypatch):
    responses: Iterator[EndpointResponse] = iter(
        [
            _response(
                {
                    "id": "run-1",
                    "status": "queued",
                    "status_url": "https://status.example/runs/run-1",
                },
                status_code=202,
            ),
            _response(
                {"id": "run-1", "status": "completed", "output": "done"},
                url="https://status.example/runs/run-1",
            ),
        ]
    )
    sent = []

    class FakeSession:
        def send(self, request, *, on_event=None):
            sent.append(request)
            return next(responses)

    monkeypatch.setattr(
        endpoint_mod,
        "resolve_target",
        lambda **kwargs: ("https://app", False, None),
    )
    monkeypatch.setattr(endpoint_mod, "HttpSession", FakeSession)
    monkeypatch.setattr(endpoint_mod.time, "sleep", lambda _: None)

    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "https://app",
            "--preset",
            "mason-durable",
            "--message",
            "hello",
            "--background",
            "--wait",
            "--no-auth",
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert sent[1].url == "https://status.example/runs/run-1"


def test_request_id_requires_durable_preset():
    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "http://localhost:8000",
            "--preset",
            "mason",
            "--id",
            "request-id",
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "--id requires --preset mason-durable" in result.output


def test_request_id_requires_uuid():
    result = CliRunner().invoke(
        endpoint,
        [
            "invoke",
            "--url",
            "http://localhost:8000",
            "--preset",
            "mason-durable",
            "--id",
            "request-id",
        ],
        obj=_Ctx(),
    )

    assert result.exit_code != 0
    assert "--id must be a valid UUID" in result.output


def test_http_session_wraps_connection_errors():
    class FailingOpener:
        def open(self, request, timeout):
            raise urllib.error.URLError("connection refused")

    session = endpoint_transport_mod.HttpSession()
    session._opener = FailingOpener()

    with pytest.raises(endpoint_mod.AgentCliError, match="Could not reach endpoint"):
        session.send(
            endpoint_mod.EndpointRequest(
                url="http://localhost:1/run",
                method="POST",
                headers={"Content-Type": "application/json"},
                body={},
                timeout=1,
            )
        )


def test_loadtest_durable_preset_generates_unique_ids(monkeypatch):
    bodies = []

    class FakeSession:
        def send(self, request, *, on_event=None):
            bodies.append(request.body)
            return _response({"status": "completed"}, url=request.url)

    monkeypatch.setattr(endpoint_loadtest_mod, "HttpSession", FakeSession)

    result = CliRunner().invoke(
        endpoint,
        [
            "loadtest",
            "--url",
            "http://localhost:8000",
            "--preset",
            "mason-durable",
            "--message",
            "hello",
            "--requests",
            "3",
            "--concurrency",
            "2",
        ],
        obj=_Ctx(),
    )

    assert result.exit_code == 0, result.output
    assert len({body["id"] for body in bodies}) == 3
    payload = json.loads(result.output)
    assert payload["requests"] == 3
    assert payload["successful"] == 3


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


def test_stream_printer_suppresses_done_marker():
    printer = endpoint_output_mod.StreamPrinter(enabled=True)

    with CliRunner().isolation() as streams:
        printer({"data": {"type": "delta", "content": "hello"}})
        printer({"data": "[DONE]"})
        printer.finish()

    assert streams[0].getvalue().decode() == "hello\n"
