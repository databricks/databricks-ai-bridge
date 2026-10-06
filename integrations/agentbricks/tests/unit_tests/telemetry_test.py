"""Unit coverage for Agent Bricks CLI FrontendLog telemetry."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest import mock

import click
from click.testing import CliRunner

from databricks_agentbricks._group import AgentBricksCommand
from databricks_agentbricks.cli import telemetry
from databricks_agentbricks.errors import AgentCliError


class _ApiClient:
    def __init__(self, error: BaseException | None = None):
        self.calls: list[tuple[str, str, dict]] = []
        self.error = error

    def do(self, method: str, path: str, *, body: dict):
        self.calls.append((method, path, body))
        if self.error is not None:
            raise self.error
        return {}


class _Context:
    command_path = "agentbricks init"

    def __init__(self, api_client: _ApiClient | None = None):
        workspace_client = None
        if api_client is not None:
            workspace_client = SimpleNamespace(api_client=api_client)
        self.obj = SimpleNamespace(
            profile=None,
            _client=SimpleNamespace(workspace_client=workspace_client)
            if workspace_client is not None
            else None,
            telemetry_framework="openai",
            telemetry_server="agentbricks",
            telemetry_tracing_configured=True,
        )

    def find_root(self):
        return self


def _inner(body: dict) -> dict:
    assert body["items"] == []
    assert len(body["protoLogs"]) == 1
    return json.loads(body["protoLogs"][0])["entry"]


def test_build_payload_matches_frontend_log_schema_and_allowlist():
    context = _Context()
    log = telemetry.build_log(
        context,
        execution_time_ms=17,
        success=True,
        exit_code=0,
    )
    payload = telemetry.build_payload(log, upload_time_ms=123)

    assert payload["uploadTime"] == 123
    encoded = json.loads(payload["protoLogs"][0])
    assert uuid.UUID(encoded["frontend_log_event_id"])
    assert _inner(payload) == {
        "agentbricks_cli_log": {
            "command_path": "agentbricks init",
            "execution_time_ms": 17,
            "exit_code": 0,
            "framework": "FRAMEWORK_OPENAI",
            "operating_system": telemetry.platform.system().lower(),
            "package_version": telemetry._package_version(),
            "server": "SERVER_AGENTBRICKS",
            "success": True,
            "tracing_configured": True,
        }
    }
    assert "/tmp" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload)


def test_failed_payload_uses_bounded_category_and_no_message():
    context = _Context()
    log = telemetry.build_log(
        context,
        execution_time_ms=4,
        success=False,
        exit_code=1,
        error_category=telemetry._error_category(
            AgentCliError("secret message /tmp/private", error_code="PERMISSION_DENIED")
        ),
    )

    assert log["error_category"] == "ERROR_CATEGORY_AUTHORIZATION"
    encoded = json.dumps(log)
    assert "secret message" not in encoded
    assert "/tmp/private" not in encoded


def test_keyboard_interrupt_uses_click_abort_exit_code():
    assert telemetry._exit_code(KeyboardInterrupt()) == 1
    assert telemetry._error_category(KeyboardInterrupt()) == "ERROR_CATEGORY_INTERRUPTED"


def test_existing_workspace_client_receives_exact_transport_request(monkeypatch):
    api_client = _ApiClient()
    context = _Context(api_client)
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_MAX_FOREGROUND_WAIT_S", 1)
    monkeypatch.setattr(telemetry.time, "time", lambda: 12.3)

    telemetry.emit_command(context, execution_time_ms=5, success=True, exit_code=0)

    assert len(api_client.calls) == 1
    method, path, body = api_client.calls[0]
    assert (method, path) == ("POST", "/telemetry-ext")
    assert body["uploadTime"] == 12300
    assert _inner(body)["agentbricks_cli_log"]["success"] is True


def test_opt_out_does_not_construct_or_send(monkeypatch):
    context = _Context(_ApiClient())
    monkeypatch.setenv("AGENTBRICKS_DISABLE_TELEMETRY", "true")
    with mock.patch.object(telemetry, "_workspace_client") as workspace_client:
        telemetry.emit_command(context, execution_time_ms=1, success=True, exit_code=0)
    workspace_client.assert_not_called()


def test_no_auth_skips_client_creation(monkeypatch):
    context = _Context()
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    with mock.patch.object(telemetry, "_workspace_client", return_value=None) as workspace_client:
        telemetry.emit_command(context, execution_time_ms=1, success=True, exit_code=0)
    workspace_client.assert_called_once_with(context)


def test_transport_failure_is_silent_and_does_not_change_result(monkeypatch):
    context = _Context(_ApiClient(RuntimeError("transport detail")))
    monkeypatch.setattr(telemetry, "_MAX_FOREGROUND_WAIT_S", 1)
    telemetry.emit_command(context, execution_time_ms=1, success=True, exit_code=0)


def test_agentbricks_command_records_success_and_failed_attempt(monkeypatch):
    records = []
    monkeypatch.setattr(
        telemetry,
        "emit_command",
        lambda ctx, **kwargs: records.append((ctx.command_path, kwargs)),
    )

    @click.command(cls=AgentBricksCommand, name="ok")
    def ok():
        return None

    @click.command(cls=AgentBricksCommand, name="bad")
    def bad():
        raise AgentCliError("private message", error_code="UNAVAILABLE")

    ok_result = CliRunner().invoke(ok)
    bad_result = CliRunner().invoke(bad)

    assert ok_result.exit_code == 0, ok_result.output
    assert bad_result.exit_code == 1
    assert len(records) == 2
    assert records[0][1]["success"] is True
    assert records[0][1]["exit_code"] == 0
    assert records[1][1]["success"] is False
    assert records[1][1]["exit_code"] == 1
    assert records[1][1]["error_category"] == "ERROR_CATEGORY_NETWORK"


def test_agentbricks_command_rethrows_unexpected_exception(monkeypatch):
    monkeypatch.setattr(telemetry, "emit_command", lambda *args, **kwargs: None)
    failure = RuntimeError("original")

    @click.command(cls=AgentBricksCommand)
    def broken():
        raise failure

    result = CliRunner().invoke(broken)

    assert result.exit_code == 1
    assert result.exception is failure
