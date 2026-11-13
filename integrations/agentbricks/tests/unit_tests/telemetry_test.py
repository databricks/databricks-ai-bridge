"""Unit coverage for Agent Bricks CLI FrontendLog telemetry."""

from __future__ import annotations

import json
import threading
import time
import uuid
from enum import Enum
from types import SimpleNamespace
from unittest import mock
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit

import click
import pytest
import requests
from click.core import ParameterSource
from click.testing import CliRunner
from deploy_test import _DeployApi, _DeployRunner
from rich.console import Console

import databricks_agentbricks.cli.app as cli
from databricks_agentbricks.cli import deploy as deploy_mod
from databricks_agentbricks.cli import telemetry
from databricks_agentbricks.clients import databricks_cli
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation import render
from databricks_agentbricks.presentation.group import AgentBricksCommand
from databricks_agentbricks.projects.agent_project import AgentProject
from databricks_agentbricks.services.deployment import deploy_state


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
        private_client = (
            SimpleNamespace(workspace_client=SimpleNamespace(api_client=api_client))
            if api_client is not None
            else None
        )
        self.obj = SimpleNamespace(
            profile=None,
            api_client_provider=SimpleNamespace(_client=private_client),
        )
        self.command = SimpleNamespace(name="init", params=())
        self.parent = SimpleNamespace(
            command=SimpleNamespace(name="agentbricks", params=()),
            parent=None,
            obj=self.obj,
        )

    def find_root(self):
        return self


def _real_context(*path: str) -> click.Context:
    command = cli.agentbricks
    context = click.Context(command, info_name="agentbricks")
    for child_name in path:
        assert isinstance(command, click.Group)
        command = command.commands[child_name]
        context = click.Context(command, info_name=child_name, parent=context)
    return context


def _explicit_option(
    context: click.Context,
    name: str,
    value: object,
    source: ParameterSource = ParameterSource.COMMANDLINE,
) -> None:
    assert any(
        isinstance(parameter, click.Option) and parameter.name == name
        for parameter in context.command.params
    )
    context.params[name] = value
    context.set_parameter_source(name, source)


def _real_options():
    def walk(command, path):
        for parameter in command.params:
            if isinstance(parameter, click.Option):
                yield path, parameter
        if isinstance(command, click.Group):
            for child_name, child in command.commands.items():
                yield from walk(child, (*path, child_name))

    yield from walk(cli.agentbricks, ())


def _inner(body: dict) -> dict:
    assert body["items"] == []
    assert len(body["protoLogs"]) == 1
    return json.loads(body["protoLogs"][0])["entry"]


def _log(context, **kwargs) -> dict:
    return telemetry._build_log(context, **kwargs).to_dict()


def test_build_payload_matches_frontend_log_schema():
    context = _Context()
    log = telemetry._build_log(
        context,
        execution_time_ms=17,
        exit_code=0,
    )
    payload = telemetry._FrontendLogPayload(
        log=log, event_id=str(uuid.uuid4()), upload_time_ms=123
    ).to_body()

    assert payload["uploadTime"] == 123
    encoded = json.loads(payload["protoLogs"][0])
    assert uuid.UUID(encoded["frontend_log_event_id"])
    assert _inner(payload) == {
        "agentbricks_cli_log": {
            "command_path": "agentbricks init",
            "execution_time_ms": 17,
            "exit_code": 0,
            "operating_system": telemetry.platform.system().lower(),
            "package_version": telemetry._package_version(),
        }
    }
    assert "/tmp" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload)


def test_build_log_uses_static_click_path_instead_of_raw_context_path():
    context = _Context()
    context.command_path = "agentbricks init --profile private-profile"

    log = _log(
        context,
        execution_time_ms=1,
        exit_code=0,
    )

    assert log["command_path"] == "agentbricks init"
    assert "private-profile" not in json.dumps(log)

    context.command = SimpleNamespace(name="init --private-profile", params=())
    fallback_log = _log(
        context,
        execution_time_ms=1,
        exit_code=0,
    )
    assert fallback_log["command_path"] == "agentbricks"


def test_real_init_telemetry_captures_safe_options_without_values(monkeypatch, tmp_path):
    records = []

    monkeypatch.setattr(
        telemetry,
        "emit_command",
        lambda ctx, **kwargs: records.append(_log(ctx, **kwargs)),
    )
    private_directory = tmp_path / "private-project-name"
    result = CliRunner().invoke(
        cli.agentbricks,
        [
            "--output",
            "json",
            "init",
            str(private_directory),
            "--framework",
            "openai",
            "--server",
            "custom",
            "--disable-chat-app",
            "--profile",
            "private-profile-name",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(records) == 1
    log = records[0]
    assert set(log) == {
        "command_path",
        "package_version",
        "operating_system",
        "execution_time_ms",
        "exit_code",
        "parameters",
    }
    assert log["command_path"] == "agentbricks init"
    assert log["exit_code"] == 0
    assert log["execution_time_ms"] >= 0
    parameters = {parameter["name"]: parameter for parameter in log["parameters"]}
    assert parameters["agentbricks.output"] == {
        "name": "agentbricks.output",
        "choice_value": "AGENTBRICKS_CLI_JSON",
    }
    assert parameters["agentbricks init.framework"] == {
        "name": "agentbricks init.framework",
        "choice_value": "AGENTBRICKS_CLI_OPENAI",
    }
    assert parameters["agentbricks init.server"] == {
        "name": "agentbricks init.server",
        "choice_value": "AGENTBRICKS_CLI_CUSTOM",
    }
    assert parameters["agentbricks init.disable_chat_app"] == {
        "name": "agentbricks init.disable_chat_app",
        "bool_value": True,
    }
    assert parameters["agentbricks init.profile"] == {
        "name": "agentbricks init.profile",
    }
    assert set(parameters) == {
        "agentbricks.output",
        "agentbricks init.framework",
        "agentbricks init.server",
        "agentbricks init.disable_chat_app",
        "agentbricks init.profile",
        "agentbricks init.existing",
        "agentbricks init.enable_chat_app",
    }
    encoded = json.dumps(log)
    assert "private-project-name" not in encoded
    assert "private-profile-name" not in encoded


def test_failed_payload_uses_bounded_category_and_no_message():
    context = _Context()
    log = _log(
        context,
        execution_time_ms=4,
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
    assert telemetry._exit_code(SystemExit()) == 0


def test_wrapped_failure_preserves_text_in_sensitive_field():
    context = _Context()
    error = AgentCliError("private message", hint="private hint")
    error.__cause__ = AgentCliError("private service detail", error_code="PERMISSION_DENIED")
    log = _log(context, exit_code=1, exception=error)
    assert log["failure_message"] == "private message\nCaused by: private service detail"
    assert log == _log(_Context(), exit_code=1, exception=error)
    assert "diagnostics" not in log
    assert telemetry._error_category(error) == "ERROR_CATEGORY_AUTHORIZATION"
    assert "private hint" not in json.dumps(log)


@pytest.mark.parametrize(
    ("cause", "category"),
    [
        (ConnectionError("private host"), "NETWORK"),
        (URLError("private host"), "NETWORK"),
        (requests.ConnectionError("private host"), "NETWORK"),
        (requests.Timeout("private host"), "NETWORK"),
        (FileNotFoundError("private path"), "CONFIGURATION"),
        (ImportError("private module"), "DEPENDENCY"),
        (ModuleNotFoundError("private module"), "DEPENDENCY"),
    ],
)
def test_failure_classification_follows_explicit_causes(cause, category):
    error = AgentCliError("Could not start")
    error.__cause__ = cause
    assert telemetry._error_category(error) == f"ERROR_CATEGORY_{category}"


@pytest.mark.parametrize(
    ("code", "category"),
    [
        ("INVALID_TOKEN", "AUTHENTICATION"),
        ("TOKEN_EXPIRED", "AUTHENTICATION"),
        ("forbidden", "AUTHORIZATION"),
        ("NOT_AUTHORIZED", "AUTHORIZATION"),
        ("TIMED_OUT", "NETWORK"),
        ("CANCELLED", "NETWORK"),
        ("CONFIGURATION_ERROR", "CONFIGURATION"),
        ("INVALID_ARGUMENT", "CONFIGURATION"),
        ("NOT_IMPLEMENTED", "RUNTIME"),
        ("ALREADY_EXISTS", "RUNTIME"),
    ],
)
def test_service_code_classification_takes_precedence_over_wrapped_transport_error(code, category):
    error = AgentCliError("service rejected request", error_code=code)
    error.__cause__ = ConnectionError("private host")
    assert telemetry._error_category(error) == f"ERROR_CATEGORY_{category}"


def test_unknown_codes_and_cyclic_causes_do_not_leak_or_loop():
    error = AgentCliError(
        "private", error_code="private-user-defined-code", error_category="private-user-category"
    )
    error.__cause__ = error
    assert telemetry._error_category(error) == "ERROR_CATEGORY_RUNTIME"
    log = _log(_Context(), exit_code=1, exception=error)
    assert log["failure_message"] == "private"
    assert "private-user-defined-code" not in json.dumps(log)
    assert "private-user-category" not in json.dumps(log)


def test_subprocess_failure_keeps_child_exit_separate_from_cli_exit(monkeypatch):
    records = []
    monkeypatch.setattr(
        telemetry, "emit_command", lambda ctx, **kwargs: records.append(_log(ctx, **kwargs))
    )

    @click.command(cls=AgentBricksCommand)
    def run():
        raise AgentCliError(
            "private child output",
            subprocess_exit_code=17,
        )

    result = CliRunner().invoke(run)
    assert result.exit_code == 1
    assert records[0]["exit_code"] == 1
    assert records[0]["failure_message"] == "private child output"
    assert "diagnostics" not in records[0]
    assert "subprocess_exit_code" not in records[0]
    assert "failure_reason" not in records[0]


def test_existing_workspace_client_receives_exact_transport_request(monkeypatch):
    api_client = _ApiClient()
    context = _Context(api_client)
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry.time, "time", lambda: 12.3)

    with mock.patch("databricks_agentkit._api_client._workspace_client") as build:
        telemetry.emit_command(context, execution_time_ms=5, exit_code=0)
    build.assert_not_called()

    assert len(api_client.calls) == 1
    method, path, body = api_client.calls[0]
    assert (method, path) == ("POST", "/telemetry-ext")
    assert body["uploadTime"] == 12300
    assert _inner(body)["agentbricks_cli_log"]["exit_code"] == 0


def test_transport_payload_contains_static_parameter_names(monkeypatch):
    api_client = _ApiClient()
    context = _real_context("deploy")
    _explicit_option(context, "workspace_path", "private-label")
    _explicit_option(context, "allow_user_scope_update", True)

    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(
        telemetry,
        "_workspace_client",
        lambda ctx: SimpleNamespace(api_client=api_client),
    )
    telemetry.emit_command(context, execution_time_ms=1, exit_code=0)

    assert len(api_client.calls) == 1
    log = _inner(api_client.calls[0][2])["agentbricks_cli_log"]
    parameters = {parameter["name"]: parameter for parameter in log["parameters"]}
    assert parameters == {
        "agentbricks deploy.workspace_path": {"name": "agentbricks deploy.workspace_path"},
        "agentbricks deploy.allow_user_scope_update": {
            "name": "agentbricks deploy.allow_user_scope_update",
            "bool_value": True,
        },
    }
    assert log["command_path"] == "agentbricks deploy"
    encoded = json.dumps(log)
    assert "private-label" not in encoded


def test_custom_click_names_are_static_and_values_are_not_sent(monkeypatch):
    captured = []

    @click.command(name="inventory", cls=AgentBricksCommand)
    @click.option("--label")
    @click.pass_context
    def inventory(ctx, label):
        del ctx, label

    monkeypatch.setattr(
        telemetry,
        "emit_command",
        lambda ctx, **kwargs: captured.append(_log(ctx, **kwargs)),
    )
    result = CliRunner().invoke(inventory, ["--label", "private-label"])

    assert result.exit_code == 0, result.output
    assert captured[0]["command_path"] == "inventory"
    assert captured[0]["parameters"] == [{"name": "inventory.label"}]
    assert "private-label" not in json.dumps(captured[0])


def test_opt_out_does_not_construct_or_send(monkeypatch):
    context = _Context(_ApiClient())
    monkeypatch.setenv("AGENTBRICKS_DISABLE_TELEMETRY", "true")
    with mock.patch.object(telemetry, "_workspace_client") as workspace_client:
        telemetry.emit_command(context, execution_time_ms=1, exit_code=0)
    workspace_client.assert_not_called()


def test_sdk_resolves_default_profile_when_no_agentbricks_profile():
    context = _Context()
    client = SimpleNamespace(api_client=_ApiClient())
    with mock.patch(
        "databricks_agentkit._api_client._workspace_client", return_value=client
    ) as build:
        assert telemetry._workspace_client(context) is client
    build.assert_called_once_with(None)


def test_sdk_receives_selected_agentbricks_profile():
    context = _Context()
    context.obj.profile = "selected"
    client = SimpleNamespace(api_client=_ApiClient())
    with mock.patch(
        "databricks_agentkit._api_client._workspace_client", return_value=client
    ) as build:
        assert telemetry._workspace_client(context) is client
    build.assert_called_once_with("selected")


def test_sdk_auth_failure_is_best_effort():
    context = _Context()
    with mock.patch(
        "databricks_agentkit._api_client._workspace_client",
        side_effect=RuntimeError("no credentials"),
    ) as build:
        assert telemetry._workspace_client(context) is None
    build.assert_called_once_with(None)


def test_transport_failure_is_silent_and_does_not_change_result(monkeypatch):
    context = _Context(_ApiClient(RuntimeError("transport detail")))
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    telemetry.emit_command(context, execution_time_ms=1, exit_code=0)


@pytest.mark.parametrize("setup_delay", [0.35, 1.15])
def test_default_allowance_waits_for_slow_client_setup_and_upload(monkeypatch, setup_delay):
    completed = threading.Event()
    workers = []

    class SlowApiClient(_ApiClient):
        def do(self, method, path, *, body):
            time.sleep(0.1)
            result = super().do(method, path, body=body)
            completed.set()
            return result

    api_client = SlowApiClient()

    def build_client(ctx):
        workers.append(threading.current_thread())
        time.sleep(setup_delay)
        return SimpleNamespace(api_client=api_client)

    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_workspace_client", build_client)
    started = time.monotonic()
    try:
        telemetry.emit_command(_Context(), execution_time_ms=1, exit_code=0)
        elapsed = time.monotonic() - started
        assert completed.is_set()
        assert elapsed >= setup_delay + 0.1
        assert len(api_client.calls) == 1
    finally:
        for worker in workers:
            worker.join(timeout=5)


@pytest.mark.parametrize("blocked_stage", ["setup", "transport"])
@pytest.mark.parametrize("exit_code", [0, 1])
def test_default_allowance_bounds_blocked_worker_and_preserves_outcome(
    monkeypatch, blocked_stage, exit_code
):
    release = threading.Event()
    entered = threading.Event()
    workers = []

    def block():
        workers.append(threading.current_thread())
        entered.set()
        release.wait(timeout=10)

    class BlockedApiClient(_ApiClient):
        def do(self, method, path, *, body):
            block()
            return super().do(method, path, body=body)

    def build_client(ctx):
        if blocked_stage == "setup":
            block()
        api_client = BlockedApiClient() if blocked_stage == "transport" else _ApiClient()
        return SimpleNamespace(api_client=api_client)

    failure = RuntimeError("original command failure")

    @click.command(cls=AgentBricksCommand)
    def command():
        if exit_code:
            raise failure
        click.echo("original command output")

    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_workspace_client", build_client)
    started = time.monotonic()
    try:
        result = CliRunner().invoke(command)
        elapsed = time.monotonic() - started
        assert entered.is_set()
        assert result.exit_code == exit_code
        if exit_code:
            assert result.exception is failure
        else:
            assert result.output == "original command output\n"
        assert telemetry._MAX_FOREGROUND_WAIT_S <= elapsed < telemetry._MAX_FOREGROUND_WAIT_S + 1
        assert len(workers) == 1
        assert workers[0].daemon
        assert workers[0].is_alive()
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=5)
            assert not worker.is_alive()


def test_agentbricks_command_records_exit_outcome(monkeypatch):
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

    class ZeroCodeError(RuntimeError):
        code = 0

    @click.command(cls=AgentBricksCommand, name="bad-zero-code")
    def bad_zero_code():
        raise ZeroCodeError("private message")

    ok_result = CliRunner().invoke(ok)
    bad_result = CliRunner().invoke(bad)
    bad_zero_code_result = CliRunner().invoke(bad_zero_code)

    assert ok_result.exit_code == 0, ok_result.output
    assert bad_result.exit_code == 1
    assert bad_zero_code_result.exit_code == 1
    assert len(records) == 3
    assert records[0][1]["exit_code"] == 0
    assert records[0][1]["execution_time_ms"] >= 0
    assert records[1][1]["exit_code"] == 1
    assert records[1][1]["error_category"] == "ERROR_CATEGORY_NETWORK"
    assert records[2][1]["exit_code"] == 1


def test_agentbricks_command_rethrows_unexpected_exception(monkeypatch):
    monkeypatch.setattr(telemetry, "emit_command", lambda *args, **kwargs: None)
    failure = RuntimeError("original")

    @click.command(cls=AgentBricksCommand)
    def broken():
        raise failure

    result = CliRunner().invoke(broken)

    assert result.exit_code == 1
    assert result.exception is failure


def test_build_log_captures_explicit_options_and_typed_values():
    context = _real_context("deploy")
    _explicit_option(context, "instance_count", 3)
    _explicit_option(context, "workspace_path", "private-secret")
    _explicit_option(context, "allow_user_scope_update", True)

    log = _log(context, exit_code=0)
    parameters = {parameter["name"]: parameter for parameter in log["parameters"]}
    assert parameters["agentbricks deploy.instance_count"] == {
        "name": "agentbricks deploy.instance_count",
        "int_value": 3,
    }
    assert parameters["agentbricks deploy.workspace_path"] == {
        "name": "agentbricks deploy.workspace_path",
    }
    assert parameters["agentbricks deploy.allow_user_scope_update"] == {
        "name": "agentbricks deploy.allow_user_scope_update",
        "bool_value": True,
    }
    assert len(parameters) == 3
    encoded = json.dumps(log)
    assert "private-secret" not in encoded


def test_build_log_includes_typed_defaults():
    context = _real_context("deploy")
    _explicit_option(context, "instance_count", 2, source=ParameterSource.DEFAULT)
    assert _log(context, exit_code=0)["parameters"] == [
        {"name": "agentbricks deploy.instance_count", "int_value": 2}
    ]

    pipeline_context = _real_context("memory", "pipeline", "create")
    _explicit_option(pipeline_context, "trigger", "manual", source=ParameterSource.DEFAULT)
    assert _log(pipeline_context, exit_code=0)["parameters"] == [
        {
            "name": "agentbricks memory pipeline create.trigger",
            "choice_value": "AGENTBRICKS_CLI_MANUAL",
        }
    ]


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("args", [[], ["--mode", "fast", "--enabled", "--count", "3"]])
def test_new_commands_emit_click_resolved_defaults_automatically(monkeypatch, fails, args):
    records = []
    timeout_default = mock.Mock(return_value="1.5")
    monkeypatch.setattr(
        telemetry, "emit_command", lambda ctx, **kwargs: records.append(_log(ctx, **kwargs))
    )

    @click.group(name="inventory")
    @click.option("--output", type=click.Choice(["text", "json"]), default="text")
    def inventory(output):
        pass

    @inventory.command(cls=AgentBricksCommand)
    @click.option("--mode", type=click.Choice(["safe", "fast"]), default="safe")
    @click.option("--enabled/--disabled", default=False)
    @click.option("--count", type=int, default=0)
    @click.option("--timeout", type=float, default=timeout_default)
    @click.option(
        "--scope", type=click.Choice(["all", "new"]), multiple=True, default=("all", "new")
    )
    @click.option("--profile", default="private-profile")
    def scan(mode, enabled, count, timeout, scope, profile):
        assert (mode, enabled, count) == (("fast", True, 3) if args else ("safe", False, 0))
        assert timeout == 1.5
        assert scope == ("all", "new")
        if fails:
            raise click.ClickException("private-error")

    result = CliRunner().invoke(inventory, ["scan", *args])

    assert result.exit_code == (1 if fails else 0), result.output
    timeout_default.assert_called_once_with()
    assert len(records) == 1
    assert records[0]["exit_code"] == result.exit_code
    assert records[0]["parameters"] == [
        {"name": "inventory.output", "choice_value": "AGENTBRICKS_CLI_TEXT"},
        {
            "name": "inventory scan.mode",
            "choice_value": "AGENTBRICKS_CLI_FAST" if args else "AGENTBRICKS_CLI_SAFE",
        },
        {"name": "inventory scan.enabled", "bool_value": bool(args)},
        {"name": "inventory scan.count", "int_value": 3 if args else 0},
        {"name": "inventory scan.timeout", "double_value": 1.5},
        {"name": "inventory scan.scope", "choice_value": "AGENTBRICKS_CLI_ALL"},
        {"name": "inventory scan.scope", "choice_value": "AGENTBRICKS_CLI_NEW"},
    ]
    message = records[0].pop("failure_message", None)
    assert message == ("private-error" if fails else None)
    assert "private-" not in json.dumps(records)


@pytest.mark.parametrize("fails", [False, True])
def test_real_init_defaults_are_emitted_for_success_and_failure(monkeypatch, tmp_path, fails):
    records = []
    monkeypatch.setattr(
        telemetry, "emit_command", lambda ctx, **kwargs: records.append(_log(ctx, **kwargs))
    )
    destination = tmp_path / "private-project"
    if fails:
        destination.mkdir()
    result = CliRunner().invoke(cli.agentbricks, ["init", str(destination)])
    assert result.exit_code == (1 if fails else 0), result.output
    assert len(records) == 1
    log = records[0]
    parameters = {p["name"]: p for p in log["parameters"]}
    assert parameters["agentbricks init.framework"]["choice_value"] == "AGENTBRICKS_CLI_LANGGRAPH"
    assert parameters["agentbricks init.server"]["choice_value"] == "AGENTBRICKS_CLI_AGENTBRICKS"
    assert parameters["agentbricks init.existing"]["bool_value"] is False
    assert parameters["agentbricks init.disable_chat_app"]["bool_value"] is False
    assert log["exit_code"] == result.exit_code
    message = log.pop("failure_message", None)
    assert (message is not None) == fails
    assert "private-project" not in json.dumps(log)
    assert "agentbricks init.profile" not in parameters
    assert "agentbricks init.memory_store" not in parameters
    assert "agentbricks init.session_store" not in parameters


def test_default_free_form_and_unresolved_values_are_omitted():
    context = _real_context("init")
    _explicit_option(context, "profile", "private-profile", source=ParameterSource.DEFAULT)
    _explicit_option(context, "framework", None, source=ParameterSource.DEFAULT)
    assert "parameters" not in _log(context, exit_code=0)


@pytest.mark.parametrize(
    "detail",
    [
        "ssh://user:synthetic-secret@host/db",
        json.dumps({"client_secret": 'synthetic-prefix"synthetic-tail'}),
        "secretKey=synthetic-opaque-value",
        "tokenValue=synthetic-opaque-value",
        "access_token_value=synthetic-opaque-value",
        json.dumps({"credentials": {"client_secret": "synthetic-opaque-value"}}),
        json.dumps({"credentials": [{"value": "synthetic-opaque-value"}]}, indent=2),
        json.dumps({"credentials": ["synthetic-opaque-value"]}),
        *[
            json.dumps({header: [{"value": "synthetic-opaque-value"}]}, indent=2)
            for header in ["Authorization", "Proxy-Authorization", "Cookie", "Set-Cookie"]
        ],
        'Cookie: [\n"synthetic-opaque-value"',
        'Authorization: {\n"value": "synthetic-opaque-value"',
        json.dumps({"message": json.dumps({"client_secret": "synthetic-opaque-value"})}),
        json.dumps({"message": json.dumps({"credentials": ["synthetic-opaque-value"]})}),
        json.dumps({"message": json.dumps({"Cookie": ["synthetic-opaque-value"]})}),
        r'{"\u0063lient_secret": "synthetic-opaque-value"}',
        "example://name:synthetic-secret@synthetic-tail@host/db",
        "client_secret: |\n  synthetic-opaque-value",
        "client_secret: >-\n  synthetic-opaque-value",
        repr({"password": b" synthetic-opaque-value"}),
        repr({"password": b",synthetic-opaque-value"}),
        repr({"password": b";synthetic-opaque-value"}),
        "credentials:\n  client_secret: synthetic-opaque-value",
        "credentials:\n  - synthetic-opaque-value",
        '"\\u0063redentials":\n  - synthetic-opaque-value',
        "token: Bearer synthetic-opaque-value",
        "token=Basic synthetic-opaque-value",
        "password=multiple words synthetic-opaque-value",
        "credentials: # configured\n  - synthetic-opaque-value",
        "client_secret: first line\n  synthetic-opaque-value",
        "client_secret: &anchor\n  synthetic-opaque-value",
        "client_secret: !!str\n  synthetic-opaque-value",
        'password="""first line\n"}synthetic-opaque-value"""',
        'credentials={nested: """first line\n"}synthetic-opaque-value"""}',
        "--password synthetic-opaque-value",
        "private_key = synthetic-opaque-value",
        "['tool', '--password', 'synthetic-opaque-value']",
    ],
)
def test_callback_error_redacts_credentials_from_uploaded_message(monkeypatch, detail):
    api = _ApiClient()
    context = _Context(api)
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    error = AgentCliError("Request failed: " + detail)
    telemetry.emit_command(context, exit_code=1, execution_time_ms=1, exception=error)
    assert len(api.calls) == 1
    payload = json.dumps(api.calls[0][2])
    assert "synthetic-secret" not in payload
    assert "synthetic-prefix" not in payload
    assert "synthetic-tail" not in payload
    assert "synthetic-opaque-value" not in payload
    assert "Request failed" in payload


def test_uploaded_failure_masks_escaped_url_credentials(monkeypatch):
    api = _ApiClient()
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    error = AgentCliError(r"Connection failed: https://user:synthetic-prefix\u0031@host/db")
    telemetry.emit_command(_Context(api), exit_code=1, exception=error)

    assert len(api.calls) == 1
    payload = api.calls[0][2]
    log = _inner(payload)["agentbricks_cli_log"]
    assert log["failure_message"] == "Connection failed: https://[REDACTED]@host/db"
    assert "synthetic-prefix" not in json.dumps(payload)
    assert log["exit_code"] == 1


@pytest.mark.parametrize("source", ["message", "detail", "cause", "combined"])
def test_oversized_failure_text_is_omitted_from_uploaded_payload(monkeypatch, source):
    api = _ApiClient()
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    oversized = "synthetic-over-budget " + "x" * 65536
    if source == "message":
        error = AgentCliError(oversized)
    elif source == "detail":
        error = AgentCliError("Request failed", failure_detail=oversized)
    elif source == "cause":
        error = AgentCliError("Request failed")
        error.__cause__ = RuntimeError(oversized)
    else:
        error = AgentCliError("a" * 32768, failure_detail="b" * 32768)
    displayed_message = str(error)

    telemetry.emit_command(
        _Context(api), exit_code=1, error_category="ERROR_CATEGORY_RUNTIME", exception=error
    )

    assert len(api.calls) == 1
    payload = api.calls[0][2]
    log = _inner(payload)["agentbricks_cli_log"]
    assert "failure_message" not in log
    assert "synthetic-over-budget" not in json.dumps(payload)
    assert log["exit_code"] == 1
    assert log["error_category"] == "ERROR_CATEGORY_RUNTIME"
    assert str(error) == displayed_message


def test_callback_error_preserves_prose_with_apostrophes_and_backslashes(monkeypatch):
    api = _ApiClient()
    context = _Context(api)
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    detail = r"Error: app doesn't exist in workspace; see .\logs\x.log; Final error: quota exceeded"
    telemetry.emit_command(
        context, exit_code=1, execution_time_ms=1, exception=AgentCliError(detail)
    )
    assert len(api.calls) == 1
    assert _inner(api.calls[0][2])["agentbricks_cli_log"]["failure_message"] == detail


@pytest.mark.parametrize(
    "opening", ["credentials:\n", "--password\n", "-----BEGIN PRIVATE KEY-----\n"]
)
@pytest.mark.parametrize("capture", [False, True])
def test_truncated_streamed_output_is_not_uploaded(monkeypatch, capsys, opening, capture):
    output = opening + " \n" * 2000 + "synthetic-opaque-value\n"

    class Process:
        returncode = 17

        def __enter__(self):
            self.stdout = iter(output.splitlines(keepends=True))
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(databricks_cli.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    with pytest.raises(AgentCliError) as raised:
        databricks_cli._databricks(
            ["sync", "source", "destination"],
            None,
            tee=True,
            capture=capture,
            action="Could not upload agent source.",
        )
    assert raised.value.subprocess_exit_code == 17
    assert capsys.readouterr().err == output

    api = _ApiClient()
    telemetry.emit_command(_Context(api), exit_code=1, execution_time_ms=1, exception=raised.value)
    assert len(api.calls) == 1
    log = _inner(api.calls[0][2])["agentbricks_cli_log"]
    assert log["exit_code"] == 1
    assert log["failure_message"].startswith("Could not upload agent source.")
    assert "synthetic-opaque-value" not in json.dumps(api.calls[0][2])


def test_uploaded_signal_diagnosis_preserves_reason_and_minimizes_assignment_paths(monkeypatch):
    api = _ApiClient()
    context = _Context(api)
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    detail = "Child failed: signal: killed; --config=/home/example/private/app.py; quota exceeded"
    telemetry.emit_command(
        context, exit_code=1, execution_time_ms=1, exception=AgentCliError(detail)
    )
    assert len(api.calls) == 1
    message = _inner(api.calls[0][2])["agentbricks_cli_log"]["failure_message"]
    assert "signal: killed" in message
    assert "quota exceeded" in message
    assert "/home/example/private" not in message


@pytest.mark.parametrize(
    "source", [ParameterSource.ENVIRONMENT, ParameterSource.DEFAULT_MAP, ParameterSource.PROMPT]
)
def test_other_option_sources_remain_omitted(source):
    context = _real_context("init")
    _explicit_option(context, "server", "custom", source=source)
    assert "parameters" not in _log(context, exit_code=0)


def test_pipeline_choice_uses_registered_data_shape_prefix():
    context = _real_context("memory", "pipeline", "create")
    _explicit_option(context, "trigger", "manual")
    log = _log(context, exit_code=0)
    assert log["parameters"] == [
        {
            "name": "agentbricks memory pipeline create.trigger",
            "choice_value": "AGENTBRICKS_CLI_MANUAL",
        }
    ]


def test_every_registered_choice_has_a_bounded_label():
    choice_count = 0
    for path, option in _real_options():
        if not isinstance(option.type, click.Choice):
            continue
        choice_count += 1
        name = f"{' '.join(('agentbricks', *path))}.{option.name}"
        for value in option.type.choices:
            assert isinstance(value, str)
            entry = telemetry._ParameterEntry.from_option(name, option, value)
            assert entry is not None
            assert entry.to_dict() == {
                "name": name,
                "choice_value": f"AGENTBRICKS_CLI_{value.upper().replace('-', '_')}",
            }
    assert choice_count > 0


def test_declared_choice_values_emit_without_extra_character_or_length_filters():
    unusual = click.Option(["--mode"], type=click.Choice(["read/write"]))
    bounded = click.Option(["--mode"], type=click.Choice(["safe"]))
    long = click.Option(["--mode"], type=click.Choice(["x" * 65]))

    unusual_entry = telemetry._ParameterEntry.from_option("agentbricks.mode", unusual, "read/write")
    long_entry = telemetry._ParameterEntry.from_option("agentbricks.mode", long, "x" * 65)
    undeclared_entry = telemetry._ParameterEntry.from_option("agentbricks.mode", bounded, "other")

    assert unusual_entry.to_dict() == {
        "name": "agentbricks.mode",
        "choice_value": "AGENTBRICKS_CLI_READ/WRITE",
    }
    assert long_entry.to_dict() == {
        "name": "agentbricks.mode",
        "choice_value": f"AGENTBRICKS_CLI_{'X' * 65}",
    }
    assert undeclared_entry.to_dict() == {"name": "agentbricks.mode"}


def test_enum_choice_uses_declared_click_label():
    class Mode(Enum):
        SAFE = 1

    option = click.Option(["--mode"], type=click.Choice(Mode))
    entry = telemetry._ParameterEntry.from_option("inventory.mode", option, Mode.SAFE)

    assert entry.to_dict() == {
        "name": "inventory.mode",
        "choice_value": "AGENTBRICKS_CLI_SAFE",
    }


def test_repeated_choice_values_emit_separate_entries():
    option = click.Option(["--mode"], type=click.Choice(["safe", "other"]), multiple=True)
    command = click.Command("inventory", params=[option])
    context = click.Context(command)
    _explicit_option(context, "mode", ("safe", "other"))

    assert _log(context, exit_code=0)["parameters"] == [
        {"name": "inventory.mode", "choice_value": "AGENTBRICKS_CLI_SAFE"},
        {"name": "inventory.mode", "choice_value": "AGENTBRICKS_CLI_OTHER"},
    ]


def test_repeated_options_respect_parameter_entry_limit():
    option = click.Option(["--mode"], type=click.Choice(["safe"]), multiple=True)
    command = click.Command("inventory", params=[option])
    context = click.Context(command)
    _explicit_option(context, "mode", ("safe",) * (telemetry._MAX_PARAMETER_ENTRIES + 1))

    assert len(_log(context, exit_code=0)["parameters"]) == telemetry._MAX_PARAMETER_ENTRIES


def test_repeated_free_form_values_remain_presence_only():
    option = click.Option(["--label"], multiple=True)
    command = click.Command("inventory", params=[option])
    context = click.Context(command)
    _explicit_option(context, "label", ("private-one", "private-two"))

    log = _log(context, exit_code=0)
    assert log["parameters"] == [{"name": "inventory.label"}] * 2
    assert "private-one" not in json.dumps(log)
    assert "private-two" not in json.dumps(log)


def test_hidden_and_deprecated_choices_are_emitted():
    hidden = click.Option(["--mode"], type=click.Choice(["safe"]), hidden=True)
    deprecated = click.Option(["--mode"], type=click.Choice(["safe"]), deprecated=True)

    expected = {"name": "agentbricks.mode", "choice_value": "AGENTBRICKS_CLI_SAFE"}
    assert (
        telemetry._ParameterEntry.from_option("agentbricks.mode", hidden, "safe").to_dict()
        == expected
    )
    assert (
        telemetry._ParameterEntry.from_option("agentbricks.mode", deprecated, "safe").to_dict()
        == expected
    )


def test_numeric_values_follow_click_types_and_wire_limits():
    count = click.Option(["--count"], type=click.INT)
    timeout = click.Option(["--timeout"], type=click.FLOAT)

    assert telemetry._ParameterEntry.from_option("inventory.count", count, 7).to_dict() == {
        "name": "inventory.count",
        "int_value": 7,
    }
    assert telemetry._ParameterEntry.from_option("inventory.timeout", timeout, 1.5).to_dict() == {
        "name": "inventory.timeout",
        "double_value": 1.5,
    }
    assert telemetry._ParameterEntry.from_option("inventory.count", count, 1 << 63).to_dict() == {
        "name": "inventory.count"
    }
    assert telemetry._ParameterEntry.from_option(
        "inventory.timeout", timeout, float("inf")
    ).to_dict() == {"name": "inventory.timeout"}


def test_real_registered_options_emit_declared_typed_values():
    for path, option in _real_options():
        assert option.name is not None
        name = f"{' '.join(('agentbricks', *path))}.{option.name}"
        if isinstance(option.type, click.Choice):
            value = option.type.choices[0]
        elif isinstance(option.type, click.types.BoolParamType):
            value = True
        elif isinstance(option.type, click.types.IntParamType):
            value = option.type.min if isinstance(option.type, click.IntRange) else 1
        elif isinstance(option.type, click.types.FloatParamType):
            value = option.type.min if isinstance(option.type, click.FloatRange) else 1.0
        else:
            value = "private-value"
        if option.multiple:
            value = (value,)

        context = _real_context(*path)
        _explicit_option(context, option.name, value)
        log = _log(context, exit_code=0)

        assert len(log["parameters"]) == 1, name
        entry = log["parameters"][0]
        assert entry["name"] == name
        if isinstance(option.type, click.Choice):
            assert isinstance(value, str)
            assert entry["choice_value"] == f"AGENTBRICKS_CLI_{value.upper().replace('-', '_')}"
        elif isinstance(option.type, click.types.BoolParamType):
            assert entry["bool_value"] is True
        elif isinstance(option.type, click.types.IntParamType):
            assert entry["int_value"] == value
        elif isinstance(option.type, click.types.FloatParamType):
            assert entry["double_value"] == value
        else:
            assert entry == {"name": name}


def test_parameter_entry_preserves_false_boolean_value():
    option = next(
        option
        for path, option in _real_options()
        if path == ("init",) and option.name == "existing"
    )
    entry = telemetry._ParameterEntry.from_option("agentbricks init.existing", option, False)
    assert entry.to_dict() == {"name": "agentbricks init.existing", "bool_value": False}


def test_failed_subprocess_detail_is_scrubbed_before_upload():
    error = AgentCliError(
        "Could not start",
        failure_detail="ModuleNotFoundError: package_missing\nAuthorization: Bearer synthetic-secret",
    )
    details = _log(_Context(), exit_code=1, exception=error)
    assert "ModuleNotFoundError: package_missing" in details["failure_message"]
    assert "synthetic-secret" not in json.dumps(details)


def test_long_subprocess_detail_preserves_final_error_after_scrubbing():
    error = AgentCliError(
        "Could not deploy",
        failure_detail=(
            "Uploading files\n" * 400 + "Final error: app is busy\npassword=synthetic-opaque-value"
        ),
    )
    message = _log(_Context(), exit_code=1, exception=error)["failure_message"]
    assert message.startswith("Could not deploy")
    assert "[truncated]" in message
    assert "Final error: app is busy" in message
    assert "synthetic-opaque-value" not in message
    assert len(message.encode("utf-8")) <= 4096


@pytest.mark.parametrize(
    "error",
    [
        click.UsageError("private arg"),
        click.Abort(),
        KeyboardInterrupt(),
        SystemExit(1),
        click.exceptions.Exit(1),
    ],
)
def test_parser_and_control_exceptions_have_no_failure_text(error):
    assert "failure_message" not in _log(_Context(), exit_code=1, exception=error)


def test_exception_string_failure_preserves_cause_text():
    class BrokenMessage(Exception):
        def __str__(self):
            raise ValueError("cannot stringify")

    error = BrokenMessage()
    error.__cause__ = ValueError("root cause")
    assert _log(_Context(), exit_code=1, exception=error)["failure_message"] == "root cause"


@pytest.mark.parametrize("outcome", ["DEPLOYMENT_OUTCOME_RUNNING", "DEPLOYMENT_OUTCOME_SUCCEEDED"])
def test_details_link_and_command_share_invocation_id(monkeypatch, outcome):
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    context = _real_context("deploy")
    with context:
        telemetry.begin_deployment()
        telemetry.set_deployment_outcome(outcome)
        url = telemetry.deployment_details_url(
            "https://workspace.example/ml/agents/details/agent-bricks-demo?o=123"
        )
        log = _log(context, exit_code=0)
    from urllib.parse import parse_qs, urlsplit

    assert isinstance(url, str)
    params = parse_qs(urlsplit(url).query)
    assert params["o"] == ["123"]
    assert params["source"] == ["agentbricks-cli"]
    assert params["cli_invocation_id"] == [log["invocation_id"]]
    assert uuid.UUID(log["invocation_id"]).version == 4
    assert log["details_link_emitted"] is True
    assert log["deployment_outcome"] == outcome
    assert log["background_child"] is False
    assert "agent-bricks-demo" not in json.dumps(log)


@pytest.mark.parametrize("background_child,opt_out", [(True, False), (False, True)])
def test_details_link_is_untracked_for_background_or_opt_out(
    monkeypatch, background_child, opt_out
):
    monkeypatch.setenv("AGENTBRICKS_DISABLE_TELEMETRY", "1" if opt_out else "0")
    context = _real_context("deploy")
    url = "https://workspace.example/ml/agents/details/agent-bricks-demo"
    with context:
        telemetry.begin_deployment(background_child=background_child)
        telemetry.set_deployment_outcome("DEPLOYMENT_OUTCOME_RUNNING")
        assert telemetry.deployment_details_url(url) == url
        assert _log(context, exit_code=0)["background_child"] is background_child


def test_missing_details_link_keeps_a_conversion_denominator():
    context = _real_context("deploy")
    with context:
        telemetry.begin_deployment()
        telemetry.set_deployment_outcome("DEPLOYMENT_OUTCOME_SUCCEEDED")
        assert telemetry.deployment_details_url(None) is None
        log = _log(context, exit_code=0)
    assert log["details_link_emitted"] is False
    assert uuid.UUID(log["invocation_id"])


def test_repeated_waits_have_distinct_invocation_ids():
    ids = []
    for _ in range(2):
        context = _real_context("deployments", "wait")
        with context:
            telemetry.begin_deployment()
            ids.append(_log(context, exit_code=75)["invocation_id"])
    assert len(set(ids)) == 2


@pytest.mark.parametrize("background_child", [False, True])
@pytest.mark.parametrize("failure_state_write", [False, True])
@pytest.mark.usefixtures("deployable_tmp_path")
def test_failed_deploy_hook_records_known_outcome(
    tmp_path, monkeypatch, background_child, failure_state_write
):
    """Run the command and service with fake workspace calls and an optional failing disk write."""
    AgentProject.create(tmp_path, framework="langgraph", server="custom").write()
    events = []
    api = _DeployApi(events)
    runner = _DeployRunner(events)
    rollout_error = RuntimeError("rollout failed")
    write_error = OSError("failure state write failed")
    invocation_ids = []
    logs = []

    def fail_rollout(args, profile, **kwargs):
        if list(args[:2]) == ["apps", "deploy"]:
            metadata = telemetry._deployment_metadata(click.get_current_context())
            assert metadata is not None
            invocation_ids.append(metadata.invocation_id)
            raise rollout_error
        return runner(args, profile, **kwargs)

    write_state = deploy_state.write_state

    def write(path, state):
        if failure_state_write and state["status"] == deploy_state.FAILED:
            raise write_error
        write_state(path, state)

    monkeypatch.setattr(deploy_mod, "_databricks", fail_rollout)
    monkeypatch.setattr(cli, "ApiClientProvider", lambda _: SimpleNamespace(get=lambda: api))
    monkeypatch.setattr(deploy_state, "write_state", write)
    monkeypatch.setattr(deploy_mod.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        telemetry, "emit_command", lambda ctx, **kwargs: logs.append(_log(ctx, **kwargs))
    )
    args = ["--profile", "fake", "deploy", "demo", "--source", str(tmp_path)]
    if background_child:
        args.append("--background-child")

    result = CliRunner().invoke(cli.agentbricks, args)

    assert result.exit_code == 1
    assert result.exception is (write_error if failure_state_write else rollout_error)
    (log,) = logs
    assert log["command_path"] == "agentbricks deploy"
    assert log["invocation_id"] == invocation_ids[0]
    assert uuid.UUID(log["invocation_id"]).version == 4
    assert log["exit_code"] == result.exit_code
    assert str(write_error if failure_state_write else rollout_error) in log["failure_message"]
    assert log["background_child"] is background_child
    assert log["details_link_emitted"] is False
    assert log["deployment_outcome"] == "DEPLOYMENT_OUTCOME_FAILED"


@pytest.mark.parametrize("background_child", [False, True])
@pytest.mark.usefixtures("deployable_tmp_path")
def test_preparation_failure_hook_keeps_outcome_unspecified(
    tmp_path, monkeypatch, background_child
):
    AgentProject.create(tmp_path, framework="langgraph", server="custom").write()
    events = []
    api = _DeployApi(events)
    logs = []
    monkeypatch.setattr(deploy_mod, "_databricks", _DeployRunner(events))
    monkeypatch.setattr(cli, "ApiClientProvider", lambda _: SimpleNamespace(get=lambda: api))
    monkeypatch.setattr(deploy_mod.signal, "signal", lambda *args: None)
    monkeypatch.setattr(
        telemetry, "emit_command", lambda ctx, **kwargs: logs.append(_log(ctx, **kwargs))
    )
    args = ["--profile", "fake", "deploy", "--source", str(tmp_path)]
    if background_child:
        args.append("--background-child")

    result = CliRunner().invoke(cli.agentbricks, args)

    assert result.exit_code == 1
    assert "No deployment name given" in result.output
    assert events == []
    (log,) = logs
    assert uuid.UUID(log["invocation_id"]).version == 4
    assert log["exit_code"] == result.exit_code
    assert log["background_child"] is background_child
    assert log["details_link_emitted"] is False
    assert log["deployment_outcome"] == "DEPLOYMENT_OUTCOME_UNSPECIFIED"


def test_outcome_setter_failure_preserves_command_exception(monkeypatch):
    error = RuntimeError("original command failure")

    def unavailable_context(*args, **kwargs):
        raise RuntimeError("telemetry context unavailable")

    @click.command(cls=AgentBricksCommand)
    def command():
        telemetry.begin_deployment()
        try:
            raise error
        finally:
            with monkeypatch.context() as patch:
                patch.setattr(click, "get_current_context", unavailable_context)
                telemetry.set_deployment_outcome("DEPLOYMENT_OUTCOME_FAILED")

    result = CliRunner().invoke(command)
    assert result.exit_code == 1
    assert result.exception is error


@pytest.mark.parametrize("command", ["get", "list"])
@pytest.mark.parametrize("enabled,opt_out", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize(
    "output,console_options,list_link_visible",
    [
        pytest.param("json", {"force_terminal": False}, True, id="json"),
        pytest.param("text", {"force_terminal": False}, False, id="redirected-text"),
        pytest.param("text", {"force_terminal": True}, True, id="terminal"),
        pytest.param(
            "text", {"force_terminal": True, "color_system": None}, False, id="plain-terminal"
        ),
        pytest.param(
            "text", {"force_terminal": True, "legacy_windows": True}, False, id="legacy-windows"
        ),
        pytest.param("text", {"force_terminal": True, "no_color": True}, True, id="no-color"),
        pytest.param(
            "text",
            {"force_terminal": False, "color_system": "standard"},
            True,
            id="redirected-ansi",
        ),
    ],
)
def test_read_command_links_join_one_command_event(
    monkeypatch, command, enabled, opt_out, output, console_options, list_link_visible
):
    monkeypatch.setenv("AGENTBRICKS_DISABLE_TELEMETRY", "1" if opt_out else "0")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(render, "_stdout", Console(width=300, **console_options))
    items = [
        {"name": "agent-bricks-one", "url": "https://one.example"},
        {"name": "agent-bricks-two", "url": "https://two.example"},
    ]
    uploads = []
    flag_requests = []

    class WorkspaceTransport:
        def do(self, method, path, *, query=None, body=None):
            if method == "GET":
                flag_requests.append(path)
                return {
                    "flags": {
                        "databricks.fe.customagents.agentCentricView.enable": enabled,
                        "agent_memory_sessions": True,
                    }
                }
            assert (method, path) == ("POST", "/telemetry-ext")
            uploads.append(body)
            return {}

    client = SimpleNamespace(
        host="https://workspace.example",
        workspace_client=SimpleNamespace(
            config=SimpleNamespace(workspace_id="123"), api_client=WorkspaceTransport()
        ),
    )
    # Fake only external Apps reads and workspace transport; the command hook, attribution,
    # event construction, and emitter all execute together.
    monkeypatch.setattr(
        cli, "ApiClientProvider", lambda _: SimpleNamespace(get=lambda: client, _client=client)
    )
    monkeypatch.setattr(
        deploy_mod,
        "_databricks",
        lambda args, *a, **kw: SimpleNamespace(
            returncode=0, stdout=json.dumps(items if command == "list" else items[0]), stderr=""
        ),
    )
    args = ["--profile", "fake", "--output", output, "deployments", command]
    if command == "get":
        args.append("agent-bricks-one")
    result = CliRunner().invoke(cli.agentbricks, args, color=True)
    assert result.exit_code == 0, result.output
    link_visible = enabled and (command == "get" or list_link_visible)
    assert flag_requests == ["/api/2.0/ui-flags"]
    assert len(uploads) == (0 if opt_out else 1)
    if not opt_out:
        log = json.loads(uploads[0]["protoLogs"][0])["entry"]["agentbricks_cli_log"]
        assert log["command_path"] == f"agentbricks deployments {command}"
        assert log["deployment_outcome"] == "DEPLOYMENT_OUTCOME_UNSPECIFIED"
        assert log["background_child"] is False
        assert log["details_link_emitted"] is link_visible
        assert uuid.UUID(log["invocation_id"]).version == 4
        assert "agent-bricks-one" not in json.dumps(log)
    if output != "json":
        assert "agent-bricks-one" in result.stdout
        assert ("/ml/agents/details/agent-bricks-one" in result.stdout) is link_visible
        if opt_out:
            assert "cli_invocation_id" not in result.stdout
        else:
            assert (log["invocation_id"] in result.stdout) is link_visible
        return
    payload = json.loads(result.stdout)
    rows = payload if command == "list" else [payload]
    for row in rows:
        assert row["url"] == next(item["url"] for item in items if item["name"] == row["name"])
        if not enabled:
            assert "details_url" not in row
            continue
        parts = urlsplit(row["details_url"])
        assert parts.path == f"/ml/agents/details/{row['name']}"
        query = parse_qs(parts.query)
        if opt_out:
            assert query == {"o": ["123"]}
        else:
            assert query == {
                "o": ["123"],
                "source": ["agentbricks-cli"],
                "cli_invocation_id": [log["invocation_id"]],
            }


@pytest.mark.parametrize("command", ["get", "list"])
def test_read_command_failure_keeps_outcome_unspecified(monkeypatch, command):
    logs = []
    monkeypatch.setattr(
        telemetry, "emit_command", lambda ctx, **kwargs: logs.append(_log(ctx, **kwargs))
    )

    def unavailable(*args, **kwargs):
        raise AgentCliError("unavailable", error_code="UNAVAILABLE")

    monkeypatch.setattr(deploy_mod, "_databricks", unavailable)
    args = ["--profile", "fake", "deployments", command]
    if command == "get":
        args.append("agent-bricks-one")
    result = CliRunner().invoke(cli.agentbricks, args)
    assert result.exit_code == 1
    (log,) = logs
    assert log["command_path"] == f"agentbricks deployments {command}"
    assert uuid.UUID(log["invocation_id"]).version == 4
    assert log["details_link_emitted"] is False
    assert log["deployment_outcome"] == "DEPLOYMENT_OUTCOME_UNSPECIFIED"
