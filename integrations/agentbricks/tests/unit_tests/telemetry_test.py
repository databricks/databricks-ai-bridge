"""Unit coverage for Agent Bricks CLI FrontendLog telemetry."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest import mock

import click
from click.testing import CliRunner

import databricks_agentbricks.cli.app as cli
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
        )
        self.command = SimpleNamespace(name="init", params=())
        self.parent = SimpleNamespace(
            command=SimpleNamespace(name="agentbricks", params=()),
            parent=None,
            obj=self.obj,
        )

    def find_root(self):
        return self


def _inner(body: dict) -> dict:
    assert body["items"] == []
    assert len(body["protoLogs"]) == 1
    return json.loads(body["protoLogs"][0])["entry"]


def test_build_payload_matches_frontend_log_schema():
    context = _Context()
    log = telemetry.build_log(
        context,
        execution_time_ms=17,
        exit_code=0,
    )
    payload = telemetry.build_payload(log, upload_time_ms=123)

    assert payload["uploadTime"] == 123
    encoded = json.loads(payload["protoLogs"][0])
    assert uuid.UUID(encoded["frontend_log_event_id"])
    assert _inner(payload) == {
        "agentbricks_cli_log": {
            "execution_context": {
                "command_path": "agentbricks init",
                "execution_time_ms": 17,
                "exit_code": 0,
                "operating_system": telemetry.platform.system().lower(),
                "package_version": telemetry._package_version(),
            }
        }
    }
    assert "/tmp" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload)


def test_build_log_uses_static_click_path_instead_of_raw_context_path():
    context = _Context()
    context.command_path = "agentbricks init --profile private-profile"

    log = telemetry.build_log(
        context,
        execution_time_ms=1,
        exit_code=0,
    )

    assert log["execution_context"]["command_path"] == "agentbricks init"
    assert "private-profile" not in json.dumps(log)

    context.command = SimpleNamespace(name="init --private-profile", params=())
    fallback_log = telemetry.build_log(
        context,
        execution_time_ms=1,
        exit_code=0,
    )
    assert fallback_log["execution_context"]["command_path"] == "agentbricks"


def test_real_init_telemetry_captures_safe_options_without_values(monkeypatch, tmp_path):
    records = []

    monkeypatch.setattr(
        telemetry,
        "emit_command",
        lambda ctx, **kwargs: records.append(telemetry.build_log(ctx, **kwargs)),
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
    assert set(log) == {"execution_context", "parameters"}
    context = log["execution_context"]
    assert set(context) == {
        "command_path",
        "package_version",
        "operating_system",
        "execution_time_ms",
        "exit_code",
    }
    assert context["command_path"] == "agentbricks init"
    assert context["exit_code"] == 0
    assert context["execution_time_ms"] >= 0
    parameters = {parameter["name"]: parameter for parameter in log["parameters"]}
    assert parameters["agentbricks.output"] == {
        "name": "agentbricks.output",
        "choice_value": "json",
    }
    assert parameters["agentbricks init.framework"] == {
        "name": "agentbricks init.framework",
        "choice_value": "openai",
    }
    assert parameters["agentbricks init.server"] == {
        "name": "agentbricks init.server",
        "choice_value": "custom",
    }
    assert parameters["agentbricks init.disable_chat_app"] == {
        "name": "agentbricks init.disable_chat_app",
        "bool_value": True,
    }
    assert parameters["agentbricks init.profile"] == {
        "name": "agentbricks init.profile",
    }
    # DIRECTORY is a positional argument and remains absent even though it was supplied.
    assert "agentbricks init.directory" not in parameters
    encoded = json.dumps(log)
    assert "private-project-name" not in encoded
    assert "private-profile-name" not in encoded


def test_failed_payload_uses_bounded_category_and_no_message():
    context = _Context()
    log = telemetry.build_log(
        context,
        execution_time_ms=4,
        exit_code=1,
        error_category=telemetry._error_category(
            AgentCliError("secret message /tmp/private", error_code="PERMISSION_DENIED")
        ),
    )

    assert log["execution_context"]["error_category"] == "ERROR_CATEGORY_AUTHORIZATION"
    encoded = json.dumps(log)
    assert "secret message" not in encoded
    assert "/tmp/private" not in encoded


def test_keyboard_interrupt_uses_click_abort_exit_code():
    assert telemetry._exit_code(KeyboardInterrupt()) == 1
    assert telemetry._error_category(KeyboardInterrupt()) == "ERROR_CATEGORY_INTERRUPTED"
    assert telemetry._exit_code(SystemExit()) == 0


def test_existing_workspace_client_receives_exact_transport_request(monkeypatch):
    api_client = _ApiClient()
    context = _Context(api_client)
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_MAX_FOREGROUND_WAIT_S", 1)
    monkeypatch.setattr(telemetry.time, "time", lambda: 12.3)

    telemetry.emit_command(context, execution_time_ms=5, exit_code=0)

    assert len(api_client.calls) == 1
    method, path, body = api_client.calls[0]
    assert (method, path) == ("POST", "/telemetry-ext")
    assert body["uploadTime"] == 12300
    assert _inner(body)["agentbricks_cli_log"]["execution_context"]["exit_code"] == 0


def test_transport_payload_contains_bounded_parameters(monkeypatch):
    api_client = _ApiClient()

    @click.command(name="inventory", cls=AgentBricksCommand)
    @click.option("--label", multiple=True)
    @click.option("--enabled", is_flag=True)
    @click.pass_context
    def inventory(ctx, label, enabled):
        del ctx, label, enabled

    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_MAX_FOREGROUND_WAIT_S", 1)
    monkeypatch.setattr(
        telemetry,
        "_workspace_client",
        lambda ctx: SimpleNamespace(api_client=api_client),
    )
    result = CliRunner().invoke(
        inventory,
        ["--label", "private-label", "--label", "private-label-two", "--enabled"],
    )

    assert result.exit_code == 0, result.output
    assert len(api_client.calls) == 1
    log = _inner(api_client.calls[0][2])["agentbricks_cli_log"]
    parameters = {parameter["name"]: parameter for parameter in log["parameters"]}
    assert parameters == {
        "inventory.label": {"name": "inventory.label"},
        "inventory.enabled": {"name": "inventory.enabled"},
    }
    assert log["execution_context"]["command_path"] == "inventory"
    encoded = json.dumps(log)
    assert "private-label" not in encoded
    assert "private-label-two" not in encoded


def test_opt_out_does_not_construct_or_send(monkeypatch):
    context = _Context(_ApiClient())
    monkeypatch.setenv("AGENTBRICKS_DISABLE_TELEMETRY", "true")
    with mock.patch.object(telemetry, "_workspace_client") as workspace_client:
        telemetry.emit_command(context, execution_time_ms=1, exit_code=0)
    workspace_client.assert_not_called()


def test_no_auth_skips_client_creation(monkeypatch):
    context = _Context()
    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("DATABRICKS_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    with mock.patch.object(telemetry, "_workspace_client", return_value=None) as workspace_client:
        telemetry.emit_command(context, execution_time_ms=1, exit_code=0)
    workspace_client.assert_called_once_with(context)


def test_transport_failure_is_silent_and_does_not_change_result(monkeypatch):
    context = _Context(_ApiClient(RuntimeError("transport detail")))
    monkeypatch.setattr(telemetry, "_MAX_FOREGROUND_WAIT_S", 1)
    telemetry.emit_command(context, execution_time_ms=1, exit_code=0)


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


def test_build_log_captures_only_explicit_options_and_allowlisted_values(monkeypatch):
    captured = []

    @click.group(name="agentbricks")
    def root():
        pass

    @root.command(name="deploy", cls=AgentBricksCommand)
    @click.option("--instances", type=click.IntRange(min=1, max=5), default=2)
    @click.option("--default-secret", default="secret-default")
    @click.option("--enabled", is_flag=True)
    @click.option("--hidden", is_flag=True, hidden=True)
    @click.option("--deprecated", is_flag=True, deprecated=True)
    @click.option("--kind", type=click.Choice(["private-choice"]))
    @click.option("--label", multiple=True)
    @click.argument("directory", required=False)
    @click.pass_context
    def deploy(ctx, instances, default_secret, enabled, hidden, deprecated, kind, label, directory):
        del ctx, instances, default_secret, enabled, hidden, deprecated, kind, label, directory

    monkeypatch.setattr(
        telemetry,
        "emit_command",
        lambda ctx, **kwargs: captured.append(telemetry.build_log(ctx, **kwargs)),
    )
    result = CliRunner().invoke(
        root,
        [
            "deploy",
            "private-directory",
            "--instances",
            "3",
            "--default-secret",
            "private-secret",
            "--enabled",
            "--hidden",
            "--deprecated",
            "--kind",
            "private-choice",
            "--label",
            "private-label-one",
            "--label",
            "private-label-two",
        ],
    )

    assert result.exit_code == 0, result.output
    parameters = {parameter["name"]: parameter for parameter in captured[0]["parameters"]}
    assert parameters["agentbricks deploy.instances"] == {
        "name": "agentbricks deploy.instances",
        "bounded_int_value": 3,
    }
    assert parameters["agentbricks deploy.default_secret"] == {
        "name": "agentbricks deploy.default_secret",
    }
    assert parameters["agentbricks deploy.enabled"] == {
        "name": "agentbricks deploy.enabled",
    }
    assert parameters["agentbricks deploy.label"] == {
        "name": "agentbricks deploy.label",
    }
    assert parameters["agentbricks deploy.kind"] == {
        "name": "agentbricks deploy.kind",
    }
    assert "agentbricks deploy.hidden" not in parameters
    assert "agentbricks deploy.deprecated" not in parameters
    assert "agentbricks deploy.directory" not in parameters
    encoded = json.dumps(captured[0])
    assert "private-secret" not in encoded
    assert "private-choice" not in encoded
    assert "private-label-one" not in encoded
    assert "private-label-two" not in encoded
    assert "private-directory" not in encoded


def test_build_log_omits_default_options(monkeypatch):
    captured = []

    @click.group(name="agentbricks")
    def root():
        pass

    @root.command(name="deploy", cls=AgentBricksCommand)
    @click.option("--instances", type=click.IntRange(min=1, max=5), default=2)
    @click.option("--enabled", is_flag=True)
    @click.pass_context
    def deploy(ctx, instances, enabled):
        del ctx, instances, enabled

    monkeypatch.setattr(
        telemetry,
        "emit_command",
        lambda ctx, **kwargs: captured.append(telemetry.build_log(ctx, **kwargs)),
    )
    result = CliRunner().invoke(root, ["deploy"])

    assert result.exit_code == 0, result.output
    assert "parameters" not in captured[0]


def test_unallowlisted_choice_is_presence_only(monkeypatch):
    captured = []

    @click.command(name="inventory", cls=AgentBricksCommand)
    @click.option("--source", type=click.Choice(["private-choice"]))
    @click.pass_context
    def inventory(ctx, source):
        del ctx, source

    monkeypatch.setattr(
        telemetry,
        "emit_command",
        lambda ctx, **kwargs: captured.append(telemetry.build_log(ctx, **kwargs)),
    )
    result = CliRunner().invoke(inventory, ["--source", "private-choice"])

    assert result.exit_code == 0, result.output
    assert captured[0]["parameters"] == [{"name": "inventory.source"}]
    assert "private-choice" not in json.dumps(captured[0])


def test_safe_choice_allowlist_is_a_subset_of_current_click_choices():
    declared_choices = {}

    def collect_parameters(command, path):
        for parameter in command.params:
            if isinstance(parameter, click.Option) and isinstance(parameter.type, click.Choice):
                declared_choices[f"{' '.join(path)}.{parameter.name}"] = set(parameter.type.choices)
        if isinstance(command, click.Group):
            for child_name, child in command.commands.items():
                collect_parameters(child, (*path, child_name))

    collect_parameters(cli.agentbricks, ("agentbricks",))

    for name, values in telemetry._SAFE_CHOICE_VALUES.items():
        assert name in declared_choices
        assert values <= declared_choices[name]


def test_safe_boolean_allowlist_contains_only_current_boolean_options():
    declared_flags = set()

    def collect_flags(command, path):
        for parameter in command.params:
            if isinstance(parameter, click.Option) and parameter.is_bool_flag:
                declared_flags.add(f"{' '.join(path)}.{parameter.name}")
        if isinstance(command, click.Group):
            for child_name, child in command.commands.items():
                collect_flags(child, (*path, child_name))

    collect_flags(cli.agentbricks, ("agentbricks",))

    assert telemetry._SAFE_BOOL_PARAMETER_NAMES <= declared_flags
