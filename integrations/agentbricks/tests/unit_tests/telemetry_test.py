"""Unit coverage for Agent Bricks CLI FrontendLog telemetry."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest import mock

import click
from click.core import ParameterSource
from click.testing import CliRunner

import databricks_agentbricks.cli.app as cli
from databricks_agentbricks.cli import telemetry
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.presentation.group import AgentBricksCommand


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


_PRESENCE_ONLY_CHOICE_OPTIONS = {
    "agentbricks memory pipeline create.trigger",
    "agentbricks memory pipeline update.trigger",
}


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

    log = telemetry.build_log(
        context,
        execution_time_ms=1,
        exit_code=0,
    )

    assert log["command_path"] == "agentbricks init"
    assert "private-profile" not in json.dumps(log)

    context.command = SimpleNamespace(name="init --private-profile", params=())
    fallback_log = telemetry.build_log(
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
    }
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

    assert log["error_category"] == "ERROR_CATEGORY_AUTHORIZATION"
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
    assert _inner(body)["agentbricks_cli_log"]["exit_code"] == 0


def test_transport_payload_contains_static_parameter_names(monkeypatch):
    api_client = _ApiClient()
    context = _real_context("deploy")
    _explicit_option(context, "workspace_path", "private-label")
    _explicit_option(context, "allow_user_scope_update", True)

    monkeypatch.delenv("AGENTBRICKS_DISABLE_TELEMETRY", raising=False)
    monkeypatch.setattr(telemetry, "_MAX_FOREGROUND_WAIT_S", 1)
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
            "name": "agentbricks deploy.allow_user_scope_update"
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
        lambda ctx, **kwargs: captured.append(telemetry.build_log(ctx, **kwargs)),
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


def test_build_log_captures_only_explicit_options_and_allowlisted_values():
    context = _real_context("deploy")
    _explicit_option(context, "instance_count", 3)
    _explicit_option(context, "workspace_path", "private-secret")
    _explicit_option(context, "allow_user_scope_update", True)

    log = telemetry.build_log(context, exit_code=0)
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
    }
    assert len(parameters) == 3
    encoded = json.dumps(log)
    assert "private-secret" not in encoded


def test_build_log_omits_default_options():
    context = _real_context("deploy")
    _explicit_option(context, "instance_count", 2, source=ParameterSource.DEFAULT)
    assert "parameters" not in telemetry.build_log(context, exit_code=0)


def test_unallowlisted_choice_is_presence_only():
    context = _real_context("memory", "pipeline", "create")
    _explicit_option(context, "trigger", "manual")
    log = telemetry.build_log(context, exit_code=0)
    assert log["parameters"] == [{"name": "agentbricks memory pipeline create.trigger"}]
    assert "manual" not in json.dumps(log)


def test_choice_policy_covers_every_registered_choice_and_value():
    declared_choices = {
        f"{' '.join(('agentbricks', *path))}.{option.name}": frozenset(option.type.choices)
        for path, option in _real_options()
        if isinstance(option.type, click.Choice)
    }

    assert not (set(telemetry._SAFE_CHOICE_VALUES) & _PRESENCE_ONLY_CHOICE_OPTIONS)
    assert set(declared_choices) == (
        set(telemetry._SAFE_CHOICE_VALUES) | _PRESENCE_ONLY_CHOICE_OPTIONS
    )
    for name, values in telemetry._SAFE_CHOICE_VALUES.items():
        assert values == declared_choices[name]


def test_safe_choice_values_use_registered_data_shape_prefix():
    for path, option in _real_options():
        name = f"{' '.join(('agentbricks', *path))}.{option.name}"
        for value in telemetry._SAFE_CHOICE_VALUES.get(name, ()):
            entry = telemetry._parameter_entry(name=name, parameter=option, value=value)
            assert entry is not None
            assert entry.as_log() == {
                "name": name,
                "choice_value": f"AGENTBRICKS_CLI_{value.upper().replace('-', '_')}",
            }


def test_real_registered_options_only_emit_reviewed_values():
    for path, option in _real_options():
        assert option.name is not None
        name = f"{' '.join(('agentbricks', *path))}.{option.name}"
        if isinstance(option.type, click.Choice):
            value = option.type.choices[0]
        elif option.is_bool_flag:
            value = True
        elif isinstance(option.type, click.IntRange):
            value = option.type.min if option.type.min is not None else 1
        else:
            value = "private-value"
        if option.multiple:
            value = (value,)

        context = _real_context(*path)
        _explicit_option(context, option.name, value)
        log = telemetry.build_log(context, exit_code=0)

        if option.hidden or option.deprecated:
            assert "parameters" not in log, name
            continue
        assert len(log["parameters"]) == 1, name
        entry = log["parameters"][0]
        assert entry["name"] == name
        if name in telemetry._SAFE_CHOICE_VALUES:
            assert isinstance(value, str)
            assert entry["choice_value"] == f"AGENTBRICKS_CLI_{value.upper().replace('-', '_')}"
        elif name in telemetry._SAFE_BOOL_PARAMETER_NAMES:
            assert entry["bool_value"] is True
        elif name in telemetry._SAFE_BOUNDED_INT_RANGES:
            assert entry["int_value"] == value
        else:
            assert entry == {"name": name}


def test_bounded_integer_policy_references_registered_options():
    declared_ranges = {
        f"{' '.join(('agentbricks', *path))}.{option.name}": option.type
        for path, option in _real_options()
        if isinstance(option.type, click.IntRange)
    }
    for name, (minimum, maximum) in telemetry._SAFE_BOUNDED_INT_RANGES.items():
        assert name in declared_ranges
        assert declared_ranges[name].min <= minimum <= maximum <= declared_ranges[name].max


def test_parameter_entry_preserves_false_boolean_value():
    option = next(
        option
        for path, option in _real_options()
        if path == ("init",) and option.name == "existing"
    )
    entry = telemetry._parameter_entry(
        name="agentbricks init.existing", parameter=option, value=False
    )
    assert entry is not None
    assert entry.as_log() == {"name": "agentbricks init.existing", "bool_value": False}


def test_safe_boolean_allowlist_contains_only_current_boolean_options():
    declared_flags = {
        f"{' '.join(('agentbricks', *path))}.{option.name}"
        for path, option in _real_options()
        if option.is_bool_flag
    }

    assert telemetry._SAFE_BOOL_PARAMETER_NAMES <= declared_flags
