"""Unit coverage for Agent Bricks CLI FrontendLog telemetry."""

from __future__ import annotations

import json
import threading
import time
import uuid
from enum import Enum
from types import SimpleNamespace
from unittest import mock

import click
import pytest
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
