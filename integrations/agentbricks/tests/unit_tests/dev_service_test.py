"""Unit coverage for the framework-agnostic local development service."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import yaml

from databricks_agentbricks.clients.apps_client import AppsClient
from databricks_agentbricks.clients.local_tracing_client import (
    LocalTracingClient,
    LocalTracingStart,
)
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.resolver import ProjectResolver
from databricks_agentbricks.projects.types import AgentServer
from databricks_agentbricks.services import dev_preflight as dev_preflight_module
from databricks_agentbricks.services import dev_service as dev_service_module
from databricks_agentbricks.services.dev_service import DevRequest, DevService
from databricks_agentkit.runtime.cors import CORS_ALLOWED_ORIGINS_ENV


@pytest.fixture
def service_fixture(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Build a local service with all external ports replaced by test doubles."""
    (tmp_path / "app.yaml").write_text(
        """command: [python, app.py]
env:
  - name: KEEP
    value: original
  - name: PIP_INDEX_URL
    value: https://deployed.example/simple/
  - name: MLFLOW_EXPERIMENT_ID
    value: stale-workspace-id
  - name: AGENT_MEMORY_STORE
    value: stale-memory
"""
    )
    project = SimpleNamespace(server=AgentServer.AGENTBRICKS, tools=[], allowed_origins=())

    resolver = Mock(spec=ProjectResolver)
    resolver.load.return_value = project
    resolver.resource_bindings.return_value = ("memory", "sessions", "/Shared/traces")

    apps = Mock(spec=AppsClient)
    tracing = Mock(spec=LocalTracingClient)
    tracing_server = cast(subprocess.Popen, Mock(spec=subprocess.Popen))
    tracing.start_dev.return_value = LocalTracingStart(
        tracing_server,
        "http://127.0.0.1:5599",
        environment={
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:5599",
            "MLFLOW_EXPERIMENT_NAME": "local-agent",
        },
    )

    port_in_use = Mock(return_value=False)
    free_port = Mock(return_value=54321)

    # Hermetic by default: the uv/Python preflight is exercised by the preflight tests below,
    # which route it through the real checks with injected fakes.
    monkeypatch.setattr(dev_service_module, "run_dev_preflight", Mock())

    service = DevService(
        project_resolver=resolver,
        apps_client=apps,
        local_tracing=tracing,
        port_in_use=port_in_use,
        free_port=free_port,
    )
    return SimpleNamespace(
        apps=apps,
        free_port=free_port,
        port_in_use=port_in_use,
        project=project,
        resolver=resolver,
        service=service,
        tracing=tracing,
        tracing_server=tracing_server,
    )


def _manifest_env(path: pathlib.Path) -> dict[str, str]:
    return {entry["name"]: entry["value"] for entry in yaml.safe_load(path.read_text())["env"]}


@pytest.mark.parametrize("origins", [("https://chat.example.com",), ()])
def test_dev_manifest_uses_current_origins_and_overrides_stale_deploy_env(
    service_fixture,
    tmp_path,
    origins,
):
    app_yaml = tmp_path / "app.yaml"
    text = app_yaml.read_text() + (
        f"  - name: {CORS_ALLOWED_ORIGINS_ENV}\n    value: '[\"https://stale.example.com\"]'\n"
    )
    app_yaml.write_text(text)
    service_fixture.project.allowed_origins = origins
    with service_fixture.service.prepare(
        DevRequest(
            source=str(tmp_path),
            prepare_environment=False,
            app_port=9000,
        )
    ) as plan:
        env = _manifest_env(plan.entry_point)
        if origins:
            assert json.loads(env[CORS_ALLOWED_ORIGINS_ENV]) == list(origins)
        else:
            assert CORS_ALLOWED_ORIGINS_ENV not in env
    assert app_yaml.read_text() == text


def test_prepare_builds_local_plan_and_run_uses_injected_apps_client(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    """Preparation resolves metadata once and keeps the temporary manifest alive through run."""
    app_yaml = tmp_path / "app.yaml"
    original_app_yaml = app_yaml.read_text()

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=None, app_port=9000)
    ) as plan:
        assert plan.preview.source_dir == tmp_path
        assert plan.preview.port == 9000
        assert plan.preview.server is AgentServer.AGENTBRICKS
        assert plan.preview.tracing_uri == "http://127.0.0.1:5599"
        assert plan.preview.local_experiment_name == "local-agent"
        assert plan.preview.tracing_warning is None
        assert plan.preview.tracing_help is None
        assert plan.preview.memory_store == "memory"
        assert plan.preview.session_store == "sessions"
        assert plan.preview.trace_experiment == "/Shared/traces"
        assert plan.preview.has_chat_ui is False
        assert plan.prepare_environment is True  # no .venv: auto-prepare
        assert plan.app_port == 54321
        assert plan.entry_point.name == "app.agentbricksdev.yaml"
        assert plan.entry_point.exists()
        assert _manifest_env(plan.entry_point) == {
            "KEEP": "original",
            "MLFLOW_TRACKING_URI": "http://127.0.0.1:5599",
            "MLFLOW_EXPERIMENT_NAME": "local-agent",
            # Display-only, so the chat app can say the bound store connects on deploy.
            "AGENTBRICKS_DEV_BOUND_MEMORY_STORE": "memory",
            "DATABRICKS_AGENTBRICKS_RUNTIME_STORE_LOCAL": "true",
        }
        # Memory itself stays off: the store setting the runtime reads is not set locally.
        assert "AGENT_MEMORY_STORE" not in _manifest_env(plan.entry_point)

        service_fixture.service.run(plan)

    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    assert app_yaml.read_text() == original_app_yaml
    service_fixture.resolver.load.assert_called_once_with(tmp_path)
    service_fixture.resolver.resource_bindings.assert_called_once_with(tmp_path)
    service_fixture.tracing.start_dev.assert_called_once_with(tmp_path)
    service_fixture.tracing.stop.assert_called_once_with(service_fixture.tracing_server)
    service_fixture.apps.run_local.assert_called_once_with(
        tmp_path,
        "app.agentbricksdev.yaml",
        prepare_environment=True,
        port=9000,
        app_port=54321,
        on_ready=None,
        quiet_status=None,
    )


def test_prepare_honors_explicit_environment_flag_and_default_port(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    (tmp_path / ".venv").mkdir()

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        assert plan.prepare_environment is False
        assert plan.preview.port == 8000
        assert plan.app_port == 54321
        service_fixture.service.run(plan)

    service_fixture.apps.run_local.assert_called_once_with(
        tmp_path,
        "app.agentbricksdev.yaml",
        prepare_environment=False,
        port=8000,
        app_port=54321,
        on_ready=None,
        quiet_status=None,
    )


def test_missing_app_yaml_fails_before_loading_project_or_starting_tracing(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    (tmp_path / "app.yaml").unlink()

    with pytest.raises(AgentCliError, match="No app.yaml found"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("prepare should fail before yielding a plan")

    service_fixture.resolver.load.assert_not_called()
    service_fixture.resolver.resource_bindings.assert_not_called()
    service_fixture.tracing.start_dev.assert_not_called()
    service_fixture.apps.run_local.assert_not_called()


def test_managed_tool_preflight_fails_before_local_resources(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_fixture.project.tools = [object()]
    require_support = Mock(side_effect=AgentCliError("managed tools are unsupported"))
    monkeypatch.setattr(dev_service_module, "require_managed_tool_support", require_support)

    with pytest.raises(AgentCliError, match="managed tools are unsupported"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("managed-tool preflight should fail before yielding a plan")

    require_support.assert_called_once_with(tmp_path)
    service_fixture.resolver.resource_bindings.assert_not_called()
    service_fixture.tracing.start_dev.assert_not_called()
    service_fixture.apps.run_local.assert_not_called()


def test_run_failure_still_removes_manifest_and_stops_tracing(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    service_fixture.apps.run_local.side_effect = RuntimeError("run-local failed")

    with pytest.raises(RuntimeError, match="run-local failed"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=False, app_port=8123)
        ) as plan:
            service_fixture.service.run(plan)

    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    service_fixture.tracing.stop.assert_called_once_with(service_fixture.tracing_server)


def test_malformed_manifest_fails_before_tracing_starts(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    # The manifest is parsed once, before tracing starts, so a malformed app.yaml fails without
    # starting (and having to tear down) the local tracing server.
    (tmp_path / "app.yaml").write_text("env: not-a-list\n")

    with pytest.raises(AgentCliError, match="env must be a list"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("malformed app.yaml should fail while preparing the manifest")

    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    service_fixture.apps.run_local.assert_not_called()
    service_fixture.tracing.start_dev.assert_not_called()
    service_fixture.tracing.stop.assert_not_called()


def test_manifest_write_failure_removes_partial_manifest_and_stops_tracing(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = pathlib.Path.open

    def failing_open(path: pathlib.Path, mode: str = "r", *args, **kwargs):
        output = real_open(path, mode, *args, **kwargs)
        if path.name != "app.agentbricksdev.yaml" or mode != "w":
            return output

        class FailingOutput:
            def __enter__(self):
                output.__enter__()
                return self

            def __exit__(self, *exit_args):
                return output.__exit__(*exit_args)

            def write(self, value: str) -> int:
                output.write("partial")
                output.flush()
                raise OSError("disk full")

        return FailingOutput()

    monkeypatch.setattr(pathlib.Path, "open", failing_open)
    app_yaml = tmp_path / "app.yaml"
    original_app_yaml = app_yaml.read_text()

    with pytest.raises(AgentCliError, match="Could not write .*disk full"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("prepare should fail when the local manifest cannot be written")

    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    assert app_yaml.read_text() == original_app_yaml
    service_fixture.apps.run_local.assert_not_called()
    service_fixture.tracing.stop.assert_called_once_with(service_fixture.tracing_server)


def test_prepare_and_run_remain_safe_when_tracing_does_not_start(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    service_fixture.tracing.start_dev.return_value = LocalTracingStart(
        None, None, warning="local tracing unavailable", help="install uv"
    )

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=8123)
    ) as plan:
        assert plan.preview.tracing_uri is None
        assert plan.preview.local_experiment_name is None
        assert plan.preview.tracing_warning == "local tracing unavailable"
        assert plan.preview.tracing_help == "install uv"
        assert plan.entry_point.exists()
        service_fixture.service.run(plan)

    assert not (tmp_path / "app.agentbricksdev.yaml").exists()
    service_fixture.tracing.stop.assert_not_called()
    service_fixture.apps.run_local.assert_called_once_with(
        tmp_path,
        "app.agentbricksdev.yaml",
        prepare_environment=False,
        port=8123,
        app_port=54321,
        on_ready=None,
        quiet_status=None,
    )


@pytest.mark.parametrize(("app_port", "expected_port"), [(None, 8000), (9100, 9100)])
def test_busy_port_fails_before_tracing_or_local_manifest(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path, app_port, expected_port
) -> None:
    service_fixture.port_in_use.return_value = True

    with pytest.raises(AgentCliError) as raised:
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=app_port)
        ):
            pytest.fail("a busy port must not yield a plan")

    assert raised.value.message == (
        f"Port {expected_port} is already in use (possibly a previous agentbricks dev run)."
    )
    assert raised.value.hint == (
        "Stop that process, or run `agentbricks dev --app-port <port>` with a free port."
    )
    service_fixture.port_in_use.assert_called_once_with(expected_port)
    service_fixture.tracing.start_dev.assert_not_called()
    service_fixture.tracing.stop.assert_not_called()
    service_fixture.apps.run_local.assert_not_called()
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()


def test_free_port_is_checked_and_run_proceeds(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=None, app_port=9100)
    ) as plan:
        service_fixture.service.run(plan)

    service_fixture.port_in_use.assert_called_once_with(9100)
    service_fixture.tracing.start_dev.assert_called_once_with(tmp_path)
    service_fixture.apps.run_local.assert_called_once()


def test_prepare_omits_the_bound_memory_store_note_when_none_is_bound(
    service_fixture: SimpleNamespace, tmp_path: pathlib.Path
) -> None:
    service_fixture.resolver.resource_bindings.return_value = (None, None, None)
    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        assert "AGENTBRICKS_DEV_BOUND_MEMORY_STORE" not in _manifest_env(plan.entry_point)


class _UvRun:
    """Fake subprocess.run for the uv probes: records calls and returns scripted outcomes.

    Outcomes are keyed by the subcommand (e.g. ``("python", "find")``); an Exception outcome is
    raised instead of returned, like a real spawn or timeout failure.
    """

    def __init__(
        self,
        results: dict[tuple[str, ...], object] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self._results = results or {}
        self._events = events

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), kwargs))
        if self._events is not None:
            self._events.append(args[2])  # "find" / "install"
        outcome = self._results[tuple(args[1:3])]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _wire_real_preflight(monkeypatch: pytest.MonkeyPatch, *, which, run, os_name="posix") -> None:
    """Route dev's preflight through the real checks with injected fakes."""
    real = dev_preflight_module.run_dev_preflight

    def wired(source_dir, command_tokens, prepare_environment):
        real(source_dir, command_tokens, prepare_environment, which=which, run=run, os_name=os_name)

    monkeypatch.setattr(dev_service_module, "run_dev_preflight", wired)


def _uv_app(tmp_path: pathlib.Path, command: str) -> None:
    (tmp_path / "app.yaml").write_text(f"command: {command}\nenv: []\n")


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["uv", "run", "start-server"], True),
        (["uv.exe", "run", "start-server"], True),
        (["/opt/homebrew/bin/uv", "run", "start-server"], True),
        (["UV", "run", "start-server"], True),
        (["python", "app.py"], False),
        (["uvicorn", "main:uv"], False),
        ([], False),
    ],
)
def test_uses_uv_only_inspects_the_executable_token(tokens, expected) -> None:
    assert dev_preflight_module.uses_uv(tokens) is expected


@pytest.mark.parametrize("command", ["[uv, run, start-server]", "uv run start-server"])
def test_missing_uv_fails_before_tracing_or_local_manifest(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    _uv_app(tmp_path, command)
    which = Mock(return_value=None)
    _wire_real_preflight(monkeypatch, which=which, run=_UvRun())

    with pytest.raises(AgentCliError) as raised:
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
        ):
            pytest.fail("a missing uv must not yield a plan")

    assert raised.value.message == (
        "uv is required to run this agent locally, but uv was not found on PATH."
    )
    which.assert_called_once_with("uv")
    service_fixture.tracing.start_dev.assert_not_called()
    service_fixture.tracing.stop.assert_not_called()
    service_fixture.apps.run_local.assert_not_called()
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()


@pytest.mark.parametrize(
    ("os_name", "install_command"),
    [
        ("posix", "curl -LsSf https://astral.sh/uv/install.sh | sh"),
        (
            "nt",
            'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"',
        ),
    ],
)
def test_missing_uv_hint_names_install_verify_and_rerun(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    os_name: str,
    install_command: str,
) -> None:
    _uv_app(tmp_path, "[uv, run, start-server]")
    _wire_real_preflight(monkeypatch, which=Mock(return_value=None), run=_UvRun(), os_name=os_name)

    with pytest.raises(AgentCliError) as raised:
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
        ):
            pytest.fail("a missing uv must not yield a plan")

    # One line per step, no baked-in indentation (the renderer indents; `-o json` stays clean).
    assert raised.value.hint == (
        f"Install it: {install_command}\n"
        "Then run `uv --version` to verify, and re-run `agentbricks dev`."
    )


def test_windows_uv_exe_command_triggers_the_uv_check(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _uv_app(tmp_path, '["uv.exe", "run", "start-server"]')
    which = Mock(return_value=None)
    _wire_real_preflight(monkeypatch, which=which, run=_UvRun(), os_name="nt")

    with pytest.raises(AgentCliError, match="uv was not found on PATH"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
        ):
            pytest.fail("a missing uv.exe must not yield a plan")

    which.assert_called_once_with("uv.exe")


def test_uv_found_probes_python_once_then_proceeds_as_before(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _uv_app(tmp_path, "[uv, run, start-server]")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "agent"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
    )
    events: list[str] = []
    runner = _UvRun(
        {
            ("python", "find"): SimpleNamespace(
                returncode=0, stdout="/usr/bin/python3.11", stderr=""
            )
        },
        events=events,
    )
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    def _start_dev(source_dir):
        events.append("start_dev")
        return service_fixture.tracing.start_dev.return_value

    service_fixture.tracing.start_dev.side_effect = _start_dev

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=9000)
    ) as plan:
        assert plan.prepare_environment is False
        service_fixture.service.run(plan)

    # The probe runs before anything starts, and a match ends the check quietly.
    assert events == ["find", "start_dev"]
    assert len(runner.calls) == 1
    args, kwargs = runner.calls[0]
    assert args == ["/usr/bin/uv", "python", "find", ">=3.11"]
    assert kwargs["cwd"] == tmp_path
    assert capsys.readouterr().err == ""
    service_fixture.apps.run_local.assert_called_once()


@pytest.mark.parametrize("prepare_environment", [None, False])
def test_non_uv_run_with_reused_venv_makes_no_uv_probes(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    prepare_environment,
) -> None:
    if prepare_environment is None:
        (tmp_path / ".venv").mkdir()  # auto-detection reuses it: prepare is False
    which = Mock(return_value=None)
    runner = _UvRun()
    _wire_real_preflight(monkeypatch, which=which, run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=prepare_environment, app_port=None)
    ) as plan:
        assert plan.prepare_environment is False
        service_fixture.service.run(plan)

    which.assert_not_called()
    assert runner.calls == []
    service_fixture.apps.run_local.assert_called_once()


def test_environment_prepare_triggers_the_uv_check_for_non_uv_commands(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # run-local's --prepare-environment (the default without a .venv) builds the venv with uv, so
    # the uv check applies even though the app command itself does not use uv.
    which = Mock(return_value=None)
    _wire_real_preflight(monkeypatch, which=which, run=_UvRun())

    with pytest.raises(AgentCliError, match="uv was not found on PATH"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=None, app_port=None)
        ):
            pytest.fail("a missing uv must not yield a plan")

    which.assert_called_once_with("uv")


@pytest.mark.parametrize(
    "pyproject",
    [
        None,  # no pyproject.toml at all
        '[project]\nname = "agent"\n',  # no requires-python key
        "not toml\n",  # unparseable
        '[project]\nrequires-python = ">=banana"\n',  # a spec uv would reject too
    ],
)
def test_unreadable_or_invalid_python_spec_skips_the_probe(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    pyproject: str | None,
) -> None:
    _uv_app(tmp_path, "[uv, run, start-server]")
    if pyproject is not None:
        (tmp_path / "pyproject.toml").write_text(pyproject)
    runner = _UvRun()
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    assert runner.calls == []
    service_fixture.apps.run_local.assert_called_once()


@pytest.mark.parametrize(
    "failure",
    [subprocess.TimeoutExpired(cmd="uv", timeout=15), OSError("spawn not permitted")],
)
def test_inconclusive_python_probe_does_not_block_the_run(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    _uv_app(tmp_path, "[uv, run, start-server]")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    runner = _UvRun({("python", "find"): failure})
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    # The find probe ran; without a conclusive answer nothing else is attempted and dev proceeds.
    assert len(runner.calls) == 1
    service_fixture.apps.run_local.assert_called_once()


def test_uv_installs_python_when_the_probe_finds_nothing(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _uv_app(tmp_path, "[uv, run, start-server]")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    runner = _UvRun(
        {
            ("python", "find"): SimpleNamespace(returncode=1, stdout="", stderr="no match"),
            ("python", "install"): SimpleNamespace(returncode=0, stdout="", stderr="Installed"),
        }
    )
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    assert [args for args, _kwargs in runner.calls] == [
        ["/usr/bin/uv", "python", "find", ">=3.11"],
        ["/usr/bin/uv", "python", "install", ">=3.11"],
    ]
    assert 'Installing Python ">=3.11" with uv (this can take a minute)...' in (
        capsys.readouterr().err
    )
    service_fixture.apps.run_local.assert_called_once()


def test_failed_python_install_fails_before_starting(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _uv_app(tmp_path, "[uv, run, start-server]")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    runner = _UvRun(
        {
            ("python", "find"): SimpleNamespace(returncode=1, stdout="", stderr="no match"),
            ("python", "install"): SimpleNamespace(
                returncode=2,
                stdout="",
                stderr="error: No download found for request: cpython->=3.11-linux-x86_64-gnu\n",
            ),
        }
    )
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with pytest.raises(AgentCliError) as raised:
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
        ):
            pytest.fail("a failed Python install must not yield a plan")

    assert raised.value.message == 'uv couldn\'t get Python ">=3.11" (required by pyproject.toml).'
    assert raised.value.hint == (
        'Run `uv python install ">=3.11"`, then re-run `agentbricks dev`.\n'
        "uv said: error: No download found for request: cpython->=3.11-linux-x86_64-gnu"
    )
    service_fixture.tracing.start_dev.assert_not_called()
    service_fixture.apps.run_local.assert_not_called()
    assert not (tmp_path / "app.agentbricksdev.yaml").exists()


def test_failed_install_names_the_spec_from_pyproject(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The version comes from the project, not a hardcoded floor, and a long last uv line is not
    # quoted (only short reasons make the `uv said:` cut).
    _uv_app(tmp_path, "[uv, run, start-server]")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.12,<3.14"\n')
    runner = _UvRun(
        {
            ("python", "find"): SimpleNamespace(returncode=1, stdout="", stderr="no match"),
            ("python", "install"): SimpleNamespace(
                returncode=2,
                stdout="",
                stderr="first line\n" + "x" * 300 + "\n",
            ),
        }
    )
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with pytest.raises(AgentCliError) as raised:
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
        ):
            pytest.fail("a failed Python install must not yield a plan")

    assert '">=3.12,<3.14"' in raised.value.message
    assert 'uv python install ">=3.12,<3.14"' in raised.value.hint
    assert "re-run `agentbricks dev`" in raised.value.hint
    assert "uv said:" not in raised.value.hint


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable bit")
def test_project_relative_uv_resolves_against_the_source_dir(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # run-local starts the command from the project, so `./bin/uv` must be found there even when
    # dev runs from another directory with --source; the Python probe then uses that same uv.
    project = tmp_path / "project"
    (project / "bin").mkdir(parents=True)
    vendored_uv = project / "bin" / "uv"
    vendored_uv.write_text("#!/bin/sh\n")
    vendored_uv.chmod(0o755)
    _uv_app(project, '["./bin/uv", "run", "start-server"]')
    (project / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    runner = _UvRun({("python", "find"): SimpleNamespace(returncode=0, stdout="", stderr="")})
    _wire_real_preflight(monkeypatch, which=shutil.which, run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(project), prepare_environment=False, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    assert [args for args, _kwargs in runner.calls] == [
        [str(vendored_uv), "python", "find", ">=3.11"]
    ]
    service_fixture.apps.run_local.assert_called_once()


def test_missing_project_relative_uv_fails_before_starting(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _uv_app(tmp_path, '["./bin/uv", "run", "start-server"]')
    which = Mock(return_value=None)
    _wire_real_preflight(monkeypatch, which=which, run=_UvRun())

    with pytest.raises(AgentCliError) as raised:
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
        ):
            pytest.fail("a missing project uv must not yield a plan")

    # Installing uv globally wouldn't fix this command, so the error names the configured path.
    assert raised.value.message == "app.yaml runs uv from ./bin/uv, but no executable uv is there."
    assert raised.value.hint == (
        "Restore ./bin/uv, or change the app.yaml command to run `uv` from PATH, "
        "then re-run `agentbricks dev`."
    )
    which.assert_called_once_with(str(tmp_path / "./bin/uv"))
    service_fixture.tracing.start_dev.assert_not_called()


def test_environment_build_needs_path_uv_even_with_a_project_uv(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # run-local builds the environment with uv from PATH, so a project-local uv in the app command
    # doesn't cover the build.
    _uv_app(tmp_path, '["./bin/uv", "run", "start-server"]')
    project_uv = str(tmp_path / "./bin/uv")
    which = Mock(side_effect=lambda name: name if name == project_uv else None)
    _wire_real_preflight(monkeypatch, which=which, run=_UvRun())

    with pytest.raises(AgentCliError, match="uv was not found on PATH"):
        with service_fixture.service.prepare(
            DevRequest(source=str(tmp_path), prepare_environment=True, app_port=None)
        ):
            pytest.fail("an environment build without PATH uv must not yield a plan")

    assert [call.args[0] for call in which.call_args_list] == [project_uv, "uv"]
    service_fixture.tracing.start_dev.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [subprocess.TimeoutExpired(cmd="uv", timeout=180), OSError("spawn not permitted")],
)
def test_inconclusive_python_install_does_not_block_the_run(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    # An install that times out or cannot spawn is inconclusive, like the find probe: fail open.
    _uv_app(tmp_path, "[uv, run, start-server]")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    runner = _UvRun(
        {
            ("python", "find"): SimpleNamespace(returncode=1, stdout="", stderr="no match"),
            ("python", "install"): failure,
        }
    )
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    assert len(runner.calls) == 2
    service_fixture.apps.run_local.assert_called_once()


@pytest.mark.parametrize(("prepare_environment", "probes"), [(False, 0), (True, 1)])
@pytest.mark.parametrize(
    "command",
    [
        '["uv", "run", "--no-project", "python", "app.py"]',
        '["uv", "run", "--project", "service", "--no-sync", "python", "app.py"]',
        '["uv", "run", "--project=service", "python", "app.py"]',
        '["uv", "--directory", "service", "run", "python", "app.py"]',
        '["uv", "run", "--script", "app.py"]',
        '["uv", "run", "-s", "app.py"]',
        '["uv", "run", "--frozen", "--project", "service", "python", "app.py"]',
        '["uv", "run", "--python", "/opt/custom/python", "--no-sync", "start-server"]',
        '["uv", "run", "--python=3.12", "start-server"]',
        '["uv", "run", "-p", "3.12", "start-server"]',
        '["uv", "run", "-p3.12", "start-server"]',
    ],
)
def test_project_override_command_skips_the_python_check_unless_the_environment_is_built(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    prepare_environment: bool,
    probes: int,
) -> None:
    # A uv command that selects another project (or none), or picks its own interpreter, is not
    # bound by uv's discovery of source_dir's requires-python, which then only matters when
    # run-local builds the project environment.
    _uv_app(tmp_path, command)
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=99"\n')
    runner = _UvRun({("python", "find"): SimpleNamespace(returncode=0, stdout="", stderr="")})
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=prepare_environment, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    assert len(runner.calls) == probes
    service_fixture.apps.run_local.assert_called_once()


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable bit")
def test_relative_source_resolves_project_uv_to_an_absolute_path(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The probe runs from the project, so a uv found via a relative --source must not stay
    # relative (it would then resolve under project/project/...).
    project = tmp_path / "project"
    (project / "bin").mkdir(parents=True)
    vendored_uv = project / "bin" / "uv"
    vendored_uv.write_text("#!/bin/sh\n")
    vendored_uv.chmod(0o755)
    _uv_app(project, '["./bin/uv", "run", "start-server"]')
    (project / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    monkeypatch.chdir(tmp_path)
    runner = _UvRun({("python", "find"): SimpleNamespace(returncode=0, stdout="", stderr="")})
    _wire_real_preflight(monkeypatch, which=shutil.which, run=runner)

    with service_fixture.service.prepare(
        DevRequest(source="project", prepare_environment=False, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    assert [args[0] for args, _kwargs in runner.calls] == [str(vendored_uv)]


@pytest.mark.parametrize(
    "command",
    [
        '["uv", "run", "start-server", "--project", "demo"]',
        '["uv", "run", "--no-sync", "start-server", "--directory", "x"]',
        '["uv", "run", "--", "start-server", "--no-project"]',
        '["uv", "run", "start-server", "--python", "3.12"]',
    ],
)
def test_project_options_after_the_app_command_still_check_python(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    # Options after the command uv runs are that command's arguments; uv still uses source_dir's
    # project, so its requires-python is still checked.
    _uv_app(tmp_path, command)
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    runner = _UvRun({("python", "find"): SimpleNamespace(returncode=0, stdout="", stderr="")})
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    assert len(runner.calls) == 1


def test_uv_probe_output_is_decoded_as_utf8_with_replacement(
    service_fixture: SimpleNamespace,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # uv writes UTF-8; decoding with the locale codec could raise on a legacy Windows code page.
    _uv_app(tmp_path, "[uv, run, start-server]")
    (tmp_path / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.11"\n')
    runner = _UvRun({("python", "find"): SimpleNamespace(returncode=0, stdout="", stderr="")})
    _wire_real_preflight(monkeypatch, which=Mock(return_value="/usr/bin/uv"), run=runner)

    with service_fixture.service.prepare(
        DevRequest(source=str(tmp_path), prepare_environment=False, app_port=None)
    ) as plan:
        service_fixture.service.run(plan)

    _args, kwargs = runner.calls[0]
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"
    assert "text" not in kwargs
