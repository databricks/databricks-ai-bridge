"""Unit tests for `agentbricks auth` (switch / describe / login) and the profile-aware root context.

The workspace boundary (`workspace_auth._validate_bounded`), `subprocess.run`, and the terminal check are
stubbed; `DATABRICKS_CONFIG_FILE` points at a tmp file (see conftest.py).
"""

from __future__ import annotations

import json
import pathlib
from unittest import mock

import pytest
from click.testing import CliRunner
from databricks.sdk.errors import Unauthenticated

from databricks_agentbricks.cli import app as cli
from databricks_agentbricks.cli import auth
from databricks_agentbricks.cli.app import CliContext
from databricks_agentbricks.clients import api_client_provider
from databricks_agentbricks.errors import AgentCliError
from databricks_agentbricks.projects.config import write_project_metadata
from databricks_agentbricks.projects.env_file import read_env
from databricks_agentbricks.projects.profiles import ProfileInfo
from databricks_agentbricks.services import workspace_auth


def _write_env(directory: pathlib.Path, profile: str) -> None:
    (directory / ".env").write_text(f"DATABRICKS_CONFIG_PROFILE={profile}\n")


@pytest.fixture
def project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """An Agent Bricks project directory (has app.yaml) that is also the cwd."""
    (tmp_path / "app.yaml").write_text("command: []\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def env_project(project: pathlib.Path) -> pathlib.Path:
    """The `project`, whose .env selects profile 'proj'."""
    _write_env(project, "proj")
    return project


@pytest.fixture
def outside_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A cwd that is not an Agent Bricks project (no app.yaml)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def fake_api_client(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Replace the workspace client the provider builds, so first use makes no SDK calls."""
    client_cls = mock.Mock()
    client_cls.return_value.host = None
    monkeypatch.setattr(api_client_provider, "_AgentBricksApiClient", client_cls)
    return client_cls


def _ctx(profile=None, output="text") -> CliContext:
    ctx = CliContext(profile, output)
    ctx.select_profile()
    return ctx


def _invoke_profile(args, **ctx_kwargs):
    return CliRunner().invoke(auth.auth_group, args, obj=_ctx(**ctx_kwargs))


def _get_json(args: list[str]) -> dict:
    result = CliRunner().invoke(cli.agentbricks, ["-o", "json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _flat(output: str) -> str:
    return " ".join(output.split())


def test_context_precedence_is_flag_then_env_file_then_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env-var")
    _write_env(tmp_path, "proj")
    ambient, project, flagged = _ctx(), _ctx(), _ctx("from-flag")

    project.select_profile(tmp_path)
    flagged.select_profile(tmp_path)

    assert ambient.profile_info == ProfileInfo("from-env-var", "DATABRICKS_CONFIG_PROFILE")
    assert project.profile_info == ProfileInfo("proj", ".env")
    assert flagged.profile_info == ProfileInfo("from-flag", "--profile")


def test_profile_flag_after_subcommand_wins(env_project):
    payload = _get_json(["-p", "before", "auth", "describe", "-p", "after"])

    assert (payload["profile"], payload["source"]) == ("after", "--profile")


def test_outside_a_project_ignores_env_file(outside_project, monkeypatch):
    _write_env(outside_project, "proj")

    payload = _get_json(["auth", "describe"])
    assert (payload["profile"], payload["source"]) == (None, "Databricks SDK default")

    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "ambient")
    payload = _get_json(["auth", "describe"])
    assert (payload["profile"], payload["source"], payload["project"]) == (
        "ambient",
        "DATABRICKS_CONFIG_PROFILE",
        None,
    )


def test_first_workspace_call_names_profile_and_host_once(
    capsys, write_databrickscfg, fake_api_client
):
    write_databrickscfg("[ws]\nhost = https://ws.example\n")
    ctx = _ctx("ws")

    ctx.client()
    ctx.client()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("Using profile 'ws' (from --profile) · https://ws.example") == 1


def test_first_workspace_call_reports_resolved_host_once(
    capsys, write_databrickscfg, fake_api_client
):
    write_databrickscfg("[ws]\nhost = https://profile.example\n")
    fake_api_client.return_value.host = "https://resolved.example"
    ctx = _ctx("ws")

    ctx.client()
    ctx.client()

    error_output = capsys.readouterr().err
    assert error_output.count("Using profile 'ws' (from --profile) · https://resolved.example") == 1
    assert "https://profile.example" not in error_output


def test_failed_client_construction_still_names_offline_workspace(
    capsys, write_databrickscfg, fake_api_client
):
    write_databrickscfg("[ws]\nhost = https://profile.example\n")
    fake_api_client.side_effect = AgentCliError("synthetic authentication failure")

    with pytest.raises(AgentCliError):
        _ctx("ws").client()

    assert (
        "Using profile 'ws' (from --profile) · https://profile.example" in capsys.readouterr().err
    )


def test_workspace_notice_names_sdk_default_auth(capsys, monkeypatch, fake_api_client):
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")

    _ctx().client()

    assert (
        "Using Databricks SDK default auth · https://env.example (from DATABRICKS_HOST)"
        in capsys.readouterr().err
    )


def test_workspace_notice_reports_environment_host_for_named_profile(
    capsys, monkeypatch, write_databrickscfg, fake_api_client
):
    write_databrickscfg("[ws]\nhost = https://profile.example\n")
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    _ctx("ws").client()

    assert "Using profile 'ws' (from --profile) · https://env.example (from DATABRICKS_HOST)" in (
        capsys.readouterr().err
    )


def test_describe_and_workspace_notice_report_switched_default(
    outside_project, capsys, write_databrickscfg, fake_api_client, validated_profile
):
    write_databrickscfg(
        "[DEFAULT]\nhost = https://default.example\n"
        "[__settings__]\ndefault_profile = switched\n"
        "[switched]\nhost = https://switched.example\n"
    )

    assert _get_json(["auth", "describe"])["workspace_host"] == "https://switched.example"
    validated_profile.assert_not_called()
    fake_api_client.assert_not_called()
    _ctx().client()

    assert "https://switched.example ([switched] profile)" in capsys.readouterr().err


def test_describe_reports_environment_host_even_with_a_flagged_profile(
    outside_project, monkeypatch, write_databrickscfg, validated_profile
):
    write_databrickscfg("[ws]\nhost = https://profile.example\n")
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")
    monkeypatch.setenv("DATABRICKS_TOKEN", "test-token")

    assert _get_json(["auth", "describe", "-p", "ws"])["workspace_host"] == "https://env.example"
    validated_profile.assert_not_called()


def test_login_notice_reports_environment_host(
    outside_project, monkeypatch, write_databrickscfg, validated_profile
):
    write_databrickscfg("[ws]\nhost = https://profile.example\n")
    monkeypatch.setenv("DATABRICKS_HOST", "https://env.example")

    result = _invoke_profile(["login", "ws"])

    assert result.exit_code == 0, result.output
    assert "Logging in to profile 'ws' (from argument) — https://env.example" in result.output


def test_workspace_notice_is_silent_for_json(capsys, fake_api_client):
    _ctx("ws", output="json").client()

    assert capsys.readouterr().err == ""


def test_get_reports_profile_source_host_and_project_without_workspace_calls(
    env_project, write_databrickscfg, validated_profile
):
    write_databrickscfg("[proj]\nhost = https://proj.example\n")

    assert _get_json(["auth", "describe"]) == {
        "profile": "proj",
        "source": ".env",
        "workspace_host": "https://proj.example",
        "project": str(env_project),
    }
    validated_profile.assert_not_called()


def test_commands_in_a_project_subfolder_use_the_root_env(
    env_project, monkeypatch, write_databrickscfg, validated_profile
):
    write_databrickscfg("[proj]\nhost = https://proj.example\n")
    write_project_metadata(env_project, framework="openai", template="agent-template")
    subfolder = env_project / "agent"
    subfolder.mkdir()
    monkeypatch.chdir(subfolder)

    assert _get_json(["auth", "describe"]) == {
        "profile": "proj",
        "source": ".env",
        "workspace_host": "https://proj.example",
        "project": str(env_project),
    }


def test_app_yaml_in_a_parent_does_not_make_a_subfolder_a_project(tmp_path, monkeypatch):
    (tmp_path / "app.yaml").write_text("command: []\n")
    _write_env(tmp_path, "proj")
    subfolder = tmp_path / "agent"
    subfolder.mkdir()
    monkeypatch.chdir(subfolder)

    payload = _get_json(["auth", "describe"])

    assert (payload["profile"], payload["source"], payload["project"]) == (
        None,
        "Databricks SDK default",
        None,
    )


def test_switch_from_a_subfolder_rewrites_the_root_env(env_project, monkeypatch):
    write_project_metadata(env_project, framework="openai", template="agent-template")
    subfolder = env_project / "agent"
    subfolder.mkdir()
    monkeypatch.chdir(subfolder)

    result = _invoke_profile(["switch", "new"])

    assert result.exit_code == 0, result.output
    assert read_env(env_project / ".env")["DATABRICKS_CONFIG_PROFILE"] == "new"
    assert not (subfolder / ".env").exists()


def test_set_rewrites_profile_line_and_keeps_other_lines(project):
    (project / ".env").write_text("OTHER=1\nDATABRICKS_CONFIG_PROFILE=old\nLAST=2\n")

    result = _invoke_profile(["switch", "new"])

    assert result.exit_code == 0, result.output
    assert (project / ".env").read_text() == "OTHER=1\nDATABRICKS_CONFIG_PROFILE=new\nLAST=2\n"


def test_set_requires_a_project(outside_project):
    result = _invoke_profile(["switch", "ml"])

    assert result.exit_code != 0
    assert "isn't an Agent Bricks project" in _flat(result.output)
    assert not (outside_project / ".env").exists()


def test_set_signs_in_when_interactive(project, interactive_terminal, validated_profile):
    result = _invoke_profile(["switch", "ml"], output="json")

    assert result.exit_code == 0, result.output
    validated_profile.assert_called_once_with("ml")
    assert json.loads(result.output)["signed_in_user"] == "me@example.com"


def test_set_noninteractive_suggests_login_without_signing_in(project, validated_profile):
    result = _invoke_profile(["switch", "ml"])

    assert result.exit_code == 0, result.output
    assert "agentbricks auth login ml" in _flat(result.output)
    validated_profile.assert_not_called()


@pytest.mark.parametrize(
    ("args", "flag", "expected_profile", "expected_source"),
    [
        (["login", "named"], "flagged", "named", "argument"),
        (["login"], "flagged", "flagged", "--profile"),
        (["login"], None, "proj", ".env"),
    ],
    ids=["argument-beats-flag", "global-flag-beats-env-file", "project-env-file"],
)
def test_login_target_precedence(
    env_project, validated_profile, args, flag, expected_profile, expected_source
):
    result = _invoke_profile(args, profile=flag, output="json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["source"] == expected_source
    validated_profile.assert_called_once_with(expected_profile)


def test_login_without_a_target_errors_with_hint_and_ignores_env_var(
    outside_project, monkeypatch, validated_profile
):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "ambient")

    out = _flat(_invoke_profile(["login"]).output)

    assert "No profile to log in to." in out
    assert "agentbricks auth login <profile>" in out
    validated_profile.assert_not_called()


def test_login_runs_databricks_cli_when_unauthenticated_and_interactive(
    monkeypatch, interactive_terminal, databricks_login
):
    client = mock.Mock(host="https://ws.example")
    validate = mock.Mock(side_effect=[(None, Unauthenticated("expired")), ((client, "me"), None)])
    monkeypatch.setattr(workspace_auth, "_validate_bounded", validate)

    result = _invoke_profile(["login", "ml"])

    assert result.exit_code == 0, result.output
    assert databricks_login.call_args.args[0] == ["databricks", "auth", "login", "--profile", "ml"]
    assert "Logged in to 'ml'" in _flat(result.output)


def test_login_noninteractive_unauthenticated_fails_without_browser(
    fail_validation, databricks_login
):
    fail_validation(Unauthenticated("expired"))

    result = _invoke_profile(["login", "ml"])

    assert result.exit_code != 0
    assert "interactive terminal" in _flat(result.output)
    databricks_login.assert_not_called()


def test_login_leaves_project_env_file_alone(env_project, validated_profile):
    result = _invoke_profile(["login", "other"])

    assert result.exit_code == 0, result.output
    assert read_env(env_project / ".env").get("DATABRICKS_CONFIG_PROFILE") == "proj"
