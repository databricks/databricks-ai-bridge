"""Unit tests for `agentbricks profile` (set / get / login) and the profile-aware root context.

The workspace boundary (`auth._validate_bounded`), `subprocess.run`, and the terminal check are
stubbed; `DATABRICKS_CONFIG_FILE` points at a tmp file (see conftest.py).
"""

from __future__ import annotations

import json
import pathlib
from unittest import mock

import pytest
from click.testing import CliRunner

from databricks_agentbricks.cli import app as cli
from databricks_agentbricks.cli import auth
from databricks_agentbricks.cli import profile as profile_mod
from databricks_agentbricks.cli.app import CliContext


@pytest.fixture
def project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """An Agent Bricks project directory (has app.yaml) that is also the cwd."""
    (tmp_path / "app.yaml").write_text("command: []\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _ctx(profile=None, output="text") -> CliContext:
    return CliContext(profile, output)


def _invoke(command, args, **ctx_kwargs):
    return CliRunner().invoke(command, args, obj=_ctx(**ctx_kwargs))


def _env_profile(directory: pathlib.Path) -> str | None:
    return auth._parse_env_file(directory / ".env").get("DATABRICKS_CONFIG_PROFILE")


def _flat(output: str) -> str:
    return " ".join(output.split())


# --- CliContext ------------------------------------------------------------------------------


def test_context_resolves_flag_then_env_var(monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "from-env-var")

    assert _ctx("from-flag").profile_info == auth.ProfileInfo("from-flag", "--profile")
    assert _ctx().profile_info == auth.ProfileInfo("from-env-var", "DATABRICKS_CONFIG_PROFILE")


def test_context_profile_is_read_only():
    with pytest.raises(AttributeError):
        _ctx("x").profile = "y"  # type: ignore[misc]


def test_use_project_profile_applies_env_file_and_rebuilds_provider(tmp_path):
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")
    ctx = _ctx()
    before = ctx.api_client_provider

    ctx.use_project_profile(tmp_path)

    assert ctx.profile_info == auth.ProfileInfo("proj", ".env")
    assert ctx.api_client_provider is not before
    assert ctx.api_client_provider._profile == "proj"


def test_use_project_profile_keeps_provider_when_name_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "same")
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=same\n")
    ctx = _ctx()
    before = ctx.api_client_provider

    ctx.use_project_profile(tmp_path)

    assert ctx.api_client_provider is before
    assert ctx.profile_info.source == ".env"


def test_use_project_profile_is_noop_for_explicit_flag(tmp_path):
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")
    ctx = _ctx("flag")
    before = ctx.api_client_provider

    ctx.use_project_profile(tmp_path)

    assert ctx.profile_info == auth.ProfileInfo("flag", "--profile")
    assert ctx.api_client_provider is before


def test_use_project_profile_accepts_pre_parsed_env_values(tmp_path):
    ctx = _ctx()

    ctx.use_project_profile(tmp_path, {"DATABRICKS_CONFIG_PROFILE": "parsed"})

    assert ctx.profile_info == auth.ProfileInfo("parsed", ".env")


def test_use_flag_profile_overrides_env_file(tmp_path):
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")
    ctx = _ctx()
    ctx.use_project_profile(tmp_path)

    ctx.use_flag_profile("flag")
    ctx.use_project_profile(tmp_path)

    assert ctx.profile_info == auth.ProfileInfo("flag", "--profile")
    assert ctx.api_client_provider._profile == "flag"


# --- root group wiring ------------------------------------------------------------------------


def _get_json(args: list[str]) -> dict:
    result = CliRunner().invoke(cli.agentbricks, ["-o", "json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def test_root_applies_cwd_project_env_file(project):
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    payload = _get_json(["profile", "get"])

    assert (payload["profile"], payload["source"]) == ("proj", ".env")


def test_root_ignores_env_file_outside_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    payload = _get_json(["profile", "get"])

    assert payload["profile"] is None


def test_global_flag_beats_project_env_file(project):
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    payload = _get_json(["-p", "flag", "profile", "get"])

    assert (payload["profile"], payload["source"]) == ("flag", "--profile")


@pytest.mark.parametrize("flag", ["-p", "--profile"])
def test_profile_flag_is_accepted_after_the_subcommand(project, flag):
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    payload = _get_json(["profile", "get", flag, "after"])

    assert (payload["profile"], payload["source"]) == ("after", "--profile")


def test_profile_flag_after_the_subcommand_wins_over_the_global_one(project):
    payload = _get_json(["-p", "before", "profile", "get", "-p", "after"])

    assert payload["profile"] == "after"


def test_init_keeps_its_own_profile_option():
    init_options = [opt for param in cli.init.params for opt in param.opts]

    assert init_options.count("--profile") == 1
    assert init_options.count("-p") == 1


# --- profile get -----------------------------------------------------------------------------


def test_get_reports_profile_source_host_and_project(project, write_databrickscfg):
    write_databrickscfg("[proj]\nhost = https://proj.example\n")
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    payload = _get_json(["profile", "get"])

    assert payload == {
        "profile": "proj",
        "source": ".env",
        "workspace_host": "https://proj.example",
        "project": str(project),
    }


def test_get_outside_project_uses_env_var_and_reports_no_project(
    tmp_path, monkeypatch, write_databrickscfg
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "ambient")
    write_databrickscfg("[ambient]\nhost = https://ambient.example\n")

    payload = _get_json(["profile", "get"])

    assert payload == {
        "profile": "ambient",
        "source": "DATABRICKS_CONFIG_PROFILE",
        "workspace_host": "https://ambient.example",
        "project": None,
    }


def test_get_without_any_profile_reports_sdk_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    payload = _get_json(["profile", "get"])

    assert (payload["profile"], payload["source"]) == (None, "Databricks SDK default")


def test_get_source_reads_another_projects_env_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "app.yaml").write_text("command: []\n")
    (other / ".env").write_text("DATABRICKS_CONFIG_PROFILE=elsewhere\n")

    payload = _get_json(["profile", "get", "--source", str(other)])

    assert (payload["profile"], payload["source"]) == ("elsewhere", ".env")
    assert payload["project"] == str(other)


def test_get_text_lists_profile_source_and_project(project, write_databrickscfg):
    write_databrickscfg("[proj]\nhost = https://proj.example\n")
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    result = CliRunner().invoke(cli.agentbricks, ["profile", "get"])

    out = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "Profile proj" in out
    assert "Source .env" in out
    assert "Workspace https://proj.example" in out


def test_get_makes_no_workspace_calls(project, validated_profile):
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    _get_json(["profile", "get"])

    validated_profile.assert_not_called()


# --- profile set -----------------------------------------------------------------------------


def test_set_rewrites_profile_line_and_keeps_other_lines(project):
    (project / ".env").write_text("OTHER=1\nDATABRICKS_CONFIG_PROFILE=old\nLAST=2\n")

    result = _invoke(profile_mod.profile, ["set", "new"])

    assert result.exit_code == 0, result.output
    assert (project / ".env").read_text() == "OTHER=1\nDATABRICKS_CONFIG_PROFILE=new\nLAST=2\n"
    assert "Project profile set" in _flat(result.output)


def test_set_same_profile_reports_unchanged(project):
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=same\n")

    result = _invoke(profile_mod.profile, ["set", "same"])

    assert result.exit_code == 0, result.output
    assert "already 'same'" in _flat(result.output)


def test_set_seeds_env_from_example_when_missing(project):
    (project / ".env.example").write_text("KEEP=1\n")

    result = _invoke(profile_mod.profile, ["set", "ml"])

    assert result.exit_code == 0, result.output
    assert (project / ".env").read_text() == "DATABRICKS_CONFIG_PROFILE=ml\nKEEP=1\n"


def test_set_requires_a_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = _invoke(profile_mod.profile, ["set", "ml"])

    assert result.exit_code != 0
    assert "isn't an Agent Bricks project" in _flat(result.output)
    assert not (tmp_path / ".env").exists()


def test_set_source_targets_another_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "app.yaml").write_text("command: []\n")

    result = _invoke(profile_mod.profile, ["set", "ml", "--source", str(other)])

    assert result.exit_code == 0, result.output
    assert _env_profile(other) == "ml"


def test_set_noninteractive_warns_on_unknown_profile_and_suggests_login(project):
    result = _invoke(profile_mod.profile, ["set", "ghost"])

    out = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "isn't in your Databricks config yet" in out
    assert "agentbricks profile login ghost" in out


def test_set_noninteractive_makes_no_workspace_calls(
    project, write_databrickscfg, validated_profile
):
    write_databrickscfg("[ml]\nhost = https://ml.example\n")

    result = _invoke(profile_mod.profile, ["set", "ml"])

    assert result.exit_code == 0, result.output
    validated_profile.assert_not_called()
    assert "isn't in your Databricks config yet" not in _flat(result.output)


def test_set_json_payload(project, write_databrickscfg):
    write_databrickscfg("[ml]\nhost = https://ml.example\n")

    result = _invoke(profile_mod.profile, ["set", "ml"], output="json")

    assert json.loads(result.output) == {
        "directory": str(project),
        "env_profile": "ml",
        "changed": True,
        "workspace_host": "https://ml.example",
        "signed_in_user": None,
    }


def test_set_signs_in_when_interactive(project, monkeypatch, validated_profile):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)

    result = _invoke(profile_mod.profile, ["set", "ml"], output="json")

    assert result.exit_code == 0, result.output
    validated_profile.assert_called_once_with("ml")
    assert json.loads(result.output)["signed_in_user"] == "me@example.com"


def test_set_text_shows_signed_in_user_without_login_next_step(
    project, monkeypatch, validated_profile
):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)

    result = _invoke(profile_mod.profile, ["set", "ml"])

    out = _flat(result.output)
    assert "Signed in as me@example.com" in out
    assert "Next step" not in out


# --- profile login ---------------------------------------------------------------------------


def test_login_argument_validates_and_reports_user(validated_profile):
    result = _invoke(profile_mod.profile, ["login", "ml"], output="json")

    assert result.exit_code == 0, result.output
    validated_profile.assert_called_once_with("ml")
    assert json.loads(result.output) == {
        "profile": "ml",
        "source": "argument",
        "user": "me@example.com",
        "host": "https://ws.example",
    }


def test_login_announces_target_on_stderr_before_signing_in(validated_profile, write_databrickscfg):
    write_databrickscfg("[ml]\nhost = https://ml.example\n")

    result = CliRunner().invoke(profile_mod.profile, ["login", "ml"], obj=_ctx())

    assert result.exit_code == 0, result.output
    assert "Logging in to profile 'ml' (from argument) — https://ml.example" in _flat(result.output)


def test_login_announce_explains_a_profile_missing_from_config(validated_profile):
    result = _invoke(profile_mod.profile, ["login", "ghost"])

    assert "not in your Databricks config yet" in _flat(result.output)


def test_login_defaults_to_global_flag(validated_profile):
    result = _invoke(profile_mod.profile, ["login"], profile="flagged", output="json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["source"] == "--profile"
    validated_profile.assert_called_once_with("flagged")


def test_login_argument_beats_global_flag(validated_profile):
    _invoke(profile_mod.profile, ["login", "named"], profile="flagged")

    validated_profile.assert_called_once_with("named")


def test_login_defaults_to_project_env_profile(project, validated_profile):
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    result = _invoke(profile_mod.profile, ["login"], output="json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["source"] == ".env"
    validated_profile.assert_called_once_with("proj")


def test_login_ignores_ambient_env_var(tmp_path, monkeypatch, validated_profile):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "ambient")

    result = _invoke(profile_mod.profile, ["login"])

    assert result.exit_code != 0
    assert "No profile to log in to." in _flat(result.output)
    validated_profile.assert_not_called()


def test_login_without_any_target_errors_with_hint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = _invoke(profile_mod.profile, ["login"])

    assert result.exit_code != 0
    assert "No profile to log in to." in _flat(result.output)
    assert "agentbricks profile login <profile>" in _flat(result.output)


def test_login_runs_databricks_cli_when_unauthenticated_and_interactive(monkeypatch):
    client = mock.Mock(host="https://ws.example")
    validate = mock.Mock(side_effect=[(None, RuntimeError("expired")), ((client, "me"), None)])
    monkeypatch.setattr(auth, "_validate_bounded", validate)
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    databricks_login = mock.Mock(return_value=mock.Mock(returncode=0))
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    result = _invoke(profile_mod.profile, ["login", "ml"])

    assert result.exit_code == 0, result.output
    assert databricks_login.call_args.args[0] == ["databricks", "auth", "login", "--profile", "ml"]
    assert "Logged in to 'ml'" in _flat(result.output)


def test_login_noninteractive_unauthenticated_fails_without_browser(monkeypatch):
    monkeypatch.setattr(auth, "_validate_bounded", lambda profile: (None, RuntimeError("expired")))
    databricks_login = mock.Mock()
    monkeypatch.setattr(auth.subprocess, "run", databricks_login)

    result = _invoke(profile_mod.profile, ["login", "ml"])

    assert result.exit_code != 0
    assert "interactive terminal" in _flat(result.output)
    databricks_login.assert_not_called()


def test_login_does_not_modify_the_project_env_file(project, validated_profile):
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    _invoke(profile_mod.profile, ["login", "other"])

    assert _env_profile(project) == "proj"


def test_login_notes_when_project_still_uses_another_profile(project, validated_profile):
    (project / ".env").write_text("DATABRICKS_CONFIG_PROFILE=proj\n")

    result = _invoke(profile_mod.profile, ["login", "other"])

    out = _flat(result.output)
    assert "This project still uses profile 'proj' (from .env)" in out
    assert "agentbricks profile set other" in out


# --- helpers ---------------------------------------------------------------------------------


def test_sign_in_if_interactive_skips_without_a_terminal(validated_profile):
    assert profile_mod.sign_in_if_interactive(_ctx(), "ml", "argument") is None
    validated_profile.assert_not_called()


def test_sign_in_if_interactive_returns_user(monkeypatch, validated_profile):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)

    assert profile_mod.sign_in_if_interactive(_ctx(), "ml", "argument") == "me@example.com"


def test_sign_in_if_interactive_swallows_sign_in_failure(monkeypatch):
    monkeypatch.setattr(auth, "_is_interactive", lambda: True)
    monkeypatch.setattr(auth, "_validate_bounded", lambda profile: (None, RuntimeError("denied")))
    monkeypatch.setattr(auth.subprocess, "run", mock.Mock(return_value=mock.Mock(returncode=2)))

    assert profile_mod.sign_in_if_interactive(_ctx(), "ml", "argument") is None


def test_warn_unknown_profile_is_silent_for_json_and_known_profiles(capsys, write_databrickscfg):
    write_databrickscfg("[known]\nhost = https://known.example\n")

    profile_mod.warn_unknown_profile("ghost", _ctx(output="json"))
    profile_mod.warn_unknown_profile("known", _ctx())

    assert capsys.readouterr().out == ""
