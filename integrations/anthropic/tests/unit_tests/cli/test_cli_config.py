"""Config directory, settings and secrets files, the TOML writer, init's key handling, and flag precedence."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from databricks_anthropic.cli import config, main
from databricks_anthropic.cli.config import ConfigError, Secrets, Settings

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # ty: ignore[unresolved-import] - only installed on Python 3.10


def mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def cfg_dir(tmp_path):
    return tmp_path / "cfg"


@pytest.fixture
def no_prompt(monkeypatch):
    """Fail if init prompts for a key (tests that need prompts set their own answers)."""

    def fail(prompt=""):  # pragma: no cover
        raise AssertionError(f"unexpected prompt: {prompt}")

    monkeypatch.setattr(main.getpass, "getpass", fail)
    monkeypatch.delenv(main.ENVIRONMENT_KEY_VAR, raising=False)
    monkeypatch.delenv(main.API_KEY_VAR, raising=False)


def answers(monkeypatch, *values: str) -> list[str]:
    """Scripted getpass answers; returns the prompts it was asked, in order."""
    asked: list[str] = []
    queue = list(values)

    def fake(prompt=""):
        asked.append(prompt)
        return queue.pop(0)

    monkeypatch.setattr(main.getpass, "getpass", fake)
    return asked


# ------------------------------------------------------------------ directory and names


def test_config_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(config.CONFIG_DIR_ENV, raising=False)
    assert str(config.config_dir()).endswith(".config/databricks-anthropic")
    monkeypatch.setenv(config.CONFIG_DIR_ENV, str(tmp_path / "env"))
    assert config.config_dir() == tmp_path / "env"
    assert (
        config.config_dir(str(tmp_path / "flag")) == tmp_path / "flag"
    )  # --config-dir beats the environment


def test_config_dir_is_created_0700(tmp_path):
    path = config.ensure_config_dir(tmp_path / "a" / "b")
    assert mode(path) == 0o700
    config.ensure_config_dir(path)  # idempotent, mode kept
    assert mode(path) == 0o700


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden", "x" * 65])
def test_config_names_cannot_escape_the_directory(bad):
    with pytest.raises(ConfigError):
        config.check_name(bad)


@pytest.mark.parametrize("bad", ["", "env_1/../x", "env id", "env_" + "x" * 200])
def test_environment_ids_are_path_safe(bad):
    with pytest.raises(ConfigError):
        config.check_environment_id(bad)
    assert config.check_environment_id("env_011CZexample") == "env_011CZexample"


def test_base_url_must_be_https():
    assert config.check_base_url("https://api.anthropic.com") == "https://api.anthropic.com"
    with pytest.raises(ConfigError):
        config.check_base_url("http://gateway.local")


# ------------------------------------------------------------------ settings round-trip and files


def test_settings_round_trip_through_toml(tmp_path):
    settings = Settings(
        profile="dev",
        environment_id="env_abc",
        backend="sdk",
        mcp_servers=["genie-one"],
        lifecycle="resumable",
        retention_days=3.5,
        export_volume="/Volumes/main/agents/out",
        sandbox_prefix="cmalocal-",
        anthropic_base_url="https://api.anthropic.com",
    )
    path = tmp_path / "default.toml"
    config.save_settings(path, settings)
    assert config.load_settings(path) == settings


def test_settings_file_is_private(tmp_path):
    path = tmp_path / "default.toml"
    config.save_settings(path, Settings(profile="dev", environment_id="env_1"))
    assert mode(path) == 0o600


def test_unknown_or_mistyped_settings_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown setting"):
        config.settings_from_toml({"profle": "dev"})
    with pytest.raises(ConfigError, match="wrong type"):
        config.settings_from_toml({"retention_days": "7"})
    with pytest.raises(ConfigError, match="wrong type"):
        config.settings_from_toml({"retention_days": True})
    with pytest.raises(ConfigError, match="backend"):
        config.settings_from_toml({"backend": "docker"})
    with pytest.raises(ConfigError, match="unknown MCP server"):
        config.settings_from_toml({"mcp_servers": ["nope"]})


def test_settings_file_must_be_valid_toml(tmp_path):
    path = tmp_path / "default.toml"
    path.write_text("profile = [unterminated\n")
    with pytest.raises(ConfigError, match="not valid TOML"):
        config.load_settings(path)


def test_secrets_are_created_exclusively_0600(tmp_path):
    path = tmp_path / "default.secrets"
    config.save_secrets(path, Secrets(environment_key="k1", sandbox_id_key="h1"))
    assert mode(path) == 0o600
    assert json.loads(path.read_text()) == {"environment_key": "k1", "sandbox_id_key": "h1"}
    assert config.load_secrets(path) == Secrets(environment_key="k1", sandbox_id_key="h1")


def test_secrets_creation_uses_o_excl(tmp_path, monkeypatch):
    """The first write must be an O_CREAT|O_EXCL open with mode 0600 (never a truncate of an existing file)."""
    seen = []
    real_open = os.open

    def spy(path, flags, perm=0o777, *args, **kwargs):
        seen.append((str(path), flags, perm))
        return real_open(path, flags, perm, *args, **kwargs)

    monkeypatch.setattr(config.os, "open", spy)
    path = tmp_path / "default.secrets"
    config.save_secrets(path, Secrets(environment_key="k"))
    target = [s for s in seen if s[0] == str(path)]
    assert (
        target and target[0][1] & os.O_EXCL and target[0][1] & os.O_CREAT and target[0][2] == 0o600
    )


def test_secrets_rewrite_replaces_permissions_and_contents(tmp_path):
    path = tmp_path / "default.secrets"
    path.write_text('{"environment_key": "old"}')
    os.chmod(path, 0o644)
    config.save_secrets(path, Secrets(environment_key="new"))
    assert mode(path) == 0o600
    assert config.load_secrets(path).environment_key == "new"
    assert [p.name for p in tmp_path.iterdir()] == ["default.secrets"]  # no temp file left behind


def test_rewrite_never_writes_through_a_symlink(tmp_path):
    target = tmp_path / "elsewhere.txt"
    target.write_text("untouched")
    path = tmp_path / "default.secrets"
    os.symlink(target, path)
    config.save_secrets(path, Secrets(environment_key="k"))
    assert target.read_text() == "untouched"  # the symlink's target was not written
    assert not path.is_symlink() and mode(path) == 0o600


def test_world_readable_secrets_are_refused(tmp_path):
    path = tmp_path / "default.secrets"
    config.save_secrets(path, Secrets(environment_key="k"))
    os.chmod(path, 0o644)
    with pytest.raises(ConfigError, match="chmod 600"):
        config.load_secrets(path)
    assert config.load_secrets(path, check_mode=False).environment_key == "k"


def test_secrets_are_not_in_reprs():
    keys = Secrets(
        environment_key="sk-ant-oat01-CANARY",
        sandbox_id_key="hmac-CANARY",
        api_key="sk-ant-api03-CANARY",
    )
    assert "CANARY" not in repr(keys)


# ------------------------------------------------------------------ TOML writer


@pytest.mark.parametrize(
    "value",
    [
        "plain",
        'quote " and backslash \\ path C:\\Volumes\\x',
        "tab\tnewline\nreturn\rformfeed\fbell\b",
        "nul\x00 del\x7f esc\x1b",
        "unicode: café, 数据, 🙂",
        "",
    ],
)
def test_toml_strings_round_trip(value):
    text = config.dumps_toml({"k": value})
    assert tomllib.loads(text) == {"k": value}


def test_toml_writer_covers_the_types_it_stores():
    values = {
        "s": "x",
        "b": False,
        "i": 42,
        "f": 7.5,
        "tiny": 1e-05,
        "lst": ["genie-one", "other"],
        "empty": [],
        "none": None,  # omitted
    }
    text = config.dumps_toml(values)
    assert "none" not in text
    assert tomllib.loads(text) == {k: v for k, v in values.items() if v is not None}


@pytest.mark.parametrize(
    "bad", [{"d": {"nested": 1}}, {"f": float("nan")}, {"f": float("inf")}, {"l": [1, 2]}]
)
def test_toml_writer_refuses_what_it_cannot_store(bad):
    with pytest.raises((TypeError, ValueError)):
        config.dumps_toml(bad)


def test_toml_writer_quotes_keys_it_cannot_write_bare():
    with pytest.raises(ValueError):
        config.dumps_toml({"bad key": "x"})


def test_lone_surrogates_are_refused():
    with pytest.raises(UnicodeEncodeError):
        config.toml_string("\ud800")


# ------------------------------------------------------------------ init: HMAC key and prompts


def test_init_creates_config_and_hmac_key(cfg_dir, monkeypatch, capsys, no_prompt):
    monkeypatch.setenv(main.ENVIRONMENT_KEY_VAR, "sk-ant-oat01-first")
    assert (
        main.main(
            ["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)]
        )
        == 0
    )
    secrets = config.load_secrets(cfg_dir / "default.secrets")
    assert secrets.environment_key == "sk-ant-oat01-first"
    assert len(secrets.sandbox_id_key or "") == 64  # secrets.token_hex(32)
    assert mode(cfg_dir) == 0o700
    assert mode(cfg_dir / "default.secrets") == 0o600
    out = capsys.readouterr().out
    assert "sk-ant-oat01-first" not in out and "Keys are not shown" in out


def test_reinit_keeps_the_hmac_key_and_updates_settings(cfg_dir, monkeypatch, no_prompt):
    monkeypatch.setenv(main.ENVIRONMENT_KEY_VAR, "sk-ant-oat01-first")
    main.main(["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)])
    first = config.load_secrets(cfg_dir / "default.secrets")

    monkeypatch.setenv(main.ENVIRONMENT_KEY_VAR, "sk-ant-oat01-rotated")
    main.main(
        [
            "init",
            "--profile",
            "other",
            "--environment",
            "env_2",
            "--config-dir",
            str(cfg_dir),
            "--backend",
            "mcp",
        ]
    )
    second = config.load_secrets(cfg_dir / "default.secrets")
    settings = config.load_settings(cfg_dir / "default.toml")

    assert second.sandbox_id_key == first.sandbox_id_key  # never regenerated
    assert second.environment_key == "sk-ant-oat01-rotated"
    assert (settings.profile, settings.environment_id, settings.backend) == (
        "other",
        "env_2",
        "mcp",
    )


def test_reinit_prompt_keeps_saved_key_on_enter(cfg_dir, monkeypatch):
    monkeypatch.delenv(main.ENVIRONMENT_KEY_VAR, raising=False)
    answers(monkeypatch, "sk-ant-oat01-saved")
    main.main(["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)])
    asked = answers(monkeypatch, "")  # Enter: keep the saved key
    main.main(["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)])
    assert "keep the saved" in asked[0]
    assert config.load_secrets(cfg_dir / "default.secrets").environment_key == "sk-ant-oat01-saved"


def test_init_prompts_hidden_and_never_echoes(cfg_dir, monkeypatch, capsys):
    monkeypatch.delenv(main.ENVIRONMENT_KEY_VAR, raising=False)
    asked = answers(monkeypatch, "sk-ant-oat01-PROMPTED")
    main.main(["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)])
    captured = capsys.readouterr()
    assert asked and "hidden" in asked[0]
    assert "PROMPTED" not in captured.out + captured.err
    assert (
        config.load_secrets(cfg_dir / "default.secrets").environment_key == "sk-ant-oat01-PROMPTED"
    )


def test_init_without_any_key_fails_clearly(cfg_dir, monkeypatch):
    monkeypatch.delenv(main.ENVIRONMENT_KEY_VAR, raising=False)
    answers(monkeypatch, "")
    with pytest.raises(SystemExit, match="no environment key"):
        main.main(
            ["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)]
        )
    assert not (cfg_dir / "default.secrets").exists()


def test_init_requires_profile_or_env_auth(cfg_dir, no_prompt):
    with pytest.raises(SystemExit, match="--profile"):
        main.main(["init", "--environment", "env_1", "--config-dir", str(cfg_dir)])


def test_init_requires_environment(cfg_dir, no_prompt):
    with pytest.raises(SystemExit, match="--environment"):
        main.main(["init", "--profile", "dev", "--config-dir", str(cfg_dir)])


def test_init_with_api_key_prompts_only_when_asked(cfg_dir, monkeypatch, no_prompt):
    monkeypatch.setenv(main.ENVIRONMENT_KEY_VAR, "sk-ant-oat01-x")
    monkeypatch.setenv(main.API_KEY_VAR, "sk-ant-api03-ENV")
    main.main(["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)])
    assert config.load_secrets(cfg_dir / "default.secrets").api_key is None
    main.main(
        [
            "init",
            "--profile",
            "dev",
            "--environment",
            "env_1",
            "--config-dir",
            str(cfg_dir),
            "--with-api-key",
        ]
    )
    assert config.load_secrets(cfg_dir / "default.secrets").api_key == "sk-ant-api03-ENV"


def test_init_warns_when_it_has_to_generate_a_missing_hmac_key(
    cfg_dir, monkeypatch, capsys, no_prompt
):
    monkeypatch.setenv(main.ENVIRONMENT_KEY_VAR, "sk-ant-oat01-x")
    main.main(["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)])
    (cfg_dir / "default.secrets").write_text(json.dumps({"environment_key": "sk-ant-oat01-x"}))
    capsys.readouterr()
    main.main(["init", "--profile", "dev", "--environment", "env_1", "--config-dir", str(cfg_dir)])
    assert "new one was generated" in capsys.readouterr().out


# ------------------------------------------------------------------ flags over config


def _write_settings(cfg_dir, **values) -> None:
    config.ensure_config_dir(cfg_dir)
    config.save_settings(cfg_dir / "default.toml", Settings(**values))


def test_flags_override_the_file_and_absent_flags_keep_it(cfg_dir, monkeypatch):
    _write_settings(
        cfg_dir,
        profile="file-profile",
        backend="sdk",
        lifecycle="ephemeral",
        retention_days=7.0,
        environment_id="env_1",
        export_volume="/Volumes/a/b/c",
    )
    args = main.build_parser().parse_args(
        [
            "run",
            "--config-dir",
            str(cfg_dir),
            "--profile",
            "flag-profile",
            "--lifecycle",
            "resumable",
            "--retention-days",
            "2",
            "--mcp",
            "genie-one",
        ]
    )
    ctx = main._load(args)
    assert ctx.settings.profile == "flag-profile"  # flag wins
    assert ctx.settings.lifecycle == "resumable"
    assert ctx.settings.retention_days == 2.0
    assert ctx.settings.mcp_servers == ["genie-one"]
    assert ctx.settings.export_volume == "/Volumes/a/b/c"  # not given: the file's value stays
    assert ctx.settings.backend == "sdk"


def test_flag_precedence_with_no_flags(cfg_dir):
    _write_settings(cfg_dir, profile="file-profile", environment_id="env_1")
    args = main.build_parser().parse_args(["list", "--config-dir", str(cfg_dir)])
    assert main._load(args).settings.profile == "file-profile"


def test_apply_overrides_skips_none_and_validates():
    base = Settings(profile="p", retention_days=7.0)
    merged = config.apply_overrides(
        base, profile=None, retention_days=3.0, mcp_servers=["genie-one"]
    )
    assert (merged.profile, merged.retention_days, merged.mcp_servers) == ("p", 3.0, ["genie-one"])
    with pytest.raises(ConfigError):
        config.apply_overrides(base, backend="docker")


def test_run_needs_a_config(cfg_dir, no_prompt):
    with pytest.raises(SystemExit, match="init"):
        main.main(["run", "--config-dir", str(cfg_dir), "--profile", "dev"])
