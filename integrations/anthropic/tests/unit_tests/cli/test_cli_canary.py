"""Canary: fake keys go into init and are never printed, logged or echoed by any command."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from cli_fakes import FakeDispatcher, FakeStats, FakeWorkspace

from databricks_anthropic.cli import auth, config, doctor, main

ENV_KEY = "sk-ant-oat01-CANARY-environment"
API_KEY = "sk-ant-api03-CANARY-api"
HMAC = "0123456789abcdef-CANARY-hmac"
ENV = "env_canary1"


@pytest.fixture
def canary(tmp_path, monkeypatch):
    directory = tmp_path / "cfg"
    monkeypatch.setenv(main.ENVIRONMENT_KEY_VAR, ENV_KEY)
    monkeypatch.setenv(main.API_KEY_VAR, API_KEY)
    monkeypatch.setattr(auth, "login", lambda **kwargs: FakeWorkspace())
    monkeypatch.setattr(doctor, "environment_api", lambda key, base: FakeStats(0).client())
    monkeypatch.setattr(main, "DatabricksSandboxDispatcher", FakeDispatcher)
    monkeypatch.setattr(main, "_launcher", lambda ctx, w, args, **kwargs: object())
    monkeypatch.setattr(main, "_pause", lambda seconds: None)

    import anthropic

    class FakeAnthropic:
        def __init__(self, **kwargs):
            self.beta = SimpleNamespace(
                environments=SimpleNamespace(
                    retrieve=lambda env_id: SimpleNamespace(
                        name="laptop", config=SimpleNamespace(type="self_hosted")
                    )
                )
            )

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    return directory


def test_no_command_prints_or_logs_a_key(canary, monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    d = str(canary)

    assert (
        main.main(
            ["init", "--profile", "dev", "--environment", ENV, "--config-dir", d, "--with-api-key"]
        )
        == 0
    )
    # The keys did reach the secrets file (the canary is live, not vacuous).
    saved = config.load_secrets(canary / "default.secrets")
    assert saved.environment_key == ENV_KEY and saved.api_key == API_KEY
    # Give the HMAC key a known canary value so it is checked as well.
    config.save_secrets(
        canary / "default.secrets",
        config.Secrets(environment_key=ENV_KEY, api_key=API_KEY, sandbox_id_key=HMAC),
    )

    assert main.main(["doctor", "--config-dir", d, "-v"]) == 0
    assert main.main(["list", "--config-dir", d, "-v"]) == 0
    assert main.main(["cleanup", "--config-dir", d, "--dry-run", "--sessions-gone", "-v"]) == 0
    assert main.main(["run", "--config-dir", d, "--drain", "--sessions-gone", "-v"]) == 0

    captured = capsys.readouterr()
    text = captured.out + captured.err + caplog.text
    for secret in (ENV_KEY, API_KEY, HMAC, "CANARY"):
        assert secret not in text, f"{secret!r} leaked into output or logs"


def test_a_failing_login_does_not_echo_keys(canary, monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    d = str(canary)
    main.main(
        ["init", "--profile", "dev", "--environment", ENV, "--config-dir", d, "--with-api-key"]
    )

    def boom(**kwargs):
        raise auth.AuthSetupError("Login for profile 'dev' failed: 401")

    monkeypatch.setattr(auth, "login", boom)
    with pytest.raises(SystemExit):
        main.main(["run", "--config-dir", d, "-v"])
    captured = capsys.readouterr()
    assert "CANARY" not in captured.out + captured.err + caplog.text
