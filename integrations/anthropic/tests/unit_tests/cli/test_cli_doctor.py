"""``doctor`` with fakes: every check, its OK or FIX line, and the exit code."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
from cli_fakes import FakeStats, FakeWorkspace

from databricks_anthropic.cli import auth, config, doctor, lock, main
from databricks_anthropic.cli.config import Secrets, Settings

ENV = "env_doc01"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    directory = tmp_path / "cfg"
    config.ensure_config_dir(directory)
    config.save_settings(directory / "default.toml", Settings(profile="dev", environment_id=ENV))
    config.save_secrets(
        directory / "default.secrets",
        Secrets(environment_key="sk-ant-oat01-DOC", sandbox_id_key="h"),
    )
    state: dict[str, Any] = {
        "dir": directory,
        "stats": FakeStats(0),
        "workspace": FakeWorkspace(),
        "login_error": None,
    }

    def login(**kwargs):
        state["login_kwargs"] = kwargs
        if state["login_error"]:
            raise auth.AuthSetupError(state["login_error"])
        return state["workspace"]

    monkeypatch.setattr(auth, "login", login)
    monkeypatch.setattr(doctor, "environment_api", lambda key, base: state["stats"].client())
    monkeypatch.setattr(doctor, "mcp_extra_missing", lambda: False)
    return state


def doctor_main(state, *extra) -> int:
    return main.main(["doctor", "--config-dir", str(state["dir"]), *extra])


def test_all_green_exits_zero(setup, capsys):
    assert doctor_main(setup) == 0
    out = capsys.readouterr().out
    for check in (
        "[OK] Config file",
        "[OK] Config directory is private (0700)",
        "[OK] Secrets file is private (0600)",
        "[OK] Databricks login",
        "[OK] Databricks Sandbox API",
        "[OK] Anthropic environment ID + key",
        "[OK] Work queue",
        "[OK] Single-instance lock",
    ):
        assert check in out, check
    assert "sk-ant-oat01-DOC" not in out


def test_login_uses_the_profile_from_the_file(setup, capsys):
    doctor_main(setup)
    assert setup["login_kwargs"]["profile"] == "dev"


def test_missing_config_is_a_fix_with_the_init_command(tmp_path, capsys):
    rc = main.main(["doctor", "--config-dir", str(tmp_path / "empty")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FIX] Config file" in out and "databricks-anthropic init" in out


def test_world_readable_secrets_is_a_fix(setup, capsys):
    os.chmod(setup["dir"] / "default.secrets", 0o644)
    assert doctor_main(setup) == 1
    out = capsys.readouterr().out
    assert "[FIX] Secrets file is private (0600)" in out and "chmod 600" in out


def test_missing_profile_stops_before_login(setup, capsys):
    config.save_settings(setup["dir"] / "default.toml", Settings(environment_id=ENV))
    assert doctor_main(setup) == 1
    out = capsys.readouterr().out
    assert "[FIX] Databricks profile" in out and "--profile" in out
    assert "login_kwargs" not in setup  # login was never attempted


def test_login_failure_is_a_fix(setup, capsys):
    setup["login_error"] = "Profile 'dev' not found in ~/.databrickscfg."
    assert doctor_main(setup) == 1
    assert "[FIX] Databricks login" in capsys.readouterr().out


def test_sandbox_api_failure_is_a_fix(setup, capsys):
    def broken(**kwargs):
        raise PermissionError("Sandboxes preview is off")

    setup["workspace"].sandbox.list_sandboxes = broken
    assert doctor_main(setup) == 1
    out = capsys.readouterr().out
    assert "[FIX] Databricks Sandbox API" in out and "Previews" in out


def test_genie_check_runs_only_with_mcp(setup, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(doctor, "_check_mcp", lambda rep, w, names: calls.append(names))
    doctor_main(setup)
    assert calls == []
    doctor_main(setup, "--mcp", "genie-one")
    assert calls == [["genie-one"]]


def test_missing_mcp_extra_gives_the_pip_hint(setup, monkeypatch, capsys):
    monkeypatch.setattr(doctor, "mcp_extra_missing", lambda: True)
    assert doctor_main(setup, "--mcp", "genie-one") == 1
    assert "pip install 'databricks-anthropic[mcp]'" in capsys.readouterr().out


def test_missing_environment_key_is_a_fix(setup, capsys):
    config.save_secrets(setup["dir"] / "default.secrets", Secrets(sandbox_id_key="h"))
    assert doctor_main(setup) == 1
    assert "[FIX] Anthropic environment ID + key" in capsys.readouterr().out


def test_other_workers_polling_is_reported_as_a_warning_not_a_failure(setup, capsys):
    setup["stats"] = FakeStats(2)
    assert doctor_main(setup) == 0
    out = capsys.readouterr().out
    assert "[WARN] Work queue" in out and "2 worker(s) polling" in out


def test_unreadable_queue_stats_are_a_warning(setup, capsys):
    setup["stats"] = FakeStats(RuntimeError("401 from the queue"))
    assert doctor_main(setup) == 0
    assert "[WARN] Work queue" in capsys.readouterr().out


def test_a_running_dispatcher_on_this_machine_is_a_fix(setup, capsys):
    holder = lock.EnvironmentLock(lock.lock_path(setup["dir"], ENV), ENV)
    holder.acquire()
    try:
        assert doctor_main(setup) == 1
        out = capsys.readouterr().out
        assert "[FIX] Single-instance lock" in out
        assert f"another dispatcher for env {ENV} is running (pid {os.getpid()})" in out
    finally:
        holder.release()


def test_self_hosted_check_runs_only_with_an_api_key(setup, monkeypatch, capsys):
    seen = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.beta = SimpleNamespace(
                environments=SimpleNamespace(
                    retrieve=lambda env_id: SimpleNamespace(
                        name="laptop", config=SimpleNamespace(type="self_hosted")
                    )
                )
            )

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    doctor_main(setup)
    assert seen == {}  # no API key saved: no call
    config.save_secrets(
        setup["dir"] / "default.secrets",
        Secrets(environment_key="k", sandbox_id_key="h", api_key="a"),
    )
    assert doctor_main(setup) == 0
    assert "[OK] Anthropic environment is self-hosted" in capsys.readouterr().out
    assert seen["api_key"] == "a"


def test_workers_polling_reads_the_environment_stats():
    assert doctor.workers_polling(FakeStats(3).client(), ENV) == 3
    assert doctor.workers_polling(FakeStats(None).client(), ENV) == 0


def test_lock_and_queue_are_reported_even_when_login_fails(setup, capsys):
    """They need no Databricks login, so a broken login does not hide them."""
    setup["login_error"] = "Profile 'dev' not found in ~/.databrickscfg."
    setup["stats"] = FakeStats(3)
    holder = lock.EnvironmentLock(lock.lock_path(setup["dir"], ENV), ENV)
    holder.acquire()
    try:
        assert doctor_main(setup) == 1
        out = capsys.readouterr().out
        assert "[FIX] Single-instance lock" in out
        assert "[WARN] Work queue" in out
        assert "[FIX] Databricks login" in out
    finally:
        holder.release()
