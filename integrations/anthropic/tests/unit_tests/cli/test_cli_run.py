"""``run``: double-serving check, signals, the lock at start-up, aliases, start-up line."""

from __future__ import annotations

import asyncio
import signal
from typing import Any, cast

import pytest
from cli_fakes import FakeDispatcher, FakeStats, FakeWorkspace

from databricks_anthropic.cli import auth, config, doctor, lock, main
from databricks_anthropic.cli.config import Secrets, Settings

ENV = "env_run01"
REAL_LOGIN = auth.login  # captured before the run_env fixture swaps it for a fake
REAL_LAUNCHER = main._launcher


# ------------------------------------------------------------------ double-serving check (pure logic)


def check(values, *, force=False):
    """Run the check with scripted counts; returns (outcome, sleeps)."""
    stats = FakeStats(*values)
    sleeps: list[float] = []
    try:
        main.check_no_other_worker(
            lambda: stats.stats(ENV).workers_polling,
            environment_id=ENV,
            force=force,
            sleep=sleeps.append,
        )
    except main.ConcurrentWorkerError:
        return "refused", sleeps
    return "proceed", sleeps


def test_no_other_worker_proceeds_without_waiting():
    assert check([0]) == ("proceed", [])


def test_a_worker_that_just_stopped_is_waited_out():
    assert check([2, 0]) == ("proceed", [main.POLL_WINDOW_S])


def test_a_worker_still_polling_after_the_wait_refuses():
    assert check([1, 1]) == ("refused", [main.POLL_WINDOW_S])


def test_force_starts_anyway_after_the_wait():
    assert check([1, 1], force=True) == ("proceed", [main.POLL_WINDOW_S])


def test_a_failed_first_read_warns_and_proceeds(caplog):
    def broken():
        raise RuntimeError("no stats for you")

    sleeps: list[float] = []
    main.check_no_other_worker(broken, environment_id=ENV, force=False, sleep=sleeps.append)
    assert sleeps == []
    assert "starting without the double-serving check" in caplog.text


def test_a_failed_second_read_warns_and_proceeds(caplog):
    calls = iter([1, RuntimeError("flaky")])

    def read():
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    sleeps: list[float] = []
    main.check_no_other_worker(read, environment_id=ENV, force=False, sleep=sleeps.append)
    assert sleeps == [main.POLL_WINDOW_S]
    assert "starting anyway" in caplog.text


def test_the_wait_is_35_seconds_by_default():
    assert main.POLL_WINDOW_S == 35


# ------------------------------------------------------------------ signals


class SlowDispatcher:
    """Claims one launch that takes a while, then waits like the real dispatcher's finally block does."""

    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []
        self.events: list[str] = []

    async def run(self):
        launch = asyncio.ensure_future(self._launch())
        try:
            await asyncio.sleep(3600)
        finally:
            await launch  # claimed launches finish before run() returns
            self.events.append("finally")

    async def _launch(self):
        await asyncio.sleep(0.2)
        self.events.append("launched")


@pytest.mark.asyncio
async def test_sigterm_cancels_the_task_and_waits_for_claimed_launches():
    dispatcher = SlowDispatcher()
    asyncio.get_running_loop().call_later(0.05, signal.raise_signal, signal.SIGTERM)
    code = await main.serve(
        dispatcher, force_exit=lambda c: pytest.fail(f"must not force exit ({c})")
    )
    assert code == 0
    assert dispatcher.events == ["launched", "finally"]  # the launch was not dropped


@pytest.mark.asyncio
async def test_sigint_is_handled_the_same_way():
    dispatcher = SlowDispatcher()
    asyncio.get_running_loop().call_later(0.05, signal.raise_signal, signal.SIGINT)
    assert await main.serve(dispatcher, force_exit=lambda c: pytest.fail("no")) == 0
    assert dispatcher.events == ["launched", "finally"]


class StuckDispatcher:
    """Its claimed launch never finishes on its own; only the forced exit releases it."""

    def __init__(self) -> None:
        self.failures: list[tuple[str, str]] = []
        self.gate = asyncio.Event()

    async def run(self):
        try:
            await asyncio.sleep(3600)
        finally:
            await self.gate.wait()


@pytest.mark.asyncio
async def test_a_second_signal_force_exits_with_130():
    dispatcher = StuckDispatcher()
    codes: list[int] = []

    def force_exit(code: int) -> None:
        codes.append(code)
        dispatcher.gate.set()  # stands in for os._exit, which would end the test process

    loop = asyncio.get_running_loop()
    loop.call_later(0.05, signal.raise_signal, signal.SIGTERM)
    loop.call_later(0.1, signal.raise_signal, signal.SIGINT)
    assert await main.serve(dispatcher, force_exit=force_exit) == 0
    assert codes == [130]


class FailingDispatcher:
    failures = [("sesn_1", "RuntimeError: boom")]

    async def run(self):
        return []


@pytest.mark.asyncio
async def test_exit_status_is_one_when_a_launch_failed():
    assert await main.serve(FailingDispatcher()) == 1


# ------------------------------------------------------------------ run end to end, offline


@pytest.fixture
def run_env(tmp_path, monkeypatch):
    """A saved config and secrets, with Databricks, the queue and the dispatcher all faked."""
    directory = tmp_path / "cfg"
    config.ensure_config_dir(directory)
    config.save_settings(directory / "default.toml", Settings(profile="dev", environment_id=ENV))
    config.save_secrets(
        directory / "default.secrets",
        Secrets(environment_key="sk-ant-oat01-RUN", sandbox_id_key="h" * 64),
    )
    state: dict[str, Any] = {
        "stats": FakeStats(0),
        "pauses": [],
        "workspace": FakeWorkspace(),
        "launcher_args": None,
    }

    monkeypatch.setattr(auth, "login", lambda **kwargs: state["workspace"])
    monkeypatch.setattr(doctor, "environment_api", lambda key, base: state["stats"].client())
    monkeypatch.setattr(main, "_pause", state["pauses"].append)
    monkeypatch.setattr(main, "DatabricksSandboxDispatcher", FakeDispatcher)
    FakeDispatcher.instances.clear()

    def launcher(ctx, w, args, **kwargs):
        state["launcher_args"] = (ctx, kwargs)
        return object()

    monkeypatch.setattr(main, "_launcher", launcher)
    state["dir"] = str(directory)
    return state


def run_argv(state, *extra):
    return ["run", "--config-dir", state["dir"], *extra]


def test_run_prints_the_identity_line_and_releases_the_lock(run_env, capsys):
    assert main.main(run_argv(run_env)) == 0
    out = capsys.readouterr().out
    assert (
        f"serving environment {ENV} as dev@example.com: every session on this environment runs with this "
        "identity's Databricks access"
    ) in out
    assert lock.holder_pid(config.config_dir(run_env["dir"]), ENV) is None  # released on exit


def test_run_passes_the_hmac_key_and_prefix_to_the_launcher(run_env):
    main.main(run_argv(run_env))
    _ctx, kwargs = run_env["launcher_args"]
    assert kwargs["environment_key"] == "sk-ant-oat01-RUN"
    ctx, _ = run_env["launcher_args"]
    assert ctx.secrets.sandbox_id_key == "h" * 64
    assert ctx.settings.sandbox_prefix == "cmalocal-"


def test_run_refuses_while_another_dispatcher_holds_the_lock(run_env):
    holder = lock.EnvironmentLock(lock.lock_path(config.config_dir(run_env["dir"]), ENV), ENV)
    holder.acquire()
    try:
        with pytest.raises(
            SystemExit, match=rf"another dispatcher for env {ENV} is running \(pid \d+\)"
        ):
            main.main(run_argv(run_env))
    finally:
        holder.release()


def test_run_refuses_when_another_worker_is_polling_twice(run_env):
    run_env["stats"] = FakeStats(1, 1)
    with pytest.raises(SystemExit, match="refusing to start"):
        main.main(run_argv(run_env))
    assert run_env["pauses"] == [main.POLL_WINDOW_S]


def test_run_force_starts_anyway(run_env, capsys):
    run_env["stats"] = FakeStats(1, 1)
    assert main.main(run_argv(run_env, "--force")) == 0
    assert run_env["pauses"] == [main.POLL_WINDOW_S]


def test_run_waits_out_a_just_stopped_worker(run_env):
    run_env["stats"] = FakeStats(1, 0)
    assert main.main(run_argv(run_env)) == 0
    assert run_env["pauses"] == [main.POLL_WINDOW_S]


def test_run_without_a_profile_never_builds_a_databricks_client(run_env, monkeypatch, tmp_path):
    config.save_settings(
        config.config_dir(run_env["dir"]) / "default.toml", Settings(environment_id=ENV)
    )
    monkeypatch.setattr(
        auth, "login", REAL_LOGIN
    )  # the real login must refuse before any client exists

    def no_client(**kwargs):  # pragma: no cover - the assertion is that this is never reached
        raise AssertionError("WorkspaceClient must not be built without a profile")

    monkeypatch.setattr(auth, "WorkspaceClient", no_client)
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    with pytest.raises(SystemExit, match="--profile"):
        main.main(run_argv(run_env))


def test_run_refuses_a_world_readable_secrets_file(run_env):
    secrets_path = config.config_dir(run_env["dir"]) / "default.secrets"
    secrets_path.chmod(0o644)
    with pytest.raises(SystemExit, match="chmod 600"):
        main.main(run_argv(run_env))


# ------------------------------------------------------------------ aliases


@pytest.mark.parametrize(
    "alias, canonical, command",
    [
        ("sandbox-dispatcher", "run", "cmd_run"),
        ("sandbox-list", "list", "cmd_list"),
        ("sandbox-cleanup", "cleanup", "cmd_cleanup"),
    ],
)
def test_aliases_parse_to_the_same_command(monkeypatch, alias, canonical, command):
    seen: dict[str, dict] = {}

    def record(name):
        def fake(args):
            seen[name] = {
                k: v for k, v in vars(args).items() if k != "func"
            }  # the bound handler differs
            return 0

        return fake

    monkeypatch.setattr(main, command, record("alias"))
    main.main([alias, "--profile", "dev", "--config-dir", "/nonexistent-ok"])
    monkeypatch.setattr(main, command, record("canonical"))
    main.main([canonical, "--profile", "dev", "--config-dir", "/nonexistent-ok"])
    assert seen["alias"] == seen["canonical"]
    assert seen["alias"]["command"] == canonical


def test_run_help_mentions_the_flags_that_stay(capsys):
    with pytest.raises(SystemExit):
        main.main(["run", "--help"])
    text = capsys.readouterr().out
    for flag in (
        "--drain",
        "--max-concurrent-launches",
        "--max-idle",
        "--inactivity-timeout",
        "--image-uri",
        "--lifecycle",
        "--retention-days",
        "--export-volume",
        "--sweep-interval",
        "--sessions-gone",
        "--index-url",
        "--mcp",
        "--force",
    ):
        assert flag in text, flag


def test_cleanup_keeps_its_flags_and_gains_legacy_prefix(capsys):
    with pytest.raises(SystemExit):
        main.main(["cleanup", "--help"])
    text = capsys.readouterr().out
    for flag in (
        "--dry-run",
        "--retention-days",
        "--sessions-gone",
        "--delete-all",
        "--force",
        "--yes",
        "--legacy-prefix",
    ):
        assert flag in text, flag


def test_a_dispatcher_crash_is_one_line_and_exit_1(run_env, monkeypatch, caplog):
    class Crashing(FakeDispatcher):
        async def run(self):
            raise ConnectionError("queue unreachable")

    monkeypatch.setattr(main, "DatabricksSandboxDispatcher", Crashing)
    assert main.main(run_argv(run_env)) == 1
    assert "dispatcher stopped: ConnectionError: queue unreachable" in caplog.text
    assert lock.holder_pid(config.config_dir(run_env["dir"]), ENV) is None  # the lock is released


def test_legacy_prefix_cleanup_uses_the_old_names_without_a_key(run_env):
    """`cleanup --legacy-prefix cma-` plans for the old unsalted names: prefix cma-, no HMAC key."""
    from cli_fakes import FakeWorkspace as _FakeWorkspace

    ctx = main._load(main.build_parser().parse_args(["cleanup", "--config-dir", run_env["dir"]]))
    launcher = REAL_LAUNCHER(
        ctx,
        _FakeWorkspace(),
        cast(Any, object()),
        environment_key="",
        mcp_servers=[],
        legacy_prefix="cma-",
    )
    assert launcher.sandbox_id_prefix == "cma-"
    assert launcher.sandbox_id_key is None
    current = REAL_LAUNCHER(
        ctx, _FakeWorkspace(), cast(Any, object()), environment_key="", mcp_servers=[]
    )
    assert current.sandbox_id_prefix == "cmalocal-" and current.sandbox_id_key == "h" * 64


def test_mcp_flag_without_the_extra_gives_the_pip_hint(run_env, monkeypatch):
    monkeypatch.setattr(doctor, "mcp_extra_missing", lambda: True)
    with pytest.raises(SystemExit, match=r"pip install 'databricks-anthropic\[mcp\]'"):
        main._mcp_servers(FakeWorkspace(), ["genie-one"])


def test_an_ambient_anthropic_base_url_is_ignored(run_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com/anthropic")
    ctx = main._load(main.build_parser().parse_args(["list", "--config-dir", run_env["dir"]]))
    assert ctx.settings.anthropic_base_url == "https://api.anthropic.com"
