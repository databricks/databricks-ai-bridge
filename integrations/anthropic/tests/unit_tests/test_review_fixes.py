"""Regression tests for the PR review findings: stale pid, finishing worker, health recovery, kept sandboxes,
start-up failures, Genie payload shapes and the env allowlist."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound

from databricks_anthropic.sandbox import _ops
from databricks_anthropic.sandbox import dispatcher as dispatcher_mod
from databricks_anthropic.sandbox import launcher as launcher_mod
from databricks_anthropic.sandbox.dispatcher import KEPT_MARKER, DatabricksSandboxDispatcher
from databricks_anthropic.sandbox.launcher import DatabricksSandboxLauncher

SECRET = "eyJzZXNzaW9uc190b2tlbiI6ICJ0b2sifQ"  # base64url of {"sessions_token": "tok"}


def load_worker():
    spec = importlib.util.find_spec("databricks_anthropic.sandbox.worker")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_launcher(**kwargs) -> DatabricksSandboxLauncher:
    defaults: dict[str, Any] = dict(
        workspace_client=MagicMock(), environment_id="env_1", environment_key="sk-env", confirm_s=0
    )
    return DatabricksSandboxLauncher(**{**defaults, **kwargs})


def work(session_id="sesn_A", work_id="sesn_A"):
    return SimpleNamespace(
        id=work_id, secret=SECRET, data=SimpleNamespace(id=session_id, type="session")
    )


def tagged(session_id="sesn_A"):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            custom_tags={
                "managed_by": "databricks-anthropic",
                "claude_session_id": session_id,
                "claude_environment_id": "env_1",
            }
        )
    )


@pytest.fixture
def ops(monkeypatch):
    calls: dict[str, list] = {}

    def record(name, result=None):
        def fn(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return result(*args, **kwargs) if callable(result) else result

        monkeypatch.setattr(_ops, name, fn)

    for name, result in [
        ("ensure_running", None),
        ("create", lambda w, sid, **kw: f"sandboxes/{sid}"),
        ("wait_for_state", None),
        ("stop_pidfile_process", None),
        ("live_worker", None),
        ("start_detached", 4242),
        ("alive_with_log", (True, "")),
        ("state_of", _ops.SandboxState.SANDBOX_STATE_RUNNING),
        ("delete_quietly", None),
        ("upload_text", None),
        ("run", SimpleNamespace(stdout="")),
    ]:
        record(name, result)
    return calls


# ---------------------------------------------------------------- 3. stale pid after stop/start


def test_live_worker_only_trusts_a_pid_running_the_worker(monkeypatch):
    scripts = []
    monkeypatch.setattr(
        _ops, "run", lambda w, n, s, **k: scripts.append(s) or SimpleNamespace(stdout="")
    )
    assert _ops.live_worker(MagicMock(), "s", "/p", "/w", worker_path="/x/worker.py") is None
    script = scripts[0]
    assert "/proc/" in script and "grep -qF -- /x/worker.py" in script
    assert script.index("/proc/") < script.index('echo "$p')
    assert "else rm -f /p /w" in script  # stale pidfile and work id are removed


# ---------------------------------------------------------------- 4. new turn while the worker is finishing


def test_finishing_worker_is_not_live(monkeypatch):
    monkeypatch.setattr(_ops, "run", lambda *a, **k: SimpleNamespace(stdout="finishing\n"))
    assert (
        _ops.live_worker(
            MagicMock(), "s", "/p", "/w", worker_path="/x/worker.py", finishing_path="/x/fin"
        )
        is None
    )


def test_finish_writes_finishing_marker_and_rechecks_replacement(monkeypatch, tmp_path):
    worker = load_worker()
    replacing, finishing = tmp_path / "replacing", tmp_path / "finishing"
    fake = MagicMock()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda **kw: fake)
    monkeypatch.setenv("CMA_ON_EXIT", "delete")
    monkeypatch.setenv("CMA_REPLACING_MARKER", str(replacing))
    monkeypatch.setenv("CMA_FINISHING_MARKER", str(finishing))

    def slow_export(session_id):  # a replacement starts while the export runs
        assert finishing.exists()
        replacing.touch()
        return True

    monkeypatch.setattr(worker, "_export_outputs", slow_export)
    worker._finish("sandboxes/cma-x", "sesn_1")
    fake.sandbox.delete_sandbox.assert_not_called()


def test_launch_recreates_a_sandbox_the_old_worker_deleted(ops, monkeypatch):
    lc = make_launcher()
    get = cast(MagicMock, lc.workspace_client).sandbox.get_sandbox
    get.side_effect = [tagged(), NotFound("gone")]  # existing at first, gone when re-created
    states = iter([NotFound("deleted by the finishing worker")])

    def state_of(*a, **k):
        raise next(states)

    monkeypatch.setattr(_ops, "state_of", state_of)
    result = make_launcher.__globals__["DatabricksSandboxLauncher"].launch(lc, work())
    assert result.created and len(ops["create"]) == 1 and len(ops["start_detached"]) == 1


def test_start_command_clears_the_finishing_marker(ops):
    lc = make_launcher()
    cast(MagicMock, lc.workspace_client).sandbox.get_sandbox.side_effect = NotFound("x")
    lc.launch(work())
    ((args, kwargs),) = ops["start_detached"]
    assert f"rm -f {launcher_mod.WORKER_FINISHING}" in args[2]
    assert kwargs["envs"]["CMA_FINISHING_MARKER"] == launcher_mod.WORKER_FINISHING


# ---------------------------------------------------------------- 5. early worker failures


def test_worker_keeps_sandbox_when_it_fails_before_attaching(monkeypatch, tmp_path):
    worker = load_worker()
    fake = MagicMock()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda **kw: fake)
    monkeypatch.setenv("CMA_FINISHING_MARKER", str(tmp_path / "fin"))
    worker._finish("sandboxes/cma-x", "sesn_1", attached=False)
    fake.sandbox.delete_sandbox.assert_not_called()
    fake.sandbox.stop_sandbox.assert_not_called()


def test_launch_error_includes_the_worker_log(ops, monkeypatch):
    monkeypatch.setattr(_ops, "alive_with_log", lambda *a, **k: (False, "RuntimeError: bad token"))
    lc = make_launcher()
    cast(MagicMock, lc.workspace_client).sandbox.get_sandbox.side_effect = NotFound("x")
    with pytest.raises(_ops.SandboxCommandError, match="bad token"):
        lc.launch(work())
    assert len(ops["delete_quietly"]) == 1  # created here, so deleted after reading the log


def test_launch_explains_a_sandbox_deleted_during_startup(ops, monkeypatch):
    def gone(*a, **k):
        raise NotFound("gone")

    monkeypatch.setattr(_ops, "alive_with_log", gone)
    lc = make_launcher()
    cast(MagicMock, lc.workspace_client).sandbox.get_sandbox.side_effect = NotFound("x")
    with pytest.raises(_ops.SandboxCommandError, match="deleted while its worker was starting"):
        lc.launch(work())


def test_alive_with_log_uses_one_exec(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _ops, "run", lambda *a, **k: calls.append(a) or SimpleNamespace(stdout="DEAD\nline1\nline2")
    )
    assert _ops.alive_with_log(MagicMock(), "s", 7, "/l/w.log") == (False, "line1\nline2")
    assert len(calls) == 1


# ---------------------------------------------------------------- 2. kept (export-failed) sandboxes


def _sandbox(display_name, updated):
    return SimpleNamespace(
        name="sandboxes/cma-k",
        display_name=display_name,
        status=SimpleNamespace(state=_ops.SandboxState.SANDBOX_STATE_STOPPED),
        update_time=SimpleNamespace(seconds=int(updated), nanos=0),
        create_time=SimpleNamespace(seconds=0, nanos=0),
        metadata=SimpleNamespace(
            custom_tags={
                "managed_by": "databricks-anthropic",
                "claude_environment_id": "env_1",
                "claude_session_id": "sesn_1",
                "lifecycle": "ephemeral",
            }
        ),
    )


@pytest.mark.parametrize(("days", "expected"), [(2, "keep"), (8, "delete")])  # retention_days = 7
def test_cleanup_keeps_export_failed_sandboxes_until_retention(days, expected):
    d = DatabricksSandboxDispatcher(make_launcher(retention_days=7, sandbox_id_prefix="cma-"))
    now = 100 * 86400
    sb = _sandbox(f"Claude session sesn_1 {KEPT_MARKER}", now - days * 86400)
    cast(MagicMock, d.launcher.workspace_client).sandbox.list_sandboxes.return_value = [sb]
    (action,) = d.plan_cleanup(now=now)
    assert action.action == expected


def test_unmarked_leftover_ephemeral_is_still_deleted_after_grace():
    d = DatabricksSandboxDispatcher(make_launcher(retention_days=7, sandbox_id_prefix="cma-"))
    now = 100 * 86400
    sb = _sandbox("Claude session sesn_1", now - dispatcher_mod.EPHEMERAL_GRACE_S - 60)
    cast(MagicMock, d.launcher.workspace_client).sandbox.list_sandboxes.return_value = [sb]
    (action,) = d.plan_cleanup(now=now)
    assert action.action == "delete"


def test_failed_export_marks_and_stops_the_sandbox(monkeypatch, tmp_path):
    worker = load_worker()
    fake = MagicMock()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda **kw: fake)
    monkeypatch.setattr(worker, "_export_outputs", lambda session_id: False)
    monkeypatch.setenv("CMA_ON_EXIT", "delete")
    monkeypatch.setenv("CMA_FINISHING_MARKER", str(tmp_path / "fin"))
    worker._finish("sandboxes/cma-x", "sesn_1")
    fake.sandbox.delete_sandbox.assert_not_called()
    fake.sandbox.stop_sandbox.assert_called_once()
    args = fake.sandbox.update_sandbox.call_args.args
    assert KEPT_MARKER in args[1].display_name and list(args[2].paths) == ["display_name"]


def test_worker_kept_marker_matches_dispatcher():
    assert load_worker().KEPT_MARKER == KEPT_MARKER


# ---------------------------------------------------------------- 1. health recovers after a crash loop


def test_health_recovers_once_a_run_stays_up(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
    import service as svc  # noqa: E402

    from databricks_anthropic.sandbox.dispatcher import DispatcherStatus

    clock = {"t": 1000.0}
    monkeypatch.setattr(svc.time, "monotonic", lambda: clock["t"])
    service = svc.DispatcherService(make_launcher())
    service.dispatcher = cast(Any, SimpleNamespace(status=DispatcherStatus(polling=True)))
    service._task = cast(Any, SimpleNamespace(done=lambda: False))
    service.consecutive_crashes = svc.UNHEALTHY_AFTER_RESTARTS  # after a quick crash loop
    service.run_started_at = clock["t"]
    assert not service.healthy
    clock["t"] += svc.RECOVERED_AFTER_S + 1
    assert service.healthy and service.consecutive_crashes == 0


# ---------------------------------------------------------------- 8. Genie payload shapes


class _Result:
    def __init__(self, text=None, structured=None):
        self.structured_content = structured
        self.content = [SimpleNamespace(type="text", text=text)] if text is not None else []
        self.is_error = False


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_Result(text="plain answer"), {"text": "plain answer"}),
        (_Result(text=json.dumps([1, 2])), {"items": [1, 2]}),
        (_Result(structured=[{"a": 1}]), {"items": [{"a": 1}]}),
        (_Result(text=json.dumps({"status": "COMPLETED"})), {"status": "COMPLETED"}),
    ],
)
def test_mcp_payload_is_always_a_dict(result, expected):
    assert load_worker()._mcp_payload(result) == expected


def test_genie_wait_returns_an_unpollable_answer_instead_of_timing_out():
    worker = load_worker()
    session = MagicMock()

    async def call_tool(name, arguments):
        assert name == "genie_ask"
        return _Result(text="42 trips")

    session.call_tool = call_tool
    tool = worker._genie_wait_tool(session)
    out = json.loads(asyncio.run(tool.call({"question": "how many?"})))
    assert "timed_out" not in json.dumps(out) and "42 trips" in json.dumps(out)


# ---------------------------------------------------------------- 10. env allowlist leaves unset vars out


def test_env_allowlist_skips_unset_variables(monkeypatch):
    scripts = []
    monkeypatch.setattr(
        _ops, "run", lambda w, n, s, **k: scripts.append(s) or SimpleNamespace(stdout="pid=1")
    )
    _ops.start_detached(
        MagicMock(), "s", "c", log_path="/l/a", pid_path="/l/p", envs={"CMA_X": "1"}
    )
    assert '${TZ+"TZ=${TZ}"}' in scripts[0] and '${CMA_X+"CMA_X=${CMA_X}"}' in scripts[0]
    assert 'TZ="${TZ}"' not in scripts[0]


# ---------------------------------------------------------------- 9. hash-locked worker dependencies


@pytest.mark.parametrize(
    ("filename", "pins"),
    [
        ("worker-requirements.txt", launcher_mod.WORKER_PACKAGES),
        (
            "worker-requirements-mcp.txt",
            (*launcher_mod.WORKER_PACKAGES, *launcher_mod.MCP_PACKAGES),
        ),
    ],
)
def test_worker_locks_match_the_pins_and_carry_hashes(filename, pins):
    import re
    from importlib import resources

    text = resources.files("databricks_anthropic.sandbox").joinpath(filename).read_text()
    entries = re.findall(r"^([A-Za-z0-9_.\-\[\]]+)==([^\s;\\]+)", text, flags=re.M)
    locked = {name.split("[")[0].lower().replace("_", "-"): version for name, version in entries}
    for pin in pins:
        name, version = pin.split("==")
        assert locked[name.split("[")[0].lower()] == version, f"{pin} not locked at that version"
    # every requirement block carries at least one hash
    blocks = re.split(r"\n(?=[A-Za-z0-9])", text.strip())
    assert all("--hash=sha256:" in b for b in blocks if not b.startswith("#"))


def test_lock_follows_mcp_configuration():
    assert make_launcher().worker_lock()[0] == "worker-requirements.txt"
    mcp = make_launcher(mcp_servers=[launcher_mod.SandboxMcpServer("m", "u")])
    assert mcp.worker_lock()[0] == "worker-requirements-mcp.txt"


def test_extra_packages_are_installed_after_the_lock(ops, monkeypatch):
    calls = []
    monkeypatch.setattr(
        _ops, "run", lambda w, n, s, **k: calls.append(s) or SimpleNamespace(stdout="old")
    )
    make_launcher(extra_packages=("polars==1.0.0",))._bootstrap("sandboxes/x")
    script = next(c for c in calls if "--require-hashes" in c)
    assert script.index("--require-hashes") < script.index("polars==1.0.0")
