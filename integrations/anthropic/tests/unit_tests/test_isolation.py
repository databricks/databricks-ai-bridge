"""Hardening for shared deployments: only the sessions token enters a sandbox (H1), no login shell and an
allow-listed worker environment (H2), unpredictable verified sandbox names (H4), idempotent launch (H5),
a verified worker file (H6) and https only (H7)."""

from __future__ import annotations

import base64
import importlib.util
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import AlreadyExists, NotFound

from databricks_anthropic.sandbox import _ops
from databricks_anthropic.sandbox import launcher as launcher_mod
from databricks_anthropic.sandbox.launcher import (
    DatabricksSandboxLauncher,
    SandboxOwnershipError,
    sandbox_id_for_session,
)


def secret(payload: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


SECRET = secret({"sessions_token": "tok-sess", "ingress_token": "tok-in"})


def work(secret_value: str | None = SECRET, work_id="work_1", session_id="sesn_A"):
    return SimpleNamespace(
        id=work_id, secret=secret_value, data=SimpleNamespace(id=session_id, type="session")
    )


def make_launcher(**kwargs) -> DatabricksSandboxLauncher:
    defaults: dict[str, Any] = dict(
        workspace_client=MagicMock(), environment_id="env_1", environment_key="sk-env", confirm_s=0
    )
    return DatabricksSandboxLauncher(**{**defaults, **kwargs})


def get_sandbox(lc: DatabricksSandboxLauncher) -> MagicMock:
    return cast(MagicMock, lc.workspace_client).sandbox.get_sandbox


def load_worker():
    path = launcher_mod.resources.files("databricks_anthropic.sandbox").joinpath("worker.py")
    spec = importlib.util.spec_from_file_location("cma_worker_isolation", str(path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tagged(session_id="sesn_A", environment_id="env_1", managed_by="databricks-anthropic"):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            custom_tags={
                "managed_by": managed_by,
                "claude_session_id": session_id,
                "claude_environment_id": environment_id,
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
        ("worker_progress", (True, True, "")),
        ("state_of", _ops.SandboxState.SANDBOX_STATE_RUNNING),
        ("delete_quietly", None),
        ("upload_text", None),
        ("tail", ""),
        ("run", SimpleNamespace(stdout="")),
    ]:
        record(name, result)
    return calls


# ---------------------------------------------------------------- H1: no environment key in a sandbox


def test_secrets_carry_only_the_sessions_token():
    out = make_launcher()._worker_secrets(work())
    assert set(out) == {"ANTHROPIC_WORK_SECRET"}
    payload = json.loads(base64.urlsafe_b64decode(out["ANTHROPIC_WORK_SECRET"]))
    assert payload == {"sessions_token": "tok-sess"}  # ingress/source tokens dropped


@pytest.mark.parametrize("bad", [None, "", "not-base64!", secret({"ingress_token": "x"})])
def test_launch_fails_closed_without_a_sessions_token(ops, bad):
    with pytest.raises(ValueError, match="sessions token"):
        make_launcher().launch(work(bad))
    assert "create" not in ops and "upload_text" not in ops  # nothing created or uploaded


def test_launch_refuses_if_environment_key_would_enter_the_sandbox(ops):
    # e.g. a misconfiguration that puts the key in the MCP server list
    lc = make_launcher(extra_packages=())
    lc.mcp_servers = [launcher_mod.SandboxMcpServer("m", "https://h/mcp?k=sk-env")]
    with pytest.raises(RuntimeError, match="environment key"):
        lc.launch(work())
    assert "create" not in ops


def test_worker_refuses_an_environment_key(monkeypatch):
    worker = load_worker()
    with pytest.raises(RuntimeError, match="environment key"):
        worker._session_credential(
            {"ANTHROPIC_ENVIRONMENT_KEY": "k", "ANTHROPIC_WORK_SECRET": SECRET}
        )
    monkeypatch.setenv("ANTHROPIC_ENVIRONMENT_KEY", "k")
    with pytest.raises(RuntimeError, match="environment key"):
        worker._session_credential({"ANTHROPIC_WORK_SECRET": SECRET})


def test_worker_uses_the_sessions_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_ENVIRONMENT_KEY", raising=False)
    worker = load_worker()
    assert worker._session_credential({"ANTHROPIC_WORK_SECRET": SECRET}) == (SECRET, "tok-sess")
    with pytest.raises(RuntimeError, match="sessions token"):
        worker._session_credential({"ANTHROPIC_WORK_SECRET": secret({"x": 1})})


def test_worker_and_launcher_decode_secrets_alike():
    worker = load_worker()
    for value in [SECRET, secret({"sessions_token": ""}), "%%%", None, secret(["a"])]:
        assert worker._sessions_token(value) == launcher_mod.sessions_token_from_secret(value)


# ---------------------------------------------------------------- H2: no login shell, env allowlist


def test_run_never_uses_a_login_or_interactive_shell():
    w = MagicMock()
    w.sandbox.execute_command_sync.return_value = MagicMock()
    _ops.run(w, "sandboxes/x", "true", check=False)
    args = w.sandbox.execute_command_sync.call_args.kwargs["args"]
    assert args[:3] == ["--noprofile", "--norc", "-c"] and "-l" not in "".join(args[:3])


def test_worker_starts_under_env_i_with_allowlist(monkeypatch):
    scripts = []
    monkeypatch.setattr(
        _ops, "run", lambda w, n, s, **k: scripts.append((s, k)) or SimpleNamespace(stdout="pid=7")
    )
    pid = _ops.start_detached(
        MagicMock(),
        "sandboxes/x",
        "exec python w.py",
        log_path="/l/w.log",
        pid_path="/l/w.pid",
        envs={"CMA_X": "1"},
    )
    script, kwargs = scripts[0]
    assert pid == 7 and " env -i " in script and "--noprofile --norc -c" in script
    assert (
        '${CMA_X+"CMA_X=${CMA_X}"}' in script
        and '${DATABRICKS_CONFIG_FILE+"DATABRICKS_CONFIG_FILE=${DATABRICKS_CONFIG_FILE}"}' in script
    )
    assert "PYTHONPATH" not in script and "LD_PRELOAD" not in script
    assert kwargs["envs"] == {"CMA_X": "1"}  # values via the exec env, never the command line


def test_start_detached_rejects_odd_env_names(monkeypatch):
    monkeypatch.setattr(_ops, "run", MagicMock())
    with pytest.raises(ValueError):
        _ops.start_detached(
            MagicMock(), "s", "c", log_path="/l/a", pid_path="/l/p", envs={"A;rm -rf /": "x"}
        )


# ---------------------------------------------------------------- H4: names and ownership


def test_hmac_ids_differ_from_plain_and_by_key():
    plain = sandbox_id_for_session("sesn_A")
    k1 = sandbox_id_for_session("sesn_A", key="k1")
    assert plain != k1 != sandbox_id_for_session("sesn_A", key="k2")
    assert k1 == sandbox_id_for_session("sesn_A", key=b"k1") and len(k1) == len(plain)
    lc = make_launcher(sandbox_id_key="k1")
    assert lc.sandbox_name_for("sesn_A") == f"sandboxes/{k1}"
    assert "k1" not in repr(lc)


@pytest.mark.parametrize(
    "found",
    [
        tagged(session_id="sesn_OTHER"),
        tagged(environment_id="env_OTHER"),
        tagged(managed_by="someone-else"),
        SimpleNamespace(metadata=None),
    ],
)
def test_existing_sandbox_with_wrong_tags_is_never_used(ops, found):
    lc = make_launcher()
    get_sandbox(lc).return_value = found
    with pytest.raises(SandboxOwnershipError):
        lc.launch(work())
    assert "ensure_running" not in ops and "upload_text" not in ops and "delete_quietly" not in ops


def test_existing_sandbox_with_matching_tags_is_used(ops):
    lc = make_launcher()
    get_sandbox(lc).return_value = tagged()
    result = lc.launch(work())
    assert not result.created and "create" not in ops and len(ops["ensure_running"]) == 1


def test_already_exists_race_verifies_then_shares(ops, monkeypatch):
    lc = make_launcher()
    get_sandbox(lc).side_effect = [NotFound("x"), tagged()]
    monkeypatch.setattr(_ops, "create", MagicMock(side_effect=AlreadyExists("taken")))
    name, created = lc.ensure_sandbox("sesn_A")
    assert not created and name == lc.sandbox_name_for("sesn_A")


def test_already_exists_squatted_name_is_refused(ops, monkeypatch):
    lc = make_launcher()
    get_sandbox(lc).side_effect = [NotFound("x"), tagged("sesn_OTHER")]
    monkeypatch.setattr(_ops, "create", MagicMock(side_effect=AlreadyExists("taken")))
    with pytest.raises(SandboxOwnershipError):
        lc.ensure_sandbox("sesn_A")


# ---------------------------------------------------------------- H5: idempotent launch


def test_redelivered_item_with_live_worker_is_left_alone(ops, monkeypatch):
    lc = make_launcher()
    get_sandbox(lc).return_value = tagged()
    monkeypatch.setattr(_ops, "live_worker", lambda *a, **k: _ops.WorkerState(99, "work_1"))
    result = lc.launch(work(work_id="work_1"))
    assert result.reused and result.pid == 99
    assert "stop_pidfile_process" not in ops and "start_detached" not in ops


def test_new_item_replaces_previous_worker(ops, monkeypatch):
    lc = make_launcher()
    get_sandbox(lc).return_value = tagged()
    monkeypatch.setattr(_ops, "live_worker", lambda *a, **k: _ops.WorkerState(99, "work_OLD"))
    result = lc.launch(work(work_id="work_2"))
    assert not result.reused and len(ops["stop_pidfile_process"]) == 1


def test_live_worker_parses_pid_and_work_id(monkeypatch):
    monkeypatch.setattr(_ops, "run", lambda *a, **k: SimpleNamespace(stdout="123 work_9\n"))
    assert _ops.live_worker(
        MagicMock(), "s", "/p", "/w", worker_path="/x/worker.py"
    ) == _ops.WorkerState(123, "work_9")
    monkeypatch.setattr(_ops, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    assert _ops.live_worker(MagicMock(), "s", "/p", "/w", worker_path="/x/worker.py") is None
    monkeypatch.setattr(_ops, "run", lambda *a, **k: SimpleNamespace(stdout="123\n"))
    assert (
        _ops.live_worker(MagicMock(), "s", "/p", "/w", worker_path="/x/worker.py") is None
    )  # no work id recorded


# ---------------------------------------------------------------- H6 / H7


def test_start_command_verifies_worker_hash_and_records_work_id(ops):
    lc = make_launcher()
    get_sandbox(lc).side_effect = NotFound("x")
    lc.launch(work())
    ((args, _),) = ops["start_detached"]
    command = args[2]
    source = launcher_mod.resources.files("databricks_anthropic.sandbox").joinpath("worker.py")
    digest = launcher_mod.hashlib.sha256(source.read_text().encode()).hexdigest()
    assert f"echo {digest}'  '{launcher_mod.WORKER_PATH} | sha256sum -c" in command
    assert command.index("sha256sum") < command.index("exec ")
    assert launcher_mod.WORKER_WORK_ID in command and " -I " in command


@pytest.mark.parametrize("url", ["http://api.anthropic.com", "ftp://x", "api.anthropic.com"])
def test_base_url_must_be_https(url):
    with pytest.raises(ValueError, match="https"):
        make_launcher(anthropic_base_url=url)
