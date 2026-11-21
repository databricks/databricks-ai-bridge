from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from databricks_anthropic.sandbox import _ops
from databricks_anthropic.sandbox.dispatcher import EPHEMERAL_GRACE_S, DatabricksSandboxDispatcher
from databricks_anthropic.sandbox.launcher import DatabricksSandboxLauncher

NOW = 2_000_000_000.0
DAY = 86400


def launcher(**kwargs) -> DatabricksSandboxLauncher:
    return DatabricksSandboxLauncher(
        workspace_client=MagicMock(),
        environment_id="env_1",
        environment_key="sk-env",
        confirm_s=0,
        **kwargs,
    )


def work(session_id="sesn_X"):
    return SimpleNamespace(
        id="work_1", secret="s", data=SimpleNamespace(id=session_id, type="session")
    )


# ---------------------------------------------------------------- launcher options


def test_default_lifecycle_is_ephemeral_delete_on_exit():
    env = launcher()._worker_env(work(), "sandboxes/cma-x")
    assert env["CMA_ON_EXIT"] == "delete"
    assert "CMA_EXPORT_VOLUME" not in env


def test_resumable_stops_on_exit_and_export_env():
    env = launcher(lifecycle="resumable", export_volume_path="/Volumes/c/s/v/out/")._worker_env(
        work(), "sb"
    )
    assert env["CMA_ON_EXIT"] == "stop"
    assert env["CMA_EXPORT_VOLUME"] == "/Volumes/c/s/v/out"
    assert env["CMA_EXPORT_DIR"] == "/home/sandbox-agent/agent-workspace/outputs"


def test_invalid_lifecycle_rejected():
    with pytest.raises(ValueError):
        launcher(lifecycle="forever")


def test_new_sandbox_is_tagged_with_session(monkeypatch):
    created = {}
    monkeypatch.setattr(_ops, "ensure_running", MagicMock(side_effect=_ops.NotFound("nope")))
    monkeypatch.setattr(
        _ops, "create", lambda w, sid, **kw: created.update(kw, sid=sid) or f"sandboxes/{sid}"
    )
    monkeypatch.setattr(_ops, "wait_for_state", lambda *a, **k: None)
    lc = launcher(lifecycle="resumable")
    cast(MagicMock, lc.workspace_client).sandbox.get_sandbox.side_effect = _ops.NotFound("nope")
    name, was_created = lc.ensure_sandbox("sesn_ABC")
    assert was_created and name == f"sandboxes/{created['sid']}"
    assert created["tags"] == {
        "managed_by": "databricks-anthropic",
        "claude_session_id": "sesn_ABC",
        "claude_environment_id": "env_1",
        "lifecycle": "resumable",
    }
    assert "sesn_ABC" in created["display_name"]


# ---------------------------------------------------------------- cleanup policy


def sandbox(name, state, *, updated, lifecycle="resumable", session="sesn_1"):
    tags = {
        "managed_by": "databricks-anthropic",
        "claude_environment_id": "env_1",
        "claude_session_id": session,
        "lifecycle": lifecycle,
    }
    return SimpleNamespace(
        name=f"sandboxes/cma-{name}",
        status=SimpleNamespace(state=state),
        update_time=SimpleNamespace(seconds=int(updated), nanos=0),
        create_time=SimpleNamespace(seconds=0, nanos=0),
        metadata=SimpleNamespace(custom_tags=tags),
    )


@pytest.fixture
def dispatcher():
    d = DatabricksSandboxDispatcher(launcher(lifecycle="resumable", retention_days=7))
    running, stopped = (
        _ops.SandboxState.SANDBOX_STATE_RUNNING,
        _ops.SandboxState.SANDBOX_STATE_STOPPED,
    )
    d.launcher.workspace_client.sandbox.list_sandboxes.return_value = [  # ty: ignore[unresolved-attribute]
        sandbox("active", running, updated=NOW - 30 * DAY),
        sandbox("recent", stopped, updated=NOW - 2 * DAY),
        sandbox("old", stopped, updated=NOW - 8 * DAY),
        sandbox("eph-left", stopped, updated=NOW - EPHEMERAL_GRACE_S - 60, lifecycle="ephemeral"),
        sandbox("eph-fresh", stopped, updated=NOW - 60, lifecycle="ephemeral"),
        SimpleNamespace(
            name="sandboxes/someone-elses", status=None, update_time=None, create_time=None
        ),
    ]
    return d


def decisions(plan):
    return {a.sandbox_name.removeprefix("sandboxes/cma-"): a.action for a in plan}


def test_retention_and_leftover_ephemeral(dispatcher):
    plan = dispatcher.plan_cleanup(now=NOW)
    assert decisions(plan) == {
        "active": "keep",
        "recent": "keep",
        "old": "delete",
        "eph-left": "delete",
        "eph-fresh": "keep",
    }  # non-cma sandboxes are ignored


def test_custom_retention_days(dispatcher):
    assert decisions(dispatcher.plan_cleanup(now=NOW, retention_days=1))["recent"] == "delete"


def test_session_gone_deletes_even_running_sandbox(dispatcher):
    api = MagicMock()
    api.beta.sessions.retrieve.return_value = SimpleNamespace(archived_at="2026-11-01T00:00:00Z")
    assert set(decisions(dispatcher.plan_cleanup(now=NOW, sessions_api=api)).values()) == {"delete"}


def test_cleanup_deletes_only_planned_and_dry_run_deletes_nothing(dispatcher, monkeypatch):
    deleted = []
    monkeypatch.setattr(_ops, "delete_quietly", lambda w, name: deleted.append(name))
    monkeypatch.setattr("databricks_anthropic.sandbox.dispatcher.time.time", lambda: NOW)
    dispatcher.cleanup(dry_run=True)
    assert deleted == []
    dispatcher.cleanup()
    assert sorted(deleted) == ["sandboxes/cma-eph-left", "sandboxes/cma-old"]


def test_delete_all_keeps_running_sandboxes_unless_forced(dispatcher):
    plan = decisions(dispatcher.plan_cleanup(now=NOW, delete_all=True))
    assert plan.pop("active") == "keep" and set(plan.values()) == {"delete"}
    assert set(
        decisions(dispatcher.plan_cleanup(now=NOW, delete_all=True, force=True)).values()
    ) == {"delete"}


def test_cleanup_never_touches_sandboxes_it_does_not_own():
    d = DatabricksSandboxDispatcher(launcher())
    stopped = _ops.SandboxState.SANDBOX_STATE_STOPPED
    ours = sandbox("ours", stopped, updated=0)
    other_env = sandbox("other-env", stopped, updated=0)
    other_env.metadata.custom_tags["claude_environment_id"] = "env_2"
    untagged = sandbox("untagged", stopped, updated=0)
    untagged.metadata.custom_tags = {}
    d.launcher.workspace_client.sandbox.list_sandboxes.return_value = [ours, other_env, untagged]  # ty: ignore[unresolved-attribute]
    assert list(decisions(d.plan_cleanup(now=NOW, delete_all=True, force=True))) == ["ours"]


@pytest.mark.parametrize("prefix", ["", "-", "CMA-", "cma", "cma_x-", "a" * 31 + "-"])
def test_unsafe_sandbox_prefix_rejected(prefix):
    with pytest.raises(ValueError, match="sandbox_id_prefix"):
        launcher(sandbox_id_prefix=prefix)


@pytest.mark.parametrize("export_dir", ["/etc", "../outside", "a/../../b"])
def test_export_dir_must_stay_inside_workdir(export_dir):
    with pytest.raises(ValueError, match="export_dir"):
        launcher(export_dir=export_dir)


# ---------------------------------------------------------------- worker exit


def load_worker():
    spec = importlib.util.find_spec("databricks_anthropic.sandbox.worker")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "on_exit,export_ok,expected",
    [
        ("delete", True, "delete"),
        ("delete", False, "stop"),
        ("stop", True, "stop"),
    ],
)
def test_worker_finish(monkeypatch, on_exit, export_ok, expected):
    worker = load_worker()
    calls = []
    fake = MagicMock()
    fake.sandbox.delete_sandbox.side_effect = lambda n: calls.append("delete")
    fake.sandbox.stop_sandbox.side_effect = lambda n: calls.append("stop")
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda **kw: fake)
    monkeypatch.setattr(worker, "_export_outputs", lambda session_id: export_ok)
    monkeypatch.setenv("CMA_ON_EXIT", on_exit)
    worker._finish("sandboxes/cma-x", "sesn_1")
    assert calls == [expected]


def test_worker_exports_files_to_volume(monkeypatch, tmp_path):
    worker = load_worker()
    (tmp_path / "sub").mkdir()
    (tmp_path / "report.csv").write_text("a,b\n")
    (tmp_path / "sub" / "chart.png").write_bytes(b"png")
    uploaded = []
    fake = MagicMock()
    fake.files.upload.side_effect = lambda path, f, overwrite: uploaded.append(path)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda **kw: fake)
    monkeypatch.setenv("CMA_EXPORT_VOLUME", "/Volumes/c/s/v/out")
    monkeypatch.setenv("CMA_EXPORT_DIR", str(tmp_path))
    assert worker._export_outputs("sesn_1") is True
    assert sorted(uploaded) == [
        "/Volumes/c/s/v/out/sesn_1/report.csv",
        "/Volumes/c/s/v/out/sesn_1/sub/chart.png",
    ]
