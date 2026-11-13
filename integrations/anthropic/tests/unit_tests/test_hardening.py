"""Secrets handling, failure paths and the copies of constants that worker.py must keep in step."""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from databricks_anthropic import _useragent
from databricks_anthropic.sandbox import _ops, mcp
from databricks_anthropic.sandbox import launcher as launcher_mod
from databricks_anthropic.sandbox.dispatcher import DatabricksSandboxDispatcher
from databricks_anthropic.sandbox.launcher import DatabricksSandboxLauncher


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


# ---------------------------------------------------------------- secrets


def test_worker_reads_and_deletes_secrets_file(monkeypatch, tmp_path):
    worker = load_worker()
    path = tmp_path / "worker.secrets"
    path.write_text(json.dumps({"ANTHROPIC_ENVIRONMENT_KEY": "k", "ANTHROPIC_WORK_SECRET": "s"}))
    monkeypatch.setenv("CMA_SECRETS_FILE", str(path))
    assert worker._load_secrets() == {
        "ANTHROPIC_ENVIRONMENT_KEY": "k",
        "ANTHROPIC_WORK_SECRET": "s",
    }
    assert not path.exists() and "CMA_SECRETS_FILE" not in os.environ


def test_worker_refuses_to_start_without_secrets_file(monkeypatch):
    monkeypatch.delenv("CMA_SECRETS_FILE", raising=False)
    with pytest.raises(RuntimeError, match="CMA_SECRETS_FILE"):
        load_worker()._load_secrets()


def test_export_skips_symlinks(monkeypatch, tmp_path):
    worker = load_worker()
    secret = tmp_path / "secret.txt"
    secret.write_text("do not export")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "report.csv").write_text("a\n")
    (outputs / "leak.txt").symlink_to(secret)
    (outputs / "linkdir").symlink_to(tmp_path, target_is_directory=True)
    uploaded = []
    fake = MagicMock()
    fake.files.upload.side_effect = lambda path, f, overwrite: uploaded.append(path)
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda **kw: fake)
    monkeypatch.setenv("CMA_EXPORT_VOLUME", "/Volumes/c/s/v/out")
    monkeypatch.setenv("CMA_EXPORT_DIR", str(outputs))
    assert worker._export_outputs("sesn_1") is True
    assert uploaded == ["/Volumes/c/s/v/out/sesn_1/report.csv"]


# ---------------------------------------------------------------- failure paths


@pytest.mark.asyncio
async def test_failed_launch_stops_its_work_item(monkeypatch):
    launcher = make_launcher()
    monkeypatch.setattr(launcher, "launch", MagicMock(side_effect=RuntimeError("quota")))
    stopped = []

    class Work:
        async def stop(self, work_id, **kwargs):
            stopped.append((work_id, kwargs))

    class Client:
        beta = SimpleNamespace(environments=SimpleNamespace(work=Work()))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    dispatcher = DatabricksSandboxDispatcher(launcher)
    monkeypatch.setattr(dispatcher, "_client", lambda: Client())
    item = SimpleNamespace(
        id="work_1", secret="s", data=SimpleNamespace(id="sesn_1", type="session")
    )
    await dispatcher._stop_work(item)  # ty: ignore[invalid-argument-type] - fake work item
    assert stopped == [("work_1", {"environment_id": "env_1", "force": True})]


def test_sweep_keeps_sandbox_when_session_lookup_fails():
    dispatcher = DatabricksSandboxDispatcher(make_launcher())
    sandbox = SimpleNamespace(
        name="sandboxes/cma-a",
        status=SimpleNamespace(state=_ops.SandboxState.SANDBOX_STATE_STOPPED),
        metadata=SimpleNamespace(
            custom_tags={
                "managed_by": "databricks-anthropic",
                "claude_environment_id": "env_1",
                "claude_session_id": "sesn_1",
                "lifecycle": "resumable",
            }
        ),
        create_time=None,
        update_time=None,
    )
    dispatcher.launcher.workspace_client.sandbox.list_sandboxes.return_value = [sandbox]  # ty: ignore[unresolved-attribute]
    api = MagicMock()
    api.beta.sessions.retrieve.side_effect = ConnectionError("blip")
    (action,) = dispatcher.plan_cleanup(sessions_api=api, now=0)
    assert action.action == "keep" and "lookup failed" in action.reason


def test_relaunch_marks_replacement_and_waits_for_old_worker(monkeypatch):
    scripts = []
    monkeypatch.setattr(_ops, "run", lambda w, n, s, **k: scripts.append(s))
    _ops.stop_pidfile_process(
        MagicMock(),
        "sb",
        "/x/worker.pid",
        marker_path="/x/worker.replacing",
        worker_path="/x/worker.py",
    )
    script = scripts[0]
    # Only a pid running our worker is signalled (a stale pidfile after a restart can name another process).
    assert (
        script.index("/proc/")
        < script.index("touch /x/worker.replacing")
        < script.index("kill -TERM")
    )
    assert "grep -qF -- /x/worker.py" in script
    assert "kill -0" in script and "kill -KILL" in script
    assert script.rstrip().endswith("rm -f /x/worker.replacing; true")


@pytest.mark.parametrize("on_exit", ["delete", "stop"])
def test_replaced_worker_keeps_the_sandbox(monkeypatch, tmp_path, on_exit):
    worker = load_worker()
    marker = tmp_path / "worker.replacing"
    marker.touch()
    fake = MagicMock()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda **kw: fake)
    monkeypatch.setattr(worker, "_export_outputs", lambda session_id: True)
    monkeypatch.setenv("CMA_ON_EXIT", on_exit)
    monkeypatch.setenv("CMA_REPLACING_MARKER", str(marker))
    worker._finish("sandboxes/cma-x", "sesn_1")
    fake.sandbox.delete_sandbox.assert_not_called()
    fake.sandbox.stop_sandbox.assert_not_called()


def test_export_skips_symlinked_output_root(monkeypatch, tmp_path):
    worker = load_worker()
    (tmp_path / "outputs").symlink_to("/", target_is_directory=True)
    fake = MagicMock()
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda **kw: fake)
    monkeypatch.setenv("CMA_EXPORT_VOLUME", "/Volumes/c/s/v/out")
    monkeypatch.setenv("CMA_EXPORT_DIR", str(tmp_path / "outputs"))
    assert worker._export_outputs("sesn_1") is True
    fake.files.upload.assert_not_called()


def test_upload_rejects_non_octal_mode():
    with pytest.raises(ValueError, match="octal"):
        _ops.upload_text(MagicMock(), "sb", "/x/f", "data", mode="600; rm -rf /")


def test_upload_is_atomic_and_private(monkeypatch):
    scripts = []
    monkeypatch.setattr(_ops, "run", lambda w, n, s, **k: scripts.append(s))
    _ops.upload_text(MagicMock(), "sb", "/x/f", "secret", mode="600")
    assert "umask 077" in scripts[0] and '"$DBX_PATH.part"' in scripts[0]
    assert "chmod 600" in scripts[-1] and 'mv -f "$DBX_PATH.part" "$DBX_PATH"' in scripts[-1]


# ---------------------------------------------------------------- genie wait tool


class _Session:
    def __init__(self, results):
        self.results, self.calls = list(results), []

    async def call_tool(self, name, arguments):
        self.calls.append(name)
        return self.results.pop(0) if self.results else _ok({"status": "in_progress"})


def _ok(payload):
    return SimpleNamespace(
        structured_content={"conversation_id": "c", "response_id": "r", **payload}, content=[]
    )


def _err(text):
    return SimpleNamespace(
        is_error=True, structured_content=None, content=[SimpleNamespace(type="text", text=text)]
    )


async def _call(tool, **args):
    out = await tool.call(args)
    return json.loads(out if isinstance(out, str) else "".join(b.get("text", "") for b in out))


@pytest.mark.asyncio
async def test_genie_error_is_reported_not_timed_out(monkeypatch):
    worker = load_worker()
    result = await _call(
        worker._genie_wait_tool(_Session([_err("PERMISSION_DENIED: no EXECUTE")])), question="q"
    )
    assert result == {"status": "error", "error": "PERMISSION_DENIED: no EXECUTE"}


@pytest.mark.asyncio
async def test_genie_timeout_cancels_query(monkeypatch):
    monkeypatch.setenv("CMA_GENIE_WAIT_TIMEOUT", "0")
    worker = load_worker()
    session = _Session([_ok({"status": "in_progress"})])
    result = await _call(worker._genie_wait_tool(session), question="q")
    assert result["status"] == "timed_out_after_0s" and session.calls[-1] == "genie_cancel_response"


@pytest.mark.asyncio
async def test_genie_large_answer_stays_valid_json():
    worker = load_worker()
    big = _ok(
        {
            "status": "completed",
            "final_answer": "x" * 50_000,
            "query_items": [{"item_id": str(i), "sql": "y" * 10_000} for i in range(30)],
        }
    )
    result = await _call(worker._genie_wait_tool(_Session([big])), question="q")
    assert result["final_answer"].endswith("[truncated]") and len(result["query_items"]) == 10


# ---------------------------------------------------------------- copies worker.py keeps


def test_worker_copies_match_package():
    worker = load_worker()
    assert (worker.PRODUCT_NAME, worker.PRODUCT_VERSION, worker.UPSTREAM_ENV) == (
        _useragent.PRODUCT_NAME,
        _useragent.PRODUCT_VERSION,
        _useragent.UPSTREAM_ENV,
    )
    assert worker._genie_wait_tool(None).name == mcp.GENIE_WAIT_DECLARATION["name"]


def test_worker_pins_are_exact_and_within_pyproject_ranges():
    # Parsed with a regex, not tomllib, so the check also runs on Python 3.10.
    from packaging.requirements import Requirement

    text = (Path(__file__).parents[2] / "pyproject.toml").read_text()
    deps = []
    for section in (r"^dependencies = \[(.*?)^\]", r"^mcp = \[(.*?)^\]"):
        block = re.search(r"(?ms)" + section, text)
        assert block, f"no {section} in pyproject.toml"
        deps += re.findall(r'^\s*"([^"]+)"', block.group(1), flags=re.M)
    declared = {Requirement(d).name: Requirement(d) for d in deps}
    for pin in (*launcher_mod.WORKER_PACKAGES, *launcher_mod.MCP_PACKAGES):
        req = Requirement(pin)
        (spec,) = req.specifier
        assert spec.operator == "==", f"{pin} is not an exact pin"
        assert declared[req.name].specifier.contains(spec.version), (
            f"{pin} is outside pyproject's {declared[req.name]}"
        )


def test_anthropic_internals_the_worker_imports_exist():
    """worker.py relies on non-public anthropic.lib modules; fail loudly if a release moves them."""
    from anthropic import beta_async_tool  # noqa: F401
    from anthropic.lib.environments import EnvironmentWorker  # noqa: F401
    from anthropic.lib.tools.agent_toolset import beta_agent_toolset_20260401  # noqa: F401

    pytest.importorskip("mcp")
    from anthropic.lib.tools.mcp import async_mcp_tool  # noqa: F401
