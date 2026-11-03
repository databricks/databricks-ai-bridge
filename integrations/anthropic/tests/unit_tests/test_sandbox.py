from __future__ import annotations

import ast
import json
import re
from importlib import resources
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from databricks.sdk.errors import NotFound

from databricks_anthropic.sandbox import _ops
from databricks_anthropic.sandbox import launcher as launcher_mod
from databricks_anthropic.sandbox.dispatcher import DatabricksSandboxDispatcher
from databricks_anthropic.sandbox.launcher import (
    DatabricksSandboxLauncher,
    SandboxMcpServer,
    sandbox_id_for_session,
)
from databricks_anthropic.sandbox.mcp import check_custom_tool_rules

AIP_122 = re.compile(r"^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$")


def work_item(session_id="sesn_01AbC_def", work_id="work_1", secret="s3cret", kind="session"):
    return SimpleNamespace(
        id=work_id, secret=secret, data=SimpleNamespace(id=session_id, type=kind)
    )


def make_launcher(**kwargs) -> DatabricksSandboxLauncher:
    defaults: dict[str, Any] = dict(
        workspace_client=MagicMock(), environment_id="env_1", environment_key="sk-env", confirm_s=0
    )
    return DatabricksSandboxLauncher(**{**defaults, **kwargs})


# ---------------------------------------------------------------- sandbox ids


def test_sandbox_id_is_aip122_deterministic_and_case_sensitive():
    a = sandbox_id_for_session("sesn_01VfLQBYovJgdc5b6KgLyNuy")
    assert AIP_122.match(a) and len(a) <= 63
    assert a == sandbox_id_for_session("sesn_01VfLQBYovJgdc5b6KgLyNuy")
    # Lower-casing would collide these; hashing must not.
    assert a != sandbox_id_for_session("sesn_01vflqbyovjgdc5b6kglynuy")


def test_custom_prefix_still_valid():
    assert AIP_122.match(sandbox_id_for_session("sesn_x", prefix="team-a-"))


# ---------------------------------------------------------------- worker env


def test_worker_env_holds_no_secrets_and_never_the_api_key():
    launcher = make_launcher(
        mcp_servers=[SandboxMcpServer("genie-one", "https://h/mcp", exclude=("view_ask",))]
    )
    env = launcher._worker_env(work_item(), "sandboxes/cma-x")
    assert "ANTHROPIC_WORK_SECRET" not in env and "ANTHROPIC_ENVIRONMENT_KEY" not in env
    assert "sk-env" not in env.values() and "s3cret" not in env.values()
    assert env["CMA_SECRETS_FILE"] == launcher_mod.WORKER_SECRETS
    assert env["ANTHROPIC_SESSION_ID"] == "sesn_01AbC_def"
    assert env["CMA_ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_BASE_URL" not in env
    assert json.loads(env["CMA_MCP_SERVERS"]) == [
        {"name": "genie-one", "url": "https://h/mcp", "exclude": ["view_ask"], "wait_tool": False}
    ]


def test_mcp_packages_only_when_servers_configured():
    assert not any(p.startswith("mcp") for p in make_launcher().packages())
    assert any(
        p.startswith("mcp")
        for p in make_launcher(mcp_servers=[SandboxMcpServer("m", "u")]).packages()
    )


# ---------------------------------------------------------------- launch


@pytest.fixture
def ops(monkeypatch):
    """Replace the sandbox ops with recorders."""
    calls: dict[str, list] = {}

    def record(name, result=None):
        def fn(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))
            return result(*args, **kwargs) if callable(result) else result

        monkeypatch.setattr(_ops, name, fn)

    record("ensure_running")
    record("create", lambda w, sid, **kw: f"sandboxes/{sid}")
    record("wait_for_state")
    record("stop_pidfile_process")
    record("start_detached", 4242)
    record("is_alive", True)
    record("delete_quietly")
    record("upload_text")
    record("tail", "log tail")
    record("run", SimpleNamespace(stdout=""))
    return calls


def test_launch_creates_sandbox_when_missing(ops, monkeypatch):
    def not_found(*a, **k):
        raise NotFound("nope")

    monkeypatch.setattr(_ops, "ensure_running", not_found)
    result = make_launcher().launch(work_item())
    assert result.created and result.pid == 4242
    assert result.sandbox_name == "sandboxes/" + sandbox_id_for_session("sesn_01AbC_def")
    ((_, kwargs),) = ops["start_detached"]
    assert "ANTHROPIC_WORK_SECRET" not in kwargs["envs"]
    secrets = [(a, k) for a, k in ops["upload_text"] if a[2] == launcher_mod.WORKER_SECRETS]
    assert len(secrets) == 1 and secrets[0][1]["mode"] == "600"
    assert json.loads(secrets[0][0][3]) == {
        "ANTHROPIC_ENVIRONMENT_KEY": "sk-env",
        "ANTHROPIC_WORK_SECRET": "s3cret",
    }
    assert "delete_quietly" not in ops


def test_create_deletes_sandbox_that_never_starts(ops, monkeypatch):
    monkeypatch.setattr(_ops, "ensure_running", MagicMock(side_effect=NotFound("nope")))
    monkeypatch.setattr(_ops, "wait_for_state", MagicMock(side_effect=TimeoutError("stuck")))
    with pytest.raises(TimeoutError):
        make_launcher().launch(work_item())
    assert len(ops["delete_quietly"]) == 1


def test_launch_deletes_sandbox_it_created_when_worker_dies(ops, monkeypatch):
    monkeypatch.setattr(_ops, "ensure_running", MagicMock(side_effect=NotFound("nope")))
    monkeypatch.setattr(_ops, "is_alive", lambda *a, **k: False)
    with pytest.raises(_ops.SandboxCommandError):
        make_launcher().launch(work_item())
    assert len(ops["delete_quietly"]) == 1


def test_launch_keeps_existing_sandbox_on_failure(ops, monkeypatch):
    monkeypatch.setattr(_ops, "is_alive", lambda *a, **k: False)
    with pytest.raises(_ops.SandboxCommandError):
        make_launcher().launch(work_item())
    assert "delete_quietly" not in ops  # resumed session: never delete the user's sandbox


def test_launch_replaces_previous_worker(ops):
    make_launcher().launch(work_item())
    assert len(ops["stop_pidfile_process"]) == 1


def test_bootstrap_skips_install_when_spec_matches(ops, monkeypatch):
    launcher = make_launcher()
    spec = " ".join(sorted(launcher.packages()))
    scripts = []

    def fake_run(w, name, script, **kwargs):
        scripts.append(script)
        return SimpleNamespace(stdout=spec if script.startswith("cat ") else "")

    monkeypatch.setattr(_ops, "run", fake_run)
    launcher._bootstrap("sandboxes/x")
    assert not any("uv pip install" in s for s in scripts)
    assert len(ops["upload_text"]) == 1  # worker is always (re)uploaded


def test_bootstrap_installs_when_spec_differs(ops, monkeypatch):
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_run(w, n, s, **k):
        calls.append((s, k))
        return SimpleNamespace(stdout="old")

    monkeypatch.setattr(_ops, "run", fake_run)
    make_launcher(index_url="https://user:tok@proxy/simple").launch(work_item())
    script, kwargs = next(c for c in calls if "uv pip install" in c[0])
    assert (
        "anthropic" in script and "proxy" not in script
    )  # index URL (maybe with credentials) not in the script
    assert kwargs["envs"]["UV_DEFAULT_INDEX"] == "https://user:tok@proxy/simple"
    assert "tok@" not in kwargs["envs"]["DBX_SPEC"]


def test_worker_module_is_packaged_and_self_contained():
    source = resources.files("databricks_anthropic.sandbox").joinpath("worker.py").read_text()
    tree = ast.parse(source)
    imported = {
        (node.module or "").split(".")[0]
        if isinstance(node, ast.ImportFrom)
        else alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "databricks_anthropic" not in imported  # not installed inside the sandbox


# ---------------------------------------------------------------- mcp


def test_custom_tool_rules():
    ok = {"type": "object", "properties": {"question": {"type": "string"}}}
    assert check_custom_tool_rules("genie_ask", ok) == []
    assert check_custom_tool_rules("mcp__x", ok)
    assert check_custom_tool_rules("bad name", ok)
    assert check_custom_tool_rules("t", {"oneOf": []})
    assert check_custom_tool_rules("t", {"properties": {"a": {"$ref": "#/x"}}})


# ---------------------------------------------------------------- dispatcher


class FakeClient:
    def __init__(self, items):
        self.items = items
        self.poller_kwargs: dict[str, Any] = {}
        self.beta = SimpleNamespace(
            environments=SimpleNamespace(work=SimpleNamespace(poller=self.poller))
        )

    async def poller(self, **kwargs):
        self.poller_kwargs = kwargs
        for item in self.items:
            yield item

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_dispatcher_launches_sessions_skips_others_and_records_failures(monkeypatch):
    launched = []

    def launch(work):
        if work.data.id == "sesn_bad":
            raise RuntimeError("boom")
        launched.append(work.data.id)
        return SimpleNamespace(sandbox_name="sandboxes/x")

    launcher = make_launcher()
    monkeypatch.setattr(launcher, "launch", launch)
    items = [work_item("sesn_a"), work_item("x", kind="other"), work_item("sesn_bad")]
    fake = FakeClient(items)
    dispatcher = DatabricksSandboxDispatcher(launcher, drain=True)
    monkeypatch.setattr(dispatcher, "_client", lambda: fake)
    await dispatcher.run()

    assert launched == ["sesn_a"]
    assert [f[0] for f in dispatcher.failures] == ["sesn_bad"]
    assert fake.poller_kwargs["auto_stop"] is False
    assert fake.poller_kwargs["block_ms"] is None and fake.poller_kwargs["drain"] is True


def test_dispatcher_client_ignores_ambient_anthropic_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stray")
    client = DatabricksSandboxDispatcher(make_launcher())._client()
    assert str(client.base_url).startswith("https://api.anthropic.com")
    assert list(client.auth_headers) == ["Authorization"]


def test_launcher_module_exports():
    assert launcher_mod.DEFAULT_WORKDIR == "/home/sandbox-agent/agent-workspace"


# ---------------------------------------------------------------- genie wait tool


@pytest.mark.asyncio
async def test_genie_wait_tool_polls_inside_worker_until_completed(monkeypatch):
    monkeypatch.setenv("CMA_GENIE_POLL_SECONDS", "0")  # no real waiting between polls
    import importlib.util

    spec = importlib.util.find_spec("databricks_anthropic.sandbox.worker")
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    calls = []

    class Session:
        async def call_tool(self, name, arguments):
            calls.append(name)
            status = "in_progress" if name == "genie_ask" or len(calls) < 4 else "completed"
            payload = {
                "conversation_id": "c1",
                "response_id": "r1",
                "status": status,
                "final_answer": "Total revenue is $66,471." if status == "completed" else None,
                "query_items": [{"item_id": "i1", "sql": "SELECT 1"}],
            }
            return SimpleNamespace(structured_content=payload, content=[])

    tool = worker._genie_wait_tool(Session())
    out = await tool.call({"question": "total revenue"})
    text = out if isinstance(out, str) else "".join(b.get("text", "") for b in out)
    result = json.loads(text)
    assert calls == [
        "genie_ask",
        "genie_poll_response",
        "genie_poll_response",
        "genie_poll_response",
    ]
    assert result["status"] == "completed" and result["final_answer"].startswith("Total revenue")
    assert result["query_items"] == [{"item_id": "i1", "sql": "SELECT 1"}]


async def _noop():
    return None


def test_genie_server_wait_mode_declares_one_tool_and_hides_raw_ask_poll():
    from databricks_anthropic.sandbox import mcp

    w = MagicMock()
    w.config.host = "https://h"
    w.get_workspace_id.return_value = 1
    server = mcp.genie_one_server(w)
    assert (
        server.wait_tool
        and "genie_ask" in server.exclude
        and "genie_get_query_result" not in server.exclude
    )
    assert server.to_env()["wait_tool"] is True
    assert (
        mcp.check_custom_tool_rules(
            "genie_ask_and_wait", mcp.GENIE_WAIT_DECLARATION["input_schema"]
        )
        == []
    )
    raw = mcp.genie_one_server(w, wait_tool=False)
    assert raw.exclude == ("view_ask",) and not raw.wait_tool


@pytest.mark.asyncio
async def test_dispatcher_accept_filter_launches_only_matching_sessions(monkeypatch):
    launched = []
    launcher = make_launcher()
    monkeypatch.setattr(launcher, "launch", lambda work: launched.append(work.data.id))
    fake = FakeClient([work_item("sesn_old"), work_item("sesn_mine"), work_item("sesn_other")])
    dispatcher = DatabricksSandboxDispatcher(
        launcher, drain=True, accept=lambda w: w.data.id == "sesn_mine"
    )
    monkeypatch.setattr(dispatcher, "_client", lambda: fake)
    await dispatcher.run()
    assert launched == ["sesn_mine"]


def test_launch_prepares_memory_dir_every_time(ops, monkeypatch):
    scripts = []
    monkeypatch.setattr(
        _ops, "run", lambda w, n, s, **k: scripts.append(s) or SimpleNamespace(stdout="ok")
    )
    make_launcher().launch(work_item())
    make_launcher().launch(work_item())
    memory_scripts = [s for s in scripts if "/mnt/memory" in s]
    assert len(memory_scripts) == 2 and all("sudo -n" in s for s in memory_scripts)
