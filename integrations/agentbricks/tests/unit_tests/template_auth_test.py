"""The scaffolded agents fail fast, and boundedly, when Databricks auth can't be resolved."""

from __future__ import annotations

import importlib
import sys
import threading
from pathlib import Path

import pytest


@pytest.fixture(params=["langgraph", "openai"])
def agent(request, monkeypatch, tmp_path):
    dependencies = (
        ("databricks_langchain", "langgraph", "langchain", "langchain_mcp_adapters")
        if request.param == "langgraph"
        else ("agents", "databricks_openai")
    )
    for dependency in dependencies:
        pytest.importorskip(dependency, reason=f"Requires databricks-agentbricks[{request.param}]")
    template_path = (
        Path(__file__).parents[2] / f"src/databricks_agentbricks/templates/agent-{request.param}"
    )
    monkeypatch.syspath_prepend(str(template_path))
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("agent.agent")
    monkeypatch.setenv("HOME", str(tmp_path))
    yield module
    for name in list(sys.modules):
        if name in {"agent", "runtime"} or name.startswith(("agent.", "runtime.")):
            sys.modules.pop(name, None)


def test_check_passes_when_workspace_client_resolves(agent, monkeypatch):
    monkeypatch.setattr(agent, "workspace_client", lambda: object())

    agent._check_databricks_auth()


def test_check_names_the_profile_when_resolution_fails(agent, monkeypatch):
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "ml")

    def fail():
        raise ValueError("bad creds")

    monkeypatch.setattr(agent, "workspace_client", fail)

    with pytest.raises(RuntimeError, match=r"(?s)Tried profile 'ml'.*bad creds"):
        agent._check_databricks_auth()


def test_check_gives_up_when_resolution_hangs(agent, monkeypatch):
    release = threading.Event()
    monkeypatch.setattr(agent, "_AUTH_CHECK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(agent, "workspace_client", lambda: release.wait(5))

    with pytest.raises(RuntimeError, match="timed out"):
        agent._check_databricks_auth()
    release.set()


def test_check_skips_resolution_for_external_browser_without_cached_token(
    agent, monkeypatch, tmp_path
):
    config = tmp_path / "databrickscfg"
    config.write_text("[ml]\nhost = https://ml.example\nauth_type = external-browser\n")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(config))
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "ml")
    monkeypatch.setattr(
        agent, "workspace_client", lambda: pytest.fail("must not block on a browser flow")
    )

    with pytest.raises(RuntimeError, match="needs an interactive login"):
        agent._check_databricks_auth()


def test_check_resolves_external_browser_profile_with_cached_token(agent, monkeypatch, tmp_path):
    config = tmp_path / "databrickscfg"
    config.write_text("[ml]\nhost = https://ml.example\nauth_type = external-browser\n")
    cache = tmp_path / ".config/databricks-sdk-py/oauth"
    cache.mkdir(parents=True)
    (cache / "token.json").write_text("{}")
    monkeypatch.setenv("DATABRICKS_CONFIG_FILE", str(config))
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "ml")
    monkeypatch.setattr(agent, "workspace_client", lambda: object())

    agent._check_databricks_auth()
