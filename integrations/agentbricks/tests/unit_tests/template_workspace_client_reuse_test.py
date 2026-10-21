"""Templates build the App's WorkspaceClient once, not on every request.

Constructing a WorkspaceClient resolves auth with blocking I/O, so doing it inside async request
handlers stalls every request on the event loop.
"""

import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates"


class _CountingClientFactory:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return SimpleNamespace(config=SimpleNamespace(host="https://example.databricks.com/"))


def _clear_cache(module, name: str) -> None:
    getattr(getattr(module, name, None), "cache_clear", lambda: None)()


def _import_template_agent(monkeypatch, template: str):
    monkeypatch.syspath_prepend(str(TEMPLATES / template))
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            monkeypatch.delitem(sys.modules, name)
    return importlib.import_module("agent.agent")


@pytest.fixture
def clean_agent_modules():
    yield
    for name in list(sys.modules):
        if name == "agent" or name.startswith("agent."):
            sys.modules.pop(name, None)


@pytest.mark.usefixtures("clean_agent_modules")
def test_langgraph_graph_reuses_one_workspace_client_across_turns(monkeypatch):
    for dependency in ("databricks_langchain", "langgraph", "langchain", "langchain_mcp_adapters"):
        pytest.importorskip(dependency, reason="Requires databricks-agentbricks[langgraph]")
    agent = _import_template_agent(monkeypatch, "agent-langgraph")
    factory = _CountingClientFactory()
    monkeypatch.setattr(agent, "workspace_client", factory)
    _clear_cache(agent, "_app_workspace_client")

    async def no_mcp_tools(*_args, **_kwargs):
        return []

    monkeypatch.setattr(agent, "mcp_tools", no_mcp_tools)
    monkeypatch.setattr(agent, "genie_tools", lambda **_kwargs: [])
    monkeypatch.setattr(agent, "memory_tools", lambda _actor: [])
    monkeypatch.setattr(agent, "checkpointer", lambda: None)
    monkeypatch.setattr(agent, "_RoutedChatDatabricks", lambda **kwargs: kwargs)
    monkeypatch.setattr(agent, "create_agent", lambda **kwargs: kwargs)

    first = asyncio.run(agent.create_agent_graph("alice"))
    second = asyncio.run(agent.create_agent_graph("bob", model="system.ai.other-model"))

    assert factory.calls == 1
    assert first["model"]["workspace_client"] is second["model"]["workspace_client"]
    # The model can still change per turn; only the client behind it is shared.
    assert second["model"]["endpoint"] == "system.ai.other-model"
    _clear_cache(agent, "_app_workspace_client")


@pytest.mark.usefixtures("clean_agent_modules")
def test_custom_langgraph_reuses_one_model_client(monkeypatch):
    pytest.importorskip("databricks_langchain", reason="Requires databricks-agentbricks[langgraph]")
    agent = _import_template_agent(monkeypatch, "custom-agent-langgraph")
    factory = _CountingClientFactory()
    monkeypatch.setattr(agent, "workspace_client", factory)
    monkeypatch.setattr(agent, "_RoutedChatDatabricks", lambda **kwargs: kwargs)
    _clear_cache(agent, "_model")

    class _Graph:
        async def ainvoke(self, _input):
            return {"messages": [{"role": "assistant", "content": "hi"}]}

    monkeypatch.setattr(agent, "create_agent", lambda **_kwargs: _Graph())

    for _ in range(2):
        asyncio.run(agent.invoke([{"role": "user", "content": "hello"}]))

    assert factory.calls == 1
    _clear_cache(agent, "_model")


@pytest.mark.usefixtures("clean_agent_modules")
def test_custom_openai_configures_client_once(monkeypatch):
    pytest.importorskip("agents", reason="Requires databricks-agentbricks[openai]")
    pytest.importorskip("databricks_openai", reason="Requires databricks-agentbricks[openai]")
    agent = _import_template_agent(monkeypatch, "custom-agent-openai")
    factory = _CountingClientFactory()
    monkeypatch.setattr(agent, "workspace_client", factory)
    monkeypatch.setattr(agent, "AsyncDatabricksOpenAI", lambda **kwargs: kwargs)
    monkeypatch.setattr(agent, "set_default_openai_client", lambda _client: None)
    monkeypatch.setattr(agent, "set_default_openai_api", lambda _api: None)
    _clear_cache(agent, "_configure_client")

    async def run(*_args, **_kwargs):
        return SimpleNamespace(final_output="hi")

    monkeypatch.setattr(agent.Runner, "run", run)

    for _ in range(2):
        asyncio.run(agent.invoke([{"role": "user", "content": "hello"}]))

    assert factory.calls == 1
    _clear_cache(agent, "_configure_client")


@pytest.mark.parametrize("framework", ["agent-langgraph", "agent-openai"])
def test_ui_workspace_host_is_resolved_once(monkeypatch, framework):
    spec = importlib.util.spec_from_file_location(
        f"reuse_ui_{framework.replace('-', '_')}",
        TEMPLATES / f"ui/{framework}/runtime/ui.py",
    )
    assert spec and spec.loader
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    factory = _CountingClientFactory()
    monkeypatch.setattr(ui, "workspace_client", factory)
    _clear_cache(ui, "_workspace_host")

    assert ui._workspace_host() == "https://example.databricks.com"
    assert ui._workspace_host() == "https://example.databricks.com"
    assert factory.calls == 1
    _clear_cache(ui, "_workspace_host")
