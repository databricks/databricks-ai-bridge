"""Behavior tests for static OpenAI and LangGraph manifest runtimes."""

from __future__ import annotations

import asyncio
import importlib
import pathlib
import shutil
import sys
import types
from types import SimpleNamespace

import pytest

from databricks_mason import init as init_mod


def _write_direct_manifest(project: pathlib.Path, framework: str) -> None:
    (project / "agent.toml").write_text(
        f'''schema_version = 1

[agent]
framework = "{framework}"

[[tools]]
id = "sandbox"
source = {{ kind = "sandbox", service = "system.ai.sandbox" }}
policy = {{ downscope = [{{ resource = "table:samples.nyctaxi.trips", permission = "read_only" }}] }}

[[tools]]
id = "web"
source = {{ kind = "mcp", service = "system.ai.web_search" }}

[[tools]]
id = "lookup"
source = {{ kind = "uc_function", function = "main.tools.lookup" }}
''',
        encoding="utf-8",
    )


def _project(tmp_path: pathlib.Path, framework: str) -> pathlib.Path:
    project = tmp_path / framework
    mason = project / "agent" / "mason"
    mason.mkdir(parents=True)
    (project / "agent" / "__init__.py").write_text("", encoding="utf-8")
    (mason / "__init__.py").write_text("", encoding="utf-8")
    (project / "agent" / "mcps.py").write_text(
        "def build_mcp_servers():\n    return []\n", encoding="utf-8"
    )
    _write_direct_manifest(project, framework)
    if framework == "langgraph":
        template_mason = (
            pathlib.Path(__file__).parents[2] / "templates" / "agent-langgraph" / "agent" / "mason"
        )
        shutil.copyfile(template_mason / "tool_manifest.py", mason / "tool_manifest.py")
        shutil.copyfile(template_mason / "mcp_runtime.py", mason / "mcp_runtime.py")
    else:
        init_mod._install_runtime_plumbing(project, framework)
    return project


def _clear_agent_modules() -> None:
    for name in tuple(sys.modules):
        if name == "agent" or name.startswith("agent."):
            del sys.modules[name]


def _load_runtime(project: pathlib.Path, monkeypatch):
    _clear_agent_modules()
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(project))
    return importlib.import_module("agent.mason.mcp_runtime")


class _FakeWorkspaceClient:
    def __init__(self):
        self.config = SimpleNamespace(host="https://df1.example.com")


def test_openai_runtime_loads_direct_manifest_and_protects_sandbox_meta(
    tmp_path: pathlib.Path, monkeypatch
):
    project = _project(tmp_path, "openai")

    class FakeMcpServer:
        def __init__(self, url=None, workspace_client=None, timeout=None, **kwargs):
            self.url = url
            self.workspace_client = workspace_client or _FakeWorkspaceClient()
            self.timeout = timeout
            self.name = kwargs.get("name", "unnamed")
            self.kwargs = kwargs

        @classmethod
        def from_uc_function(
            cls, catalog, schema, function_name=None, workspace_client=None, **kwargs
        ):
            client = workspace_client or _FakeWorkspaceClient()
            return cls(
                url=(
                    f"{client.config.host}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}"
                ),
                workspace_client=client,
                **kwargs,
            )

        async def call_tool(self, name, arguments, **kwargs):
            return name, arguments, kwargs

    class FakeManager:
        def __init__(self, servers):
            self.active_servers = servers

    class FakeStack:
        async def enter_async_context(self, manager):
            return manager

    agents = types.ModuleType("agents")
    agents_mcp = types.ModuleType("agents.mcp")
    agents_mcp.__dict__["MCPServer"] = object
    agents_mcp.__dict__["MCPServerManager"] = FakeManager
    databricks = types.ModuleType("databricks")
    databricks_sdk = types.ModuleType("databricks.sdk")
    databricks_sdk.__dict__["WorkspaceClient"] = _FakeWorkspaceClient
    databricks_openai = types.ModuleType("databricks_openai")
    databricks_openai_agents = types.ModuleType("databricks_openai.agents")
    databricks_openai_agents.__dict__["McpServer"] = FakeMcpServer
    monkeypatch.setitem(sys.modules, "agents", agents)
    monkeypatch.setitem(sys.modules, "agents.mcp", agents_mcp)
    monkeypatch.setitem(sys.modules, "databricks", databricks)
    monkeypatch.setitem(sys.modules, "databricks.sdk", databricks_sdk)
    monkeypatch.setitem(sys.modules, "databricks_openai", databricks_openai)
    monkeypatch.setitem(sys.modules, "databricks_openai.agents", databricks_openai_agents)

    runtime = _load_runtime(project, monkeypatch)
    servers = asyncio.run(runtime.connect(FakeStack()))

    assert [server.name for server in servers] == ["sandbox", "web", "lookup"]
    assert servers[0].url == ("https://df1.example.com/ai-gateway/mcp-services/system.ai.sandbox")
    assert servers[1].url == (
        "https://df1.example.com/ai-gateway/mcp-services/system.ai.web_search"
    )
    assert servers[2].url == ("https://df1.example.com/api/2.0/mcp/functions/main/tools/lookup")
    name, arguments, kwargs = asyncio.run(
        servers[0].call_tool(
            "sandbox",
            {"code": 'print("ok")'},
            meta={"downscope": {"tables": []}, "trace_id": "trace"},
        )
    )
    assert name == "sandbox"
    assert arguments == {"code": 'print("ok")'}
    assert kwargs["meta"] == {
        "trace_id": "trace",
        "downscope": {"tables": [{"name": "samples.nyctaxi.trips", "permission": "read_only"}]},
    }


def test_langgraph_runtime_loads_direct_manifest_and_protects_sandbox_meta(
    tmp_path: pathlib.Path, monkeypatch
):
    project = _project(tmp_path, "langgraph")

    class FakeDatabricksMCPServer:
        def __init__(self, name, url, workspace_client=None, **kwargs):
            self.name = name
            self.url = url
            self.workspace_client = workspace_client
            self.kwargs = kwargs

        @classmethod
        def from_uc_function(
            cls, catalog, schema, name, function_name=None, workspace_client=None, **kwargs
        ):
            client = workspace_client or _FakeWorkspaceClient()
            return cls(
                name=name,
                url=(
                    f"{client.config.host}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}"
                ),
                workspace_client=client,
                **kwargs,
            )

        def to_connection_dict(self):
            return {"transport": "streamable_http", "url": self.url}

    class FakeMultiServerClient:
        last = None

        def __init__(self, servers, **kwargs):
            self.servers = servers
            self.kwargs = kwargs
            FakeMultiServerClient.last = self

        async def get_tools(self):
            return [server.name for server in self.servers]

    class FakeSession:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return name, arguments, kwargs

    class FakeSessionContext:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            return False

    databricks = types.ModuleType("databricks")
    databricks_sdk = types.ModuleType("databricks.sdk")
    databricks_sdk.__dict__["WorkspaceClient"] = _FakeWorkspaceClient
    databricks_langchain = types.ModuleType("databricks_langchain")
    databricks_langchain.__dict__["DatabricksMCPServer"] = FakeDatabricksMCPServer
    databricks_langchain.__dict__["DatabricksMultiServerMCPClient"] = FakeMultiServerClient
    adapters = types.ModuleType("langchain_mcp_adapters")
    sessions = types.ModuleType("langchain_mcp_adapters.sessions")
    sessions.__dict__["create_session"] = lambda connection: FakeSessionContext()
    monkeypatch.setitem(sys.modules, "databricks", databricks)
    monkeypatch.setitem(sys.modules, "databricks.sdk", databricks_sdk)
    monkeypatch.setitem(sys.modules, "databricks_langchain", databricks_langchain)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", adapters)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.sessions", sessions)

    runtime = _load_runtime(project, monkeypatch)
    assert asyncio.run(runtime.mcp_tools()) == ["sandbox", "web", "lookup"]
    client = FakeMultiServerClient.last
    assert client is not None
    assert [server.url for server in client.servers] == [
        "https://df1.example.com/ai-gateway/mcp-services/system.ai.sandbox",
        "https://df1.example.com/ai-gateway/mcp-services/system.ai.web_search",
        "https://df1.example.com/api/2.0/mcp/functions/main/tools/lookup",
    ]
    assert client.kwargs["tool_interceptors"] == [runtime._sandbox_tool_interceptor]

    request = SimpleNamespace(
        server_name="sandbox",
        name="sandbox",
        args={"code": 'print("ok")'},
    )

    async def unexpected_handler(request):
        raise AssertionError("sandbox calls must use the fixed-meta session")

    name, arguments, kwargs = asyncio.run(
        runtime._sandbox_tool_interceptor(request, unexpected_handler)
    )
    assert name == "sandbox"
    assert arguments == {"code": 'print("ok")'}
    assert kwargs["meta"] == {
        "downscope": {"tables": [{"name": "samples.nyctaxi.trips", "permission": "read_only"}]}
    }


def test_manifest_reader_rejects_wrong_framework(tmp_path: pathlib.Path, monkeypatch):
    project = _project(tmp_path, "openai")
    (project / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "langgraph"\n', encoding="utf-8"
    )
    _clear_agent_modules()
    monkeypatch.syspath_prepend(str(project))
    monkeypatch.setenv("MASON_PROJECT_ROOT", str(project))
    manifest = importlib.import_module("agent.mason.tool_manifest")

    with pytest.raises(RuntimeError, match="framework"):
        manifest.load_tools(expected_framework="openai")
