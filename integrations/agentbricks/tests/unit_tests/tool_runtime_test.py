"""Behavior tests for the manifest-driven MCP runtime (databricks_agentkit.langgraph.mcp)."""

from __future__ import annotations

import asyncio
import importlib
import logging
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest


def _write_direct_manifest(project: pathlib.Path) -> None:
    (project / "agent.toml").write_text(
        """schema_version = 1

[agent]
framework = "langgraph"
server = "agentbricks"

[[tools]]
id = "sandbox"
source = { kind = "sandbox", service = "system.ai.sandbox" }
policy = { downscope = [{ resource = "table:samples.nyctaxi.trips", permission = "read_only" }], include_databricks_token_env = true }

[[tools]]
id = "web"
source = { kind = "mcp", service = "system.ai.web_search" }

[[tools]]
id = "lookup"
source = { kind = "uc_function", function = "main.tools.lookup" }
""",
        encoding="utf-8",
    )


def _project(tmp_path: pathlib.Path) -> pathlib.Path:
    project = tmp_path / "langgraph"
    project.mkdir(parents=True)
    _write_direct_manifest(project)
    return project


class _FakeWorkspaceClient:
    def __init__(self):
        self.config = SimpleNamespace(host="https://df1.example.com")


def _reload_mcp():
    # The runtime modules read env at call time, but re-import so patched sys.modules take effect.
    for name in (
        "databricks_agentkit.langgraph.mcp",
        "databricks_agentkit.runtime.tool_manifest",
    ):
        sys.modules.pop(name, None)
    return importlib.import_module("databricks_agentkit.langgraph.mcp")


def _reload_tool_manifest():
    sys.modules.pop("databricks_agentkit.runtime.tool_manifest", None)
    return importlib.import_module("databricks_agentkit.runtime.tool_manifest")


def test_langgraph_runtime_loads_direct_manifest_and_protects_sandbox_meta(
    tmp_path: pathlib.Path, monkeypatch
):
    project = _project(tmp_path)

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

        async def get_tools(self, server_name=None):
            return [
                server.name
                for server in self.servers
                if server_name is None or server.name == server_name
            ]

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
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(project))

    mcp = _reload_mcp()
    monkeypatch.setattr(mcp, "workspace_client", _FakeWorkspaceClient)
    assert mcp.load_tools(expected_framework="langgraph")[0].include_databricks_token_env is True

    # _declared_servers() builds one server per manifest tool, with the right URLs.
    monkeypatch.setattr(mcp, "workspace_client", _FakeWorkspaceClient)
    servers = mcp._declared_servers()
    assert [s.url for s in servers] == [
        "https://df1.example.com/ai-gateway/mcp-services/system.ai.sandbox",
        "https://df1.example.com/ai-gateway/mcp-services/system.ai.web_search",
        "https://df1.example.com/api/2.0/mcp/functions/main/tools/lookup",
    ]

    # mcp_tools() includes the manifest servers and fetches from a client that carries the sandbox
    # interceptor (the manifest declares a sandbox tool).
    assert asyncio.run(mcp.mcp_tools()) == ["sandbox", "web", "lookup"]
    client = FakeMultiServerClient.last
    assert client is not None
    assert len(client.kwargs["tool_interceptors"]) == 1

    # the interceptor downscopes sandbox calls with the fixed meta from the manifest policy.
    interceptor = client.kwargs["tool_interceptors"][0]
    request = SimpleNamespace(
        server_name="sandbox",
        name="sandbox",
        args={"code": 'print("ok")'},
    )

    async def unexpected_handler(request):
        raise AssertionError("sandbox calls must use the fixed-meta session")

    name, arguments, kwargs = asyncio.run(interceptor(request, unexpected_handler))
    assert name == "sandbox"
    assert arguments == {"code": 'print("ok")'}
    assert kwargs["meta"] == {
        "downscope": {"tables": [{"name": "samples.nyctaxi.trips", "permission": "read_only"}]},
        "include_databricks_token_env": True,
    }


def test_runtime_manifest_defaults_legacy_sandbox_token_env_policy_to_false(
    tmp_path: pathlib.Path, monkeypatch
):
    project = tmp_path / "legacy"
    project.mkdir()
    _write_direct_manifest(project)
    manifest = project / "agent.toml"
    manifest.write_text(manifest.read_text().replace(", include_databricks_token_env = true", ""))
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(project))

    record = _reload_tool_manifest().load_tools(expected_framework="langgraph")[0]

    assert record.include_databricks_token_env is False


@pytest.mark.parametrize("value", ['"true"', "1", "[]"])
def test_runtime_manifest_rejects_non_boolean_sandbox_token_env_policy(
    tmp_path: pathlib.Path, monkeypatch, value
):
    project = tmp_path / "invalid"
    project.mkdir()
    _write_direct_manifest(project)
    manifest = project / "agent.toml"
    manifest.write_text(manifest.read_text().replace("true }", f"{value} }}"))
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(project))

    with pytest.raises(RuntimeError, match="include_databricks_token_env"):
        _reload_tool_manifest().load_tools(expected_framework="langgraph")


@pytest.fixture
def mcp_discovery(tmp_path: pathlib.Path, monkeypatch):
    project = tmp_path / "langgraph"
    project.mkdir(parents=True)
    (project / "agent.toml").write_text(
        """schema_version = 1

[agent]
framework = "langgraph"
server = "agentbricks"

[[tools]]
id = "alpha"
source = { kind = "mcp", service = "system.ai.alpha" }

[[tools]]
id = "bad"
source = { kind = "mcp", service = "system.ai.bad" }

[[tools]]
id = "gamma"
source = { kind = "mcp", service = "system.ai.gamma" }
""",
        encoding="utf-8",
    )

    class FakeDatabricksMCPServer:
        def __init__(self, name, url, workspace_client=None, **kwargs):
            self.name = name
            self.url = url
            self.workspace_client = workspace_client

        def to_connection_dict(self):
            return {"transport": "streamable_http", "url": self.url}

    class FakeMultiServerClient:
        inflight = 0
        max_inflight = 0

        def __init__(self, servers, **kwargs):
            self.servers = servers
            self.kwargs = kwargs

        async def get_tools(self, server_name=None):
            FakeMultiServerClient.inflight += 1
            FakeMultiServerClient.max_inflight = max(
                FakeMultiServerClient.max_inflight, FakeMultiServerClient.inflight
            )
            try:
                await asyncio.sleep(0)  # yield so peers start -> proves concurrent fetch
                if server_name == "bad":
                    raise RuntimeError("401 Unauthorized Bearer must-not-be-logged")
                return [f"{server_name}-tool"]
            finally:
                FakeMultiServerClient.inflight -= 1

    databricks_langchain = types.ModuleType("databricks_langchain")
    databricks_langchain.__dict__["DatabricksMCPServer"] = FakeDatabricksMCPServer
    databricks_langchain.__dict__["DatabricksMultiServerMCPClient"] = FakeMultiServerClient
    adapters = types.ModuleType("langchain_mcp_adapters")
    sessions = types.ModuleType("langchain_mcp_adapters.sessions")
    sessions.__dict__["create_session"] = lambda connection: None
    monkeypatch.setitem(sys.modules, "databricks_langchain", databricks_langchain)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters", adapters)
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.sessions", sessions)
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(project))

    mcp = _reload_mcp()
    monkeypatch.setattr(mcp, "workspace_client", _FakeWorkspaceClient)
    return project, mcp, FakeDatabricksMCPServer, FakeMultiServerClient


@pytest.mark.parametrize(
    "auth,optional,strict",
    [(None, False, False), ("user", False, True), ("app", False, False), (None, True, False)],
)
def test_mcp_tools_isolates_legacy_and_app_but_rejects_user_auth_failures(
    mcp_discovery, monkeypatch, caplog, auth, optional, strict
):
    project, mcp, FakeDatabricksMCPServer, FakeMultiServerClient = mcp_discovery
    if auth is not None:
        manifest = project / "agent.toml"
        manifest.write_text(
            manifest.read_text().replace('id = "bad"', f'id = "bad"\nauth = "{auth}"')
        )

    extra_servers = None
    if optional:
        monkeypatch.setattr(mcp, "load_tools", lambda **kwargs: [])
        extra_servers = [
            FakeDatabricksMCPServer(name, "https://workspace") for name in ("alpha", "bad", "gamma")
        ]

    with caplog.at_level(logging.WARNING):
        if strict:
            from databricks_agentkit.runtime.auth import AuthError

            with pytest.raises(AuthError):
                asyncio.run(mcp.mcp_tools(workspace_client_for=lambda mode: _FakeWorkspaceClient()))
            assert FakeMultiServerClient.inflight == 0
            assert "must-not-be-logged" not in caplog.text
            return
        tools = asyncio.run(
            mcp.mcp_tools(
                extra_servers,
                **(
                    {"workspace_client_for": lambda mode: _FakeWorkspaceClient()}
                    if auth is not None
                    else {}
                ),
            )
        )

    # the failing server drops only its own tools; the other two survive, order preserved.
    assert tools == ["alpha-tool", "gamma-tool"]
    # all three servers were fetched concurrently rather than serially.
    assert FakeMultiServerClient.max_inflight == 3
    failures = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "server 'bad'" in r.getMessage()
    ]
    assert len(failures) == 1
    assert failures[0].exc_info is None
    assert "must-not-be-logged" not in caplog.text


def test_mcp_tools_waits_for_blocked_sibling_before_user_auth_failure(mcp_discovery, monkeypatch):
    from databricks_agentkit.runtime.auth import AuthError

    project, mcp, _, client_type = mcp_discovery
    manifest = project / "agent.toml"
    manifest.write_text(manifest.read_text().replace('id = "bad"', 'id = "bad"\nauth = "user"'))

    async def exercise():
        sibling_started = asyncio.Event()
        failure_raised = asyncio.Event()
        release_sibling = asyncio.Event()
        completed = set()

        async def get_tools(self, server_name=None):
            if server_name == "bad":
                await sibling_started.wait()
                failure_raised.set()
                raise PermissionError("Bearer must-not-be-exposed")
            if server_name == "gamma":
                sibling_started.set()
                await release_sibling.wait()
            completed.add(server_name)
            return [f"{server_name}-tool"]

        monkeypatch.setattr(client_type, "get_tools", get_tools)
        discovery = asyncio.create_task(
            mcp.mcp_tools(workspace_client_for=lambda mode: _FakeWorkspaceClient())
        )
        try:
            await failure_raised.wait()
            done, _ = await asyncio.wait({discovery}, timeout=0.01)
            assert not done
            assert "gamma" not in completed
            release_sibling.set()
            with pytest.raises(AuthError) as raised:
                await discovery
            assert completed == {"alpha", "gamma"}
            assert raised.value.integration_id == "bad"
            assert raised.value.code == "MCP_PERMISSION_DENIED"
            assert "must-not-be-exposed" not in str(raised.value)
        finally:
            release_sibling.set()
            discovery.cancel()
            await asyncio.gather(discovery, return_exceptions=True)

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


def test_mcp_tools_isolates_mixed_declared_and_optional_servers(mcp_discovery, monkeypatch, caplog):
    project, mcp, server_type, client_type = mcp_discovery
    manifest = project / "agent.toml"
    manifest.write_text(
        manifest.read_text()
        .replace('id = "alpha"', 'id = "alpha"\nauth = "user"')
        .replace('id = "gamma"', 'id = "gamma"\nauth = "app"')
    )
    user_client = _FakeWorkspaceClient()
    app_client = _FakeWorkspaceClient()
    customer_client = object()
    resolved_modes = []
    discovered_clients = {}
    extra_servers = [
        server_type(name, "https://customer.example/mcp", workspace_client=customer_client)
        for name in ("alpha", "bad", "gamma")
    ]

    def resolve(mode):
        resolved_modes.append(mode)
        return user_client if mode == "user" else app_client

    async def exercise():
        started = {"declared": set(), "optional": set()}
        ready = {kind: asyncio.Event() for kind in started}

        async def get_tools(self, server_name=None):
            server = next(server for server in self.servers if server.name == server_name)
            kind = "optional" if server.workspace_client is customer_client else "declared"
            discovered_clients[kind, server_name] = server.workspace_client
            started[kind].add(server_name)
            if len(started[kind]) == 3:
                ready[kind].set()
            await ready[kind].wait()
            if server_name == "bad":
                raise RuntimeError("Bearer must-not-be-logged")
            return [f"{kind}-{server_name}-tool"]

        monkeypatch.setattr(client_type, "get_tools", get_tools)
        tools = await mcp.mcp_tools(extra_servers, workspace_client_for=resolve)
        assert tools == [
            "declared-alpha-tool",
            "declared-gamma-tool",
            "optional-alpha-tool",
            "optional-gamma-tool",
        ]

    with caplog.at_level(logging.WARNING):
        asyncio.run(asyncio.wait_for(exercise(), timeout=5))
    assert resolved_modes == ["user", "app", "app"]
    assert discovered_clients == {
        ("declared", "alpha"): user_client,
        ("declared", "bad"): app_client,
        ("declared", "gamma"): app_client,
        ("optional", "alpha"): customer_client,
        ("optional", "bad"): customer_client,
        ("optional", "gamma"): customer_client,
    }
    assert all(server.workspace_client is customer_client for server in extra_servers)
    assert all(server.url == "https://customer.example/mcp" for server in extra_servers)
    failures = [record for record in caplog.records if "server 'bad'" in record.getMessage()]
    assert len(failures) == 2
    assert all(record.exc_info is None for record in failures)
    assert "must-not-be-logged" not in caplog.text


@pytest.mark.parametrize("phase", ["declared", "optional"])
def test_mcp_tools_caller_cancellation_drains_discovery_tasks(mcp_discovery, monkeypatch, phase):
    project, mcp, server_type, client_type = mcp_discovery
    manifest = project / "agent.toml"
    manifest.write_text(manifest.read_text().replace("source =", 'auth = "user"\nsource ='))
    extra_servers = [
        server_type(name, "https://customer.example/mcp")
        for name in ("optional-alpha", "optional-beta")
    ]
    expected = (
        {"alpha", "bad", "gamma"} if phase == "declared" else {"optional-alpha", "optional-beta"}
    )

    async def exercise():
        all_started = asyncio.Event()
        all_cleaning = asyncio.Event()
        release_cleanup = asyncio.Event()
        blocked = asyncio.Event()
        tasks = {}
        cleaning = set()
        finished = set()

        async def get_tools(self, server_name=None):
            if server_name not in expected:
                assert phase == "optional"
                return [f"{server_name}-tool"]
            tasks[server_name] = asyncio.current_task()
            if tasks.keys() == expected:
                all_started.set()
            try:
                await blocked.wait()
            finally:
                cleaning.add(server_name)
                if cleaning == expected:
                    all_cleaning.set()
                await release_cleanup.wait()
                finished.add(server_name)

        monkeypatch.setattr(client_type, "get_tools", get_tools)
        discovery = asyncio.create_task(
            mcp.mcp_tools(extra_servers, workspace_client_for=lambda mode: _FakeWorkspaceClient())
        )
        try:
            await all_started.wait()
            discovery.cancel()
            await all_cleaning.wait()
            done, _ = await asyncio.wait({discovery}, timeout=0.01)
            assert not done
            assert not finished
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await discovery
            assert finished == expected
            assert all(task.done() and task.cancelled() for task in tasks.values())
        finally:
            release_cleanup.set()
            discovery.cancel()
            for task in tasks.values():
                task.cancel()
            await asyncio.gather(discovery, *tasks.values(), return_exceptions=True)

    asyncio.run(asyncio.wait_for(exercise(), timeout=5))


def test_manifest_reader_rejects_wrong_framework(tmp_path: pathlib.Path, monkeypatch):
    project = _project(tmp_path)
    (project / "agent.toml").write_text(
        'schema_version = 1\n\n[agent]\nframework = "openai"\nserver = "agentbricks"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(project))
    sys.modules.pop("databricks_agentkit.runtime.tool_manifest", None)
    manifest = importlib.import_module("databricks_agentkit.runtime.tool_manifest")

    with pytest.raises(RuntimeError, match="framework"):
        manifest.load_tools(expected_framework="langgraph")


def test_manifest_reader_rejects_python_tool_entries_with_code_first_migration(
    tmp_path: pathlib.Path, monkeypatch
):
    project = _project(tmp_path)
    (project / "agent.toml").write_text(
        """schema_version = 1

[agent]
framework = "langgraph"
server = "agentbricks"

[[tools]]
id = "lookup-ticket"
source = { kind = "python", entrypoint = "agent.tools.lookup_ticket:lookup_ticket" }
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(project))
    sys.modules.pop("databricks_agentkit.runtime.tool_manifest", None)
    manifest = importlib.import_module("databricks_agentkit.runtime.tool_manifest")

    with pytest.raises(RuntimeError, match="Python tools are code-first") as error:
        manifest.load_tools(expected_framework="langgraph")

    assert "Remove this entry" in str(error.value)
    assert "agent/tools" in str(error.value)


def test_openai_adapter_propagates_manifest_validation_errors(tmp_path: pathlib.Path, monkeypatch):
    project = _project(tmp_path)
    (project / "agent.toml").write_text(
        """schema_version = 1

[agent]
framework = "openai"
server = "agentbricks"

[[tools]]
id = "lookup-ticket"
source = { kind = "python", entrypoint = "agent.tools.lookup_ticket:lookup_ticket" }
""",
        encoding="utf-8",
    )

    databricks_openai = types.ModuleType("databricks_openai")
    databricks_openai.__path__ = []
    databricks_openai_agents = types.ModuleType("databricks_openai.agents")
    databricks_openai_agents.__dict__["McpServer"] = type("McpServer", (), {})
    monkeypatch.setitem(sys.modules, "databricks_openai", databricks_openai)
    monkeypatch.setitem(sys.modules, "databricks_openai.agents", databricks_openai_agents)
    monkeypatch.setenv("AGENTBRICKS_PROJECT_ROOT", str(project))
    for name in ("databricks_agentkit.openai.mcp", "databricks_agentkit.runtime.tool_manifest"):
        sys.modules.pop(name, None)
    mcp = importlib.import_module("databricks_agentkit.openai.mcp")

    with pytest.raises(RuntimeError, match="Python tools are code-first"):
        asyncio.run(mcp.mcp_servers())
