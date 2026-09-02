from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from databricks_mason.integrations import MCPService, Sandbox, Scope, UCFunction
from databricks_mason.langgraph import load_tools
from databricks_mason.langgraph import mcp as mcp_module


class _Server:
    def __init__(self, name: str, url: str, **kwargs) -> None:
        self.name = name
        self.url = url
        self.kwargs = kwargs

    @classmethod
    def from_uc_function(
        cls,
        catalog: str,
        schema: str,
        function_name: str,
        name: str,
        workspace_client,
        **kwargs,
    ):
        return cls(
            name,
            f"{workspace_client.config.host}/api/2.0/mcp/functions/{catalog}/{schema}/{function_name}",
            **kwargs,
        )

    def to_connection_dict(self):
        return {"transport": "streamable_http", "url": self.url}


class _Client:
    last: _Client | None = None

    def __init__(self, servers, **kwargs) -> None:
        self.servers = list(servers)
        self.kwargs = kwargs
        _Client.last = self

    async def get_tools(self):
        return [server.name for server in self.servers]


@pytest.fixture
def runtime(monkeypatch):
    client = SimpleNamespace(config=SimpleNamespace(host="https://workspace.example.com"))
    monkeypatch.setattr(mcp_module, "DatabricksMCPServer", _Server)
    monkeypatch.setattr(mcp_module, "DatabricksMultiServerMCPClient", _Client)
    monkeypatch.setattr(mcp_module, "_default_workspace_client", lambda: client)
    monkeypatch.setattr(mcp_module, "workspace_headers", lambda: {})
    return client


def test_load_tools_materializes_only_explicit_integrations_and_extra_servers(runtime) -> None:
    custom = _Server("custom", "https://custom.example.com/mcp")

    tools = asyncio.run(
        load_tools(
            [
                Sandbox(id="sandbox", scopes=(Scope.table("samples.nyctaxi.trips"),)),
                MCPService(id="web", service="system.ai.web_search"),
                UCFunction(id="lookup", function="main.tools.lookup"),
            ],
            extra_servers=cast(Any, [custom]),
        )
    )

    assert tools == ["sandbox", "web", "lookup", "custom"]
    client = cast(Any, _Client.last)
    assert [server.url for server in client.servers] == [
        "https://workspace.example.com/ai-gateway/mcp-services/system.ai.sandbox",
        "https://workspace.example.com/ai-gateway/mcp-services/system.ai.web_search",
        "https://workspace.example.com/api/2.0/mcp/functions/main/tools/lookup",
        "https://custom.example.com/mcp",
    ]


def test_sandbox_interceptor_closes_over_selection_and_protects_downscope(
    runtime, monkeypatch
) -> None:
    call = {}

    class _Session:
        async def initialize(self):
            call["initialized"] = True

        async def call_tool(self, name, arguments, **kwargs):
            call.update(name=name, arguments=arguments, kwargs=kwargs)
            return "ok"

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(mcp_module, "create_session", lambda connection: _SessionContext())
    asyncio.run(
        mcp_module.load_tools([Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),))])
    )
    client = cast(Any, _Client.last)
    interceptor = client.kwargs["tool_interceptors"][0]
    request = SimpleNamespace(
        server_name="sandbox",
        name="sandbox",
        args={"code": "print('ok')", "downscope": "model-controlled"},
    )

    result = asyncio.run(interceptor(request, lambda request: None))

    assert result == "ok"
    assert call["arguments"] is request.args
    assert call["kwargs"] == {
        "meta": {"downscope": {"volumes": [{"name": "main.data.files", "permission": "read_only"}]}}
    }


def test_request_workspace_client_is_reused_by_sandbox_interceptor(runtime, monkeypatch) -> None:
    request_client = SimpleNamespace(
        config=SimpleNamespace(host="https://request-workspace.example.com")
    )
    monkeypatch.setattr(
        mcp_module,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(AssertionError("must use the request client")),
    )

    class _Session:
        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, **kwargs):
            return "ok"

    class _SessionContext:
        async def __aenter__(self):
            return _Session()

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(mcp_module, "create_session", lambda connection: _SessionContext())

    asyncio.run(
        mcp_module.load_tools(
            [Sandbox(id="sandbox", scopes=(Scope.volume("main.data.files"),))],
            workspace_client=cast(Any, request_client),
        )
    )
    client = cast(Any, _Client.last)
    server = client.servers[0]
    interceptor = client.kwargs["tool_interceptors"][0]

    assert server.kwargs["workspace_client"] is request_client
    assert (
        asyncio.run(
            interceptor(
                SimpleNamespace(server_name="sandbox", name="sandbox", args={"code": "1"}),
                lambda request: None,
            )
        )
        == "ok"
    )


def test_empty_selection_returns_without_constructing_a_client(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_module,
        "_default_workspace_client",
        lambda: (_ for _ in ()).throw(AssertionError("must not resolve auth")),
    )

    assert asyncio.run(mcp_module.load_tools([])) == []


def test_configured_integration_discovery_failure_is_not_silently_dropped(
    runtime, monkeypatch
) -> None:
    async def fail(self):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(_Client, "get_tools", fail)

    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(mcp_module.load_tools([MCPService(id="web", service="system.ai.web_search")]))


def test_load_tools_rejects_server_name_collisions_before_attaching_sandbox_policy(runtime) -> None:
    custom = _Server("shared", "https://custom.example.com/mcp")
    _Client.last = None

    with pytest.raises(ValueError, match="unique.*shared"):
        asyncio.run(
            mcp_module.load_tools(
                [
                    Sandbox(
                        id="shared",
                        scopes=(Scope.volume("main.data.files"),),
                    )
                ],
                extra_servers=cast(Any, [custom]),
            )
        )

    assert _Client.last is None


def test_direct_mcp_client_rejects_duplicate_server_names(runtime) -> None:
    with pytest.raises(ValueError, match="server names.*shared"):
        mcp_module.mcp_client(
            cast(
                Any,
                [
                    _Server("shared", "https://one.example.com/mcp"),
                    _Server("shared", "https://two.example.com/mcp"),
                ],
            )
        )


def test_load_tools_rejects_duplicate_advertised_tool_names(runtime, monkeypatch) -> None:
    async def duplicate_tools(self):
        return [
            SimpleNamespace(name="lookup"),
            SimpleNamespace(name="lookup"),
        ]

    monkeypatch.setattr(_Client, "get_tools", duplicate_tools)

    with pytest.raises(ValueError, match="tool names.*lookup"):
        asyncio.run(
            mcp_module.load_tools(
                [
                    MCPService(id="first", service="main.tools.first"),
                    MCPService(id="second", service="main.tools.second"),
                ]
            )
        )


def test_load_tools_rejects_remote_name_that_collides_with_existing_agent_tool(
    runtime, monkeypatch
) -> None:
    async def remote_tools(self):
        return [SimpleNamespace(name="lookup")]

    monkeypatch.setattr(_Client, "get_tools", remote_tools)

    with pytest.raises(ValueError, match="tool names.*lookup"):
        asyncio.run(
            mcp_module.load_tools(
                [MCPService(id="remote", service="main.tools.remote")],
                existing_tools=[SimpleNamespace(name="lookup")],
            )
        )


@pytest.mark.parametrize(
    "extra_servers",
    [None, [_Server("custom", "https://custom.example.com/mcp")]],
)
def test_legacy_mcp_tools_call_fails_with_migration_guidance(runtime, extra_servers) -> None:
    with pytest.raises(RuntimeError, match=r"mcp_tools\(\).*agent\.toml.*load_tools"):
        asyncio.run(mcp_module.mcp_tools(extra_servers))
